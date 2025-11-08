import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import numpy as np

from .base_trainer import BaseTrainer
from .memory_bank import MemoryBank
from .utils import coerce_config_scalars, save_checkpoint
from .id_loss import InstanceDiscriminationLoss as IDLoss


class InstanceDiscriminationTrainer(BaseTrainer):
    """
    Instance Discrimination (self-supervised) training.

    Notes:
    - Requires a memory bank of embeddings (size = num_train_samples, dim = embedding_dim).
    - The loss produces updated entries for the current batch to write back into the memory bank.
    - Evaluation measures nearest-neighbour accuracy in the embedding space against the memory bank.

    Expected loss API (InstanceDiscriminationLoss):
      forward(outputs, indices) -> (loss, entries_to_update, data_loss, noise_loss)
      trainable_parameters() -> parameters of the projection head
      update_memory_bank(memory_bank_tensor)
    """

    def __init__(self, *args, **kwargs):
        # Keep a copy of optimizer configuration to rebuild with loss params later
        self._optimizer_name = kwargs.get("optimizer", "AdamW")
        self._optimizer_params = dict(kwargs.get("optimizer_params", {}) or {})
        super().__init__(*args, **kwargs)

        # Make backbone output high-dimensional features (e.g., remove final classifier layer)
        self._adapt_model_for_ir()

        # Initialize memory bank (requires embedding_dim from loss params)
        self._init_memory_bank()
        # Attach real memory bank to loss
        self._finalize_loss_with_memory_bank()
        # Precompute ordered training labels aligned with memory bank indices
        self._init_training_ordered_labels()

        # If loss exposes extra trainable parameters (e.g., projection head),
        # rebuild optimizer immediately to include them from the start.
        if hasattr(self, "loss") and hasattr(self.loss, "trainable_parameters"):
            try:
                self._rebuild_optimizer_with_loss_params()
            except Exception:
                pass

    # Override to build the custom loss instead of a torch.nn loss
    def _build_loss(self, loss: str, loss_params):
        # Build loss with a temporary 1-row memory bank; will be replaced after loaders exist
        params = coerce_config_scalars(loss_params or {})
        model_output_dim = int(params.get("model_output_dim", 4096))
        embedding_dim = int(params.get("embedding_dim", 128))
        mode = str(params.get("mode", "dynamic"))
        tmp_mem = torch.zeros(1, embedding_dim, dtype=torch.float32)
        
        return IDLoss(tmp_mem, model_output_dim,
                      m=int(params.get("m", 4096)),
                      gamma=float(params.get("gamma", 0.5)),
                      tau=float(params.get("tau", 0.07)),
                      embedding_dim=embedding_dim,
                      z_const=float(params.get("z_const", 1232.82)),
                      mode=mode)

    def train_epoch(self, epoch: int):
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", leave=False)

        mem_update = self.memory_bank.update
        get_bank = self.memory_bank.get_memory_bank
        loss_fn = self.loss
        model = self.model
        device = self.device
        scaler = self.scaler
        use_cuda = (device.type == 'cuda')

        for indices, images, _ in progress_bar:
            indices = indices.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            if use_cuda:
                images = images.contiguous(memory_format=torch.channels_last)

            self.optimizer.zero_grad(set_to_none=True)

            # No autocast for self-supervised training (match Nayebi dynamics)
            with autocast(device_type=device.type, enabled=False):
                outputs = model(images)
                loss, entries_to_update, data_loss, noise_loss = loss_fn(outputs, indices)

            # Disable GradScaler: straight fp32 backward/step, no clipping
            loss.backward()
            
            all_params = list(self.model.parameters())
            if hasattr(loss_fn, 'trainable_parameters'):
                all_params.extend(list(loss_fn.trainable_parameters()))
            
            # Apply the clipping to all parameters
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            
            self.optimizer.step()

            with torch.no_grad():
                mem_update(indices, entries_to_update)
                if hasattr(loss_fn, "update_memory_bank"):
                    loss_fn.update_memory_bank(get_bank())

            total_loss += float(loss.detach())
            progress_bar.set_postfix({"Loss": f"{float(loss.detach()):.4f}"})

        epoch_loss = total_loss / max(1, len(self.train_loader))
        return epoch_loss, 0.0
    
    def evaluate(self, loader: DataLoader, epoch: int, mode: str = "Val"):
        assert hasattr(self, "training_ordered_labels"), "training_ordered_labels not initialized"
        assert hasattr(self, "memory_bank"), "memory_bank not initialized"

        self.model.eval()
        if hasattr(self.loss, "eval"):
            self.loss.eval()

        total_correct = 0
        total_samples = 0

        desc = f"Testing" if mode == "Test" else f"Validating epoch {epoch+1}"

        get_all_ip = self.memory_bank.get_all_inner_products
        embed_fn = getattr(self.loss, "_embedding_func", None)
        if embed_fn is None:
            raise RuntimeError("Loss does not expose _embedding_func needed for evaluation")

        device = self.device
        use_cuda = (device.type == 'cuda')

        with torch.no_grad():
            progress_bar = tqdm(loader, desc=desc, leave=False)
            for batch in progress_bar:
                # Unpack batch robustly: (idx, images, labels) or (images, labels) or dict
                batch_indices = None
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        batch_indices, images, labels = batch
                    elif len(batch) == 2:
                        images, labels = batch
                    else:
                        raise ValueError(f"Unexpected batch structure with length {len(batch)}")
                elif isinstance(batch, dict):
                    images = batch.get('imgs') or batch.get('image')
                    labels = batch.get('lbls') or batch.get('label')
                else:
                    raise ValueError(f"Unsupported batch type: {type(batch)}")

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if batch_indices is not None:
                    batch_indices = batch_indices.to(device, non_blocking=True).long()
                if use_cuda:
                    images = images.contiguous(memory_format=torch.channels_last)

                with autocast(device_type=device.type, enabled=False):
                    outputs = self.model(images)
                    emb = embed_fn(outputs)
                emb = emb.float()
                emb = emb / torch.sqrt(torch.sum(emb ** 2, dim=1, keepdim=True)).clamp_min(1e-12)

                all_inner = get_all_ip(emb)

                # If evaluating on the training loader that yields indices, exclude self-match
                if batch_indices is not None and all_inner.shape[1] >= batch_indices.max().item() + 1:
                    all_inner[torch.arange(all_inner.size(0), device=all_inner.device), batch_indices] = float('-inf')

                _, nearest_idx = torch.topk(all_inner, k=1, dim=1)
                nearest_idx = nearest_idx.squeeze(1)

                nn_labels = self.training_ordered_labels[nearest_idx.detach().cpu().numpy()]
                nn_labels = torch.from_numpy(nn_labels).to(labels.device)

                total_correct += torch.eq(nn_labels, labels).sum().item()
                total_samples += labels.size(0)

                acc = 100.0 * total_correct / max(1, total_samples)
                progress_bar.set_postfix({"Acc": f"{acc:.2f}%"})

        epoch_acc = 100.0 * total_correct / max(1, total_samples)
        return 0.0, epoch_acc

    # ----------------------- helpers -----------------------
    def _init_memory_bank(self) -> None:
        assert isinstance(self.loss_params, dict), "loss_params must be provided for IR"
        params = coerce_config_scalars(self.loss_params)
        embedding_dim = int(params.get("embedding_dim", 128))
        num_entries = len(self.train_loader.dataset)
        # If IndexedDataset wraps the underlying dataset, its length matches train set size
        self.memory_bank = MemoryBank(num_entries=num_entries, embedding_dim=embedding_dim, device=self.device)

    def _finalize_loss_with_memory_bank(self) -> None:
        # Replace placeholder memory bank in the already-built loss with the real one
        if hasattr(self.loss, "update_memory_bank"):
            real_bank = self.memory_bank.get_memory_bank()
            # Ensure device/dtype alignment
            if real_bank.device != next(self.loss.parameters()).device:
                self.loss.to(real_bank.device)
            self.loss.update_memory_bank(real_bank)

    def _init_training_ordered_labels(self) -> None:
        ds = self.train_loader.dataset
        # Unwrap IndexedDataset if present
        underlying = getattr(ds, "dataset", ds)
        labels_np: np.ndarray
        try:
            from torch.utils.data import Subset
            from torchvision.datasets import ImageFolder
            if isinstance(underlying, Subset) and isinstance(underlying.dataset, ImageFolder):
                base: ImageFolder = underlying.dataset
                idxs = underlying.indices
                labels_np = np.array([int(base.targets[i]) for i in idxs], dtype=np.int64)
            elif hasattr(underlying, "base_dataset") and hasattr(underlying.base_dataset, "files"):
                base = underlying.base_dataset  # e.g., MiniImageNet
                labels_np = np.array([int(lbl) for _, lbl in getattr(base, "files")], dtype=np.int64)
            else:
                # Fallback: enumerate dataset once (may decode images)
                labels = []
                for i in range(len(underlying)):
                    item = underlying[i]
                    if isinstance(item, tuple) and len(item) >= 2:
                        labels.append(int(item[1]))
                    elif isinstance(item, dict) and ("lbls" in item or "label" in item):
                        labels.append(int(item.get("lbls", item.get("label"))))
                    else:
                        labels.append(-1)
                labels_np = np.array(labels, dtype=np.int64)
        except Exception:
            # Robust fallback
            labels_np = np.zeros(len(underlying), dtype=np.int64)

        self.training_ordered_labels = labels_np

    def _rebuild_optimizer_with_loss_params(self) -> None:
        from torch import optim as _optim
        opt_name = self._optimizer_name if isinstance(self._optimizer_name, str) else "AdamW"
        opt_params = coerce_config_scalars(dict(self._optimizer_params)) if isinstance(self._optimizer_params, dict) else {}
        opt_class = getattr(_optim, opt_name, None)
        if opt_class is None:
            opt_class = self.optimizer.__class__
        combined_params = list(self.model.parameters())
        try:
            combined_params += list(self.loss.trainable_parameters())
        except Exception:
            pass
        self.optimizer = opt_class(combined_params, **opt_params)

    def _adapt_model_for_ir(self) -> None:
        """Modify the model to output backbone features instead of class logits.

        For torchvision AlexNet, drop the final Linear to num_classes.
        Safe no-op for models without a classifier Sequential.
        """
        m = self.model
        try:
            if hasattr(m, "classifier") and isinstance(m.classifier, torch.nn.Sequential):
                modules = list(m.classifier.children())
                if len(modules) >= 1 and isinstance(modules[-1], torch.nn.Linear):
                    # Remove the last classification layer
                    m.classifier = torch.nn.Sequential(*modules[:-1])
        except Exception:
            pass

    # ------------------- warmup -------------------
    def _warmup_memory_bank(self, warmup_epochs: int) -> None:
        if warmup_epochs <= 0:
            return
        self.model.eval()
        if hasattr(self.loss, "eval"):
            self.loss.eval()
        device = self.device
        use_cuda = (device.type == 'cuda')
        embed_fn = getattr(self.loss, "_embedding_func", None)
        if embed_fn is None:
            return
        mem_update = self.memory_bank.update
        get_bank = self.memory_bank.get_memory_bank

        for we in range(int(warmup_epochs)):
            with torch.no_grad():
                progress_bar = tqdm(self.train_loader, desc=f"Warmup {we+1}/{warmup_epochs}", leave=False)
                for indices, images, _ in progress_bar:
                    indices = indices.to(device, non_blocking=True)
                    images = images.to(device, non_blocking=True)
                    if use_cuda:
                        images = images.contiguous(memory_format=torch.channels_last)
                    with autocast(device_type=device.type, enabled=False):
                        outputs = self.model(images)
                        emb = embed_fn(outputs)
                    emb = emb.float()
                    emb = emb / torch.sqrt(torch.sum(emb ** 2, dim=1, keepdim=True)).clamp_min(1e-12)
                    mem_update(indices, emb)
                    if hasattr(self.loss, "update_memory_bank"):
                        self.loss.update_memory_bank(get_bank())

    # override to insert warmup before training loop
    def train(self):
        # optional warmup epochs from loss params (default 1)
        try:
            warmup_epochs = int(coerce_config_scalars(self.loss_params).get("warmup_epochs", 1))
        except Exception:
            warmup_epochs = 1
        if getattr(self, "start_epoch", 0) == 0:
            self._warmup_memory_bank(warmup_epochs)

        # Train without early stopping (match Nayebi)
        best_val_acc = 0.0

        # Reinitialize scheduler fresh for this training run
        self.scheduler = self._build_scheduler(self._scheduler_name, self._scheduler_params_template)

        if self.start_epoch > 0:
            self.logger.info(f"Resuming training from epoch {self.start_epoch}.")

        save_every_n = self.save_every_n

        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            self.logger.info("-" * 10)

            # Train one epoch
            train_loss, _ = self.train_epoch(epoch)

            # Evaluate KNN accuracy only on validation set (skip train kNN)
            val_loss, val_acc = self.evaluate(self.val_loader, epoch, mode="Val")

            # Step scheduler (maximize accuracy if using ReduceLROnPlateau)
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(-val_acc)
            else:
                self.scheduler.step()

            # Record history (val loss is undefined in IR; store 0.0 for plotting continuity)
            self.train_losses.append(train_loss)
            self.val_losses.append(0.0)
            # Placeholder to keep history aligned with plots
            self.train_accuracies.append(0.0)
            self.val_accuracies.append(val_acc)

            # Logging
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {0.0000:.4f}, Val Acc: {val_acc:.2f}%")
            self.logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Track best, but do not early stop
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=0.0,
                    accuracy=val_acc,
                    history={
                        "train_losses": self.train_losses,
                        "val_losses": self.val_losses,
                        "train_accuracies": self.train_accuracies,
                        "val_accuracies": self.val_accuracies,
                    },
                    name="best_model.pth",
                    logger=self.logger,
                    msg=f"Saved best model",
                )

            # Milestone checkpoints
            if (epoch + 1) % save_every_n == 0:
                save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=0.0,
                    accuracy=val_acc,
                    history={
                        "train_losses": self.train_losses,
                        "val_losses": self.val_losses,
                        "train_accuracies": self.train_accuracies,
                        "val_accuracies": self.val_accuracies,
                    },
                    name=f"checkpoint_epoch_{epoch+1}.pth",
                    logger=self.logger,
                    msg=f"Saved milestone checkpoint",
                )

            # No early stopping

            self.logger.info("-" * 10)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }


