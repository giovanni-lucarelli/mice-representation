import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import models
from typing import Optional, Dict, Any
import os
from pathlib import Path
from copy import deepcopy

# append src to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..datasets.DataManager import DataManager
from .utils import _get_run_logger, coerce_config_scalars, save_checkpoint


class BaseTrainer():
    def __init__(self,
                 model          : Optional[torch.nn.Module]             = None,
                 data_manager   : Optional[DataManager]                 = None,
                 num_epochs     : int                                   = 100,
                 dropout_rate   : float                                 = 0.3,
                 patience       : int                                   = 15,
                 log_file       : Optional[Path]                        = None,
                 checkpoint_dir : Optional[Path]                        = None,
                 artifacts_dir  : Optional[Path]                        = None,
                 use_cuda       : bool                                  = True,
                 save_every_n   : int                                   = 10,
                 optimizer      : Optional[optim.Optimizer] | str       = "AdamW",
                 optimizer_params: Optional[Dict[str, Any]]             = None,
                 scheduler      : str  = "MultiStepLR",
                 scheduler_params: Optional[dict]                       = None,
                 loss           : Optional[str]                         = "CrossEntropyLoss",
                 loss_params    : Optional[Dict[str, Any]]             = None,
             ):
        self.model          : Optional[torch.nn.Module]                 = model
        self.data_manager   : Optional[DataManager]                     = data_manager
        self.num_epochs     : int                                       = num_epochs
        self.dropout_rate   : float                                     = dropout_rate
        self.patience       : int                                       = patience
        self.save_every_n   : int                                       = save_every_n
        self.start_epoch    : int                                       = 0
        self.use_cuda       : bool                                      = use_cuda
        self.checkpoint_dir : Path                                      = (checkpoint_dir or Path(os.environ.get("CHECKPOINT_DIR_OVERRIDE", "checkpoints")).resolve())
        self.log_file       : Path                                      = (log_file or (self.checkpoint_dir / "train.log")).resolve()
        self.artifacts_dir  : Path                                      = (artifacts_dir or (self.checkpoint_dir / "artifacts")).resolve()
        self.scheduler      : str                                       = scheduler
        self.optimizer      : str                                       = optimizer
        self.loss           : Optional[str]                             = loss
        self.loss_params    : Optional[Dict[str, Any]]                  = loss_params
        # Cache originals so we can re-init per train() call
        self._scheduler_name = scheduler
        self._scheduler_params_template: Dict[str, Any] = deepcopy(scheduler_params) if isinstance(scheduler_params, dict) else {}

        # initialize file logger (file-only, no console output)
        self.logger, self.log_file_path = _get_run_logger(self.log_file)

        if model is None:
            if data_manager is None:
                raise ValueError("Data manager not provided")
            self.model = models.alexnet(weights=None, num_classes=self.data_manager.num_classes, dropout=self.dropout_rate)

        #? ------------ Initialize Device ------------ #
        self.device = torch.device("cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu")
        self.logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass

        # Build optimizer
        self.optimizer = self._build_optimizer(optimizer, optimizer_params)

        # Build loss
        self.loss = self._build_loss(loss, loss_params)

        if self.data_manager is None:
            raise ValueError("Data manager not provided")

        self.model.to(self.device)

        self.train_losses = []
        self.val_losses   = []

        self.train_accuracies = []
        self.val_accuracies   = []

        self.train_loader = self.data_manager.train_loader
        self.val_loader   = self.data_manager.val_loader
        self.test_loader  = self.data_manager.test_loader

        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')

    # Orchestration remains common across trainers
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0

        # Reinitialize scheduler fresh for this training run
        self.scheduler = self._build_scheduler(self._scheduler_name, self._scheduler_params_template)

        save_every_n = self.save_every_n

        if self.start_epoch > 0:
            self.logger.info(f"Resuming training from epoch {self.start_epoch}.")
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            self.logger.info("-" * 10)

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader, epoch)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            self.logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                save_checkpoint(epoch, val_loss, val_acc,
                                name = "best_model.pth",
                                msg = f"Saved best model")
            else:
                patience_counter += 1
                self.logger.info(f"Patience: {patience_counter}/{self.patience}")

            # Milestone checkpoints
            if (epoch + 1) % save_every_n == 0:
                save_checkpoint(epoch, val_loss, val_acc,
                                name = f"checkpoint_epoch_{epoch+1}.pth",
                                msg = f"Saved milestone checkpoint")

            # Stop if validation loss doesn't improve
            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

            self.logger.info("-" * 10)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies
        }

    def train_epoch(self, epoch: int):
        raise NotImplementedError

    def evaluate(self, loader: DataLoader, epoch: int, mode: str = "Val"):
        raise NotImplementedError

    def test(self):
        return self.evaluate(self.test_loader, self.num_epochs, "Test")

    def load_model(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            default_best = self.checkpoint_dir / "best_model.pth"
            print(f"Path {path} does not exist, checking {default_best}")
            self.logger.warning(f"Path {path} does not exist, checking {default_best}")
            if default_best.exists():
                path = default_best
            else:
                alt_path = self.checkpoint_dir / path.name
                if alt_path.exists():
                    path = alt_path
        if not path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        checkpoint = torch.load(path.as_posix(), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        try:
            saved_epoch = int(checkpoint.get('epoch', -1))
        except Exception:
            saved_epoch = -1
        self.start_epoch = max(0, saved_epoch + 1)
        history = checkpoint.get('history')
        if isinstance(history, dict):
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
            self.train_accuracies = history.get('train_accuracies', [])
            self.val_accuracies = history.get('val_accuracies', [])

        print(f"Model loaded from {path}")
        print(f"Best validation loss: {checkpoint['loss']:.4f}")
        print(f"Best validation accuracy: {checkpoint['accuracy']:.2f}%")
        self.logger.info(f"Model loaded from {path}")
        self.logger.info(f"Best validation loss: {checkpoint['loss']:.4f}")
        self.logger.info(f"Best validation accuracy: {checkpoint['accuracy']:.2f}%")
        self.logger.info(f"start_epoch set to {self.start_epoch}")

    def plot_training_history(self, show: bool = False, max_loss: Optional[float] = None, max_acc: Optional[int] = None, max_epoch: Optional[int] = None):
        import matplotlib.pyplot as plt
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        if max_loss is not None:
            plt.ylim(0, max_loss)
        if max_epoch is not None:
            plt.xlim(0, max_epoch)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b-', label='Train Acc')
        plt.plot(epochs, self.val_accuracies, 'r-', label='Val Acc')
        if max_acc is not None:
            plt.ylim(0, max_acc)
        if max_epoch is not None:
            plt.xlim(0, max_epoch)
        plt.title('Accuracy (Top-1)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        out_path = self.artifacts_dir / 'training_history.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        try:
            import csv
            csv_path = self.artifacts_dir / 'training_history.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
                for i, e in enumerate(epochs):
                    tl = self.train_losses[i] if i < len(self.train_losses) else ''
                    vl = self.val_losses[i] if i < len(self.val_losses) else ''
                    ta = self.train_accuracies[i] if i < len(self.train_accuracies) else ''
                    va = self.val_accuracies[i] if i < len(self.val_accuracies) else ''
                    writer.writerow([int(e), float(tl), float(vl), float(ta), float(va)])
        except Exception:
            pass

    #? ------------------------ Build Components -------------------------
    def _build_optimizer(self, optimizer: str, optimizer_params: Optional[Dict[str, Any]]) -> optim.Optimizer:
        optimizer_name = optimizer if isinstance(optimizer, str) else "AdamW"
        optimizer_params: Dict[str, Any] = dict(optimizer_params) if isinstance(optimizer_params, dict) else {"lr": 3e-4, "weight_decay": 1e-4}
        optimizer_params = coerce_config_scalars(optimizer_params)
        if "learning_rate" in optimizer_params and "lr" not in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")
        optimizer_class = getattr(optim, optimizer_name, None)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer_class(self.model.parameters(), **optimizer_params)

    def _build_scheduler(self, scheduler: str, scheduler_params: Optional[Dict[str, Any]]) -> optim.lr_scheduler:
        scheduler_name = scheduler if isinstance(scheduler, str) else "ReduceLROnPlateau"
        scheduler_params: Dict[str, Any] = dict(scheduler_params) if isinstance(scheduler_params, dict) else {}
        scheduler_params = coerce_config_scalars(scheduler_params)
        if scheduler_name == "ReduceLROnPlateau" and "mode" not in scheduler_params:
            scheduler_params["mode"] = "min"
        scheduler_class = getattr(optim.lr_scheduler, scheduler_name, None)
        if scheduler_class is None:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        self.scheduler = scheduler_name
        return scheduler_class(self.optimizer, **scheduler_params)

    def _build_loss(self, loss: str, loss_params: Optional[Dict[str, Any]]) -> torch.nn.Module:
        loss_name = loss if isinstance(loss, str) else "CrossEntropyLoss"
        loss_params: Dict[str, Any] = dict(loss_params) if isinstance(loss_params, dict) else {}
        loss_params = coerce_config_scalars(loss_params)
        loss_class = getattr(nn, loss_name, None)
        if loss_class is None:
            raise ValueError(f"Unsupported loss: {loss_name}")
        return loss_class(**loss_params)


