import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.amp import autocast, GradScaler
from torchvision import models
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Optional
import os
from pathlib import Path
import sys

# append src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .datasets.DataManager import DataManager

# Reduce /dev/shm usage to avoid DataLoader bus errors
try:
    torch.multiprocessing.set_sharing_strategy(
        os.environ.get("PYTORCH_SHARING_STRATEGY", "file_system")
    )
except Exception:
    pass

def _get_run_logger(log_file: Path):
    """Create and return a file-only logger writing to the provided file path."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_file
    run_id = log_file.parent.name
    logger = logging.getLogger(f"mice_repr.train.{run_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path.as_posix())
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False
    return logger, log_path

    
#? -------------------------------------------------------------- #
#?                        Model - AlexNet                         #
#? -------------------------------------------------------------- #
        
class AlexNet():
    def __init__(self,
                 model       : Optional[torch.nn.Module]                = None,
                 device      : Optional[torch.device]                   = None,
                 criterion   : Optional[torch.nn.Module]                = None,
                 data_manager: Optional[DataManager]                    = None,
                 optimizer   : Optional[torch.optim.Optimizer] | str    = "AdamW",
                 num_epochs  : int                                      = 100,
                 learning_rate: float                                   = 3e-4,
                 weight_decay: float                                    = 1e-4,
                 dropout_rate: float                                    = 0.3,
                 patience    : int                                      = 15,
                 label_smoothing: float                                 = 0.1,
                 log_file: Optional[Path]                               = None,
                 checkpoint_dir: Optional[Path]                         = None,
                 artifacts_dir: Optional[Path]                          = None,
                 use_cuda: bool                                         = True,
                 save_every_n: int                                      = 10,
             ):
        
        self.model        : Optional[torch.nn.Module]       = model
        self.device       : Optional[torch.device]          = device
        self.criterion    : Optional[torch.nn.Module]       = criterion
        self.data_manager : Optional[DataManager]           = data_manager
        self.num_epochs   : int                             = num_epochs
        self.learning_rate: float                           = learning_rate
        self.weight_decay : float                           = weight_decay
        self.dropout_rate : float                           = dropout_rate
        self.patience     : int                             = patience
        self.save_every_n : int                             = save_every_n
        self.start_epoch  : int                             = 0
        self.use_cuda     : bool                            = use_cuda
        self.checkpoint_dir: Path                           = (checkpoint_dir or Path(os.environ.get("CHECKPOINT_DIR_OVERRIDE", "checkpoints")).resolve())
        self.log_file: Path                                 = (log_file or (self.checkpoint_dir / "train.log")).resolve()
        self.artifacts_dir: Path                            = (artifacts_dir or (self.checkpoint_dir / "artifacts")).resolve()
        
        # initialize file logger (file-only, no console output)
        self.logger, self.log_file_path = _get_run_logger(self.log_file)

        if model is None:
            self.model = models.alexnet(weights=None, num_classes=self.data_manager.num_classes, dropout=self.dropout_rate)
        
        if device is None:
            self.device = torch.device("cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu")
            self.logger.info(f"Using device: {self.device}")
            # Performance toggles for CUDA GPUs (RTX 40xx supports TF32)
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision('high')
                except Exception:
                    pass

        if criterion is None:
            self.criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if data_manager is None:
            raise ValueError("Data manager not provided")
        
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses   = []
        
        self.train_accuracies = []
        self.val_accuracies   = []
        
        if self.data_manager is None:
            raise ValueError("Data manager not provided")
        
        self.train_loader = self.data_manager.train_loader
        self.val_loader   = self.data_manager.val_loader
        self.test_loader  = self.data_manager.test_loader
        
        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')
        
    #? ------------------------ Train Epoch -------------------------
        
    def train_epoch(self, epoch):
        self.model.train()
        
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
            
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            if self.device.type == 'cuda':
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixed precision training
            with autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (labels == predicted).sum().item()
           
            total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss":  f'{loss.item():.4f}',
                "Acc":   f'{100 * total_correct / total_samples:.2f}%'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc  = 100 * total_correct / total_samples
            
        return epoch_loss, epoch_acc
    
    #? ------------------------ Validate Epoch -------------------------
    
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(loader, desc="Validating", leave=False)
            
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device, non_blocking=True)
                if self.device.type == 'cuda':
                    images = images.contiguous(memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
                
                # Mixed precision inference
                with autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (labels == predicted).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "Loss":  f'{loss.item():.4f}',
                    "Acc":   f'{100 * total_correct / total_samples:.2f}%'
                })
                
        epoch_loss = total_loss / len(loader)
        epoch_acc  = 100 * total_correct / total_samples
        
        return epoch_loss, epoch_acc
    
    #? ------------------------ Train -------------------------
    
    def train(self):
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        save_every_n = self.save_every_n
        
        # If resuming, start from self.start_epoch
        if self.start_epoch > 0:
            self.logger.info(f"Resuming training from epoch {self.start_epoch}.")
        for epoch in range(self.start_epoch, self.num_epochs):
            
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            self.logger.info("-" * 10)
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            self.logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            def save_checkpoint(epoch, loss, accuracy, name = None, msg = None):
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                if name is None:
                    name = f"checkpoint_epoch_{epoch+1}.pth"
                ckpt_path = self.checkpoint_dir / name
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                    "accuracy": accuracy,
                    "history": {
                        "train_losses": self.train_losses,
                        "val_losses": self.val_losses,
                        "train_accuracies": self.train_accuracies,
                        "val_accuracies": self.val_accuracies,
                    }
                }, ckpt_path.as_posix())
                self.logger.info(f"{msg}: {ckpt_path}")
            
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
        
    #? ------------------------ Test -------------------------
        
    def test(self):
        return self.evaluate(self.test_loader)
    
    def load_model(self, model_path: str):
        # Normalize to Path and fallback to configured checkpoint path
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
        # Set start_epoch to next epoch after the one stored in checkpoint
        try:
            saved_epoch = int(checkpoint.get('epoch', -1))
        except Exception:
            saved_epoch = -1
        self.start_epoch = max(0, saved_epoch + 1)
        # Restore history if available so plots are not empty
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
        """Plot training and validation curves."""
            
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(12, 4))
        
        # Plot losses
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
            
        # Plot accuracies
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
        # Also save CSV with history for programmatic access
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
    