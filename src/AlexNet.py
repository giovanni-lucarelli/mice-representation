import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.amp import autocast, GradScaler
from torchvision import models

from tqdm import tqdm
from typing import Optional
import os
from pathlib import Path

from config import *
from DataManager import DataManager

# Reduce /dev/shm usage to avoid DataLoader bus errors
try:
    torch.multiprocessing.set_sharing_strategy(
        os.environ.get("PYTORCH_SHARING_STRATEGY", "file_system")
    )
except Exception:
    pass

    
#? -------------------------------------------------------------- #
#?                        Model - AlexNet                         #
#? -------------------------------------------------------------- #
        
class AlexNet():
    def __init__(self,
                 model       : Optional[torch.nn.Module]                = None,
                 device      : Optional[torch.device]                   = None,
                 criterion   : Optional[torch.nn.Module]                = None,
                 data_manager: Optional[DataManager]                    = None,
                 optimizer   : Optional[torch.optim.Optimizer] | str    = OPTIMIZER,
                 num_epochs  : int                                      = NUM_EPOCHS,
                 learning_rate: float                                   = LEARNING_RATE,
                 weight_decay: float                                    = WEIGHT_DECAY,
                 dropout_rate: float                                    = DROPOUT_RATE,
                 patience    : int                                      = PATIENCE,
                 label_smoothing: float                                 = LABEL_SMOOTHING
             ):
        
        self.model        : Optional[torch.nn.Module]       = model
        self.device       : Optional[torch.device]          = device
        self.criterion    : Optional[torch.nn.Module]       = criterion
        self.optimizer    : Optional[torch.optim.Optimizer] = None # set below
        self.num_epochs   : int                             = num_epochs
        self.data_manager : Optional[DataManager]           = data_manager
        self.learning_rate: float                           = learning_rate
        self.weight_decay : float                           = weight_decay
        self.dropout_rate : float                           = dropout_rate
        self.patience     : int                             = patience
        
        if model is None:
            self.model = models.alexnet(weights=None)
        
        if device is None:
            self.device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
            print(f"Using device: {self.device}")
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
        
        if isinstance(optimizer, str):
            if optimizer == "SGD":
                self.optimizer = optim.SGD(
                    self.model.parameters(), 
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    momentum=0.9
                )

            elif optimizer == "AdamW":
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")
        
        if self.optimizer is None:
            raise ValueError("Error: Optimizer not set")
            
        if data_manager is None:
            raise ValueError("Data manager not provided")
        
        # Modify final layer to match number of classes and add dropout
        if self.model is not None and self.data_manager is not None:
            num_classes = self.data_manager.num_classes
            
            # Create a new classifier that adapts to the input size
            # First, let's create a dummy input to get the actual feature dimension
            input_size = int(os.environ.get("INPUT_SIZE", "64"))
            # For 64×64, shrink avgpool to 1×1 (Nayebi-style). Keep default for larger inputs
            if input_size <= 64:
               self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            dummy_input = torch.randn(1, 3, input_size, input_size)
            with torch.no_grad():
                # Get features before classifier
                features = self.model.features(dummy_input)
                features = self.model.avgpool(features)
                features = torch.flatten(features, 1)
                actual_feature_dim = features.shape[1]
                print(f"Actual feature dimension: {actual_feature_dim}")

            self.model.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(actual_feature_dim, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
        
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses   = []
        
        self.train_accuracies = []
        self.val_accuracies   = []
        self.train_top5_accuracies = []
        self.val_top5_accuracies   = []
        
        if self.data_manager is None:
            raise ValueError("Data manager not provided")
        
        self.train_loader = self.data_manager.train_loader
        self.val_loader   = self.data_manager.val_loader
        self.test_loader  = self.data_manager.test_loader
        
        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')
        
    #? ------------------------ Train Epoch -------------------------
        
    def train_epoch(self):
        self.model.train()
        
        total_loss    = 0.0
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
            
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
            # Top-5
            top5 = torch.topk(outputs, k=5, dim=1).indices
            total_top5_correct += top5.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss":  f'{loss.item():.4f}',
                "Acc":   f'{100 * total_correct / total_samples:.2f}%'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc  = 100 * total_correct / total_samples
        epoch_acc5 = 100 * total_top5_correct / total_samples
            
        return epoch_loss, epoch_acc, epoch_acc5
    
    #? ------------------------ Validate Epoch -------------------------
    
    def validate_epoch(self):
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
            
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
                top5 = torch.topk(outputs, k=5, dim=1).indices
                total_top5_correct += top5.eq(labels.view(-1,1)).any(dim=1).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "Loss":  f'{loss.item():.4f}',
                    "Acc":   f'{100 * total_correct / total_samples:.2f}%'
                })
                
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc  = 100 * total_correct / total_samples
        epoch_acc5 = 100 * total_top5_correct / total_samples
        
        return epoch_loss, epoch_acc, epoch_acc5
    
    #? ------------------------ Train -------------------------
    
    def train(self):
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Learning rate scheduler
        if SCHEDULER == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5
            )
        elif SCHEDULER == "MultiStepLR":
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=SCHEDULER_MILESTONES,
                gamma=SCHEDULER_GAMMA
            )
        else:
            raise ValueError(f"Invalid scheduler: {SCHEDULER}")
        
        for epoch in range(self.num_epochs):
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 10)
            
            train_loss, train_acc, train_acc5 = self.train_epoch()
            val_loss, val_acc, val_acc5 = self.validate_epoch()
            
            # Update learning rate based on validation loss
            if SCHEDULER == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            elif SCHEDULER == "MultiStepLR":
                scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.train_top5_accuracies.append(train_acc5)
            self.val_top5_accuracies.append(val_acc5)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc@1: {train_acc:.2f}%, Acc@5: {train_acc5:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc@1: {val_acc:.2f}%, Acc@5: {val_acc5:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # ensure directory exists and save to configured checkpoint path
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                best_ckpt_path = CHECKPOINT_PATH
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": val_loss,
                    "accuracy": val_acc,
                    "history": {
                        "train_losses": self.train_losses,
                        "val_losses": self.val_losses,
                        "train_accuracies": self.train_accuracies,
                        "val_accuracies": self.val_accuracies,
                        "train_top5_accuracies": self.train_top5_accuracies,
                        "val_top5_accuracies": self.val_top5_accuracies,
                    }
                }, best_ckpt_path.as_posix())
                print(f"Model saved to {best_ckpt_path}")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{self.patience}")
                
            # Milestone checkpoints
            if (epoch + 1) in [25, 50, 75, 100]:
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path = CHECKPOINT_PATH.parent / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": val_loss,
                    "accuracy": val_acc,
                    "history": {
                        "train_losses": self.train_losses,
                        "val_losses": self.val_losses,
                        "train_accuracies": self.train_accuracies,
                        "val_accuracies": self.val_accuracies,
                        "train_top5_accuracies": self.train_top5_accuracies,
                        "val_top5_accuracies": self.val_top5_accuracies,
                    }
                }, ckpt_path.as_posix())
                print(f"Saved milestone checkpoint: {ckpt_path}")

            # Stop if validation loss doesn't improve
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            print("-" * 10)
            
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies
        }
        
    #? ------------------------ Test -------------------------
        
    def test(self):
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        correct5 = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Testing")
            
            for images, labels in progress_bar:
                images = images.to(self.device, non_blocking=True)
                if self.device.type == 'cuda':
                    images = images.contiguous(memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
                
                # Mixed precision inference
                with autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                top5 = torch.topk(outputs, k=5, dim=1).indices
                correct5 += top5.eq(labels.view(-1,1)).any(dim=1).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc@1': f'{100 * correct / total:.2f}%',
                    'Acc@5': f'{100 * correct5 / total:.2f}%'
                })
                
        test_loss = total_loss / len(self.test_loader)
        test_accuracy = 100 * correct / total
        test_accuracy5 = 100 * correct5 / total
        
        return test_loss, test_accuracy, test_accuracy5
    
    def load_model(self, model_path: str):
        # Normalize to Path and fallback to configured checkpoint path
        path = Path(model_path)
        if not path.exists():
            print(f"Path {path} does not exist, checking {CHECKPOINT_PATH}")
            if CHECKPOINT_PATH.exists():
                path = CHECKPOINT_PATH
            else:
                alt_path = CHECKPOINT_PATH.parent / path.name
                if alt_path.exists():
                    path = alt_path
        if not path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        checkpoint = torch.load(path.as_posix(), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Restore history if available so plots are not empty
        history = checkpoint.get('history')
        if isinstance(history, dict):
            self.train_losses = history.get('train_losses', [])
            self.val_losses = history.get('val_losses', [])
            self.train_accuracies = history.get('train_accuracies', [])
            self.val_accuracies = history.get('val_accuracies', [])
            self.train_top5_accuracies = history.get('train_top5_accuracies', [])
            self.val_top5_accuracies = history.get('val_top5_accuracies', [])
        print(f"Model loaded from {path}")
        print(f"Best validation loss: {checkpoint['loss']:.4f}")
        print(f"Best validation accuracy: {checkpoint['accuracy']:.2f}%")
        
    def plot_training_history(self):
        """Plot training and validation curves."""
        try:
            import matplotlib.pyplot as plt
            
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
            
            # Plot accuracies
            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.train_accuracies, 'b-', label='Train Acc@1')
            plt.plot(epochs, self.val_accuracies, 'r-', label='Val Acc@1')
            if len(self.train_top5_accuracies) == len(self.train_accuracies):
                plt.plot(epochs, self.train_top5_accuracies, 'b--', label='Train Acc@5')
            if len(self.val_top5_accuracies) == len(self.val_accuracies):
                plt.plot(epochs, self.val_top5_accuracies, 'r--', label='Val Acc@5')
            plt.title('Accuracy (Top-1 and Top-5)')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
            print("Training losses:", self.train_losses)
            print("Validation losses:", self.val_losses)
            print("Training accuracies:", self.train_accuracies)
            print("Validation accuracies:", self.val_accuracies)
                
if __name__ == "__main__":

    # detect device
    device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    print("Optimizations applied: Mixed Precision, Optimized DataLoader, Larger Batch Size")

    data_manager = DataManager(
        data_path=MINI_IMAGENET_PATH,
        labels_path=LABELS_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        split_seed=SPLIT_SEED,
    )
    data_manager.setup()
    
    model_manager = AlexNet(
        data_manager=data_manager,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        dropout_rate=DROPOUT_RATE,
        patience=PATIENCE,
        label_smoothing=LABEL_SMOOTHING
    )
    
    # Optional warm-start from best checkpoint for fine-tuning
    if os.environ.get("FINETUNE_FROM_BEST", "0") == "1" and CHECKPOINT_PATH.exists():
        try:
            model_manager.load_model(CHECKPOINT_PATH.as_posix())
            print("Warm-started from best_model.pth (weights only). Optimizer reset for fine-tuning.")
        except Exception as e:
            print(f"Could not warm-start from best_model.pth: {e}")
    
    # Train the model
    training_history = model_manager.train()
    
    # Plot training history
    model_manager.plot_training_history()
    
    # Test the model
    model_manager.test()