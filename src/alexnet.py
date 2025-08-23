import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

import os
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

from pathlib import Path


# find root path
root_path = Path(__file__).parent

# Reduce /dev/shm usage to avoid DataLoader bus errors
try:
    torch.multiprocessing.set_sharing_strategy(
        os.environ.get("PYTORCH_SHARING_STRATEGY", "file_system")
    )
except Exception:
    pass

# define data paths
data_path = root_path / "data" / "miniimagenet"
labels_path = root_path / "data" / "labels.txt"

#? -------------------------------------------------------------- #
#?                         Data Manager                           #
#? -------------------------------------------------------------- #

class DataManager():
    def __init__(self,
                 data_path      : str                           = data_path, 
                 labels_path    : str                           = labels_path,
                 batch_size     : int                           = 128,
                 train_transform: Optional[transforms.Compose]  = None,
                 eval_transform : Optional[transforms.Compose]  = None,
                 num_workers    : int                           = 4,
                 train_split    : float                         = 0.7,
                 val_split      : float                         = 0.15):
        
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        
        if train_transform is None:
            # Train transform
            self.train_transform = transforms.Compose([
                transforms.Resize((64, 64)),   # resize diretto, no crop
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.train_transform = train_transform

        if eval_transform is None:
            # Eval transform
            self.eval_transform = transforms.Compose([
                transforms.Resize((64, 64)),   # resize diretto, no crop
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.eval_transform = eval_transform
            
        # Dataset attributes
        self.dataset        : Optional[ImageFolder]     = None
        self._base_dataset  : Optional[ImageFolder]     = None
        self.train_dataset  : Optional[Subset]          = None
        self.val_dataset    : Optional[Subset]          = None
        self.test_dataset   : Optional[Subset]          = None
        
        self.id_to_labels   : Dict[int, str]            = {}
        self.labels_to_id   : Dict[str, int]            = {}
        
        # DataLoader attributes
        self.train_loader   : Optional[DataLoader] = None
        self.val_loader     : Optional[DataLoader] = None
        self.test_loader    : Optional[DataLoader] = None
        
        # Dataset info
        self.num_classes : Optional[int]    = None
        self.class_names : Optional[list]   = None
        
    def load_labels(self):
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels path {self.labels_path} does not exist")
        
        with open(self.labels_path, "r") as f:
            labels_lines = f.readlines()
        
        # Parse labels: format is "n02119789 1 kit_fox"
        for line in labels_lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                class_id = int(parts[1]) - 1  # 0-based index
                class_name = parts[2].replace('_', ' ')
                # forward lookup dictionary
                self.id_to_labels[class_id] = class_name
                # reverse lookup dictionary
                self.labels_to_id[class_name] = class_id
        
    def _get_class_name(self, class_id: int) -> str:
        if self.id_to_labels is None:
            raise ValueError("Labels not loaded. Call load_labels() first.")
        
        try:
            return self.id_to_labels[class_id]
        except KeyError:
            raise ValueError(f"Class ID '{class_id}' not found in labels.")
    
    def _get_class_id(self, class_name: str) -> int:
        if self.labels_to_id is None:
            raise ValueError("Labels not loaded. Call load_labels() first.")

        try:
            return self.labels_to_id[class_name]
        except KeyError:
            raise ValueError(f"Class name '{class_name}' not found in labels.")
        
    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset path {self.data_path} does not exist")
        
        print(f"Loading dataset from {self.data_path}")
        self._base_dataset = ImageFolder(root=self.data_path, transform=None)
        self.dataset = self._base_dataset
        
        self.load_labels()
        
        self.num_classes = len(self.dataset.classes)
        self.class_names = [self._get_class_name(i) for i in range(self.num_classes)]
        
        print(f"Dataset loaded: {len(self.dataset)} samples, {self.num_classes} classes")
        print(f"Classes: {self.class_names}")
        
    def split_data(self):
        
        if self._base_dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        print(f"Splitting dataset into train, val, and test sets")
        
        indices = list(range(len(self._base_dataset)))
        
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=1 - self.train_split,
            stratify=[self._base_dataset.targets[i] for i in indices]
        )
        val_idx, test_idx   = train_test_split(
            temp_idx,
            test_size=self.val_split /(1 - self.train_split),
            stratify=[self._base_dataset.targets[i] for i in temp_idx]
        )

        # Build three datasets with their own transforms and then subset by the same indices
        train_full = ImageFolder(root=self.data_path, transform=self.train_transform)
        eval_full  = ImageFolder(root=self.data_path, transform=self.eval_transform)

        self.train_dataset = Subset(train_full, train_idx)
        self.val_dataset   = Subset(eval_full,  val_idx)
        self.test_dataset  = Subset(eval_full,  test_idx)
        
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Val dataset: {len(self.val_dataset)} samples")
        print(f"Test dataset: {len(self.test_dataset)} samples")
        
    def create_loaders(self):
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            raise ValueError("Train, val, and test datasets not provided")
        
        use_cuda = torch.cuda.is_available()
        pin = use_cuda
        # Clamp workers/prefetch to avoid shared-memory exhaustion
        effective_workers = min(self.num_workers, int(os.environ.get("NUM_WORKERS_OVERRIDE", "8")))
        persistent = False  # safer for stability across restarts
        prefetch = int(os.environ.get("PREFETCH_FACTOR", "2")) if effective_workers > 0 else None

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,  
            num_workers=effective_workers,
            pin_memory=pin,
            persistent_workers=persistent,  # stability over absolute throughput
            prefetch_factor=prefetch
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,  
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=effective_workers,
            pin_memory=pin,
            persistent_workers=persistent,  # stability over absolute throughput
            prefetch_factor=prefetch
        )
        
        self.test_loader  = DataLoader(
            self.test_dataset,  
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=effective_workers,
            pin_memory=pin,
            persistent_workers=persistent,  # stability over absolute throughput
            prefetch_factor=prefetch
        )
        
    #? ------------------------ Setup -------------------------
        
    def setup(self):
        self.load_data()
        self.split_data()
        self.create_loaders()
        return self.train_loader, self.val_loader, self.test_loader
    
    
    

    
#? -------------------------------------------------------------- #
#?                         Model Manager                          #
#? -------------------------------------------------------------- #
        
class ModelManager():
    def __init__(self,
                 model       : Optional[torch.nn.Module]        = None,
                 device      : Optional[torch.device]           = None,
                 criterion   : Optional[torch.nn.Module]        = None,
                 optimizer   : Optional[torch.optim.Optimizer]  = None,
                 data_manager: Optional[DataManager]            = None,
                 num_epochs  : int                              = 100,
                 learning_rate: float                           = 0.0003,
                 weight_decay: float                            = 3e-4,
                 dropout_rate: float                            = 0.5,
                 patience    : int                              = 10,
                 label_smoothing: float                         = 0.1
             ):
        
        self.model        : Optional[torch.nn.Module]       = model
        self.device       : Optional[torch.device]          = device
        self.criterion    : Optional[torch.nn.Module]       = criterion
        self.optimizer    : Optional[torch.optim.Optimizer] = optimizer
        self.num_epochs   : int                             = num_epochs
        self.data_manager : Optional[DataManager]           = data_manager
        self.learning_rate: float                           = learning_rate
        self.weight_decay : float                           = weight_decay
        self.dropout_rate : float                           = dropout_rate
        self.patience     : int                             = patience
        
        if model is None:
            self.model = models.alexnet(weights=None)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        for epoch in range(self.num_epochs):
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 10)
            
            train_loss, train_acc, train_acc5 = self.train_epoch()
            val_loss, val_acc, val_acc5 = self.validate_epoch()
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
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
                from pathlib import Path
                checkpoints_dir = Path("checkpoints")
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                best_ckpt_path = checkpoints_dir / "best_model.pth"
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
                from pathlib import Path
                checkpoints_dir = Path("checkpoints")
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pth"
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
        # Try exact path first; if missing, try under checkpoints/
        resolved_path = model_path
        if not os.path.exists(resolved_path):
            alt_path = os.path.join("checkpoints", os.path.basename(model_path))
            if os.path.exists(alt_path):
                resolved_path = alt_path
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        checkpoint = torch.load(resolved_path, map_location=self.device)
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
        print(f"Model loaded from {resolved_path}")
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

    data_path = os.path.expanduser("~/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1")
    
    # detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Optimizations applied: Mixed Precision, Optimized DataLoader, Larger Batch Size")

    data_manager = DataManager(
        data_path=data_path,
        batch_size=512,
        num_workers=12
    )
    data_manager.setup()
    
    model_manager = ModelManager(
        data_manager=data_manager,
        num_epochs=100,
        learning_rate=6e-4,
        weight_decay=3e-4,
        dropout_rate=0.3,
        patience=15,
        label_smoothing=0.1
    )
    
    # Optional warm-start from best checkpoint for fine-tuning
    if os.environ.get("FINETUNE_FROM_BEST", "0") == "1" and (
        os.path.exists("best_model.pth") or os.path.exists(os.path.join("checkpoints", "best_model.pth"))
    ):
        try:
            model_manager.load_model("best_model.pth")
            print("Warm-started from best_model.pth (weights only). Optimizer reset for fine-tuning.")
        except Exception as e:
            print(f"Could not warm-start from best_model.pth: {e}")
    
    # Train the model
    training_history = model_manager.train()
    
    # Plot training history
    model_manager.plot_training_history()
    
    # Test the model
    model_manager.test()