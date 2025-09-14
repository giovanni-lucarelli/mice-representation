import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from .base_trainer import BaseTrainer


class SupervisedTrainer(BaseTrainer):
    
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
                loss = self.loss(outputs, labels)

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

            progress_bar.set_postfix({
                "Loss":  f'{loss.item():.4f}',
                "Acc":   f'{100 * total_correct / total_samples:.2f}%'
            })

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc  = 100 * total_correct / total_samples

        return epoch_loss, epoch_acc

    def evaluate(self, loader: DataLoader, epoch: int, mode: str = "Val"):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        desc = f"Testing" if mode == "Test" else f"Validating {mode} epoch {epoch+1}"

        with torch.no_grad():
            progress_bar = tqdm(loader, desc=desc, leave=False)

            for batch_idx, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device, non_blocking=True)
                if self.device.type == 'cuda':
                    images = images.contiguous(memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    outputs = self.model(images)
                    loss = self.loss(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (labels == predicted).sum().item()

                progress_bar.set_postfix({
                    "Loss":  f'{loss.item():.4f}',
                    "Acc":   f'{100 * total_correct / total_samples:.2f}%'
                })

        epoch_loss = total_loss / len(loader)
        epoch_acc  = 100 * total_correct / total_samples

        return epoch_loss, epoch_acc


