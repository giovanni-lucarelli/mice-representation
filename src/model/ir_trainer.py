import torch
from torch.utils.data import DataLoader

from .base_trainer import BaseTrainer


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
        super().__init__(*args, **kwargs)

    # Override to build the custom loss instead of a torch.nn loss
    def _build_loss(self, loss: str, loss_params):
        raise NotImplementedError(
            "InstanceDiscriminationTrainer requires a custom loss implementation. "
            "Integrate src/model/instance_discrimination_loss.py and a MemoryBank, "
            "then return the instantiated loss here."
        )

    def train_epoch(self, epoch: int):
        raise NotImplementedError(
            "Implement training loop that: (1) forwards model, (2) computes ID loss, "
            "(3) optimizer step over model + loss.trainable_parameters(), and (4) updates memory bank."
        )



    def evaluate(self, loader: DataLoader, epoch: int, mode: str = "Val"):
        raise NotImplementedError(
            "Implement nearest-neighbour evaluation in the embedding space using the memory bank."
        )


