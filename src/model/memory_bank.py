import torch
import numpy as np


def _l2_normalize(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    assert x.ndim == 2
    return x / torch.sqrt(torch.sum(x ** 2, dim=dim, keepdim=True)).clamp_min(1e-12)


class MemoryBank:
    """Simple memory bank to store and update per-sample embeddings.

    - Stores a tensor of shape (num_entries, embedding_dim)
    - Supports in-place updates for a set of indices
    - Provides utilities to compute similarities
    """

    def __init__(self, num_entries: int, embedding_dim: int, device: torch.device | str):
        self.device = torch.device(device)
        self.num_entries = num_entries
        self.embedding_dim = embedding_dim
        
        # Initialize memory using the same logic as _initialize_bank
        self._memory = self._initialize_bank()

    def _initialize_bank(self):
        """Initialize memory bank with the same logic as the reference implementation."""
        memory_bank = torch.rand(
            self.num_entries, self.embedding_dim, device=self.device, requires_grad=False
        )
        std_dev = 1.0 / np.sqrt(self.embedding_dim / 3)
        memory_bank = memory_bank * (2 * std_dev) - std_dev
        memory_bank = _l2_normalize(memory_bank, dim=1)
        
        return memory_bank

    def get_memory_bank(self) -> torch.Tensor:
        return self._memory

    def set_memory_bank(self, memory: torch.Tensor) -> None:
        assert memory.ndim == 2
        self._memory = memory.to(self.device)

    def as_tensor(self) -> torch.Tensor:
        return self._memory.detach().clone()

    @torch.no_grad()
    def update(self, indices: torch.Tensor, entries: torch.Tensor) -> None:
        assert indices.ndim == 1
        indices = indices.long().to(self.device)
        assert entries.shape[0] == indices.shape[0]
        self._memory.index_copy_(0, indices, entries.to(self.device))

    def get_all_inner_products(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return dot-products between embeddings (B, D) and memory bank (N, D) -> (B, N)."""
        return embeddings @ self._memory.t()



