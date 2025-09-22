import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    assert x.ndim == 2
    return F.normalize(x, p=2.0, dim=dim, eps=1e-12)


class InstanceDiscriminationLoss(nn.Module):
    """
    Instance Discrimination using a memory bank and InfoNCE.

    Args:
        memory_bank: Tensor (N, D) memory bank living on the correct device
        model_output_dim: int, feature dimensionality before projection head
        m: int, number of negative samples per example
        gamma: float, momentum for memory updates
        tau: float, temperature
        embedding_dim: int, projection head output dim
    """

    def __init__(
        self,
        memory_bank: torch.Tensor,
        model_output_dim: int,
        m: int = 4096,
        gamma: float = 0.5,
        tau: float = 0.07,
        embedding_dim: int = 128,
        z_const: float = 1232.82,
        mode: str = "dynamic",
    ) -> None:
        super().__init__()
        assert isinstance(memory_bank, torch.Tensor) and memory_bank.ndim == 2
        self.register_buffer("memory_bank", memory_bank)
        self.m = int(m)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self._total_samples = int(memory_bank.shape[0])
        self._embedding_func = nn.Linear(int(model_output_dim), int(embedding_dim))
        # Normalization constant per Nayebi/Wu et al.; Z = z_const * N
        self.z_const = float(z_const)
        # Mode selection: "dynamic" uses per-batch negatives to approximate Z; "static" uses fixed z_const
        mode_lc = str(mode).strip().lower()
        if mode_lc not in ("dynamic", "static"):
            raise ValueError(f"Unsupported mode: {mode}. Expected 'dynamic' or 'static'.")
        self.mode = mode_lc

    def trainable_parameters(self):
        return self._embedding_func.parameters()
    
    def _softmax(self, inner_products: torch.Tensor) -> torch.Tensor:
        Z = float(self.z_const) # * float(self._total_samples)
        # Clamp the exponent argument to avoid overflow/underflow
        x = inner_products / self.tau
        x = x.clamp_(-20.0, 20.0)
        probs = torch.exp(x) / Z
        return probs

    @torch.no_grad()
    def set_z_const(self, value: float) -> None:
        self.z_const = float(value)

    @torch.no_grad()
    def update_memory_bank(self, memory_bank: torch.Tensor) -> None:
        assert isinstance(memory_bank, torch.Tensor) and memory_bank.ndim == 2
        # Replace the buffer reference safely
        self.memory_bank = memory_bank.detach()
        # If number of entries changed, update counters
        new_total = memory_bank.shape[0]
        if new_total != self._total_samples:
            print(f"Memory bank size changed from {self._total_samples} to {new_total}")
            self._total_samples = new_total

    def _get_new_memory_bank_entries(self, embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        curr = torch.index_select(self.memory_bank, 0, indices)
        updated = self.gamma * curr + (1.0 - self.gamma) * embeddings
        return l2_normalize(updated)
    
    def _compute_inner_products(self, embeddings: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        mb = self.memory_bank
        if idxs.ndim == 1:
            with torch.no_grad():
                mem = torch.index_select(mb, 0, idxs)
                assert mem.ndim == 2 and mem.shape == embeddings.shape
        else:
            with torch.no_grad():
                bs, m = idxs.shape
                flat = idxs.view(-1)
                mem = torch.index_select(mb, 0, flat)
                mem = mem.view(bs, m, mb.size(1))
                
            # Important! Outside of torch.no_grad() space
            embeddings = embeddings.view(bs, 1, -1)
            
        inner = torch.mul(embeddings, mem)
        assert inner.shape == mem.shape
        inner = torch.sum(inner, dim=-1)
        # inner = inner.clamp_(-1.0, 1.0)
        return inner

    def forward(self, outputs: torch.Tensor, indices: torch.Tensor):
        # Run in float32 for numerical stability
        from torch.amp import autocast as _autocast
        with _autocast(device_type=outputs.device.type, enabled=False):
            batch_size = outputs.shape[0]
            assert batch_size == indices.shape[0]
            indices = indices.detach().long()

            # Project to embedding space (float32)
            embeddings = self._embedding_func(outputs.float())
            
            # L2 normalize
            embeddings = l2_normalize(embeddings)

            # Calculate similarity for positive data
            pos_inner = self._compute_inner_products(embeddings, indices)
            
            # Calculate similarity for noise (negatives)
            random_idxs = torch.randint(0, self._total_samples, (batch_size, self.m), device=embeddings.device).long()
            neg_inner = self._compute_inner_products(embeddings, random_idxs)

            # --- Stable log-space loss calculation (as in Nayebi's code) ---
            
            # Calculate unnormalized log-probabilities
            unnormalized_pos_log_prob = pos_inner / self.tau                 # shape [B]
            unnormalized_neg_log_prob = neg_inner / self.tau                 # shape [B, m]

            if self.mode == "dynamic":
                # Dynamic approximation: use current batch negatives to estimate C ≈ sum_j exp(u'_j)
                # log_C_hat has shape [B]
                log_C_hat = torch.logsumexp(unnormalized_neg_log_prob, dim=1)

                # log h(i,v) = u - logsumexp([u, log(C_hat)])
                log_data_denominator = torch.logsumexp(
                    torch.stack([
                        unnormalized_pos_log_prob,
                        log_C_hat,
                    ], dim=0),
                    dim=0,
                )
                log_data_prob = unnormalized_pos_log_prob - log_data_denominator

                # For each negative: log(1 - h(i,v')) = log(C_hat) - logsumexp([u', log(C_hat)])
                log_noise_denominator = torch.logsumexp(
                    torch.stack([
                        unnormalized_neg_log_prob,
                        log_C_hat.view(-1, 1).expand_as(unnormalized_neg_log_prob),
                    ], dim=0),
                    dim=0,
                )
                log_noise_prob = log_C_hat.view(-1, 1) - log_noise_denominator
            else:
                # Static mode: use a fixed constant C = (Z/N)*m with Z approximated offline
                const_term = (self.z_const) * self.m
                log_const_term = torch.log(torch.tensor(const_term, device=embeddings.device))

                # log h(i,v) = u - logsumexp([u, log(C)])
                log_data_denominator = torch.logsumexp(
                    torch.stack([
                        unnormalized_pos_log_prob,
                        torch.full_like(unnormalized_pos_log_prob, log_const_term),
                    ], dim=0),
                    dim=0,
                )
                log_data_prob = unnormalized_pos_log_prob - log_data_denominator

                # log(1 - h(i,v')) = log(C) - logsumexp([u', log(C)])
                log_noise_denominator = torch.logsumexp(
                    torch.stack([
                        unnormalized_neg_log_prob,
                        torch.full_like(unnormalized_neg_log_prob, log_const_term),
                    ], dim=0),
                    dim=0,
                )
                log_noise_prob = log_const_term - log_noise_denominator
            
            # J_{NCE}(\theta)
            loss = -(torch.sum(log_data_prob) + torch.sum(log_noise_prob)) / batch_size

            # Momentum update for the memory bank
            entries_to_update = self._get_new_memory_bank_entries(embeddings, indices)

            # Auxiliary metrics
            data_loss = -(torch.sum(log_data_prob) / batch_size)
            noise_loss = -(torch.sum(log_noise_prob) / batch_size)

            return (
                loss,
                entries_to_update.detach(),
                data_loss.detach(),
                noise_loss.detach(),
            )


