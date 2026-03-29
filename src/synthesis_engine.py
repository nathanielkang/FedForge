"""
Diffusion-based Synthesis Engine for FedSynth-Engine.

Trains a denoising diffusion model on aggregated marginals with
workload-guided loss. Generates synthetic tabular data that matches
the target marginal distributions.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual MLP block with timestep conditioning."""

    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h = F.silu(h)
        h = self.fc1(h)
        h = h + self.time_proj(t_emb)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.fc2(h)
        return h + residual


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        emb = F.silu(self.fc1(emb))
        emb = self.fc2(emb)
        return emb


class DenoisingNetwork(nn.Module):
    """MLP-based denoising network for tabular diffusion."""

    def __init__(self, data_dim: int, hidden_dim: int = 256, num_layers: int = 4, time_dim: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.time_emb = TimestepEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = self.input_proj(x) + self.time_proj(t_emb)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)


def _safe_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s < 1e-15:
        return np.ones_like(p) / len(p)
    p = p / s
    p = np.clip(p, 0.0, None)
    return p / p.sum()


class SynthesisEngine:
    """
    Central diffusion model trained on aggregated marginals with workload-guided loss.

    The forward diffusion adds Gaussian noise over T steps. The denoising network
    predicts the noise, and x_0 is reconstructed within the computation graph so
    workload loss gradients flow back to model parameters.
    """

    def __init__(
        self,
        data_dim: int,
        domain_sizes: Dict[str, int],
        column_names: List[str],
        diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.2,
        hidden_dim: int = 256,
        num_layers: int = 4,
        learning_rate: float = 1e-3,
        workload_weight: float = 0.5,
        tau_scale: float = 0.5,
        device: str = "cpu",
    ):
        self.data_dim = data_dim
        self.domain_sizes = domain_sizes
        self.column_names = column_names
        self.device = torch.device(device)

        self.T = diffusion_steps
        self.workload_weight = workload_weight
        self.tau_scale = tau_scale

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.float().to(self.device)
        self.alphas = alphas.float().to(self.device)
        self.alpha_bar = alpha_bar.float().to(self.device)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).float().to(self.device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).float().to(self.device)

        self.model = DenoisingNetwork(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _encode_data(self, data: np.ndarray) -> torch.Tensor:
        """Encode discrete data to continuous [0, 1] representation."""
        encoded = np.zeros_like(data, dtype=np.float64)
        for i, col_name in enumerate(self.column_names):
            d = self.domain_sizes[col_name]
            if d > 1:
                encoded[:, i] = data[:, i].astype(np.float64) / (d - 1)
            else:
                encoded[:, i] = 0.5
        return torch.tensor(encoded, dtype=torch.float32, device=self.device)

    def _decode_data(self, x: torch.Tensor) -> np.ndarray:
        """Decode continuous [0, 1] representation back to discrete values."""
        x_np = x.detach().cpu().numpy()
        decoded = np.zeros_like(x_np, dtype=int)
        for i, col_name in enumerate(self.column_names):
            d = self.domain_sizes[col_name]
            if d > 1:
                decoded[:, i] = np.clip(np.round(x_np[:, i] * (d - 1)), 0, d - 1).astype(int)
            else:
                decoded[:, i] = 0
        return decoded

    def _soft_histogram(
        self, x: torch.Tensor, col_indices: List[int], domain_sizes_list: List[int]
    ) -> torch.Tensor:
        """Differentiable soft histogram using softmax over bin assignments."""
        shape = tuple(domain_sizes_list)
        total_bins = 1
        for s in shape:
            total_bins *= s

        flat_hist = torch.zeros(total_bins, device=self.device, dtype=torch.float32)
        batch_size = x.shape[0]

        assignment_probs = []
        for idx, (col, d) in enumerate(zip(col_indices, domain_sizes_list)):
            vals = x[:, col]
            bin_centers = torch.linspace(0, 1, d, device=self.device)
            tau = self.tau_scale / max(d, 2)
            diffs = -(vals.unsqueeze(1) - bin_centers.unsqueeze(0)) ** 2 / (2 * tau ** 2)
            probs = F.softmax(diffs, dim=1)
            assignment_probs.append(probs)

        if len(col_indices) == 1:
            flat_hist = assignment_probs[0].mean(dim=0)
        elif len(col_indices) == 2:
            d0, d1 = domain_sizes_list
            joint = torch.einsum("bi,bj->bij", assignment_probs[0], assignment_probs[1])
            flat_hist = joint.mean(dim=0).reshape(-1)
        elif len(col_indices) == 3:
            d0, d1, d2 = domain_sizes_list
            joint = torch.einsum(
                "bi,bj,bk->bijk",
                assignment_probs[0], assignment_probs[1], assignment_probs[2]
            )
            flat_hist = joint.mean(dim=0).reshape(-1)
        else:
            prob = assignment_probs[0]
            for ap in assignment_probs[1:]:
                prob = torch.einsum("b...i,bj->b...ij", prob, ap)
            flat_hist = prob.mean(dim=0).reshape(-1)

        return flat_hist

    def _workload_loss(
        self,
        x0_pred: torch.Tensor,
        target_marginals: Dict[str, np.ndarray],
        queries: List,
    ) -> torch.Tensor:
        """Compute workload-guided loss on predicted x_0."""
        total_loss = torch.tensor(0.0, device=self.device)
        n_queries = 0

        for q in queries:
            if q.key not in target_marginals:
                continue

            target = target_marginals[q.key]
            col_indices = q.columns
            ds_list = [self.domain_sizes[self.column_names[c]] for c in col_indices]

            pred_hist = self._soft_histogram(x0_pred, col_indices, ds_list)
            target_t = torch.tensor(target, dtype=torch.float32, device=self.device)

            if pred_hist.shape != target_t.shape:
                continue

            total_loss = total_loss + F.l1_loss(pred_hist, target_t)
            n_queries += 1

        if n_queries > 0:
            total_loss = total_loss / n_queries

        return total_loss

    def train_epoch(
        self,
        data: np.ndarray,
        target_marginals: Dict[str, np.ndarray],
        queries: List,
        batch_size: int = 4096,
    ) -> Dict[str, float]:
        """Train the diffusion model for one epoch."""
        self.model.train()
        x_encoded = self._encode_data(data)
        dataset = TensorDataset(x_encoded)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        epoch_denoise_loss = 0.0
        epoch_wl_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(self.device)
            B = batch.shape[0]

            t = torch.randint(0, self.T, (B,), device=self.device)
            noise = torch.randn_like(batch)

            sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(1)
            sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(1)
            x_noisy = sqrt_ab * batch + sqrt_1_ab * noise

            noise_pred = self.model(x_noisy, t)
            denoise_loss = F.mse_loss(noise_pred, noise)

            x0_pred = (x_noisy - sqrt_1_ab * noise_pred) / (sqrt_ab + 1e-8)
            x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

            wl_loss = self._workload_loss(x0_pred, target_marginals, queries)

            loss = denoise_loss + self.workload_weight * wl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_denoise_loss += denoise_loss.item()
            epoch_wl_loss += wl_loss.item()
            n_batches += 1

        return {
            "denoise_loss": epoch_denoise_loss / max(n_batches, 1),
            "workload_loss": epoch_wl_loss / max(n_batches, 1),
        }

    def train(
        self,
        data: np.ndarray,
        target_marginals: Dict[str, np.ndarray],
        queries: List,
        num_epochs: int = 200,
        batch_size: int = 4096,
        log_interval: int = 10,
    ) -> List[Dict[str, float]]:
        """Train the diffusion model for multiple epochs."""
        history = []
        for epoch in range(num_epochs):
            metrics = self.train_epoch(data, target_marginals, queries, batch_size)
            history.append(metrics)
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"denoise={metrics['denoise_loss']:.4f}, "
                    f"workload={metrics['workload_loss']:.4f}"
                )
        return history

    @torch.no_grad()
    def generate(self, num_samples: int, batch_size: int = 4096) -> np.ndarray:
        """Generate synthetic data using DDPM reverse diffusion."""
        self.model.eval()
        all_samples = []

        for start in range(0, num_samples, batch_size):
            B = min(batch_size, num_samples - start)
            x = torch.randn(B, self.data_dim, device=self.device)

            for t_idx in range(self.T - 1, -1, -1):
                t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)
                noise_pred = self.model(x, t)

                beta_t = self.betas[t_idx]
                alpha_t = self.alphas[t_idx]
                alpha_bar_t = self.alpha_bar[t_idx]

                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
                )

                if t_idx > 0:
                    sigma = torch.sqrt(beta_t)
                    z = torch.randn_like(x)
                    x = mean + sigma * z
                else:
                    x = mean

            x = torch.clamp(x, 0.0, 1.0)
            all_samples.append(x.cpu())

        x_all = torch.cat(all_samples, dim=0)
        return self._decode_data(x_all)

    def finetune_streaming(
        self,
        data: np.ndarray,
        new_marginals: Dict[str, np.ndarray],
        new_queries: List,
        old_marginals: Dict[str, np.ndarray],
        old_queries: List,
        num_epochs: int = 20,
        mu: float = 0.3,
        batch_size: int = 4096,
    ) -> List[Dict[str, float]]:
        """Fine-tune for streaming workload adaptation."""
        self.model.train()
        x_encoded = self._encode_data(data)
        dataset = TensorDataset(x_encoded)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        history = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in loader:
                batch = batch.to(self.device)
                B = batch.shape[0]
                t = torch.randint(0, self.T, (B,), device=self.device)
                noise = torch.randn_like(batch)

                sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(1)
                sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(1)
                x_noisy = sqrt_ab * batch + sqrt_1_ab * noise

                noise_pred = self.model(x_noisy, t)
                denoise_loss = F.mse_loss(noise_pred, noise)

                x0_pred = (x_noisy - sqrt_1_ab * noise_pred) / (sqrt_ab + 1e-8)
                x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

                new_wl = self._workload_loss(x0_pred, new_marginals, new_queries)
                old_wl = self._workload_loss(x0_pred, old_marginals, old_queries)

                loss = denoise_loss + self.workload_weight * (new_wl + mu * old_wl)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            history.append({"total_loss": epoch_loss / max(n_batches, 1)})

        return history

    def get_state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict)
