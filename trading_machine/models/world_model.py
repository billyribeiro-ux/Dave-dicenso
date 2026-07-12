"""
models/world_model.py — PriceVAE: Variational Autoencoder for price sequences.

This is the most critical component. The VAE compresses 500-tick price windows
into 256-dim latent states. It knows NOTHING about financial markets — it only
sees closing prices. All structure (trends, volatility, patterns) must be
discovered through unsupervised reconstruction learning.

Architecture:
  Encoder: Linear proj → 2 Temporal Conv Blocks → 4-layer Transformer → mu/logvar
  Decoder: Latent proj → 2-layer Transformer → 3 Conv1d layers → price output
"""

import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.logger import get_logger


# ============================================================================
# PriceVAE — Variational Autoencoder for price sequences
# ============================================================================

class PriceVAE(nn.Module):
    """Variational Autoencoder that learns to compress and reconstruct price sequences.

    Input:  (batch, 500, 1)  — 500 close prices
    Latent: (batch, 256)     — compressed representation
    Output: (batch, 100, 1)  — reconstructed future 100 prices
    """

    def __init__(
        self,
        input_dim: int = 1,
        latent_dim: int = 256,
        sequence_length: int = 500,
        future_length: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.future_length = future_length

        # ==================================================================
        # ENCODER
        # ==================================================================

        # 1. Input projection: 1 → 64
        self.enc_input_proj = nn.Linear(input_dim, 64)

        # 2. Temporal Convolution Block 1
        self.enc_conv1a = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.enc_bn1a = nn.BatchNorm1d(128)
        self.enc_conv1b = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.enc_bn1b = nn.BatchNorm1d(128)

        # 3. Temporal Convolution Block 2
        self.enc_conv2a = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.enc_bn2a = nn.BatchNorm1d(256)
        self.enc_conv2b = nn.Conv1d(256, 256, kernel_size=7, padding=3)
        self.enc_bn2b = nn.BatchNorm1d(256)

        # 4. Transformer Encoder: 4 layers, 8 heads, 256 dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 5. Final projection to latent parameters (mu and logvar)
        self.enc_fc_mu_logvar = nn.Linear(256, latent_dim * 2)

        # ==================================================================
        # DECODER
        # ==================================================================

        # 1. Latent projection
        self.dec_latent_proj = nn.Linear(latent_dim, 256)

        # 2. Expand to sequence
        self.dec_expand = nn.Linear(256, future_length * 256)

        # 3 & 4. Transformer Decoder: 2 layers, 8 heads, 256 dim
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # 5. Convolutional Decoder
        self.dec_conv1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv1d(64, input_dim, kernel_size=3, padding=1)

        # ==================================================================
        # Additional components
        # ==================================================================

        # Positional encoding buffer
        max_len = max(sequence_length, future_length) + 100
        pos_enc = self.encode_position(max_len, 256)
        self.register_buffer("positional_encoding", pos_enc)

        # Reconstruction loss
        self.reconstruction_loss_fn = nn.MSELoss()

        # KL weight (annealed from 0 to 1 during training)
        self.register_buffer("kl_weight", torch.tensor(0.0))

    # ------------------------------------------------------------------
    # Positional encoding (sinusoidal, as in "Attention Is All You Need")
    # ------------------------------------------------------------------

    def encode_position(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Sinusoidal positional encoding.

        Returns tensor of shape (1, seq_len, d_model).
        """
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, d_model)

    # ------------------------------------------------------------------
    # Reparameterization trick
    # ------------------------------------------------------------------

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard VAE reparameterization: z = mu + eps * exp(0.5 * logvar)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------
    # Encoder (public, used by RL agent during inference)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a price window to latent parameters.

        Args:
            x: (batch_size, sequence_length, 1) — normalized price window

        Returns:
            (mu, logvar, z) — latent distribution parameters and sample
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Input projection
        h = self.enc_input_proj(x)  # (batch, seq_len, 64)

        # Step 2: Transpose for Conv1d
        h = h.transpose(1, 2)  # (batch, 64, seq_len)

        # Step 3: Conv Block 1
        h = self.enc_conv1a(h)
        h = self.enc_bn1a(h)
        h = F.relu(h)
        h = self.enc_conv1b(h)
        h = self.enc_bn1b(h)
        h = F.relu(h)  # (batch, 128, seq_len)

        # Step 4: Conv Block 2
        h = self.enc_conv2a(h)
        h = self.enc_bn2a(h)
        h = F.relu(h)
        h = self.enc_conv2b(h)
        h = self.enc_bn2b(h)
        h = F.relu(h)  # (batch, 256, seq_len)

        # Step 5: Transpose back
        h = h.transpose(1, 2)  # (batch, seq_len, 256)

        # Step 6: Add positional encoding
        pos_enc = self.positional_encoding[:, :seq_len, :]
        h = h + pos_enc

        # Step 7: Transformer Encoder
        h = self.transformer_encoder(h)  # (batch, seq_len, 256)

        # Step 8: Mean over sequence dimension
        h = h.mean(dim=1)  # (batch, 256)

        # Step 9: Project to mu, logvar
        params = self.enc_fc_mu_logvar(h)  # (batch, latent_dim * 2)
        mu = params[:, :self.latent_dim]
        logvar = params[:, self.latent_dim:]
        # Softplus on logvar to ensure positivity
        logvar = F.softplus(logvar)

        # Step 10: Sample z
        z = self.reparameterize(mu, logvar)

        return mu, logvar, z

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to future price sequence.

        Args:
            z: (batch_size, latent_dim)

        Returns:
            (batch_size, future_length, 1) — reconstructed future prices
        """
        batch_size = z.shape[0]

        # Step 1: Latent projection
        h = self.dec_latent_proj(z)  # (batch, 256)

        # Step 2: Expand to sequence
        h = self.dec_expand(h)  # (batch, future_length * 256)
        h = h.view(batch_size, self.future_length, 256)  # (batch, future_length, 256)

        # Add positional encoding
        pos_enc = self.positional_encoding[:, :self.future_length, :]
        h = h + pos_enc

        # Step 3 & 4: Transformer Decoder (self-attention on future sequence)
        # Use a zero memory for the decoder (we're generating from latent alone)
        memory = torch.zeros(batch_size, 1, 256, device=z.device)
        h = self.transformer_decoder(h, memory)  # (batch, future_length, 256)

        # Step 5: Transpose for Conv1d
        h = h.transpose(1, 2)  # (batch, 256, future_length)

        # Convolutional Decoder
        h = self.dec_conv1(h)
        h = F.relu(h)
        h = self.dec_conv2(h)
        h = F.relu(h)
        h = self.dec_conv3(h)  # (batch, 1, future_length)

        # Transpose back
        h = h.transpose(1, 2)  # (batch, future_length, 1)

        return h

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterize → decode.

        Args:
            x: (batch_size, sequence_length, 1)

        Returns:
            (reconstructed_future, mu, logvar, z)
        """
        mu, logvar, z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        reconstructed: torch.Tensor,
        future_actual: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute VAE loss: reconstruction + KL divergence.

        Args:
            reconstructed: (batch, future_length, 1)
            future_actual: (batch, future_length, 1)
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)

        Returns:
            (total_loss, reconstruction_loss_value, kl_loss_value)
        """
        batch_size = mu.shape[0]

        # Reconstruction loss (MSE)
        recon_loss = self.reconstruction_loss_fn(reconstructed, future_actual)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / batch_size
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # Total loss with annealed KL weight
        total_loss = recon_loss + self.kl_weight * kl_loss

        return total_loss, recon_loss.item(), kl_loss.item()

    # ------------------------------------------------------------------
    # KL weight annealing
    # ------------------------------------------------------------------

    def anneal_kl_weight(self, step: int, total_steps: int) -> None:
        """Linearly increase kl_weight from 0.0 to 1.0 over training."""
        self.kl_weight.fill_(min(step / max(total_steps, 1), 1.0))


# ============================================================================
# PriceDataset — Sliding window dataset from raw price array
# ============================================================================

class PriceDataset(Dataset):
    """Creates sliding windows from a 1D price array.

    Each sample: (input_window: 500 prices, future_window: 100 prices)
    Normalizes within each window using mean/std of the INPUT window only.
    """

    def __init__(
        self,
        prices_array: np.ndarray,
        sequence_length: int = 500,
        future_length: int = 100,
    ):
        self.prices = prices_array.astype(np.float32)
        self.sequence_length = sequence_length
        self.future_length = future_length

        total_len = sequence_length + future_length
        if len(self.prices) < total_len:
            raise ValueError(
                f"Price array length ({len(self.prices)}) must be >= "
                f"sequence_length + future_length ({total_len})"
            )

        self.num_windows = len(self.prices) - total_len + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end_input = start + self.sequence_length
        end_future = end_input + self.future_length

        input_window = self.prices[start:end_input].copy()
        future_window = self.prices[end_input:end_future].copy()

        # Normalize using mean/std of INPUT window only
        input_mean = input_window.mean()
        input_std = input_window.std()
        if input_std < 1e-8:
            input_std = 1.0  # Avoid division by zero for flat prices

        input_normalized = (input_window - input_mean) / input_std
        future_normalized = (future_window - input_mean) / input_std

        # Reshape to (seq_len, 1)
        input_tensor = torch.tensor(input_normalized, dtype=torch.float32).unsqueeze(-1)
        future_tensor = torch.tensor(future_normalized, dtype=torch.float32).unsqueeze(-1)

        return input_tensor, future_tensor


# ============================================================================
# prepare_dataloaders — Create train/val DataLoaders
# ============================================================================

def prepare_dataloaders(
    prices_array: np.ndarray,
    batch_size: int = 64,
    sequence_length: int = 500,
    future_length: int = 100,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:
    """Split data chronologically (first 80% train, last 20% val).

    Args:
        prices_array: 1D numpy array of close prices.
        batch_size: Batch size for DataLoaders.
        sequence_length: Number of past ticks (input window).
        future_length: Number of future ticks (target window).
        train_split: Fraction of data for training.

    Returns:
        (train_dataloader, val_dataloader)
    """
    total_len = len(prices_array)
    split_idx = int(total_len * train_split)

    train_prices = prices_array[:split_idx + future_length]
    val_prices = prices_array[split_idx - sequence_length:]

    train_dataset = PriceDataset(train_prices, sequence_length, future_length)
    val_dataset = PriceDataset(val_prices, sequence_length, future_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, val_loader


# ============================================================================
# WorldModelTrainer — Training loop with early stopping
# ============================================================================

class WorldModelTrainer:
    """Trains the PriceVAE with AdamW, KL annealing, and early stopping."""

    def __init__(self, model: PriceVAE, device: torch.device | None = None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device = device
        self.model = model.to(device)
        self.logger = get_logger()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0001,
            weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        total_batches = len(dataloader)
        total_steps = total_epochs * total_batches

        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}/{total_epochs}", leave=False)
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)

            # Anneal KL weight
            global_step = epoch * total_batches + batch_idx
            self.model.anneal_kl_weight(global_step, total_steps)

            # Forward pass
            reconstructed, mu, logvar, z = self.model(x)
            loss, recon_val, kl_val = self.model.compute_loss(reconstructed, y, mu, logvar)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "recon": f"{recon_val:.6f}",
                "kl": f"{kl_val:.6f}",
                "kl_w": f"{self.model.kl_weight.item():.3f}",
            })

        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Run validation. Returns average validation loss."""
        self.model.eval()
        total_loss = 0.0
        total_batches = len(dataloader)

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            reconstructed, mu, logvar, z = self.model(x)
            loss, _, _ = self.model.compute_loss(reconstructed, y, mu, logvar)
            total_loss += loss.item()

        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss

    def train_full(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
        checkpoint_dir: str | None = None,
    ) -> dict:
        """Full training loop with early stopping.

        Args:
            train_dataloader: Training data.
            val_dataloader: Validation data.
            epochs: Maximum number of epochs.
            patience: Early stopping patience.
            checkpoint_dir: Directory to save best model. If None, no saving.

        Returns:
            Training history dictionary.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "best_val_loss": float("inf"),
        }

        self.logger.info(f"Starting training for {epochs} epochs (patience={patience})")
        self.logger.info(f"Device: {self.device}")

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader, epoch, epochs)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = self.validate(val_dataloader)
            history["val_loss"].append(val_loss)

            # Scheduler step
            self.scheduler.step(val_loss)

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} — "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                history["best_epoch"] = epoch + 1
                history["best_val_loss"] = best_val_loss

                # Save checkpoint
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, "world_model_best.pt")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                        },
                        checkpoint_path,
                    )
                    self.logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement for {patience_counter}/{patience} epochs"
                )
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.logger.info(
            f"Training complete. Best val loss: {best_val_loss:.6f} at epoch {history['best_epoch']}"
        )
        return history
