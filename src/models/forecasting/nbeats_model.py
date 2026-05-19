"""
src/models/forecasting/nbeats_model.py

N-BEATS (Neural Basis Expansion Analysis for Time Series) — Pure PyTorch
=========================================================================

Architecture Reference:
    Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
    interpretable time series forecasting", ICLR 2020.

This implementation follows the GENERIC architecture:
    - Multiple stacks, each containing multiple blocks
    - Each block produces a backcast (explaining the past) and a
      forecast (predicting the future) using fully-connected layers
    - Residual connections between blocks (doubly-residual stacking)
    - The final forecast is the sum of all block-level forecasts

HPE Blueprint Phase 4.1:
    - Train one model on capacity_used_pct histories of all 50 volumes
    - input_chunk_length (lookback) and output_chunk_length (horizon)
      are configurable; defaults match blueprint (20 days in, 7 days out)
    - Output: per-volume 7-day capacity forecast used to compute DTF
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class NBeatsBlock(nn.Module):
    """
    A single N-BEATS block.

    FC layers → θ_backcast → Backcast (residual subtracted from input)
    FC layers → θ_forecast → Forecast (summed into output)
    """

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        hidden_size: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size

        # Shared FC trunk
        layers = []
        in_dim = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        self.trunk = nn.Sequential(*layers)

        # Backcast and forecast heads
        self.backcast_head = nn.Linear(hidden_size, input_size)
        self.forecast_head = nn.Linear(hidden_size, forecast_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, input_size)

        Returns
        -------
        backcast : (batch, input_size)
        forecast : (batch, forecast_size)
        """
        h = self.trunk(x)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


class NBeatsStack(nn.Module):
    """A stack of N-BEATS blocks with residual connections."""

    def __init__(
        self,
        n_blocks: int,
        input_size: int,
        forecast_size: int,
        hidden_size: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, forecast_size, hidden_size, n_layers, dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        residual   : (batch, input_size) — remaining signal after backcasts
        stack_forecast : (batch, forecast_size) — sum of all block forecasts
        """
        stack_forecast = torch.zeros(x.size(0), self.blocks[0].forecast_size, device=x.device)
        residual = x
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast  # residual stacking
            stack_forecast = stack_forecast + forecast
        return residual, stack_forecast


# ─────────────────────────────────────────────────────────────────────────────
# Full N-BEATS model
# ─────────────────────────────────────────────────────────────────────────────

class NBeatsModel(nn.Module):
    """
    Complete N-BEATS model (generic architecture).

    Parameters
    ----------
    input_size : int
        Lookback window length (e.g. 20 daily time steps).
    forecast_size : int
        Forecast horizon (e.g. 7 daily time steps).
    n_stacks : int
        Number of stacks (default 3).
    n_blocks : int
        Number of blocks per stack (default 3).
    hidden_size : int
        Hidden layer width in each block (default 128).
    n_layers : int
        Number of FC layers per block (default 4).
    dropout : float
        Dropout probability (default 0.0).
    """

    def __init__(
        self,
        input_size: int = 20,
        forecast_size: int = 7,
        n_stacks: int = 3,
        n_blocks: int = 3,
        hidden_size: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size

        self.stacks = nn.ModuleList([
            NBeatsStack(n_blocks, input_size, forecast_size, hidden_size, n_layers, dropout)
            for _ in range(n_stacks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_size) — lookback window values

        Returns
        -------
        forecast : (batch, forecast_size) — predicted future values
        """
        global_forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device)
        residual = x
        for stack in self.stacks:
            residual, stack_forecast = stack(residual)
            global_forecast = global_forecast + stack_forecast
        return global_forecast

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

class CapacityDataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset from multiple capacity time series.

    Each sample is (lookback_window, forecast_window) pair extracted
    from one volume's daily capacity_used_pct series.
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        input_size: int,
        forecast_size: int,
    ) -> None:
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.samples: list[Tuple[torch.Tensor, torch.Tensor]] = []

        window = input_size + forecast_size
        for series in series_list:
            if len(series) < window:
                continue
            for i in range(len(series) - window + 1):
                x = series[i : i + input_size].astype(np.float32)
                y = series[i + input_size : i + window].astype(np.float32)
                self.samples.append((torch.from_numpy(x), torch.from_numpy(y)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def train_nbeats(
    model: NBeatsModel,
    train_dataset: CapacityDataset,
    val_dataset: Optional[CapacityDataset] = None,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    device: Optional[str] = None,
) -> dict:
    """
    Train N-BEATS model with early stopping.

    Returns
    -------
    dict with keys: train_losses, val_losses, best_epoch, best_val_loss
    """
    if len(train_dataset) == 0:
        raise ValueError("Cannot train N-BEATS with an empty train_dataset. Ensure history is long enough for training windows.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )
    criterion = nn.MSELoss()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)

        # ── Validate ──────────────────────────────────────────────────────
        val_loss = epoch_loss  # fallback if no val set
        if val_loader:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_hat = model(x_batch)
                    val_total += criterion(y_hat, y_batch).item() * x_batch.size(0)
            val_loss = val_total / len(val_dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # ── Early stopping ────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | train_loss={epoch_loss:.6f} "
                  f"| val_loss={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)

    print(f"  Training complete. Best epoch={best_epoch}, "
          f"best_val_loss={best_val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def forecast_volume(
    model: NBeatsModel,
    history: np.ndarray,
    n_steps_ahead: int = 7,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Autoregressive multi-step forecast beyond model's native horizon.

    Parameters
    ----------
    model : trained NBeatsModel
    history : 1D array, at least model.input_size values
    n_steps_ahead : total steps to forecast

    Returns
    -------
    forecast : 1D array of length n_steps_ahead
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    input_size = model.input_size
    forecast_size = model.forecast_size

    if len(history) == 0:
        return np.zeros(n_steps_ahead, dtype=np.float32)

    if len(history) < input_size:
        # Pad left with first value if history is too short
        pad = np.full(input_size - len(history), history[0], dtype=np.float32)
        history = np.concatenate([pad, history])

    # Use last input_size values
    current_input = history[-input_size:].astype(np.float32)
    forecasts: list = []

    steps_remaining = n_steps_ahead
    while steps_remaining > 0:
        x = torch.from_numpy(current_input).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy().flatten()

        take = min(forecast_size, steps_remaining)
        forecasts.append(pred[:take])
        steps_remaining -= take

        if steps_remaining > 0:
            # Shift window forward by forecast_size
            current_input = np.concatenate([current_input[take:], pred[:take]])

    return np.concatenate(forecasts)[:n_steps_ahead]
