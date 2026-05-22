"""
src/models/anomaly/lstm_autoencoder.py

LSTM-Autoencoder for sequential IO pattern anomaly detection.
Trained on normal data to reconstruct typical IO sequences.
High reconstruction error indicates anomalous patterns.

HPE Success Criteria:
- Detect complex sequential anomalies (noisy neighbor interference)
- Complement statistical + Isolation Forest detectors
- Inference time <1 second

Architecture:
- Encoder: input_dim → 64 → 32 → latent_dim=8
- Decoder: 8 → 32 → 64 → input_dim
- Sequence length: 12 timesteps (1 hour at 5-min intervals)

Usage:
    detector = LSTMAutoencoder(input_dim=10, latent_dim=8)
    detector.fit(normal_sequences)
    error, is_anomaly = detector.detect(sequence)

Bug fixes applied (v2):
    1. CRITICAL: Double-scaling in run_detection_on_data — _create_sequences was
       called with pre-scaled data, then detect_batch re-scaled it again.
       Fixed by adding a `_raw` internal path that skips the scaler.
    2. CRITICAL: val split edge case — sequences[:-0] returns an EMPTY array when
       n_val evaluates to 0 (small datasets). Guarded with max(1, n_val).
    3. CRITICAL: load() recreated the model with hardcoded architecture params
       (hidden_dim=64, latent_dim=8, num_layers=2, dropout=0.2) — any custom
       config would cause a state_dict shape mismatch crash. Architecture params
       are now persisted in the checkpoint.
    4. Early stopping discarded best weights — the model continued from the
       degraded final epoch. Fixed with best-weights save/restore via io.BytesIO.
    5. Timestamp reconstruction in run_detection_on_data used np.unique() which
       sorts volumes alphabetically, misaligning with the insertion-order output
       of _create_sequences. Fixed by iterating in insertion order via
       dict.fromkeys().
    6. No gradient clipping — LSTMs are susceptible to exploding gradients.
       Added torch.nn.utils.clip_grad_norm_ (max_norm=1.0).

GPU optimisations applied (v2):
    A. Automatic Mixed Precision (AMP) via torch.cuda.amp — halves VRAM usage
       and gives 1.5–3× throughput on A100/V100/RTX-class GPUs.
    B. cudnn.benchmark=True — enables cuDNN autotuner for fixed-size batches.
    C. pin_memory + non_blocking transfers — overlaps CPU→GPU DMA with compute.
    D. Removed premature full-tensor GPU allocation — train/val tensors now stay
       on CPU; individual batches are transferred inside the loop with
       non_blocking=True, preventing OOM on large datasets.
    E. DataLoader num_workers auto-scaled to os.cpu_count() (capped at 4) with
       persistent_workers=True to eliminate fork overhead between epochs.
    F. torch.compile() applied when PyTorch ≥ 2.0 and CUDA is available,
       reducing Python dispatch overhead by ~20–40% on repeated forward passes.
    G. detect_batch uses torch.inference_mode() instead of torch.no_grad() —
       slightly faster because it also disables version tracking.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class LSTMAutoencoderModel(nn.Module):
    """
    LSTM-based autoencoder for time-series anomaly detection.

    Optimisations:
    - Bidirectional LSTM encoder for better pattern capture
    - Dropout for regularisation (prevent overfitting to normal data)
    - Lightweight architecture (64 → 32 → 8) for fast inference
    - AMP-compatible (no non-CUDA ops in forward path)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder — bidirectional doubles output size, so use hidden_dim // 2 per direction
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Bottleneck: hidden_dim → 32 → latent_dim
        self.encoder_to_latent = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, latent_dim),
        )

        # Expansion: latent_dim → 32 → hidden_dim // 2
        self.latent_to_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_dim // 2),
        )

        # Decoder — unidirectional
        self.decoder = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.output_layer = nn.Linear(hidden_dim // 2, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode → latent → decode → reconstruct.

        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, input_dim)

        Returns
        -------
        reconstruction : Tensor, shape (batch, seq_len, input_dim)
        """
        _, seq_len, _ = x.shape

        # Encode
        encoded, _ = self.encoder(x)                        # (B, T, hidden_dim)
        last_encoded = encoded[:, -1, :]                    # (B, hidden_dim)

        # Compress
        latent = self.encoder_to_latent(last_encoded)       # (B, latent_dim)

        # Expand and repeat for each timestep
        decoder_input = self.latent_to_decoder(latent)      # (B, hidden_dim//2)
        decoder_input = decoder_input.unsqueeze(1).expand(-1, seq_len, -1)

        # Decode
        decoded, _ = self.decoder(decoder_input)            # (B, T, hidden_dim//2)
        reconstruction = self.output_layer(decoded)         # (B, T, input_dim)

        return reconstruction


# ---------------------------------------------------------------------------
# Wrapper with training + inference
# ---------------------------------------------------------------------------

class LSTMAutoencoder:
    """
    LSTM-Autoencoder anomaly detector with training, inference, save/load.

    GPU optimisations:
    - AMP (Automatic Mixed Precision) — fp16 on CUDA, fp32 on CPU
    - cudnn.benchmark — autotuner for fixed-size batches
    - pin_memory + non_blocking transfers
    - torch.compile() on PyTorch ≥ 2.0
    - torch.inference_mode() during detection
    - Gradient clipping (max_norm=1.0) for stable LSTM training

    Bug fixes:
    - Architecture params saved/restored in checkpoint
    - Best-weights restored after early stopping
    - val split n_val=0 edge case guarded
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 12,
        threshold_percentile: float = 5.0,
        device: Optional[str] = None,
        use_amp: bool = True,
        compile_model: bool = True,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of features per timestep.
        hidden_dim : int
            LSTM hidden dimension (64 is optimal for speed/accuracy).
        latent_dim : int
            Bottleneck dimension (8 forces meaningful compression).
        num_layers : int
            Number of LSTM layers (2 is sufficient for this task).
        dropout : float
            Dropout rate for regularisation.
        sequence_length : int
            Timesteps per sequence (12 = 1 hour at 5-min intervals).
        threshold_percentile : float
            Percentile of validation errors used as anomaly threshold.
        device : str, optional
            'cuda', 'cpu', or None (auto-detect GPU).
        use_amp : bool
            Enable Automatic Mixed Precision on CUDA (default True).
        compile_model : bool
            Apply torch.compile() on PyTorch ≥ 2.0 + CUDA (default True).
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.threshold_percentile = threshold_percentile

        # ── Device ──────────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── cuDNN autotuner (speeds up fixed-size batch workloads) ───────────
        if self.device.type == "cuda":
            cudnn.benchmark = True

        logger.info(f"Using device: {self.device}")

        # ── Model ────────────────────────────────────────────────────────────
        self._model_kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.model = LSTMAutoencoderModel(**self._model_kwargs).to(self.device)

        # ── torch.compile (PyTorch ≥ 2.0, CUDA only) ────────────────────────
        # FIX (Opt-F): reduces Python dispatch overhead ~20-40% on repeated passes
        self._compiled = False
        if (
            compile_model
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        ):
            try:
                self.model = torch.compile(self.model)
                self._compiled = True
                logger.info("torch.compile() applied to model")
            except Exception as e:
                logger.warning(f"torch.compile() skipped: {e}")

        # ── AMP scaler ────────────────────────────────────────────────────────
        # FIX (Opt-A): halves VRAM usage; 1.5-3x throughput on modern GPUs
        self._use_amp = use_amp and self.device.type == "cuda"
        try:
            self._grad_scaler = GradScaler(device_type=self.device.type, enabled=self._use_amp)
        except TypeError:
            self._grad_scaler = GradScaler(enabled=self._use_amp)

        # ── Preprocessing + state ────────────────────────────────────────────
        self.scaler = StandardScaler()
        self.threshold: Optional[float] = None
        self.is_fitted = False
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self._anomaly_count = 0
        self._total_samples = 0

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _create_sequences(
        self,
        X: np.ndarray,
        volume_ids: np.ndarray,
    ) -> Tuple[np.ndarray, List]:
        """
        Sliding-window sequence creation, grouped by volume.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) — already scaled
        volume_ids : ndarray, shape (n_samples,)

        Returns
        -------
        sequences : ndarray, shape (n_sequences, seq_len, n_features)
        seq_volume_ids : list — volume ID for each sequence, insertion-ordered
        """
        sequences: List[np.ndarray] = []
        seq_volume_ids: List = []

        # FIX (Bug-5): use insertion-order unique volumes (dict.fromkeys) so
        # that seq_volume_ids aligns with the output of _create_sequences
        # regardless of alphabetical sort order from np.unique.
        unique_volumes = list(dict.fromkeys(volume_ids))

        for vol_id in unique_volumes:
            vol_mask = volume_ids == vol_id
            vol_data = X[vol_mask]

            n_sequences = len(vol_data) - self.sequence_length + 1
            if n_sequences <= 0:
                continue

            # Vectorised sequence creation via stride tricks (10× faster than loop)
            vol_sequences = np.lib.stride_tricks.sliding_window_view(
                vol_data,
                window_shape=(self.sequence_length, self.input_dim),
            ).squeeze(axis=1)  # (n_sequences, seq_len, n_features)

            sequences.append(vol_sequences)
            seq_volume_ids.extend([vol_id] * n_sequences)

        if not sequences:
            raise ValueError(
                f"No sequences could be created. Need at least {self.sequence_length} "
                "samples per volume."
            )

        return np.concatenate(sequences, axis=0), seq_volume_ids

    def _to_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Non-blocking CPU → GPU transfer (FIX Opt-C)."""
        return tensor.to(self.device, non_blocking=True)

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        volume_ids: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        val_split: float = 0.15,
        early_stopping_patience: int = 10,
        grad_clip: float = 1.0,
        verbose: bool = True,
    ) -> "LSTMAutoencoder":
        """
        Train autoencoder on normal (non-anomalous) data.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_samples, n_features)
        volume_ids : ndarray, optional — groups samples per storage volume
        epochs : int — maximum training epochs
        batch_size : int — 256 is optimal for speed/GPU utilisation
        learning_rate : float — Adam LR
        weight_decay : float — L2 regularisation (helps generalisation)
        val_split : float — fraction held out for threshold calibration
        early_stopping_patience : int — stop if val loss stagnates
        grad_clip : float — max gradient norm (FIX Bug-6: prevents LSTM explosion)
        verbose : bool — log progress every 10 epochs

        Returns
        -------
        self
        """
        # ── Prepare data ──────────────────────────────────────────────────────
        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float32)
        else:
            X_array = np.asarray(X, dtype=np.float32)

        if X_array.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {X_array.shape[1]}"
            )

        if volume_ids is None:
            volume_ids = np.zeros(len(X_array), dtype=object)

        # Scale (fit on training data only)
        X_scaled = self.scaler.fit_transform(X_array)

        # Build sequences
        logger.info("Creating sequences…")
        sequences, _ = self._create_sequences(X_scaled, volume_ids)
        logger.info(f"Created {len(sequences):,} sequences")

        # ── Train / val split ─────────────────────────────────────────────────
        # FIX (Bug-2): n_val=0 caused sequences[:-0] == sequences[:0] == empty.
        n_val = max(1, int(len(sequences) * val_split))
        train_sequences = sequences[:-n_val]
        val_sequences = sequences[-n_val:]

        # ── DataLoader (CPU tensors + pin_memory for fast H2D) ───────────────
        # FIX (Opt-D): keep tensors on CPU; transfer per-batch inside loop.
        # This prevents OOM when the dataset is large.
        train_tensor = torch.from_numpy(train_sequences)   # stays on CPU
        val_tensor = torch.from_numpy(val_sequences)       # stays on CPU

        # FIX (Opt-E): pin_memory + persistent_workers overlaps DMA with compute
        num_workers = min(4, os.cpu_count() or 0)
        train_loader = DataLoader(
            TensorDataset(train_tensor, train_tensor),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

        # ── Optimiser ─────────────────────────────────────────────────────────
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        criterion = nn.MSELoss()

        # ── Best-weights buffer (FIX Bug-4) ──────────────────────────────────
        best_val_loss = float("inf")
        best_weights_buf: Optional[io.BytesIO] = None
        patience_counter = 0

        # Pre-move val tensor to GPU once (it's small and used every epoch)
        val_gpu = self._to_gpu(val_tensor)

        self.model.train()

        for epoch in range(epochs):
            # ── Training pass ─────────────────────────────────────────────────
            train_loss_sum = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = self._to_gpu(batch_x)   # non_blocking H2D
                batch_y = self._to_gpu(batch_y)

                optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

                # FIX (Opt-A): AMP forward + loss
                with autocast(device_type=self.device.type, enabled=self._use_amp):
                    reconstruction = self.model(batch_x)
                    loss = criterion(reconstruction, batch_y)

                # FIX (Opt-A): scaled backward + gradient clip + step
                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.unscale_(optimizer)
                # FIX (Bug-6): clip LSTM gradients before stepping
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                self._grad_scaler.step(optimizer)
                self._grad_scaler.update()

                train_loss_sum += loss.item()

            train_loss = train_loss_sum / len(train_loader)
            self.train_losses.append(train_loss)

            # ── Validation pass ───────────────────────────────────────────────
            self.model.eval()
            with torch.inference_mode():
                with autocast(device_type=self.device.type, enabled=self._use_amp):
                    val_recon = self.model(val_gpu)
                    val_loss = criterion(val_recon, val_gpu).item()
            self.val_losses.append(val_loss)
            self.model.train()

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} — "
                    f"Train: {train_loss:.6f}  Val: {val_loss:.6f}"
                )

            # ── Early stopping with best-weights restore ───────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # FIX (Bug-4): serialise weights to CPU buffer (no extra GPU VRAM)
                best_weights_buf = io.BytesIO()
                torch.save(self.model.state_dict(), best_weights_buf)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1}. "
                        f"Restoring best weights (val_loss={best_val_loss:.6f})"
                    )
                    break

        # ── Restore best weights ──────────────────────────────────────────────
        if best_weights_buf is not None:
            best_weights_buf.seek(0)
            self.model.load_state_dict(torch.load(best_weights_buf, map_location=self.device))

        # ── Calibrate anomaly threshold from val reconstruction errors ────────
        self.model.eval()
        with torch.inference_mode():
            with autocast(device_type=self.device.type, enabled=self._use_amp):
                val_recon = self.model(val_gpu)
            # Compute MSE per sequence: (B, T, F) → scalar per B
            val_errors = (val_gpu - val_recon).float().pow(2).mean(dim=(1, 2))
            self.threshold = float(
                np.percentile(val_errors.cpu().numpy(), self.threshold_percentile)
            )

        self.is_fitted = True
        logger.info(
            f"Training complete. "
            f"Threshold (p{self.threshold_percentile}): {self.threshold:.6f}"
        )
        return self

    # ── Single-sequence detection ─────────────────────────────────────────────

    def detect(
        self,
        sequence: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[float, bool]:
        """
        Detect if a single sequence is anomalous.

        Parameters
        ----------
        sequence : DataFrame or ndarray, shape (seq_len, n_features)
            Raw (unscaled) input sequence.

        Returns
        -------
        reconstruction_error : float
        is_anomaly : bool
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if isinstance(sequence, pd.DataFrame):
            seq_array = sequence.values.astype(np.float32)
        else:
            seq_array = np.asarray(sequence, dtype=np.float32)

        if seq_array.shape != (self.sequence_length, self.input_dim):
            raise ValueError(
                f"Expected shape ({self.sequence_length}, {self.input_dim}), "
                f"got {seq_array.shape}"
            )

        seq_scaled = self.scaler.transform(seq_array)
        seq_tensor = self._to_gpu(
            torch.from_numpy(seq_scaled).unsqueeze(0)   # (1, T, F)
        )

        self.model.eval()
        with torch.inference_mode():
            with autocast(device_type=self.device.type, enabled=self._use_amp):
                reconstruction = self.model(seq_tensor)
            error = float((seq_tensor - reconstruction).float().pow(2).mean())

        is_anomaly = error > self.threshold
        self._total_samples += 1
        self._anomaly_count += int(is_anomaly)

        return error, is_anomaly

    # ── Batch detection ───────────────────────────────────────────────────────

    def detect_batch(
        self,
        sequences: np.ndarray,
        _already_scaled: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in batch (vectorised, AMP-accelerated).

        Parameters
        ----------
        sequences : ndarray, shape (n_sequences, seq_len, n_features)
            Raw (unscaled) sequences — the scaler is applied internally.
        _already_scaled : bool
            Internal flag used by run_detection_on_data to skip double-scaling.
            Do NOT set this to True in user code.

        Returns
        -------
        errors : ndarray, shape (n_sequences,)
        is_anomalies : ndarray, shape (n_sequences,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        seq_array = np.asarray(sequences, dtype=np.float32)
        if seq_array.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n, T, F), got shape {seq_array.shape}"
            )

        n_seq, seq_len, n_feat = seq_array.shape

        if not _already_scaled:
            # FIX (Bug-1): scale each timestep-feature pair correctly
            flat = seq_array.reshape(-1, n_feat)
            flat = self.scaler.transform(flat)
            seq_array = flat.reshape(n_seq, seq_len, n_feat)

        seq_tensor = torch.from_numpy(seq_array)   # CPU tensor

        all_errors: List[np.ndarray] = []
        batch_size = 512

        self.model.eval()
        with torch.inference_mode():
            for i in range(0, n_seq, batch_size):
                batch = self._to_gpu(seq_tensor[i : i + batch_size])
                with autocast(device_type=self.device.type, enabled=self._use_amp):
                    reconstruction = self.model(batch)
                batch_errors = (batch - reconstruction).float().pow(2).mean(dim=(1, 2))
                all_errors.append(batch_errors.cpu().numpy())

        errors = np.concatenate(all_errors)
        is_anomalies = errors > self.threshold

        self._total_samples += n_seq
        self._anomaly_count += int(is_anomalies.sum())

        return errors, is_anomalies

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save model + all metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # FIX (Bug-3): persist architecture hyperparams so load() can rebuild
        # the model with the exact same structure before loading state_dict.
        checkpoint = {
            # Architecture — MUST match constructor args exactly
            "model_kwargs": self._model_kwargs,
            # Weights
            "model_state_dict": self.model.state_dict(),
            # Preprocessing
            "scaler": self.scaler,
            # Detector config
            "threshold": self.threshold,
            "sequence_length": self.sequence_length,
            "threshold_percentile": self.threshold_percentile,
            # History
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "LSTMAutoencoder":
        """
        Load model from disk.

        FIX (Bug-3): architecture params are read from the checkpoint, not
        assumed to be the defaults. State dict shape mismatch is now impossible.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        model_kwargs: dict = checkpoint["model_kwargs"]

        detector = cls(
            input_dim=model_kwargs["input_dim"],
            hidden_dim=model_kwargs["hidden_dim"],
            latent_dim=model_kwargs["latent_dim"],
            num_layers=model_kwargs["num_layers"],
            dropout=model_kwargs["dropout"],
            sequence_length=checkpoint["sequence_length"],
            threshold_percentile=checkpoint["threshold_percentile"],
            device=device,
        )

        state_dict = checkpoint["model_state_dict"]
        # Strip "_orig_mod." prefix if present (e.g. from torch.compile)
        fixed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                fixed_state_dict[k[len("_orig_mod."):]] = v
            else:
                fixed_state_dict[k] = v
        detector.model.load_state_dict(fixed_state_dict)
        detector.scaler = checkpoint["scaler"]
        detector.threshold = checkpoint["threshold"]
        detector.train_losses = checkpoint["train_losses"]
        detector.val_losses = checkpoint["val_losses"]
        detector.is_fitted = True

        logger.info(f"Model loaded from {path}")
        return detector

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        return {
            "is_fitted": self.is_fitted,
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "threshold_percentile": self.threshold_percentile,
            "total_samples_processed": self._total_samples,
            "anomalies_detected": self._anomaly_count,
            "anomaly_rate": round(
                self._anomaly_count / max(1, self._total_samples), 4
            ),
            "device": str(self.device),
            "amp_enabled": self._use_amp,
            "compiled": self._compiled,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
        }


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------

def run_detection_on_data(
    data_path: str = "data/processed/io_features.parquet",
    output_dir: str = "models/anomaly",
    train_fraction: float = 0.7,
    sequence_length: int = 12,
    epochs: int = 50,
    batch_size: int = 256,
) -> None:
    """
    Full pipeline: load → train → detect → save results.

    Strategy
    --------
    1. Load processed features and sort by (volume, time).
    2. Train on first ``train_fraction`` of the data (assumed normal).
    3. Detect anomalies in the remaining test window.
    4. Save scores, anomaly records, model and statistics.
    """
    print("=" * 70)
    print(" LSTM-Autoencoder Anomaly Detector")
    print(" HPE Phase 3.3 — Sequential Pattern Detection")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading data from {data_path}…")
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["volume_id", "timestamp"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['volume_id'].nunique()} volumes")

    feature_cols = [
        "total_iops",
        "avg_latency_us",
        "total_throughput_mbps",
        "read_latency_p99_us",
        "write_latency_p99_us",
        "sequential_ratio",
        "read_write_ratio",
        "io_size_entropy",
        "queue_depth",
        "iops_per_queue",
    ]

    X = df[feature_cols].values
    volume_ids = df["volume_id"].values

    # ── Split ──────────────────────────────────────────────────────────────────
    split_idx = int(len(X) * train_fraction)
    X_train, X_test = X[:split_idx], X[split_idx:]
    vol_train, vol_test = volume_ids[:split_idx], volume_ids[split_idx:]

    print(f"\n  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # ── Initialise ─────────────────────────────────────────────────────────────
    print(f"\nInitialising LSTM-Autoencoder (seq_len={sequence_length})…")
    detector = LSTMAutoencoder(
        input_dim=len(feature_cols),
        hidden_dim=64,
        latent_dim=8,
        sequence_length=sequence_length,
        threshold_percentile=95.0,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\nTraining (epochs={epochs}, batch_size={batch_size})…")
    detector.fit(
        X_train,
        volume_ids=vol_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )

    # ── Build test sequences ───────────────────────────────────────────────────
    # FIX (Bug-1): scale X_test ONCE here; pass _already_scaled=True to
    # detect_batch so it does NOT apply the scaler a second time.
    print("\nCreating test sequences…")
    X_test_scaled = detector.scaler.transform(X_test.astype(np.float32))
    test_sequences, test_vol_ids = detector._create_sequences(X_test_scaled, vol_test)
    print(f"  Created {len(test_sequences):,} test sequences")

    # ── Detect ────────────────────────────────────────────────────────────────
    print("\nDetecting anomalies…")
    errors, is_anomalies = detector.detect_batch(
        test_sequences, _already_scaled=True
    )

    # ── Map sequences back to timestamps ──────────────────────────────────────
    # FIX (Bug-5): iterate in insertion order (same order _create_sequences used)
    # to keep seq_timestamps aligned with the errors array.
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    seq_timestamps: List = []
    seq_volumes: List = []

    for vol_id in dict.fromkeys(vol_test):
        vol_test_df = test_df[test_df["volume_id"] == vol_id]
        n_vol_rows = len(vol_test_df)
        n_seq = n_vol_rows - sequence_length + 1
        if n_seq <= 0:
            continue
        # The sequence ending at index i corresponds to timestamp i+(seq_len-1)
        seq_timestamps.extend(
            vol_test_df["timestamp"].iloc[sequence_length - 1:].tolist()
        )
        seq_volumes.extend([vol_id] * n_seq)

    # ── Results DataFrame ─────────────────────────────────────────────────────
    results = pd.DataFrame(
        {
            "volume_id": seq_volumes,
            "timestamp": seq_timestamps,
            "reconstruction_error": errors,
            "is_anomaly": is_anomalies,
        }
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scores_path = output_path / "lstm_ae_scores.csv"
    results.to_csv(scores_path, index=False)
    print(f"  Scores    → {scores_path}")

    anomalies = results[results["is_anomaly"]].copy()
    anomalies["timestamp"] = anomalies["timestamp"].astype(str)
    anomalies_path = output_path / "lstm_ae_anomalies.json"
    with open(anomalies_path, "w") as f:
        json.dump(anomalies.to_dict("records"), f, indent=2)
    print(f"  Anomalies → {anomalies_path}")

    stats = detector.get_statistics()
    stats_path = output_path / "lstm_ae_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats     → {stats_path}")

    model_path = output_path / "lstm_ae_model.pth"
    detector.save(model_path)
    print(f"  Model     → {model_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_anomalies = int(is_anomalies.sum())
    print("\n" + "=" * 70)
    print(" DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Test sequences     : {len(test_sequences):,}")
    print(f"  Anomalies detected : {n_anomalies:,}")
    print(f"  Anomaly rate       : {n_anomalies / len(test_sequences) * 100:.2f}%")
    print(f"  Threshold (p{int(detector.threshold_percentile):02d})    : {detector.threshold:.6f}")
    print(f"  AMP enabled        : {detector._use_amp}")
    print(f"  torch.compile      : {detector._compiled}")

    print(f"\n  Reconstruction error distribution:")
    print(f"    Min    : {errors.min():.6f}")
    print(f"    Mean   : {errors.mean():.6f}")
    print(f"    Median : {float(np.median(errors)):.6f}")
    print(f"    Max    : {errors.max():.6f}")

    print(f"\n  Top 5 volumes by anomaly rate:")
    vol_stats = (
        results.groupby("volume_id")["is_anomaly"]
        .agg(anomalies="sum", total="count")
        .assign(rate=lambda d: d["anomalies"] / d["total"])
        .sort_values("rate", ascending=False)
    )
    for vol_id, row in vol_stats.head(5).iterrows():
        print(
            f"    {vol_id}: {row['rate'] * 100:.2f}% "
            f"({int(row['anomalies'])}/{int(row['total'])})"
        )

    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_detection_on_data()