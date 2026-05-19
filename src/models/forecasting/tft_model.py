"""
src/models/forecasting/tft_model.py

Temporal Fusion Transformer (TFT) customized for multi-quantile
tail latency risk forecasting (HPE Blueprint Phase 4.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class GLU(nn.Module):
    """Gated Linear Unit (GLU) for adaptive feature gating."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.fc(x)
        val, gate = torch.chunk(res, 2, dim=-1)
        return val * torch.sigmoid(gate)

class GRN(nn.Module):
    """Gated Residual Network (GRN) to control information flow and skip unused paths."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1, context_dim: Optional[int] = None):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        if context_dim is not None:
            self.fc_context = nn.Linear(context_dim, d_hidden, bias=False)
        else:
            self.fc_context = None
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.glu = GLU(d_out, d_out)
        self.gate_norm = nn.LayerNorm(d_out)
        
        if d_in != d_out:
            self.res_proj = nn.Linear(d_in, d_out)
        else:
            self.res_proj = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.fc_context is not None and context is not None:
            h = h + self.fc_context(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        gated = self.glu(h)
        residual = self.res_proj(x)
        return self.gate_norm(gated + residual)

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) for selecting relevant variables dynamically."""
    def __init__(self, num_vars: int, d_in: int, d_model: int, dropout: float = 0.1, context_dim: Optional[int] = None):
        super().__init__()
        self.num_vars = num_vars
        self.grns = nn.ModuleList([
            GRN(d_in, d_model, d_model, dropout, context_dim) for _ in range(num_vars)
        ])
        self.vsn_grn = GRN(num_vars * d_in, d_model, num_vars, dropout, context_dim)

    def forward(self, vars_list: List[torch.Tensor], context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # vars_list: list of [batch, seq_len, d_in]
        stacked_vars = torch.cat(vars_list, dim=-1) # [batch, seq_len, num_vars * d_in]
        
        # Compute weights
        weights = self.vsn_grn(stacked_vars, context) # [batch, seq_len, num_vars]
        weights = F.softmax(weights, dim=-1).unsqueeze(-1) # [batch, seq_len, num_vars, 1]
        
        # Process individual variables
        processed = [self.grns[i](vars_list[i], context).unsqueeze(-2) for i in range(self.num_vars)]
        processed = torch.cat(processed, dim=-2) # [batch, seq_len, num_vars, d_model]
        
        selected = torch.sum(weights * processed, dim=-2) # [batch, seq_len, d_model]
        return selected

class TemporalFusionTransformer(nn.Module):
    """
    Core Temporal Fusion Transformer (TFT) customized for multi-quantile
    tail latency risk forecasting.
    """
    def __init__(
        self,
        input_size: int,         # Lookback sequence length (number of history steps)
        num_features: int,       # Number of time-series features
        forecast_size: int,      # Forecast horizon (output steps)
        d_model: int = 64,       # Representation size
        n_heads: int = 4,        # Number of attention heads
        dropout: float = 0.1,    # Dropout rate
        quantiles: List[float] = [0.5, 0.9, 0.95] # Quantiles for tail latency
    ):
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.forecast_size = forecast_size
        self.d_model = d_model
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # 1. Feature Embedding layer
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])

        # 2. Variable Selection Network for history variables
        self.vsn = VariableSelectionNetwork(
            num_vars=num_features,
            d_in=d_model,
            d_model=d_model,
            dropout=dropout
        )

        # 3. Local temporal processing via LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_norm = nn.LayerNorm(d_model)

        # 4. Global temporal processing via Self-Attention Decoder
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_gate = GLU(d_model, d_model)

        # 5. Position-wise Gated Residual Connection (Dense)
        self.dense_grn = GRN(d_model, d_model, d_model, dropout)

        # 6. Multi-quantile projection head
        self.output_proj = nn.Linear(d_model, self.num_quantiles)
        
        # Dense layer to map sequential representation of size input_size to forecast_size
        self.horizon_projection = nn.Linear(input_size, forecast_size)

    @property
    def n_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_size, num_features]
        batch_size, seq_len, num_feats = x.shape
        
        # 1. Embed each feature individually
        embedded_features = []
        for i in range(self.num_features):
            feat_slice = x[:, :, i].unsqueeze(-1) # [batch, seq_len, 1]
            embedded_features.append(self.feature_embeddings[i](feat_slice)) # [batch, seq_len, d_model]

        # 2. Apply VSN to select variables
        vsn_out = self.vsn(embedded_features) # [batch, seq_len, d_model]

        # 3. Local Temporal Processing (LSTM)
        lstm_out, _ = self.lstm(vsn_out) # [batch, seq_len, d_model]
        lstm_out = self.lstm_norm(lstm_out + vsn_out) # Residual + Norm

        # 4. Global Self-Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out) # [batch, seq_len, d_model]
        attn_out = self.attn_gate(attn_out)
        attn_out = self.attn_norm(attn_out + lstm_out) # Residual + Norm

        # 5. Position-wise Feedforward GRN
        ff_out = self.dense_grn(attn_out) # [batch, seq_len, d_model]

        # 6. Horizon projection (from seq_len to forecast_size)
        ff_out = ff_out.transpose(1, 2) # [batch, d_model, seq_len]
        horizon_out = self.horizon_projection(ff_out) # [batch, d_model, forecast_size]
        horizon_out = horizon_out.transpose(1, 2) # [batch, forecast_size, d_model]

        # 7. Quantile forecasting projection
        quantiles_out = self.output_proj(horizon_out) # [batch, forecast_size, num_quantiles]
        return quantiles_out

class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for training multi-quantile models.
    """
    def __init__(self, quantiles: List[float] = [0.5, 0.9, 0.95]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: [batch, forecast_size, num_quantiles]
        # y_true: [batch, forecast_size]
        y_true = y_true.unsqueeze(-1) # [batch, forecast_size, 1]
        losses = []
        for i, q in enumerate(self.quantiles):
            error = y_true - y_pred[:, :, i].unsqueeze(-1)
            loss_q = torch.max((q - 1) * error, q * error)
            losses.append(loss_q.mean())
        return torch.stack(losses).mean()
