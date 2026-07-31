# model.py — LSTM with Temporal Attention for stock price prediction

import torch
import torch.nn as nn
from config import CONFIG


class TemporalAttention(nn.Module):
    """Soft attention over LSTM time steps — learns which days matter most."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_out):           # (batch, seq, hidden)
        scores  = self.attention(lstm_out) # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden)
        return context, weights


class RelLSTMPredictor(nn.Module):
    """
    Multi-layer LSTM + Temporal Attention → MLP head.
    Input  : (batch, seq_len=60, input_size=27)
    Output : (batch, 1)  — scaled next-day close price
    """
    def __init__(
        self,
        input_size:  int  = CONFIG["feature_count"],
        hidden_size: int  = CONFIG["hidden_size"],
        num_layers:  int  = CONFIG["num_layers"],
        dropout:     float = CONFIG["dropout"],
        use_attention: bool = CONFIG["use_attention"],
    ):
        super().__init__()
        self.use_attention = use_attention

        # Input projection (stabilises training with 27 heterogeneous features)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.attention = TemporalAttention(hidden_size) if use_attention else None

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):                          # (B, T, F)
        x           = self.input_proj(x)           # (B, T, H)
        lstm_out, _ = self.lstm(x)                 # (B, T, H)

        if self.use_attention:
            context, _ = self.attention(lstm_out)  # (B, H)
        else:
            context = lstm_out[:, -1, :]           # last timestep

        return self.head(context)                  # (B, 1)
