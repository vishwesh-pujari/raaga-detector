"""M1: a small CNN + BiGRU over tonic-normalised pitch *sequences* (see
``features/pitch.chunk_pitch_sequences``). Where B1's histogram deliberately throws away note
order, M1 exists specifically to pick that up: melodic direction (aroha/avaroha), short repeated
phrases, glides -- see CONCEPTS.md.

Input per chunk: (T, 3) -- (mean_sin, mean_cos, voiced_frac) per time step, T fixed by
chunk_s/step_s (see chunk_pitch_sequences' docstring for why this representation).
Architecture: a small 1D-conv frontend (local melodic movement, a few notes at a time) feeding a
bidirectional GRU (longer-range structure, phrase-level) with attention pooling over time, then a
linear classifier head. Small and CPU-testable on toy data; meant to actually train on a GPU.
"""

from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    n_classes: int
    in_channels: int = 3
    conv_channels: Tuple[int, ...] = (32, 64)
    conv_kernel: int = 5
    gru_hidden: int = 128
    gru_layers: int = 1
    dropout: float = 0.3


class PitchSequenceModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        c_in = cfg.in_channels
        for c_out in cfg.conv_channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=cfg.conv_kernel, padding=cfg.conv_kernel // 2),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(
            c_in,
            cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        gru_out = cfg.gru_hidden * 2
        self.attn = nn.Linear(gru_out, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(gru_out, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, in_channels) -> logits (B, n_classes). No masking needed: every chunk in
        the cache is already fixed-length (unvoiced steps are exactly zero, not padding)."""
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T', c_in) after conv+pooling
        out, _ = self.gru(x)  # (B, T', 2*hidden)
        weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)  # (B, T')
        pooled = (out * weights.unsqueeze(-1)).sum(1)  # (B, 2*hidden)
        return self.head(self.dropout(pooled))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SequenceDataset(torch.utils.data.Dataset):
    """Wraps a ``ChunkSet`` (from ``features/sequence_cache.py``) for a PyTorch DataLoader."""

    def __init__(self, X, y_idx):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y_idx, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5  # early stopping on val loss
    class_weighted: bool = True  # like B1's class_weight="balanced"
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def class_weights(y_idx: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(y_idx, minlength=n_classes).float().clamp(min=1)
    w = counts.sum() / (n_classes * counts)
    return w


def fit(
    model: PitchSequenceModel,
    train_X, train_y_idx,
    val_X, val_y_idx,
    cfg: TrainConfig,
    verbose: bool = True,
) -> dict:
    """Trains in place, restores the best (lowest val loss) weights, and returns a small history
    dict. Early-stops after ``cfg.patience`` epochs without a val-loss improvement."""
    torch.manual_seed(cfg.seed)
    model.to(cfg.device)
    train_dl = torch.utils.data.DataLoader(
        SequenceDataset(train_X, train_y_idx), batch_size=cfg.batch_size, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(SequenceDataset(val_X, val_y_idx), batch_size=cfg.batch_size)

    weight = class_weights(torch.as_tensor(train_y_idx), model.cfg.n_classes).to(cfg.device) if cfg.class_weighted else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_dl.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= len(val_dl.dataset)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"epoch {epoch + 1}/{cfg.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"early stopping at epoch {epoch + 1} (no val improvement for {cfg.patience} epochs)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    return history


@torch.no_grad()
def predict_proba(model: PitchSequenceModel, X, device: str, batch_size: int = 256) -> "torch.Tensor":
    model.eval().to(device)
    dl = torch.utils.data.DataLoader(torch.as_tensor(X, dtype=torch.float32), batch_size=batch_size)
    out = []
    for xb in dl:
        out.append(torch.softmax(model(xb.to(device)), dim=1).cpu())
    return torch.cat(out).numpy()
