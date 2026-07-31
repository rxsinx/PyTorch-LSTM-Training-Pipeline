# dataset.py — PyTorch Dataset for sliding-window sequences

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import CONFIG, FEATURE_COLS


class StockSequenceDataset(Dataset):
    """
    Converts scaled feature matrix into (X, y) sequence pairs.
    X shape : (seq_len, num_features)
    y shape : (1,) — next-day close (scaled)
    """
    def __init__(self, data: np.ndarray, seq_len: int, target_idx: int):
        self.seq_len    = seq_len
        self.target_idx = target_idx
        self.X, self.y  = self._make_sequences(data)

    def _make_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i : i + self.seq_len])
            y.append(data[i + self.seq_len, self.target_idx])
        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def prepare_data_loaders(feature_df):
    """
    Scale features → split train/val/test → return DataLoaders + scaler.
    Walk-forward split (NO random shuffle — preserves temporal order).
    """
    from torch.utils.data import DataLoader

    values = feature_df.values
    n      = len(values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    joblib.dump(scaler, CONFIG["scaler_save_path"])

    # Walk-forward split
    train_end = int(n * CONFIG["train_split"])
    val_end   = int(n * (CONFIG["train_split"] + CONFIG["val_split"]))

    target_idx = FEATURE_COLS.index("close")
    seq_len    = CONFIG["seq_len"]

    train_ds = StockSequenceDataset(scaled[:train_end],          seq_len, target_idx)
    val_ds   = StockSequenceDataset(scaled[train_end:val_end],   seq_len, target_idx)
    test_ds  = StockSequenceDataset(scaled[val_end:],            seq_len, target_idx)

    loader_kwargs = dict(batch_size=CONFIG["batch_size"], pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
        scaler,
    )
