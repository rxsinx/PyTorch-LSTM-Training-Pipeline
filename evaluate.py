# evaluate.py — Compute MSE, RMSE, MAE, R², Directional Accuracy on test set

import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import CONFIG, FEATURE_COLS
from model import RelLSTMPredictor
from dataset import StockSequenceDataset


def inverse_close(scaled_vals, scaler, close_idx):
    """Inverse-transform only the close column."""
    dummy = np.zeros((len(scaled_vals), len(FEATURE_COLS)))
    dummy[:, close_idx] = scaled_vals.squeeze()
    return scaler.inverse_transform(dummy)[:, close_idx]


@torch.no_grad()
def run_evaluation(test_loader: DataLoader, model, device, scaler):
    model.eval()
    preds_scaled, actuals_scaled = [], []

    for X, y in test_loader:
        preds_scaled.append(model(X.to(device)).cpu().numpy())
        actuals_scaled.append(y.numpy())

    preds_s   = np.concatenate(preds_scaled).flatten()
    actuals_s = np.concatenate(actuals_scaled).flatten()
    close_idx = FEATURE_COLS.index("close")

    preds   = inverse_close(preds_s,   scaler, close_idx)
    actuals = inverse_close(actuals_s, scaler, close_idx)

    # ── Metrics ───────────────────────────────────────────────────────
    mse  = np.mean((preds - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(preds - actuals))
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2   = 1 - ss_res / (ss_tot + 1e-9)
    da   = np.mean(np.sign(np.diff(preds)) == np.sign(np.diff(actuals))) * 100

    print("\n📊 ── Evaluation Metrics ─────────────────────")
    print(f"   MSE                : {mse:.4f}")
    print(f"   RMSE               : {rmse:.4f}")
    print(f"   MAE                : {mae:.4f}")
    print(f"   R²                 : {r2:.4f}")
    print(f"   Directional Acc.   : {da:.2f}%")

    # ── Plot ──────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 5))
    plt.plot(actuals, label="Actual Close", color="#1f77b4")
    plt.plot(preds,   label="Predicted Close", color="#ff7f0e", alpha=0.8)
    plt.title("RELIANCE NSE — LSTM (27 Features) Prediction vs Actual")
    plt.xlabel("Test Trading Days")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_plot.png", dpi=150)
    plt.show()
    print("📈 Plot saved to evaluation_plot.png")

    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, directional_acc=da)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RelLSTMPredictor().to(device)
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device))
    scaler = joblib.load(CONFIG["scaler_save_path"])

    # Rebuild test loader (need raw data — run data pipeline first)
    print("ℹ️  Run train.py first, then call evaluate.py independently.")
