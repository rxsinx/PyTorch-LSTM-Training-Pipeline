# train.py — Full training loop with early stopping, LR scheduler, MLflow logging

import os, time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from config import CONFIG, FEATURE_COLS
from data_fetcher import get_kite_session, fetch_historical_data, fetch_order_book_imbalance
from feature_engineering import build_features
from dataset import prepare_data_loaders
from model import RelLSTMPredictor


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += criterion(pred, y).item() * len(X)
    return total_loss / len(loader.dataset)


def main():
    # ── 1. Authenticate & fetch data ─────────────────────────────────
    kite       = get_kite_session()
    raw_df     = fetch_historical_data(kite)
    obi        = fetch_order_book_imbalance(kite)   # Live OBI for latest row
    feature_df = build_features(raw_df, obi=obi)

    print(f"📊 Feature matrix shape: {feature_df.shape}")
    print(f"   Columns: {list(feature_df.columns)}\n")

    # ── 2. DataLoaders ───────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler = prepare_data_loaders(feature_df)

    # ── 3. Model, loss, optimiser ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Training on: {device}\n")

    model     = RelLSTMPredictor().to(device)
    criterion = nn.HuberLoss(delta=0.01)           # Robust to outliers
    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, verbose=True)

    # ── 4. Training loop ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    mlflow.set_experiment("RELIANCE_LSTM_27F")
    with mlflow.start_run():
        mlflow.log_params({
            "hidden_size":   CONFIG["hidden_size"],
            "num_layers":    CONFIG["num_layers"],
            "dropout":       CONFIG["dropout"],
            "seq_len":       CONFIG["seq_len"],
            "features":      CONFIG["feature_count"],
            "epochs":        CONFIG["epochs"],
            "batch_size":    CONFIG["batch_size"],
            "learning_rate": CONFIG["learning_rate"],
        })

        for epoch in range(1, CONFIG["epochs"] + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss   = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            elapsed = time.time() - t0

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            print(f"Epoch {epoch:03d}/{CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"⏱ {elapsed:.1f}s")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), CONFIG["model_save_path"])
                print(f"  ✅ Best model saved (val_loss={val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["patience"]:
                    print(f"\n⏹️ Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
                    break

        # Log best model artifact
        mlflow.pytorch.log_model(model, "lstm_model")
        print(f"\n🏁 Training complete. Best val loss: {best_val_loss:.6f}")

    # ── 5. Final test evaluation ──────────────────────────────────────
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"🧪 Test Loss (Huber): {test_loss:.6f}")


if __name__ == "__main__":
    main()
