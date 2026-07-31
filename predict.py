# predict.py — Real-time next-day close price prediction using live Kite data

import numpy as np
import torch
import joblib

from config import CONFIG, FEATURE_COLS
from data_fetcher import get_kite_session, fetch_historical_data, fetch_order_book_imbalance
from feature_engineering import build_features
from model import RelLSTMPredictor


def predict_next_close():
    # ── 1. Authenticate & fetch ──────────────────────────────────────
    kite       = get_kite_session()
    raw_df     = fetch_historical_data(kite)
    obi        = fetch_order_book_imbalance(kite)
    feature_df = build_features(raw_df, obi=obi)

    # ── 2. Load scaler & model ───────────────────────────────────────
    scaler = joblib.load(CONFIG["scaler_save_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = RelLSTMPredictor().to(device)
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device))
    model.eval()

    # ── 3. Scale & take last seq_len rows ────────────────────────────
    scaled   = scaler.transform(feature_df.values)
    last_seq = scaled[-CONFIG["seq_len"]:]                 # (60, 27)
    X        = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

    # ── 4. Predict ───────────────────────────────────────────────────
    with torch.no_grad():
        pred_scaled = model(X).cpu().numpy().flatten()[0]

    # Inverse transform the close price
    close_idx   = FEATURE_COLS.index("close")
    dummy       = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, close_idx] = pred_scaled
    pred_price  = scaler.inverse_transform(dummy)[0, close_idx]

    last_close  = feature_df["close"].iloc[-1]
    change_pct  = (pred_price - last_close) / last_close * 100
    signal      = "🟢 BUY" if change_pct > 0.5 else ("🔴 SELL" if change_pct < -0.5 else "🟡 HOLD")

    print("\n🎯 ── RELIANCE Next-Day Prediction ──────────────")
    print(f"   Last Close   : ₹{last_close:.2f}")
    print(f"   Predicted    : ₹{pred_price:.2f}")
    print(f"   Expected Δ  : {change_pct:+.2f}%")
    print(f"   Signal       : {signal}")

    return pred_price, signal


if __name__ == "__main__":
    predict_next_close()
