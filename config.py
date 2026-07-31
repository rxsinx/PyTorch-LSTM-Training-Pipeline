# config.py — Central configuration for all hyperparameters

CONFIG = {
    # Kite API
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "access_token": None,  # Set after OAuth login

    # Stock
    "instrument_token": 738561,   # RELIANCE NSE token
    "tradingsymbol": "NSE:RELIANCE",
    "from_date": "2022-01-01",
    "to_date": "2026-07-31",
    "interval": "day",

    # Feature Engineering
    "seq_len": 60,                # Lookback window (60 trading days)
    "target_col": "close",        # Prediction target
    "feature_count": 27,          # Total input features

    # LSTM Model
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "bidirectional": False,
    "use_attention": True,

    # Training
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "patience": 15,               # Early stopping patience
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,

    # Paths
    "model_save_path": "models/best_model.pt",
    "scaler_save_path": "models/scaler.pkl",
    "log_dir": "logs/",
}

# All 27 feature column names (in order)
FEATURE_COLS = [
    # Raw OHLCV (5)
    "open", "high", "low", "close", "volume",
    # Returns (4)
    "returns", "log_returns", "hl_range", "gap_open",
    # Trend — Moving Averages (4)
    "sma_20", "sma_50", "ema_12", "ema_26",
    # Momentum (4)
    "macd", "macd_signal", "macd_hist", "rsi",
    # Volatility (4)
    "bb_pct_b", "bb_width", "atr_pct", "hist_vol_20",
    # Volume (3)
    "vol_ratio", "obv", "vwap_approx",
    # Open Interest / F&O (2)
    "oi", "oi_change",
    # Microstructure (1)
    "obi",
]
