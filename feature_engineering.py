# feature_engineering.py — Build all 27 features from raw OHLCV + OI data

import numpy as np
import pandas as pd
from config import FEATURE_COLS


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame, obi: float = 0.0) -> pd.DataFrame:
    """
    Compute all 27 features. Modifies df in-place and returns it.

    Parameters
    ----------
    df   : DataFrame with columns [open, high, low, close, volume, oi]
    obi  : Latest order book imbalance scalar (from live API or 0.0 for backtest)
    """
    # ── 1. Returns ──────────────────────────────────────────────────
    df["returns"]     = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"]    = (df["high"] - df["low"]) / df["close"]
    df["gap_open"]    = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # ── 2. Moving Averages ───────────────────────────────────────────
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # ── 3. MACD ─────────────────────────────────────────────────────
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── 4. RSI ──────────────────────────────────────────────────────
    df["rsi"] = compute_rsi(df["close"], 14)

    # ── 5. Bollinger Bands ───────────────────────────────────────────
    bb_mid         = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    bb_upper       = bb_mid + 2 * bb_std
    bb_lower       = bb_mid - 2 * bb_std
    df["bb_pct_b"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)

    # ── 6. ATR & Historical Volatility ──────────────────────────────
    df["atr_pct"]      = compute_atr(df, 14) / df["close"]
    df["hist_vol_20"]  = df["log_returns"].rolling(20).std() * np.sqrt(252)

    # ── 7. Volume Features ──────────────────────────────────────────
    vol_sma20        = df["volume"].rolling(20).mean()
    df["vol_ratio"]  = df["volume"] / (vol_sma20 + 1e-9)
    df["obv"]        = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    df["vwap_approx"]= (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    # ── 8. Open Interest ────────────────────────────────────────────
    if "oi" not in df.columns:
        df["oi"] = 0.0
    df["oi_change"] = df["oi"].pct_change().fillna(0)

    # ── 9. Order Book Imbalance (static scalar for backtesting) ─────
    df["obi"] = obi   # Use rolling live feed in production

    # ── Drop NaN rows created by rolling windows ──────────────────
    df.dropna(inplace=True)

    # Ensure column order matches FEATURE_COLS
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features after engineering: {missing}")

    return df[FEATURE_COLS]
