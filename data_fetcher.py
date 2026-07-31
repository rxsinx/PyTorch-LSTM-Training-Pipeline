# data_fetcher.py — Kite Connect OAuth login + historical data fetch

import webbrowser
from kiteconnect import KiteConnect
import pandas as pd
from config import CONFIG


def get_kite_session() -> KiteConnect:
    """
    Full OAuth login flow for Kite Connect.
    Opens browser → user logs in → pastes request_token back.
    """
    kite = KiteConnect(api_key=CONFIG["api_key"])
    login_url = kite.login_url()
    print(f"\n🔗 Opening Kite login URL:\n{login_url}\n")
    webbrowser.open(login_url)

    request_token = input("📋 Paste the request_token from the redirect URL: ").strip()
    session = kite.generate_session(request_token, api_secret=CONFIG["api_secret"])
    kite.set_access_token(session["access_token"])
    CONFIG["access_token"] = session["access_token"]
    print("✅ Kite session authenticated successfully.\n")
    return kite


def fetch_historical_data(kite: KiteConnect) -> pd.DataFrame:
    """
    Fetch daily OHLCV + OI data for the configured instrument.
    """
    print(f"📥 Fetching historical data for token {CONFIG['instrument_token']} ...")
    records = kite.historical_data(
        instrument_token=CONFIG["instrument_token"],
        from_date=CONFIG["from_date"],
        to_date=CONFIG["to_date"],
        interval=CONFIG["interval"],
        oi=True,
    )
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    print(f"✅ Fetched {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}\n")
    return df


def fetch_order_book_imbalance(kite: KiteConnect) -> float:
    """
    Fetch live order book imbalance (OBI) from the order book snapshot.
    OBI = (total_bid_qty - total_ask_qty) / (total_bid_qty + total_ask_qty)
    """
    quote = kite.quote([CONFIG["tradingsymbol"]])
    depth = quote[CONFIG["tradingsymbol"]]["depth"]
    buy_qty  = sum(d["quantity"] for d in depth["buy"])
    sell_qty = sum(d["quantity"] for d in depth["sell"])
    obi = (buy_qty - sell_qty) / (buy_qty + sell_qty + 1e-9)
    return obi
