#!/usr/bin/env python3
"""
portfolio_manager.py

FULL WORKING SCRIPT
- Reads positions.csv
- Computes ATR-based exits
- Frees cash from sells
- Allocates freed cash using volatility-adjusted sizing (1 / ATR%)
- Generates portfolio_plan.csv
- Python 3.11 / GitHub Actions safe
"""

# =========================
# IMPORTS
# =========================
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# CONFIG
# =========================
CSV_FILE = os.getenv("CSV_FILE", "positions.csv")
PLAN_FILE = os.getenv("PLAN_FILE", "portfolio_plan.csv")

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))

MAX_NEW_PER_RUN = int(os.getenv("MAX_NEW_PER_RUN", "3"))
MAX_CONTRACTS_PER_POSITION = int(os.getenv("MAX_CONTRACTS_PER_POSITION", "6"))
CONTRACT_MULTIPLIER = int(os.getenv("CONTRACT_MULTIPLIER", "100"))

# =========================
# HELPERS
# =========================
def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def get_daily_data(ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period="9mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return None


def atr_pct_for_ticker(ticker: str, period: int, cache: dict[str, pd.DataFrame]) -> Optional[float]:
    if ticker not in cache:
        df = get_daily_data(ticker)
        if df is None:
            return None
        cache[ticker] = df

    df = cache[ticker]
    if len(df) < period + 5:
        return None

    atr = calculate_atr(df, period).iloc[-1]
    price = float(df["Close"].iloc[-1])

    if not np.isfinite(atr) or atr <= 0 or price <= 0:
        return None

    return float(atr / price)


def estimate_contract_cost(option_price: float) -> float:
    return option_price * CONTRACT_MULTIPLIER


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} not found")

    positions = pd.read_csv(CSV_FILE)

    required_cols = {
        "ticker",
        "option_price",
        "contracts",
        "underlying_entry_price",
    }
    missing = required_cols - set(positions.columns)
    if missing:
        raise ValueError(f"positions.csv missing columns: {missing}")

    ticker_cache: dict[str, pd.DataFrame] = {}

    # -------------------------
    # EXIT LOGIC → FREED CASH
    # -------------------------
    freed_cash = 0.0
    plan_rows = []

    for _, row in positions.iterrows():
        ticker = row["ticker"].upper()
        contracts = int(row["contracts"])
        option_price = float(row["option_price"])
        entry_under = float(row["underlying_entry_price"])

        df = get_daily_data(ticker)
        if df is None:
            continue

        atr = calculate_atr(df, ATR_PERIOD).iloc[-1]
        current_price = float(df["Close"].iloc[-1])

        stop_price = entry_under - ATR_MULTIPLIER * atr

        sell_contracts = 0
        if current_price < stop_price:
            sell_contracts = contracts

        if sell_contracts > 0:
            freed_cash += sell_contracts * estimate_contract_cost(option_price)

        plan_rows.append(
            {
                "Type": "SELL" if sell_contracts > 0 else "HOLD",
                "Ticker": ticker,
                "SellContracts": sell_contracts,
                "PositionContracts": contracts,
                "Reason": "ATR stop breach" if sell_contracts > 0 else "Trend intact",
            }
        )

    # -------------------------
    # SCAN FOR NEW ENTRIES
    # (simple universe example)
    # -------------------------
    universe = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    candidates = []

    for t in universe:
        atrp = atr_pct_for_ticker(t, ATR_PERIOD, ticker_cache)
        if atrp is None:
            continue

        # mock option price (replace with real option chain if desired)
        option_price = round(0.03 * ticker_cache[t]["Close"].iloc[-1], 2)
        est_cost = estimate_contract_cost(option_price)

        candidates.append(
            {
                "Ticker": t,
                "ATRpct": atrp,
                "OptionPrice": option_price,
                "EstCost1": est_cost,
            }
        )

    cand_df = pd.DataFrame(candidates)
    if cand_df.empty or freed_cash <= 0:
        pd.DataFrame(plan_rows).to_csv(PLAN_FILE, index=False)
        print("No buys. Plan saved.")
        return

    # -------------------------
    # VOL-ADJUSTED ALLOCATION
    # -------------------------
    cand_df = cand_df.sort_values("ATRpct").head(MAX_NEW_PER_RUN)
    cand_df["Weight"] = 1 / cand_df["ATRpct"]
    cand_df["AllocCash"] = freed_cash * cand_df["Weight"] / cand_df["Weight"].sum()
    cand_df["BuyContracts"] = (cand_df["AllocCash"] / cand_df["EstCost1"]).astype(int)
    cand_df["BuyContracts"] = cand_df["BuyContracts"].clip(
        lower=0, upper=MAX_CONTRACTS_PER_POSITION
    )
    cand_df["EstCostTotal"] = cand_df["BuyContracts"] * cand_df["EstCost1"]

    # trim if over budget
    while cand_df["EstCostTotal"].sum() > freed_cash:
        idx = cand_df["ATRpct"].idxmax()
        if cand_df.at[idx, "BuyContracts"] <= 0:
            break
        cand_df.at[idx, "BuyContracts"] -= 1
        cand_df.at[idx, "EstCostTotal"] = (
            cand_df.at[idx, "BuyContracts"] * cand_df.at[idx, "EstCost1"]
        )

    for _, r in cand_df.iterrows():
        if r["BuyContracts"] > 0:
            plan_rows.append(
                {
                    "Type": "BUY",
                    "Ticker": r["Ticker"],
                    "BuyContracts": int(r["BuyContracts"]),
                    "EstCost": round(r["EstCostTotal"], 2),
                    "Reason": "Vol-adjusted allocation (1/ATR%)",
                }
            )

    # -------------------------
    # SAVE PLAN
    # -------------------------
    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(PLAN_FILE, index=False)

    print(f"Portfolio plan generated: {PLAN_FILE}")
    print(f"Freed cash used: ${cand_df['EstCostTotal'].sum():,.2f}")


if __name__ == "__main__":
    main()