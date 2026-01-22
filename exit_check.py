#!/usr/bin/env python3
"""
exit_check.py

Nightly / intraday Turtle-style exit checker with TWO evaluation modes:

A) Based on "CLOSE" (last available daily close in yfinance data)
B) Based on "CURRENT" (live-ish last price from yfinance fast_info, when available)

For each mode, we output TWO independent trend checks:
- 10 Days: trend intact/broken + action HOLD/SELL
- 5 Days : trend intact/broken + action HOLD/SELL

We also compute ATR(14) from completed daily bars (excluding today's partial bar when present).
ATR stop is included in actions for both CLOSE and CURRENT modes:
- CALL: SELL if price <= entry - 1.5*ATR
- PUT : SELL if price >= entry + 1.5*ATR

CSV required columns:
ticker, option_name, underlying_entry_price
(option_name must contain " C " for calls or " P " for puts)
"""

import os
import smtplib
import ssl
from email.mime.text import MIMEText
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

# =========================
# CONFIG
# =========================
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5

STRUCTURE_10 = 10
STRUCTURE_5 = 5

CSV_FILE = "positions.csv"

# SMTP / EMAIL (FROM GITHUB SECRETS)
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT_RAW = (os.getenv("SMTP_PORT") or "").strip()
SMTP_PORT = int(SMTP_PORT_RAW) if SMTP_PORT_RAW.isdigit() else 0
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
EMAIL_TO = os.getenv("EMAIL_TO", "").strip()

# Email mode:
# - "always"     => send daily summary
# - "exits_only" => send only when at least one SELL exists (in any mode)
EMAIL_MODE = os.getenv("EMAIL_MODE", "always").strip().lower()

HISTORY_PERIOD = os.getenv("HISTORY_PERIOD", "9mo").strip()


# =========================
# HELPERS
# =========================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def infer_direction(option_name: str) -> Tuple[Optional[bool], str]:
    name = f" {option_name} "
    if " C " in name:
        return True, "CALL"
    if " P " in name:
        return False, "PUT"
    return None, "UNKNOWN"


def to_float(x) -> Optional[float]:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def smtp_ready() -> bool:
    return all([SMTP_HOST, SMTP_PORT > 0, SMTP_USER, SMTP_PASS, EMAIL_TO])


def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO

    if SMTP_PORT == 465:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR = rolling mean of TR
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def get_current_price(ticker: str, fallback: float) -> float:
    """
    Try to fetch a live-ish current/last price.
    If unavailable, return fallback (usually latest daily close).
    """
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi:
            # yfinance fast_info often exposes last_price
            lp = fi.get("last_price", None)
            if lp is not None:
                return float(lp)
    except Exception:
        pass
    return float(fallback)


def remove_today_partial_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    If yfinance includes a row for today's date (often partial intraday),
    drop it so indicators/structure windows are based on completed bars only.
    """
    if df.empty:
        return df
    try:
        today = pd.Timestamp.utcnow().date()
        last_date = pd.Timestamp(df.index[-1]).date()
        if last_date == today:
            return df.iloc[:-1]
    except Exception:
        # If index isn't date-like, do nothing
        pass
    return df


def prior_window(df_completed: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Return prior n rows excluding the most recent completed bar:
    - If df_completed ends at yesterday, we exclude that last completed bar to get "prior n".
    This matches "prior N days excluding the evaluation day".
    """
    return df_completed.iloc[-(n + 1):-1]


def action_and_trend(broken: bool, atr_hit: bool) -> Tuple[str, str]:
    """
    Returns (action, trend_status) for a structure window.
    - trend_status: INTACT/BROKEN based on structure only
    - action: SELL if (broken OR atr_hit) else HOLD
    """
    trend = "BROKEN" if broken else "INTACT"
    action = "SELL" if (broken or atr_hit) else "HOLD"
    return action, trend


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError("positions.csv not found in repo root")

    positions = pd.read_csv(CSV_FILE)

    required_cols = {"ticker", "option_name", "underlying_entry_price"}
    missing = required_cols - set(positions.columns)
    if missing:
        raise ValueError(f"positions.csv missing required columns: {sorted(missing)}")

    report = []
    any_sell = False

    min_needed = max(ATR_PERIOD, STRUCTURE_10) + 10  # safety buffer

    for _, row in positions.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        option_name = str(row.get("option_name", "")).strip()
        entry_underlying = to_float(row.get("underlying_entry_price"))

        is_call, direction = infer_direction(option_name)

        if not ticker or entry_underlying is None or is_call is None:
            report.append(f"{ticker or 'UNKNOWN'}: SKIPPED (invalid row: ticker/entry/direction)")
            continue

        # Pull daily data
        df = yf.download(ticker, period=HISTORY_PERIOD, interval="1d", progress=False)
        if df is None or df.empty:
            report.append(f"{ticker}: NO DATA")
            continue

        df = flatten_columns(df).dropna()

        # "Close-based" uses the last available daily close in df (may be today-so-far if market open)
        close_price = float(df["Close"].iloc[-1])

        # "Current-based" tries to fetch last_price; falls back to close_price
        current_price = get_current_price(ticker, fallback=close_price)

        # For ATR/structure levels, use completed bars (drop today's partial if present)
        df_completed = remove_today_partial_bar(df).dropna()

        if len(df_completed) < min_needed:
            report.append(f"{ticker}: SKIPPED (insufficient completed history: {len(df_completed)} rows)")
            continue

        # ATR from completed bars
        atr_series = calculate_atr(df_completed, ATR_PERIOD)
        atr_last = atr_series.iloc[-1]
        if pd.isna(atr_last):
            report.append(f"{ticker}: SKIPPED (ATR NaN)")
            continue
        atr = float(atr_last)

        # Structure levels from completed bars (exclude the evaluation day by using prior_window)
        w10 = prior_window(df_completed, STRUCTURE_10)
        w5 = prior_window(df_completed, STRUCTURE_5)

        if w10.empty or w5.empty:
            report.append(f"{ticker}: SKIPPED (structure windows empty)")
            continue

        low10, high10 = float(w10["Low"].min()), float(w10["High"].max())
        low5, high5 = float(w5["Low"].min()), float(w5["High"].max())

        # ATR stop value (same for both modes)
        if is_call:
            atr_stop = float(entry_underlying - ATR_MULTIPLIER * atr)
            # Structure breaks for CALL: price < prior N-day low
            broken10_close = bool(close_price < low10)
            broken5_close = bool(close_price < low5)
            broken10_current = bool(current_price < low10)
            broken5_current = bool(current_price < low5)

            atr_hit_close = bool(close_price <= atr_stop)
            atr_hit_current = bool(current_price <= atr_stop)
        else:
            atr_stop = float(entry_underlying + ATR_MULTIPLIER * atr)
            # Structure breaks for PUT: price > prior N-day high
            broken10_close = bool(close_price > high10)
            broken5_close = bool(close_price > high5)
            broken10_current = bool(current_price > high10)
            broken5_current = bool(current_price > high5)

            atr_hit_close = bool(close_price >= atr_stop)
            atr_hit_current = bool(current_price >= atr_stop)

        # Build status/action for CLOSE mode
        action10_close, trend10_close = action_and_trend(broken10_close, atr_hit_close)
        action5_close, trend5_close = action_and_trend(broken5_close, atr_hit_close)

        # Build status/action for CURRENT mode
        action10_current, trend10_current = action_and_trend(broken10_current, atr_hit_current)
        action5_current, trend5_current = action_and_trend(broken5_current, atr_hit_current)

        if (
            action10_close == "SELL"
            or action5_close == "SELL"
            or action10_current == "SELL"
            or action5_current == "SELL"
        ):
            any_sell = True

        atr_note_close = " (ATR STOP HIT)" if atr_hit_close else ""
        atr_note_current = " (ATR STOP HIT)" if atr_hit_current else ""

        report.append(
            f"""Ticker: {ticker} ({direction})
Option: {option_name}

Completed-bars levels:
ATR({ATR_PERIOD}): {atr:.2f}
ATR Stop ({ATR_MULTIPLIER}x): {atr_stop:.2f}
Prior 10d Low/High: {low10:.2f} / {high10:.2f}
Prior 5d  Low/High: {low5:.2f} / {high5:.2f}

Prices:
Close price used:   {close_price:.2f}{atr_note_close}
Current price used: {current_price:.2f}{atr_note_current}

=== Based on CLOSE price ===
10 Days action : {action10_close}
10 Days trend  : {trend10_close}
5 Days action  : {action5_close}
5 Days trend   : {trend5_close}

=== Based on CURRENT price ===
10 Days action : {action10_current}
10 Days trend  : {trend10_current}
5 Days action  : {action5_current}
5 Days trend   : {trend5_current}
"""
        )

    subject = "🚨 SELL SIGNALS – Daily Check" if any_sell else "✅ Daily Trend Check – All Intact"
    body = "\n-------------------------\n".join(report) if report else "No valid positions found in positions.csv."

    if not smtp_ready():
        print("SMTP secrets not set — printing report instead\n")
        print(subject)
        print(body)
        return

    if EMAIL_MODE == "exits_only" and not any_sell:
        return

    send_email(subject, body)


if __name__ == "__main__":
    main()