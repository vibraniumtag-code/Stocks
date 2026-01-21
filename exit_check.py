#!/usr/bin/env python3
"""
exit_check.py

Nightly Turtle-style exit checker with TWO independent structure (trend) checks:
- ATR stop (1.5x ATR from underlying entry) - reported, but does not override the per-window statuses
- 10-day structure status (trend intact/broken) + action (HOLD/SELL)
- 5-day structure status (trend intact/broken) + action (HOLD/SELL)

Structure windows EXCLUDE today.
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

EMAIL_MODE = os.getenv("EMAIL_MODE", "always").strip().lower()  # always | exits_only
HISTORY_PERIOD = os.getenv("HISTORY_PERIOD", "9mo").strip()


# =========================
# HELPERS
# =========================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
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


def infer_direction(option_name: str) -> Tuple[Optional[bool], str]:
    name = f" {option_name} "
    if " C " in name:
        return True, "CALL"
    if " P " in name:
        return False, "PUT"
    return None, "UNKNOWN"


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


def to_float(x) -> Optional[float]:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def prior_window(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return prior n rows excluding today."""
    return df.iloc[-(n + 1):-1]


def fmt_action(is_sell: bool) -> str:
    return "SELL" if is_sell else "HOLD"


def fmt_trend(is_broken: bool) -> str:
    return "BROKEN" if is_broken else "INTACT"


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError("positions.csv not found")

    positions = pd.read_csv(CSV_FILE)

    report = []
    any_sell_signal = False

    # Need enough rows for ATR and both structure windows
    min_needed = max(ATR_PERIOD, STRUCTURE_10) + 5

    for _, row in positions.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        option_name = str(row.get("option_name", "")).strip()
        entry_underlying = to_float(row.get("underlying_entry_price"))

        is_call, direction = infer_direction(option_name)

        if not ticker or entry_underlying is None or is_call is None:
            report.append(f"{ticker or 'UNKNOWN'}: SKIPPED (invalid row)")
            continue

        df = yf.download(ticker, period=HISTORY_PERIOD, interval="1d", progress=False)
        if df is None or df.empty:
            report.append(f"{ticker}: NO DATA")
            continue

        df = flatten_columns(df).dropna()

        if len(df) < min_needed:
            report.append(f"{ticker}: SKIPPED (insufficient history: {len(df)} rows)")
            continue

        # ATR
        atr_series = calculate_atr(df, ATR_PERIOD)
        atr_last = atr_series.iloc[-1]
        if pd.isna(atr_last):
            report.append(f"{ticker}: SKIPPED (ATR NaN)")
            continue

        atr = float(atr_last)
        close = float(df["Close"].iloc[-1])

        # Prior windows (exclude today)
        w10 = prior_window(df, STRUCTURE_10)
        w5 = prior_window(df, STRUCTURE_5)

        low10, high10 = float(w10["Low"].min()), float(w10["High"].max())
        low5, high5 = float(w5["Low"].min()), float(w5["High"].max())

        # ATR stop (reported)
        if is_call:
            atr_stop = float(entry_underlying - ATR_MULTIPLIER * atr)
            atr_hit = bool(close <= atr_stop)

            broken10 = bool(close < low10)
            broken5 = bool(close < low5)
        else:
            atr_stop = float(entry_underlying + ATR_MULTIPLIER * atr)
            atr_hit = bool(close >= atr_stop)

            broken10 = bool(close > high10)
            broken5 = bool(close > high5)

        # Two independent statuses/actions
        action10 = fmt_action(broken10 or atr_hit)  # include ATR hit as SELL signal too
        trend10 = fmt_trend(broken10)

        action5 = fmt_action(broken5 or atr_hit)    # include ATR hit as SELL signal too
        trend5 = fmt_trend(broken5)

        # If either window says SELL, flag for email subject
        if action10 == "SELL" or action5 == "SELL":
            any_sell_signal = True

        # Add context on why SELL if ATR is the reason
        atr_note = " (ATR STOP HIT)" if atr_hit else ""

        report.append(
            f"""Ticker: {ticker} ({direction})
Option: {option_name}
Close: {close:.2f}
ATR({ATR_PERIOD}): {atr:.2f}
ATR Stop ({ATR_MULTIPLIER}x): {atr_stop:.2f}{atr_note}
Prior 10d Low/High: {low10:.2f} / {high10:.2f}
Prior 5d  Low/High: {low5:.2f} / {high5:.2f}

10 Days action: {action10}
10 Days trend : {trend10}

5 Days action : {action5}
5 Days trend  : {trend5}
"""
        )

    subject = "🚨 SELL SIGNALS – Daily Check" if any_sell_signal else "✅ Daily Trend Check – All Intact"
    body = "\n-------------------------\n".join(report) if report else "No valid positions found in positions.csv."

    if not smtp_ready():
        print("SMTP secrets not set — printing report instead\n")
        print(subject)
        print(body)
        return

    if EMAIL_MODE == "exits_only" and not any_sell_signal:
        return

    send_email(subject, body)


if __name__ == "__main__":
    main()