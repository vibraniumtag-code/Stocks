#!/usr/bin/env python3
"""
exit_check.py

Nightly Turtle-style exit checker with dual structure exits:
- ATR stop (1.5x ATR from underlying entry)
- 10-day structure break (core trend)
- 5-day structure break (profit protection)

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

STRUCTURE_PRIMARY = 10   # core trend
STRUCTURE_TIGHT = 5      # profit protection

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


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError("positions.csv not found")

    positions = pd.read_csv(CSV_FILE)

    report = []
    exit_found = False

    min_needed = max(ATR_PERIOD, STRUCTURE_PRIMARY) + 5

    for _, row in positions.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        option_name = str(row["option_name"]).strip()
        entry_underlying = to_float(row["underlying_entry_price"])

        is_call, direction = infer_direction(option_name)

        if not ticker or entry_underlying is None or is_call is None:
            report.append(f"{ticker}: SKIPPED (invalid row)")
            continue

        df = yf.download(ticker, period=HISTORY_PERIOD, interval="1d", progress=False)
        if df.empty:
            report.append(f"{ticker}: NO DATA")
            continue

        df = flatten_columns(df).dropna()

        if len(df) < min_needed:
            report.append(f"{ticker}: SKIPPED (insufficient history)")
            continue

        atr_series = calculate_atr(df, ATR_PERIOD)
        atr_last = atr_series.iloc[-1]
        if pd.isna(atr_last):
            report.append(f"{ticker}: SKIPPED (ATR NaN)")
            continue

        atr = float(atr_last)
        close = float(df["Close"].iloc[-1])

        # === Structure windows (exclude today)
        w10 = prior_window(df, STRUCTURE_PRIMARY)
        w5 = prior_window(df, STRUCTURE_TIGHT)

        low10, high10 = float(w10["Low"].min()), float(w10["High"].max())
        low5, high5 = float(w5["Low"].min()), float(w5["High"].max())

        # === ATR stop
        if is_call:
            atr_stop = entry_underlying - ATR_MULTIPLIER * atr
            atr_hit = close <= atr_stop
            break_10 = close < low10
            break_5 = close < low5
        else:
            atr_stop = entry_underlying + ATR_MULTIPLIER * atr
            atr_hit = close >= atr_stop
            break_10 = close > high10
            break_5 = close > high5

        # === Decision priority
        if atr_hit:
            action = "EXIT"
            reason = "ATR stop hit"
            exit_found = True
        elif break_10:
            action = "EXIT"
            reason = f"{STRUCTURE_PRIMARY}-day structure break (trend)"
            exit_found = True
        elif break_5:
            action = "EXIT"
            reason = f"{STRUCTURE_TIGHT}-day structure break (profit protection)"
            exit_found = True
        else:
            action = "HOLD"
            reason = "Trend intact"

        report.append(
            f"""Ticker: {ticker} ({direction})
Option: {option_name}
Action: {action}
Reason: {reason}
Close: {close:.2f}
ATR({ATR_PERIOD}): {atr:.2f}
ATR Stop: {atr_stop:.2f}
Prior 10d Low/High: {low10:.2f} / {high10:.2f}
Prior 5d  Low/High: {low5:.2f} / {high5:.2f}
"""
        )

    subject = "🚨 EXIT SIGNALS – Daily Check" if exit_found else "✅ Daily Trend Check – No Action"
    body = "\n-------------------------\n".join(report)

    if not smtp_ready():
        print("SMTP secrets not set — printing report instead\n")
        print(subject)
        print(body)
        return

    if EMAIL_MODE == "exits_only" and not exit_found:
        return

    send_email(subject, body)


if __name__ == "__main__":
    main()