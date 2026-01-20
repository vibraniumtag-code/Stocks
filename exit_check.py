#!/usr/bin/env python3
"""
exit_check.py

Nightly Turtle-style exit checker (swing / daily bars):
- Pulls daily OHLC for each ticker in positions.csv
- Calculates ATR(14)
- Calculates ATR stop (1.5x ATR from underlying entry)
- Checks structure break using N-day window (default 10), EXCLUDING today
- Sends an email summary (SMTP generic)

CSV required columns:
ticker, option_name, option_entry_price, entry_date, underlying_entry_price

Direction inference:
- If option_name contains " C " => CALL (bullish)
- If option_name contains " P " => PUT (bearish)

Structure break (default):
- CALL: exit if Close < prior N-day LOW (or optionally prior N-day CLOSE low)
- PUT : exit if Close > prior N-day HIGH (or optionally prior N-day CLOSE high)
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
STRUCTURE_DAYS = 10
ATR_MULTIPLIER = 1.5
CSV_FILE = "positions.csv"

# Structure method:
# - "lowhigh" => compare Close vs prior N-day LOW/HIGH (stricter)
# - "close"   => compare Close vs prior N-day CLOSE low/high (often smoother)
STRUCTURE_METHOD = os.getenv("STRUCTURE_METHOD", "lowhigh").strip().lower()

# SMTP / EMAIL (FROM GITHUB SECRETS)
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "0"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
EMAIL_TO = os.getenv("EMAIL_TO", "").strip()

# Email mode:
# - "always"     => send daily summary
# - "exits_only" => send only when at least one EXIT exists
EMAIL_MODE = os.getenv("EMAIL_MODE", "always").strip().lower()

# yfinance download window
HISTORY_PERIOD = os.getenv("HISTORY_PERIOD", "9mo").strip()


# =========================
# HELPERS
# =========================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance can return MultiIndex columns; flatten to single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


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


def infer_direction(option_name: str) -> Tuple[Optional[bool], str]:
    """Return (is_call, label). is_call True=CALL, False=PUT, None=unknown."""
    name = f" {option_name} "
    if " C " in name:
        return True, "CALL"
    if " P " in name:
        return False, "PUT"
    return None, "UNKNOWN"


def smtp_ready() -> bool:
    return all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO])


def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO

    # STARTTLS is typical on 587. If your provider uses 465 SSL,
    # set SMTP_PORT=465 and STARTTLS will still work on many providers,
    # but best practice is to use SMTP_SSL for 465.
    if SMTP_PORT == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
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
    """
    Return the PRIOR n rows excluding today's row.
    Uses df.iloc[-(n+1):-1].
    """
    return df.iloc[-(n + 1) : -1]


# =========================
# MAIN
# =========================
def main() -> None:
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError("positions.csv not found in repo root")

    positions = pd.read_csv(CSV_FILE)

    required_cols = {"ticker", "option_name", "underlying_entry_price"}
    missing = required_cols - set(positions.columns)
    if missing:
        raise ValueError(f"positions.csv missing required columns: {sorted(missing)}")

    report = []
    exit_found = False

    min_needed = max(ATR_PERIOD, STRUCTURE_DAYS) + 5  # extra buffer for safety

    for _, row in positions.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        option_name = str(row.get("option_name", "")).strip()
        entry_underlying = to_float(row.get("underlying_entry_price"))
        option_entry = to_float(row.get("option_entry_price"))  # informational only

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

        if len(df) < min_needed:
            report.append(f"{ticker}: SKIPPED (not enough history: {len(df)} rows)")
            continue

        # Calculate ATR safely
        atr_series = calculate_atr(df, ATR_PERIOD)
        atr_last = atr_series.iloc[-1]
        if pd.isna(atr_last):
            report.append(f"{ticker}: SKIPPED (ATR is NaN)")
            continue

        atr = float(atr_last)
        close = float(df["Close"].iloc[-1])

        # Structure window EXCLUDING today
        w = prior_window(df, STRUCTURE_DAYS)
        if w.empty or len(w) < STRUCTURE_DAYS:
            report.append(f"{ticker}: SKIPPED (structure window too small)")
            continue

        if STRUCTURE_METHOD == "close":
            low_level = float(w["Close"].min())
            high_level = float(w["Close"].max())
            level_label = f"Prior {STRUCTURE_DAYS}-day Close Low/High"
        else:
            # default: "lowhigh"
            low_level = float(w["Low"].min())
            high_level = float(w["High"].max())
            level_label = f"Prior {STRUCTURE_DAYS}-day Low/High"

        # Signals
        if is_call:
            atr_stop = float(entry_underlying - ATR_MULTIPLIER * atr)
            atr_hit = bool(close <= atr_stop)
            structure_hit = bool(close < low_level)
        else:
            atr_stop = float(entry_underlying + ATR_MULTIPLIER * atr)
            atr_hit = bool(close >= atr_stop)
            structure_hit = bool(close > high_level)

        # Decision priority
        if atr_hit:
            action = "EXIT"
            reason = f"ATR stop hit ({ATR_MULTIPLIER}x ATR)"
            exit_found = True
        elif structure_hit:
            action = "EXIT"
            reason = f"Structure break ({STRUCTURE_METHOD}): {level_label}"
            exit_found = True
        else:
            action = "HOLD"
            reason = "Trend intact"

        opt_entry_txt = f"{option_entry:.2f}" if option_entry is not None else "N/A"

        report.append(
            f"""Ticker: {ticker} ({direction})
Option: {option_name}
Action: {action}
Reason: {reason}
Underlying Entry: {entry_underlying:.2f}
Close: {close:.2f}
ATR({ATR_PERIOD}): {atr:.2f}
ATR Stop: {atr_stop:.2f}
{level_label}: {low_level:.2f} / {high_level:.2f}
Option Entry (ref): {opt_entry_txt}
"""
        )

    subject = "🚨 EXIT SIGNALS – Daily Check" if exit_found else "✅ Daily Trend Check – No Action"
    body = "\n-------------------------\n".join(report) if report else "No valid positions found in positions.csv."

    if not smtp_ready():
        print("SMTP secrets not set — printing report instead\n")
        print(subject)
        print(body)
        return

    if EMAIL_MODE == "exits_only" and not exit_found:
        print("No exits found — email suppressed (EMAIL_MODE=exits_only)")
        return

    send_email(subject, body)
    print("Email sent successfully")


if __name__ == "__main__":
    main()
    