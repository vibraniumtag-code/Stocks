#!/usr/bin/env python3
"""
exit_check.py

Nightly Turtle-style exit checker:
- Pulls daily OHLC for each ticker in positions.csv
- Calculates ATR(14)
- Calculates ATR stop (1.5x ATR from underlying entry)
- Checks structure break using N-day low/high (default 10)
- Sends an email summary (or prints to logs if email env vars missing)

CSV expected columns:
ticker, option_name, option_entry_price, entry_date, underlying_entry_price

Notes:
- ATR/structure are computed on the UNDERLYING, not the option premium.
- option_name is used to infer CALL/PUT if it contains " C " or " P ".
"""

import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# CONFIG
# =========================
ATR_PERIOD = 14
STRUCTURE_DAYS = 10
ATR_MULTIPLIER = 1.5
CSV_FILE = "positions.csv"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT_SSL = 465

EMAIL_USER = os.getenv("EMAIL_USER", "").strip()
EMAIL_PASS = os.getenv("EMAIL_PASS", "").strip()
EMAIL_TO = os.getenv("EMAIL_TO", "").strip()

# Email mode:
# - "always": send daily even if no exits (default)
# - "exits_only": send only when at least one EXIT exists
EMAIL_MODE = os.getenv("EMAIL_MODE", "always").strip().lower()


# =========================
# HELPERS
# =========================
def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns; flatten to single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR using Wilder-style True Range rolling mean (simple rolling mean here).
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
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

    atr = tr.rolling(period).mean()
    return atr


def infer_direction(option_name: str) -> Tuple[Optional[bool], str]:
    """
    Returns (is_call, label)
      is_call=True => bullish (CALL)
      is_call=False => bearish (PUT)
      is_call=None => cannot infer
    """
    name = f" {option_name} "
    if " C " in name:
        return True, "CALL"
    if " P " in name:
        return False, "PUT"
    return None, "UNKNOWN"


def send_email(subject: str, body: str) -> None:
    """Send email via Gmail SMTP SSL. Requires EMAIL_USER/PASS/TO to be set."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT_SSL) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)


def email_ready() -> bool:
    return bool(EMAIL_USER and EMAIL_PASS and EMAIL_TO)


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


# =========================
# MAIN LOGIC
# =========================
def main() -> None:
    # Load positions
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"Could not find {CSV_FILE}. Place it in the repo root (same folder as exit_check.py)."
        )

    positions = pd.read_csv(CSV_FILE)

    required_cols = {"ticker", "option_name", "option_entry_price", "underlying_entry_price"}
    missing = required_cols - set(positions.columns)
    if missing:
        raise ValueError(
            f"positions.csv missing columns: {sorted(missing)}. "
            f"Required: {sorted(required_cols)}"
        )

    report_blocks = []
    exit_found = False

    for idx, row in positions.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        option_name = str(row["option_name"]).strip()

        underlying_entry = safe_float(row.get("underlying_entry_price"))
        option_entry = safe_float(row.get("option_entry_price"))

        is_call, dir_label = infer_direction(option_name)

        # Basic validation
        if not ticker or ticker == "NAN":
            report_blocks.append(f"Row {idx+1}: Invalid ticker.")
            continue

        if underlying_entry is None:
            report_blocks.append(
                f"""Ticker: {ticker}
Option: {option_name}
Action: HOLD (NO CHECK)
Reason: Missing underlying_entry_price in CSV (cannot compute ATR stop / structure properly).
"""
                .strip()
            )
            continue

        if is_call is None:
            report_blocks.append(
                f"""Ticker: {ticker}
Option: {option_name}
Action: HOLD (NO CHECK)
Reason: Cannot infer CALL/PUT from option_name. Include ' C ' or ' P ' in the name.
"""
                .strip()
            )
            continue

        # Pull daily data
        df = yf.download(ticker, period="9mo", interval="1d", progress=False)
        if df is None or df.empty:
            report_blocks.append(
                f"""Ticker: {ticker}
Option: {option_name}
Action: HOLD (NO CHECK)
Reason: No market data returned from yfinance.
"""
                .strip()
            )
            continue

        df = flatten_yf_columns(df)
        df = df.dropna()

        min_needed = max(ATR_PERIOD, STRUCTURE_DAYS) + 2
        if len(df) < min_needed:
            report_blocks.append(
                f"""Ticker: {ticker}
Option: {option_name}
Action: HOLD (NO CHECK)
Reason: Not enough data ({len(df)} rows). Need at least {min_needed}.
"""
                .strip()
            )
            continue

        # Compute indicators (force scalars to avoid "Series is ambiguous" errors)
        atr_series = calculate_atr(df, ATR_PERIOD)
        atr_val = atr_series.iloc[-1]
        if pd.isna(atr_val):
            report_blocks.append(
                f"""Ticker: {ticker}
Option: {option_name}
Action: HOLD (NO CHECK)
Reason: ATR is NaN (insufficient rolling window / data issues).
"""
                .strip()
            )
            continue

        atr = float(atr_val)
        close = float(df["Close"].iloc[-1])

        recent = df.iloc[-STRUCTURE_DAYS:]
        low_n = float(recent["Low"].min())
        high_n = float(recent["High"].max())

        # ATR stop and structure break logic
        if is_call:
            # Bullish: stop below entry, structure break if close < N-day low
            atr_stop = float(underlying_entry - ATR_MULTIPLIER * atr)
            atr_hit = bool(close <= atr_stop)
            structure_hit = bool(close < low_n)
            structure_desc = f"Close < {STRUCTURE_DAYS}-day low"
        else:
            # Bearish: stop above entry, structure break if close > N-day high
            atr_stop = float(underlying_entry + ATR_MULTIPLIER * atr)
            atr_hit = bool(close >= atr_stop)
            structure_hit = bool(close > high_n)
            structure_desc = f"Close > {STRUCTURE_DAYS}-day high"

        # Decision priority: ATR stop first, then structure
        if atr_hit:
            action = "EXIT"
            reason = f"ATR stop hit ({ATR_MULTIPLIER}x ATR)"
            exit_found = True
        elif structure_hit:
            action = "EXIT"
            reason = f"Structure break: {structure_desc}"
            exit_found = True
        else:
            action = "HOLD"
            reason = "Trend intact"

        # Optional context: unrealized option PnL (informational only)
        opt_pnl_txt = ""
        if option_entry is not None:
            # We don't fetch option price here (needs option chain). Keep it simple.
            opt_pnl_txt = f"\nOption entry (for reference): {option_entry:.2f}"

        report_blocks.append(
            f"""Ticker: {ticker} ({dir_label})
Option: {option_name}
Action: {action}
Reason: {reason}
Underlying Entry: {underlying_entry:.2f}
Close: {close:.2f}
ATR({ATR_PERIOD}): {atr:.2f}
ATR Stop ({ATR_MULTIPLIER}x): {atr_stop:.2f}
{STRUCTURE_DAYS}-Day Low/High: {low_n:.2f} / {high_n:.2f}{opt_pnl_txt}
"""
            .strip()
        )

    # Build email/log output
    subject = "🚨 EXIT SIGNALS – Daily Check" if exit_found else "✅ Daily Trend Check – No Action"
    body = "\n\n" + ("-" * 28) + "\n\n"
    body = body.join(report_blocks) if report_blocks else "No positions found in positions.csv."

    # Email mode logic
    should_send = True
    if EMAIL_MODE == "exits_only" and not exit_found:
        should_send = False

    if not email_ready():
        # No secrets set; print to logs so workflow still succeeds for testing
        print("EMAIL secrets not set (EMAIL_USER/EMAIL_PASS/EMAIL_TO). Printing report to logs.\n")
        print(subject)
        print(body)
        return

    if should_send:
        send_email(subject, body)
        print(f"Email sent: {subject}")
    else:
        print("No exits found and EMAIL_MODE=exits_only. No email sent.")
        print(subject)
        print(body)


if __name__ == "__main__":
    main()