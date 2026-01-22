#!/usr/bin/env python3
"""
exit_check.py

Exit checker with TWO evaluation modes:

A) Based on "CLOSE"  (last available daily close from yfinance 1d bars)
B) Based on "CURRENT" (best-effort intraday/last price)

For each mode, outputs TWO independent trend checks:
- 10 Days: action HOLD/SELL + trend INTACT/BROKEN
- 5 Days : action HOLD/SELL + trend INTACT/BROKEN

Structure levels + ATR are computed from COMPLETED daily bars only
(today's partial bar is excluded when present).

CSV required columns:
ticker, option_name, underlying_entry_price
(option_name must contain " C " for calls or " P " for puts)

SMTP env vars (GitHub Secrets):
SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO
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


def remove_today_partial_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    If yfinance includes a row for today's date (often partial intraday),
    drop it so indicators/structure levels use completed bars only.
    """
    if df.empty:
        return df
    try:
        today_utc = pd.Timestamp.utcnow().date()
        last_date = pd.Timestamp(df.index[-1]).date()
        if last_date == today_utc:
            return df.iloc[:-1]
    except Exception:
        pass
    return df


def prior_window(df_completed: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return prior n rows excluding the most recent completed bar."""
    return df_completed.iloc[-(n + 1):-1]


def action_and_trend(structure_broken: bool, atr_hit: bool) -> Tuple[str, str]:
    trend = "BROKEN" if structure_broken else "INTACT"
    action = "SELL" if (structure_broken or atr_hit) else "HOLD"
    return action, trend


def get_current_price(ticker: str, fallback: float) -> Tuple[float, str]:
    """
    Best-effort current/last price with a source label:
    1) fast_info.last_price
    2) info.regularMarketPrice
    3) 1m intraday history last close
    Fallback to provided value if all fail.
    """
    t = yf.Ticker(ticker)

    # 1) fast_info
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            lp = fi.get("last_price", None)
            if lp is not None:
                return float(lp), "fast_info.last_price"
    except Exception:
        pass

    # 2) info
    try:
        info = getattr(t, "info", None)
        if isinstance(info, dict):
            rmp = info.get("regularMarketPrice", None)
            if rmp is not None:
                return float(rmp), "info.regularMarketPrice"
    except Exception:
        pass

    # 3) 1-minute intraday last bar
    try:
        intraday = t.history(period="1d", interval="1m")
        if intraday is not None and not intraday.empty and "Close" in intraday.columns:
            return float(intraday["Close"].iloc[-1]), "history(1d,1m).last_close"
    except Exception:
        pass

    return float(fallback), "fallback_daily_close"


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

        # Daily bars (may include today's partial row depending on timing)
        df = yf.download(ticker, period=HISTORY_PERIOD, interval="1d", progress=False)
        if df is None or df.empty:
            report.append(f"{ticker}: NO DATA")
            continue

        df = flatten_columns(df).dropna()

        # Close-based price is the last daily close from the downloaded daily bars
        close_price = float(df["Close"].iloc[-1])

        # Current-based price is best-effort intraday/last
        current_price, current_src = get_current_price(ticker, fallback=close_price)

        # Compute levels from completed bars only
        df_completed = remove_today_partial_bar(df).dropna()

        if len(df_completed) < min_needed:
            report.append(f"{ticker}: SKIPPED (insufficient completed history: {len(df_completed)} rows)")
            continue

        atr_series = calculate_atr(df_completed, ATR_PERIOD)
        atr_last = atr_series.iloc[-1]
        if pd.isna(atr_last):
            report.append(f"{ticker}: SKIPPED (ATR NaN)")
            continue
        atr = float(atr_last)

        # Structure levels (exclude the evaluation day) using completed bars
        w10 = prior_window(df_completed, STRUCTURE_10)
        w5 = prior_window(df_completed, STRUCTURE_5)

        if w10.empty or w5.empty:
            report.append(f"{ticker}: SKIPPED (structure windows empty)")
            continue

        low10, high10 = float(w10["Low"].min()), float(w10["High"].max())
        low5, high5 = float(w5["Low"].min()), float(w5["High"].max())

        # ATR stop value (same for both modes; ATR from completed bars)
        if is_call:
            atr_stop = float(entry_underlying - ATR_MULTIPLIER * atr)

            # Structure breaks
            broken10_close = bool(close_price < low10)
            broken5_close = bool(close_price < low5)
            broken10_current = bool(current_price < low10)
            broken5_current = bool(current_price < low5)

            # ATR hits
            atr_hit_close = bool(close_price <= atr_stop)
            atr_hit_current = bool(current_price <= atr_stop)
        else:
            atr_stop = float(entry_underlying + ATR_MULTIPLIER * atr)

            # Structure breaks
            broken10_close = bool(close_price > high10)
            broken5_close = bool(close_price > high5)
            broken10_current = bool(current_price > high10)
            broken5_current = bool(current_price > high5)

            # ATR hits
            atr_hit_close = bool(close_price >= atr_stop)
            atr_hit_current = bool(current_price >= atr_stop)

        # Close-mode actions/trends
        action10_close, trend10_close = action_and_trend(broken10_close, atr_hit_close)
        action5_close, trend5_close = action_and_trend(broken5_close, atr_hit_close)

        # Current-mode actions/trends
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

Levels (from completed daily bars):
ATR({ATR_PERIOD}): {atr:.2f}
ATR Stop ({ATR_MULTIPLIER}x): {atr_stop:.2f}
Prior 10d Low/High: {low10:.2f} / {high10:.2f}
Prior 5d  Low/High: {low5:.2f} / {high5:.2f}

Prices:
Close price used:   {close_price:.2f}{atr_note_close}
Current price used: {current_price:.2f}{atr_note_current}
Current source:     {current_src}

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