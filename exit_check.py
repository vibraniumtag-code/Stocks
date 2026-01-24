#!/usr/bin/env python3
"""
exit_check.py

Outputs:
- CLOSE-based: 10d & 5d trend (INTACT/BROKEN) and actions (HOLD/SELL)
- CURRENT-based: 10d & 5d trend (INTACT/BROKEN) and actions (HOLD/SELL)
- ATR advice (CLOSE & CURRENT): OK / CLOSE / VERY CLOSE / STOP HIT (informational)
- RECOMMENDATION: HOLD / SELL_1 / SELL_2 / ... / SELL_ALL (based on contracts)

Structure levels + ATR are computed from COMPLETED daily bars only
(today's partial bar is excluded when present).

CSV required columns:
ticker, option_name, underlying_entry_price, option_entry_price, contracts

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

# ATR proximity advice thresholds (in ATR units)
ATR_VERY_CLOSE = float(os.getenv("ATR_VERY_CLOSE", "0.25"))
ATR_CLOSE = float(os.getenv("ATR_CLOSE", "0.50"))

CSV_FILE = "positions.csv"

# If true, script will reduce contracts in CSV based on RECOMMENDATION (paper execution)
APPLY_RECOMMENDATIONS = os.getenv("APPLY_RECOMMENDATIONS", "false").strip().lower() == "true"

# Recommendation mode: which price to use for the recommendation
# "current" (default) or "close"
RECO_MODE = os.getenv("RECO_MODE", "current").strip().lower()

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


def to_int(x, default: int = 0) -> int:
    try:
        v = int(float(x))
        return max(v, 0)
    except Exception:
        return default


def smtp_ready() -> bool:
    return all([SMTP_HOST, SMTP_PORT > 0, SMTP_USER, SMTP_PASS, EMAIL_TO])


def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    
    msg["To"] = EMAIL_TO
    msg["From"] = f"Scanner <{SMTP_USER}>"
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
    # prior N bars excluding the most recent completed bar
    return df_completed.iloc[-(n + 1):-1]


def action_and_trend(structure_broken: bool) -> Tuple[str, str]:
    trend = "BROKEN" if structure_broken else "INTACT"
    action = "SELL" if structure_broken else "HOLD"
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

    # 3) intraday
    try:
        intraday = t.history(period="1d", interval="1m")
        if intraday is not None and not intraday.empty and "Close" in intraday.columns:
            return float(intraday["Close"].iloc[-1]), "history(1d,1m).last_close"
    except Exception:
        pass

    return float(fallback), "fallback_daily_close"


def atr_advice(is_call: bool, price: float, atr_stop: float, atr: float) -> Tuple[str, float, float, bool]:
    """
    Returns: (advice_text, distance_dollars, distance_atr_units, stop_hit_bool)
    distance is "room until stop" (positive=room left, negative=beyond stop):
      CALL: dist = price - stop
      PUT : dist = stop - price
    """
    if atr <= 0:
        return "ATR unavailable", 0.0, 0.0, False

    if is_call:
        dist = float(price - atr_stop)
        stop_hit = bool(price <= atr_stop)
    else:
        dist = float(atr_stop - price)
        stop_hit = bool(price >= atr_stop)

    dist_atr = float(dist / atr)

    if stop_hit:
        return "STOP HIT", dist, dist_atr, True
    if dist_atr <= ATR_VERY_CLOSE:
        return f"VERY CLOSE (≤ {ATR_VERY_CLOSE:.2f} ATR)", dist, dist_atr, False
    if dist_atr <= ATR_CLOSE:
        return f"CLOSE (≤ {ATR_CLOSE:.2f} ATR)", dist, dist_atr, False
    return "OK", dist, dist_atr, False


def decide_sell_count(contracts: int, broken10: bool, broken5: bool) -> int:
    """
    Sell sizing logic (edit as you like):
      - 10d broken => sell all
      - else 5d broken => sell 2 if >=3, sell 1 if ==2, sell 1 if ==1 (all)
      - else => 0
    """
    if contracts <= 0:
        return 0
    if broken10:
        return contracts
    if broken5:
        if contracts >= 3:
            return 2
        return 1
    return 0


def label_action(sell_count: int, contracts: int) -> str:
    if contracts <= 0:
        return "NO_POSITION"
    if sell_count <= 0:
        return "HOLD"
    if sell_count >= contracts:
        return "SELL_ALL"
    return f"SELL_{sell_count}"


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError("positions.csv not found in repo root")

    positions = pd.read_csv(CSV_FILE)

    required_cols = {"ticker", "option_name", "underlying_entry_price", "option_entry_price", "contracts"}
    missing = required_cols - set(positions.columns)
    if missing:
        raise ValueError(f"positions.csv missing required columns: {sorted(missing)}")

    report = []
    any_action = False

    updated_positions = positions.copy()

    min_needed = max(ATR_PERIOD, STRUCTURE_10) + 10  # buffer

    for idx, row in positions.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        option_name = str(row.get("option_name", "")).strip()
        entry_underlying = to_float(row.get("underlying_entry_price"))
        contracts = to_int(row.get("contracts"), default=0)

        is_call, direction = infer_direction(option_name)

        if not ticker or entry_underlying is None or is_call is None:
            report.append(f"{ticker or 'UNKNOWN'}: SKIPPED (invalid row)")
            continue

        if contracts <= 0:
            report.append(f"{ticker}: NO_POSITION (contracts=0)")
            continue

        df = yf.download(ticker, period=HISTORY_PERIOD, interval="1d", progress=False)
        if df is None or df.empty:
            report.append(f"{ticker}: NO DATA")
            continue

        df = flatten_columns(df).dropna()

        close_price = float(df["Close"].iloc[-1])
        current_price, current_src = get_current_price(ticker, fallback=close_price)

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

        w10 = prior_window(df_completed, STRUCTURE_10)
        w5 = prior_window(df_completed, STRUCTURE_5)
        if w10.empty or w5.empty:
            report.append(f"{ticker}: SKIPPED (structure windows empty)")
            continue

        low10, high10 = float(w10["Low"].min()), float(w10["High"].max())
        low5, high5 = float(w5["Low"].min()), float(w5["High"].max())

        # ATR stop (from underlying entry)
        if is_call:
            atr_stop = float(entry_underlying - ATR_MULTIPLIER * atr)

            # Structure break rules (CALL)
            broken10_close = bool(close_price < low10)
            broken5_close = bool(close_price < low5)
            broken10_current = bool(current_price < low10)
            broken5_current = bool(current_price < low5)
        else:
            atr_stop = float(entry_underlying + ATR_MULTIPLIER * atr)

            # Structure break rules (PUT)
            broken10_close = bool(close_price > high10)
            broken5_close = bool(close_price > high5)
            broken10_current = bool(current_price > high10)
            broken5_current = bool(current_price > high5)

        # Build the exact sections you asked for
        action10_close, trend10_close = action_and_trend(broken10_close)
        action5_close, trend5_close = action_and_trend(broken5_close)

        action10_current, trend10_current = action_and_trend(broken10_current)
        action5_current, trend5_current = action_and_trend(broken5_current)

        # ATR advice for both prices
        atr_adv_close, atr_dist_close, atr_dist_close_atr, atr_hit_close = atr_advice(is_call, close_price, atr_stop, atr)
        atr_adv_cur, atr_dist_cur, atr_dist_cur_atr, atr_hit_cur = atr_advice(is_call, current_price, atr_stop, atr)

        # Recommendation (uses RECO_MODE)
        if RECO_MODE == "close":
            sell_count = decide_sell_count(contracts, broken10_close, broken5_close)
            reco_basis = "CLOSE"
        else:
            sell_count = decide_sell_count(contracts, broken10_current, broken5_current)
            reco_basis = "CURRENT"

        recommendation = label_action(sell_count, contracts)

        if recommendation != "HOLD":
            any_action = True

        # Optional: paper update contracts in CSV
        if APPLY_RECOMMENDATIONS and sell_count > 0:
            new_contracts = max(contracts - sell_count, 0)
            updated_positions.at[idx, "contracts"] = new_contracts

        report.append(
            f"""Ticker: {ticker} ({direction})
Option: {option_name}
Contracts: {contracts}

Levels (completed daily bars):
ATR({ATR_PERIOD}): {atr:.2f}
ATR Stop ({ATR_MULTIPLIER}x): {atr_stop:.2f}
Prior 10d Low/High: {low10:.2f} / {high10:.2f}
Prior 5d  Low/High: {low5:.2f} / {high5:.2f}

Prices:
Close price used:   {close_price:.2f}
Current price used: {current_price:.2f}  (source: {current_src})

ATR advice:
CLOSE:   {atr_adv_close}   dist {atr_dist_close:+.2f} USD ({atr_dist_close_atr:+.2f} ATR)
CURRENT: {atr_adv_cur}     dist {atr_dist_cur:+.2f} USD ({atr_dist_cur_atr:+.2f} ATR)

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

RECOMMENDATION ({reco_basis}): {recommendation}
"""
        )

    subject = "🚨 ACTION NEEDED – Recommendations" if any_action else "✅ No Action – Hold"
    body = "\n-------------------------\n".join(report) if report else "No valid positions found in positions.csv."

    if APPLY_RECOMMENDATIONS and not updated_positions.equals(positions):
        updated_positions.to_csv(CSV_FILE, index=False)
        body += "\n\nNOTE: APPLY_RECOMMENDATIONS=true — positions.csv contracts were reduced (paper execution).\n"

    if not smtp_ready():
        print("SMTP secrets not set — printing report instead\n")
        print(subject)
        print(body)
        return

    if EMAIL_MODE == "exits_only" and not any_action:
        return

    send_email(subject, body)


if __name__ == "__main__":
    main()