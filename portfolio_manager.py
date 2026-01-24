#!/usr/bin/env python3
"""
portfolio_manager.py

Keeps your current positions.csv format.
Reads existing positions and optimizes portfolio:

- Values existing option positions using option_name parsing + yfinance option_chain
- Runs exit checks on underlying (structure 10/5 on CLOSE and CURRENT + ATR advice)
- Imports and runs your entry scanner (unified_turtle_entries_only.py) to get tomorrow entries
- Builds a plan:
    * SELL some/all existing positions if structure broken OR oversized vs slot budget
    * BUY new entries if there is capacity + cash budget
- Outputs:
    * portfolio_plan.csv
    * email summary

Positions CSV: your existing format, plus optional contracts column.
Required columns (must already exist in your file):
ticker, option_name, option_entry_price, underlying_entry_price
Optional:
contracts (defaults to 1 if missing), entry_date (ignored)

Example option_name format expected:
"NFLX 2026-02-20 P 88"
"AAPL 2026-03-20 C 200"
"""

import os
import re
import ssl
import smtplib
from datetime import datetime, date
from email.mime.text import MIMEText
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# ENV CONFIG
# =========================
CSV_FILE = os.getenv("CSV_FILE", "positions.csv").strip()
PLAN_FILE = os.getenv("PLAN_FILE", "portfolio_plan.csv").strip()

# Portfolio controls
MAX_POSITIONS = int(float(os.getenv("MAX_POSITIONS", "4")))
CASH_BUFFER_PCT = float(os.getenv("CASH_BUFFER_PCT", "0.05"))  # keep 5% as cash buffer target
MAX_NEW_PER_RUN = int(float(os.getenv("MAX_NEW_PER_RUN", "2"))) # don't add too many at once
MAX_CONTRACTS_PER_POSITION = int(float(os.getenv("MAX_CONTRACTS_PER_POSITION", "6")))
CONTRACT_MULTIPLIER = int(float(os.getenv("CONTRACT_MULTIPLIER", "100")))

# Exit logic
ATR_PERIOD = int(float(os.getenv("ATR_PERIOD", "14")))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))
STRUCTURE_10 = int(float(os.getenv("STRUCTURE_10", "10")))
STRUCTURE_5 = int(float(os.getenv("STRUCTURE_5", "5")))

ATR_VERY_CLOSE = float(os.getenv("ATR_VERY_CLOSE", "0.25"))
ATR_CLOSE = float(os.getenv("ATR_CLOSE", "0.50"))

RECO_MODE = os.getenv("RECO_MODE", "current").strip().lower()  # current | close

# Oversize redistribution rule
OVERSIZE_MULT = float(os.getenv("OVERSIZE_MULT", "1.40"))  # if position value > slot_budget*1.40, trim
TRIM_TO_SLOT = os.getenv("TRIM_TO_SLOT", "true").strip().lower() == "true"

# Email
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT_RAW = (os.getenv("SMTP_PORT") or "").strip()
SMTP_PORT = int(SMTP_PORT_RAW) if SMTP_PORT_RAW.isdigit() else 0
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
EMAIL_TO = os.getenv("EMAIL_TO", "").strip()
EMAIL_MODE = os.getenv("EMAIL_MODE", "always").strip().lower()  # always | action_only


# =========================
# EMAIL HELPERS
# =========================
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


# =========================
# DATA HELPERS
# =========================
def to_float(x) -> Optional[float]:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None

def to_int(x, default=1) -> int:
    try:
        return max(int(float(x)), 0)
    except Exception:
        return default

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

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

def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat(
        [(high - low),
         (high - close.shift(1)).abs(),
         (low - close.shift(1)).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()

def prior_window(df_completed: pd.DataFrame, n: int) -> pd.DataFrame:
    return df_completed.iloc[-(n + 1):-1]

def get_current_price(ticker: str, fallback: float) -> Tuple[float, str]:
    t = yf.Ticker(ticker)
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            lp = fi.get("last_price", None)
            if lp is not None:
                return float(lp), "fast_info.last_price"
    except Exception:
        pass
    try:
        info = getattr(t, "info", None)
        if isinstance(info, dict):
            rmp = info.get("regularMarketPrice", None)
            if rmp is not None:
                return float(rmp), "info.regularMarketPrice"
    except Exception:
        pass
    try:
        intraday = t.history(period="1d", interval="1m")
        if intraday is not None and not intraday.empty and "Close" in intraday.columns:
            return float(intraday["Close"].iloc[-1]), "history(1d,1m).last_close"
    except Exception:
        pass
    return float(fallback), "fallback_daily_close"

def atr_advice(is_call: bool, price: float, atr_stop: float, atr: float) -> Tuple[str, float, float, bool]:
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

def action_and_trend(structure_broken: bool) -> Tuple[str, str]:
    trend = "BROKEN" if structure_broken else "INTACT"
    action = "SELL" if structure_broken else "HOLD"
    return action, trend


# =========================
# OPTION PRICING FROM option_name
# =========================
_OPT_RE = re.compile(r"^\s*([A-Z]{1,6})\s+(\d{4}-\d{2}-\d{2})\s+([CP])\s+(\d+(\.\d+)?)\s*$")

def parse_option_name(option_name: str) -> Optional[Tuple[str, str, str, float]]:
    """
    Returns (ticker, expiry, type, strike)
    option_name example: "NFLX 2026-02-20 P 88"
    """
    m = _OPT_RE.match((option_name or "").strip().upper())
    if not m:
        return None
    ticker = m.group(1)
    expiry = m.group(2)
    opt_type = "CALL" if m.group(3) == "C" else "PUT"
    strike = float(m.group(4))
    return ticker, expiry, opt_type, strike

def fetch_option_quote_from_name(option_name: str) -> Dict[str, Any]:
    """
    Best effort:
      - match strike row in option_chain
      - use mid = (bid+ask)/2 if available and >0
      - else use lastPrice if >0
    """
    parsed = parse_option_name(option_name)
    if not parsed:
        return {"ok": False, "reason": "bad_option_name_format"}

    ticker, expiry, opt_type, strike = parsed
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        tab = chain.calls if opt_type == "CALL" else chain.puts
        if tab is None or tab.empty:
            return {"ok": False, "reason": "empty_chain"}

        # find closest strike (in case of float formatting)
        tab2 = tab.copy()
        tab2["strike_diff"] = (tab2["strike"] - strike).abs()
        row = tab2.nsmallest(1, "strike_diff").iloc[0].to_dict()
        if float(row.get("strike", strike)) != float(row.get("strike", strike)):
            pass

        bid = row.get("bid", None)
        ask = row.get("ask", None)
        last = row.get("lastPrice", None)

        mid = None
        try:
            b = float(bid) if bid is not None else np.nan
            a = float(ask) if ask is not None else np.nan
            if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0:
                mid = (b + a) / 2.0
        except Exception:
            mid = None

        price = None
        src = None
        if mid is not None and mid > 0:
            price = float(mid)
            src = "mid(bid,ask)"
        else:
            try:
                l = float(last) if last is not None else np.nan
                if np.isfinite(l) and l > 0:
                    price = float(l)
                    src = "lastPrice"
            except Exception:
                price = None

        if price is None:
            return {"ok": False, "reason": "no_price"}

        return {
            "ok": True,
            "ticker": ticker,
            "expiry": expiry,
            "opt_type": opt_type,
            "strike": strike,
            "price": price,
            "source": src,
            "contractSymbol": row.get("contractSymbol", ""),
        }
    except Exception:
        return {"ok": False, "reason": "exception"}


# =========================
# ENTRY SCAN (IMPORT YOUR EXISTING SCRIPT)
# =========================
def run_entry_scan() -> pd.DataFrame:
    """
    Uses your existing scanner file without modifying it.
    It must be in the repo root and named unified_turtle_entries_only.py
    and must expose generate_new_entries(...) or main pipeline is similar.
    """
    try:
        import TotalNarrow as scan
    except Exception:
        return pd.DataFrame()

    # Use its generate_new_entries directly if present
    if hasattr(scan, "generate_new_entries"):
        return scan.generate_new_entries(
            top=int(os.getenv("TOP_N_BY_DEFAULT", "300")),
            system=int(os.getenv("SYSTEM_DEFAULT", "1")),
            atr_period=int(os.getenv("ATR_PERIOD_DEFAULT", "20")),
            k_stop_atr=float(os.getenv("K_STOP_ATR_DEFAULT", "2.0")),
            k_take_atr=float(os.getenv("K_TAKE_ATR_DEFAULT", "3.0")),
            allow_shorts=bool(int(os.getenv("ALLOW_SHORTS_DEFAULT", "1"))),
            opt_min_dte=int(os.getenv("OPT_MIN_DTE_DEFAULT", "30")),
            opt_max_dte=int(os.getenv("OPT_MAX_DTE_DEFAULT", "60")),
            opt_target_dte=int(os.getenv("OPT_TARGET_DTE_DEFAULT", "45")),
            opt_sl=float(os.getenv("OPT_SL_PCT_DEFAULT", "0.50")),
            opt_tp=float(os.getenv("OPT_TP_PCT_DEFAULT", "1.00")),
        )

    # If not available, return empty (no new entries)
    return pd.DataFrame()


# =========================
# CORE: EXIT + VALUE + OPTIMIZE
# =========================
def decide_sell_count(contracts: int, broken10: bool, broken5: bool) -> int:
    """
    Exit logic:
      - 10d broken => SELL_ALL
      - 5d broken  => trim if possible (>=3 sell 2, ==2 sell 1, ==1 sell 1)
      - intact     => 0
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

def label_sell(sell_count: int, contracts: int) -> str:
    if contracts <= 0:
        return "NO_POSITION"
    if sell_count <= 0:
        return "HOLD"
    if sell_count >= contracts:
        return "SELL_ALL"
    return f"SELL_{sell_count}"

def option_position_value(option_price: Optional[float], contracts: int) -> float:
    if option_price is None or contracts <= 0:
        return 0.0
    return float(option_price) * CONTRACT_MULTIPLIER * contracts

def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} not found")

    positions = pd.read_csv(CSV_FILE)

    # Keep CSV format: contracts optional
    if "contracts" not in positions.columns:
        positions["contracts"] = 1

    required = {"ticker","option_name","option_entry_price","underlying_entry_price"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(f"{CSV_FILE} missing required columns: {sorted(missing)}")

    report_lines = []
    plan_rows = []
    any_action = False

    # 1) Value existing positions and compute signals
    enriched = positions.copy()

    # Fetch underlying history once per ticker
    ticker_cache = {}

    total_value = 0.0
    for i, row in positions.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        option_name = str(row["option_name"]).strip()
        entry_under = to_float(row["underlying_entry_price"])
        contracts = to_int(row.get("contracts", 1), 1)

        # Option mark price from option_name (best effort)
        oq = fetch_option_quote_from_name(option_name)
        option_mark = None
        option_src = ""
        if oq.get("ok"):
            option_mark = float(oq["price"])
            option_src = str(oq.get("source",""))
        else:
            # fallback to entry price if we can't quote (so we still have a value baseline)
            option_mark = to_float(row.get("option_entry_price"))
            option_src = f"fallback_entry_price({oq.get('reason')})"

        pos_value = option_position_value(option_mark, contracts)
        total_value += pos_value

        # Underlying data
        if ticker not in ticker_cache:
            df = yf.download(ticker, period="9mo", interval="1d", progress=False)
            df = flatten_columns(df).dropna() if df is not None else pd.DataFrame()
            ticker_cache[ticker] = df

        df = ticker_cache[ticker]
        if df.empty or entry_under is None:
            enriched.at[i, "recommendation"] = "SKIP"
            enriched.at[i, "pos_value"] = pos_value
            continue

        close_price = float(df["Close"].iloc[-1])
        current_price, current_src = get_current_price(ticker, fallback=close_price)

        df_completed = remove_today_partial_bar(df).dropna()
        if len(df_completed) < max(ATR_PERIOD, STRUCTURE_10) + 10:
            enriched.at[i, "recommendation"] = "NO_HISTORY"
            enriched.at[i, "pos_value"] = pos_value
            continue

        atr_series = calculate_atr(df_completed, ATR_PERIOD)
        atr_last = atr_series.iloc[-1]
        if pd.isna(atr_last):
            enriched.at[i, "recommendation"] = "NO_ATR"
            enriched.at[i, "pos_value"] = pos_value
            continue
        atr = float(atr_last)

        w10 = prior_window(df_completed, STRUCTURE_10)
        w5 = prior_window(df_completed, STRUCTURE_5)
        low10, high10 = float(w10["Low"].min()), float(w10["High"].max())
        low5, high5 = float(w5["Low"].min()), float(w5["High"].max())

        # Determine CALL/PUT from option_name parsing
        parsed = parse_option_name(option_name)
        is_call = None
        direction = "UNKNOWN"
        if parsed:
            _, _, opt_type, _ = parsed
            is_call = (opt_type == "CALL")
            direction = opt_type

        if is_call is None:
            enriched.at[i, "recommendation"] = "SKIP_BAD_OPTION_NAME"
            enriched.at[i, "pos_value"] = pos_value
            continue

        # ATR stop (on underlying entry)
        if is_call:
            atr_stop = float(entry_under - ATR_MULTIPLIER * atr)
            broken10_close = bool(close_price < low10)
            broken5_close = bool(close_price < low5)
            broken10_cur = bool(current_price < low10)
            broken5_cur = bool(current_price < low5)
        else:
            atr_stop = float(entry_under + ATR_MULTIPLIER * atr)
            broken10_close = bool(close_price > high10)
            broken5_close = bool(close_price > high5)
            broken10_cur = bool(current_price > high10)
            broken5_cur = bool(current_price > high5)

        # Action/trend sections (as you wanted)
        action10_close, trend10_close = action_and_trend(broken10_close)
        action5_close, trend5_close = action_and_trend(broken5_close)
        action10_cur, trend10_cur = action_and_trend(broken10_cur)
        action5_cur, trend5_cur = action_and_trend(broken5_cur)

        adv_c, dist_c, dist_c_atr, _ = atr_advice(is_call, close_price, atr_stop, atr)
        adv_u, dist_u, dist_u_atr, _ = atr_advice(is_call, current_price, atr_stop, atr)

        # Exit recommendation (SELL_n) based on RECO_MODE
        if RECO_MODE == "close":
            sell_count = decide_sell_count(contracts, broken10_close, broken5_close)
            reco_basis = "CLOSE"
        else:
            sell_count = decide_sell_count(contracts, broken10_cur, broken5_cur)
            reco_basis = "CURRENT"
        reco_exit = label_sell(sell_count, contracts)

        enriched.at[i, "option_mark"] = option_mark
        enriched.at[i, "option_src"] = option_src
        enriched.at[i, "pos_value"] = pos_value
        enriched.at[i, "close_price"] = close_price
        enriched.at[i, "current_price"] = current_price
        enriched.at[i, "current_src"] = current_src
        enriched.at[i, "atr"] = atr
        enriched.at[i, "atr_stop"] = atr_stop
        enriched.at[i, "trend10_close"] = trend10_close
        enriched.at[i, "trend5_close"] = trend5_close
        enriched.at[i, "trend10_current"] = trend10_cur
        enriched.at[i, "trend5_current"] = trend5_cur
        enriched.at[i, "sell_count_exit"] = sell_count
        enriched.at[i, "recommendation_exit"] = reco_exit

        if reco_exit != "HOLD":
            any_action = True

        report_lines.append(
            f"""Ticker: {ticker} ({direction})
Option: {option_name}
Contracts: {contracts}
Option Mark: {option_mark if option_mark is not None else ''} (src: {option_src})
Position Value: {pos_value:.2f}

Underlying Entry: {entry_under:.2f}
Close: {close_price:.2f}
Current: {current_price:.2f} (src: {current_src})

ATR({ATR_PERIOD}): {atr:.2f}
ATR Stop ({ATR_MULTIPLIER}x): {atr_stop:.2f}

ATR advice:
CLOSE:   {adv_c} dist {dist_c:+.2f} ({dist_c_atr:+.2f} ATR)
CURRENT: {adv_u} dist {dist_u:+.2f} ({dist_u_atr:+.2f} ATR)

=== Based on CLOSE price ===
10 Days action : {action10_close}
10 Days trend  : {trend10_close}
5 Days action  : {action5_close}
5 Days trend   : {trend5_close}

=== Based on CURRENT price ===
10 Days action : {action10_cur}
10 Days trend  : {trend10_cur}
5 Days action  : {action5_cur}
5 Days trend   : {trend5_cur}

EXIT RECOMMENDATION ({reco_basis}): {reco_exit}
"""
        )

    # 2) Compute slot budget from CSV-derived total value
    # We assume portfolio value is sum of option positions; cash is unknown -> we model as 0 cash at start.
    usable_value = total_value * (1.0 - CASH_BUFFER_PCT)
    slot_budget = usable_value / max(MAX_POSITIONS, 1) if total_value > 0 else 0.0

    # 3) Oversize trimming (redistribution) on top of exit-based sells
    #    This is what lets it "optimize" even when trends intact.
    enriched["sell_count_opt"] = 0
    enriched["recommendation_opt"] = "HOLD"

    if TRIM_TO_SLOT and slot_budget > 0:
        for i, row in enriched.iterrows():
            contracts = to_int(row.get("contracts", 1), 1)
            if contracts <= 0:
                continue

            pos_value = float(row.get("pos_value", 0.0) or 0.0)
            option_mark = to_float(row.get("option_mark", None))
            if option_mark is None or option_mark <= 0:
                continue

            # Only trim if NOT already being exited fully by structure
            exit_sell = to_int(row.get("sell_count_exit", 0), 0)
            if exit_sell >= contracts:
                continue

            if pos_value > slot_budget * OVERSIZE_MULT:
                # target contracts to get pos_value close to slot_budget
                est_cost_per_contract = option_mark * CONTRACT_MULTIPLIER
                target_contracts = int(max(1, min(MAX_CONTRACTS_PER_POSITION, slot_budget // est_cost_per_contract)))
                target_contracts = max(0, target_contracts)

                sell_needed = max(0, contracts - target_contracts)
                # If 1 contract only, cannot trim -> either keep or sell all (we keep by default)
                if contracts == 1:
                    sell_needed = 0

                if sell_needed > 0:
                    enriched.at[i, "sell_count_opt"] = sell_needed
                    enriched.at[i, "recommendation_opt"] = f"SELL_{sell_needed}"
                    any_action = True

    # 4) Merge sell decisions: take max per row (exit sells first), cap to contracts
    def merged_sell(row) -> int:
        c = to_int(row.get("contracts", 1), 1)
        s1 = to_int(row.get("sell_count_exit", 0), 0)
        s2 = to_int(row.get("sell_count_opt", 0), 0)
        return min(c, max(s1, s2))

    enriched["sell_count_final"] = enriched.apply(merged_sell, axis=1)

    def label_final(row) -> str:
        c = to_int(row.get("contracts", 1), 1)
        s = to_int(row.get("sell_count_final", 0), 0)
        if c <= 0:
            return "NO_POSITION"
        if s <= 0:
            return "HOLD"
        if s >= c:
            return "SELL_ALL"
        return f"SELL_{s}"

    enriched["recommendation_final"] = enriched.apply(label_final, axis=1)

    # Cash freed by sells (approx at option mark)
    freed_cash = 0.0
    for _, row in enriched.iterrows():
        option_mark = to_float(row.get("option_mark", None))
        s = to_int(row.get("sell_count_final", 0), 0)
        if option_mark is not None and s > 0:
            freed_cash += option_mark * CONTRACT_MULTIPLIER * s

    # 5) Entry scan for tomorrow
    entries = run_entry_scan()
    if entries is None:
        entries = pd.DataFrame()

    # Determine how many positions remain after planned SELL_ALL
    def will_be_open(row) -> bool:
        c = to_int(row.get("contracts", 1), 1)
        s = to_int(row.get("sell_count_final", 0), 0)
        return (c - s) > 0

    open_positions = enriched[enriched.apply(will_be_open, axis=1)]
    open_tickers = set(open_positions["ticker"].astype(str).str.upper().str.strip().tolist())

    remaining_slots = max(MAX_POSITIONS - len(open_positions), 0)

    # 6) Build BUY plan from entries using freed_cash (since we assume no cash otherwise)
    buy_rows = []
    if remaining_slots > 0 and not entries.empty and freed_cash > 0:
        # exclude tickers already held
        cand = entries.copy()
        cand["Ticker"] = cand["Ticker"].astype(str).str.upper().str.strip()
        cand = cand[~cand["Ticker"].isin(open_tickers)]

        # Use OptionLast from scanner as estimate of premium cost
        def est_cost(row):
            try:
                p = float(row.get("OptionLast", ""))
                return p * CONTRACT_MULTIPLIER if p > 0 else np.inf
            except Exception:
                return np.inf

        cand["EstCost1"] = cand.apply(est_cost, axis=1)
        cand = cand.sort_values(["EstCost1", "Ticker"])

        adds = 0
        cash_left = freed_cash

        for _, r in cand.iterrows():
            if adds >= min(remaining_slots, MAX_NEW_PER_RUN):
                break

            ticker = str(r["Ticker"]).upper()
            est1 = float(r["EstCost1"]) if np.isfinite(float(r["EstCost1"])) else np.inf
            if not np.isfinite(est1) or est1 <= 0 or est1 == np.inf:
                continue

            # per-slot budget based on total (optional); but we’ll mainly use cash_left
            # Buy as many as fit in slot_budget and cash_left, capped
            max_by_cash = int(cash_left // est1)
            max_by_slot = int(slot_budget // est1) if slot_budget > 0 else max_by_cash
            buy_n = max(0, min(max_by_cash, max_by_slot, MAX_CONTRACTS_PER_POSITION))
            if buy_n <= 0:
                continue

            cash_left -= buy_n * est1
            adds += 1
            open_tickers.add(ticker)

            buy_rows.append({
                "Type": "BUY",
                "Ticker": ticker,
                "Action": str(r.get("Action","")),
                "Expiry": str(r.get("Expiry","")),
                "OptionSymbol": str(r.get("OptionSymbol","")),
                "OptionLast": str(r.get("OptionLast","")),
                "BuyContracts": buy_n,
                "EstCostTotal": round(buy_n * est1, 2),
                "Reason": "New entry + budget available (from sells)"
            })

    # 7) Build plan rows for existing positions
    for _, row in enriched.iterrows():
        ticker = str(row.get("ticker","")).strip().upper()
        option_name = str(row.get("option_name","")).strip()
        contracts = to_int(row.get("contracts", 1), 1)
        sell_n = to_int(row.get("sell_count_final", 0), 0)
        reco = str(row.get("recommendation_final","HOLD"))
        pos_value = float(row.get("pos_value", 0.0) or 0.0)

        if contracts <= 0:
            continue

        plan_rows.append({
            "Type": "SELL" if sell_n > 0 else "HOLD",
            "Ticker": ticker,
            "Option": option_name,
            "ContractsHeld": contracts,
            "SellContracts": sell_n,
            "Recommendation": reco,
            "PositionValue": round(pos_value, 2),
            "OptionMark": row.get("option_mark",""),
            "Reason": ("Structure/Trim" if sell_n > 0 else "No action")
        })

    # Add buys to plan
    plan_rows.extend(buy_rows)

    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(PLAN_FILE, index=False)

    # 8) Email report
    header = []
    header.append(f"PORTFOLIO MANAGER — {datetime.now().strftime('%Y-%m-%d')}")
    header.append(f"Total portfolio value (from CSV positions): {total_value:.2f}")
    header.append(f"Usable (after cash buffer {CASH_BUFFER_PCT:.0%}): {usable_value:.2f}")
    header.append(f"Slot budget (MAX_POSITIONS={MAX_POSITIONS}): {slot_budget:.2f}")
    header.append(f"Freed cash from sells (estimated): {freed_cash:.2f}")
    header.append(f"Remaining slots after SELL_ALL: {remaining_slots}")
    header.append("")
    body = "\n".join(header)
    body += "\nEXIT REPORT\n==========\n"
    body += "\n-------------------------\n".join(report_lines) if report_lines else "No valid positions.\n"
    body += "\n\nPLAN (also saved to portfolio_plan.csv)\n======================================\n"
    body += plan_df.to_string(index=False) if not plan_df.empty else "No actions.\n"

    subject = "🚨 Portfolio Plan – Action Needed" if any_action or (len(buy_rows) > 0) else "✅ Portfolio Plan – No Action"
    if smtp_ready():
        if EMAIL_MODE == "action_only" and subject.startswith("✅"):
            return
        send_email(subject, body)
    else:
        print("SMTP secrets not set — printing report instead\n")
        print(subject)
        print(body)


if __name__ == "__main__":
    main()