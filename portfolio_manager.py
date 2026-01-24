#!/usr/bin/env python3
"""
portfolio_manager.py  (FULL SCRIPT — HTML email tables + plain-text fallback)

Fixes & features included:
- ✅ HTML email tables (nice table view) + plain text fallback
- ✅ Works on newer pandas where DataFrame.applymap() is removed
- ✅ Keeps your current positions.csv format (no changes required)
- ✅ Scanner import name: TotalNarrow.py
- ✅ Robust env parsing (no int('') / float('') crashes)
- ✅ Avoids pandas strict string dtype issues by using object dtype for mixed cols
- ✅ Saves portfolio_plan.csv in repo workspace

Required positions.csv columns:
  ticker, option_name, option_entry_price, underlying_entry_price
Optional:
  contracts (defaults to 1)

Email env vars (GitHub secrets):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO
Optional:
  EMAIL_MODE = always | action_only
"""

import os
import re
import ssl
import smtplib
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# =========================
# SAFE ENV PARSERS
# =========================
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return default if v == "" else v

def env_float(name: str, default: float) -> float:
    v = env_str(name, "")
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default

def env_int(name: str, default: int) -> int:
    v = env_str(name, "")
    if v == "":
        return default
    try:
        return int(float(v))
    except ValueError:
        return default

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

def money2(x) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return ""
        return f"${v:,.2f}"
    except Exception:
        return ""

def money0(x) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return ""
        return f"${v:,.0f}"
    except Exception:
        return ""

def num(x, n=2) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return ""
        return f"{v:.{n}f}"
    except Exception:
        return ""


# =========================
# CONFIG (safe)
# =========================
CSV_FILE = env_str("CSV_FILE", "positions.csv")
PLAN_FILE = env_str("PLAN_FILE", "portfolio_plan.csv")

MAX_POSITIONS = env_int("MAX_POSITIONS", 4)
CASH_BUFFER_PCT = env_float("CASH_BUFFER_PCT", 0.05)
MAX_NEW_PER_RUN = env_int("MAX_NEW_PER_RUN", 2)
MAX_CONTRACTS_PER_POSITION = env_int("MAX_CONTRACTS_PER_POSITION", 6)
CONTRACT_MULTIPLIER = env_int("CONTRACT_MULTIPLIER", 100)

ATR_PERIOD = env_int("ATR_PERIOD", 14)
ATR_MULTIPLIER = env_float("ATR_MULTIPLIER", 1.5)
STRUCTURE_10 = env_int("STRUCTURE_10", 10)
STRUCTURE_5 = env_int("STRUCTURE_5", 5)
RECO_MODE = env_str("RECO_MODE", "current").lower()  # current | close

ATR_VERY_CLOSE = env_float("ATR_VERY_CLOSE", 0.25)
ATR_CLOSE = env_float("ATR_CLOSE", 0.50)

OVERSIZE_MULT = env_float("OVERSIZE_MULT", 1.40)
TRIM_TO_SLOT = env_str("TRIM_TO_SLOT", "true").lower() == "true"

# Email env
SMTP_HOST = env_str("SMTP_HOST", "")
SMTP_PORT = env_int("SMTP_PORT", 0)
SMTP_USER = env_str("SMTP_USER", "")
SMTP_PASS = env_str("SMTP_PASS", "")
EMAIL_TO = env_str("EMAIL_TO", "")
EMAIL_MODE = env_str("EMAIL_MODE", "always").lower()  # always | action_only


# =========================
# EMAIL (HTML + TEXT)
# =========================
def smtp_ready() -> bool:
    return all([SMTP_HOST, SMTP_PORT > 0, SMTP_USER, SMTP_PASS, EMAIL_TO])

def send_email(subject: str, text_body: str, html_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

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
# HTML TABLE BUILDER (pandas-safe, no applymap)
# =========================
def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def df_to_html_table(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h3 style='margin:16px 0 6px;'>{html_escape(title)}</h3><div>No rows.</div>"

    safe = df.copy().fillna("")
    safe = safe.astype(str)
    # pandas-3-safe: apply + map instead of applymap
    safe = safe.apply(lambda col: col.map(html_escape))

    ths = "".join(
        f"<th style='border:1px solid #ddd; padding:8px; text-align:left; background:#f5f5f5;'>{html_escape(c)}</th>"
        for c in safe.columns
    )

    trs = []
    for i in range(len(safe)):
        tds = "".join(
            f"<td style='border:1px solid #ddd; padding:8px; vertical-align:top;'>{safe.iat[i, j]}</td>"
            for j in range(safe.shape[1])
        )
        bg = "#ffffff" if (i % 2 == 0) else "#fafafa"
        trs.append(f"<tr style='background:{bg};'>{tds}</tr>")

    return f"""
    <h3 style="margin:16px 0 6px;">{html_escape(title)}</h3>
    <table style="border-collapse:collapse; width:100%; font-family:Arial, sans-serif; font-size:14px;">
      <thead><tr>{ths}</tr></thead>
      <tbody>
        {''.join(trs)}
      </tbody>
    </table>
    """


# =========================
# MARKET DATA HELPERS
# =========================
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
    # prior N completed bars, excluding the most recent completed bar
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
    return ("SELL", "BROKEN") if structure_broken else ("HOLD", "INTACT")


# =========================
# OPTION PRICING FROM option_name
# =========================
_OPT_RE = re.compile(r"^\s*([A-Z]{1,6})\s+(\d{4}-\d{2}-\d{2})\s+([CP])\s+(\d+(\.\d+)?)\s*$", re.IGNORECASE)

def parse_option_name(option_name: str) -> Optional[Tuple[str, str, str, float]]:
    m = _OPT_RE.match((option_name or "").strip())
    if not m:
        return None
    ticker = m.group(1).upper()
    expiry = m.group(2)
    cp = m.group(3).upper()
    opt_type = "CALL" if cp == "C" else "PUT"
    strike = float(m.group(4))
    return ticker, expiry, opt_type, strike

def fetch_option_quote_from_name(option_name: str) -> Dict[str, Any]:
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

        tab2 = tab.copy()
        tab2["strike_diff"] = (tab2["strike"] - strike).abs()
        row = tab2.nsmallest(1, "strike_diff").iloc[0].to_dict()

        bid, ask, last = row.get("bid"), row.get("ask"), row.get("lastPrice")

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
# ENTRY SCAN IMPORT (TotalNarrow)
# =========================
def run_entry_scan() -> pd.DataFrame:
    try:
        import TotalNarrow as scan
    except Exception:
        return pd.DataFrame()

    if hasattr(scan, "generate_new_entries"):
        try:
            return scan.generate_new_entries()
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


# =========================
# OPT LOGIC
# =========================
def decide_sell_count(contracts: int, broken10: bool, broken5: bool) -> int:
    if contracts <= 0:
        return 0
    if broken10:
        return contracts
    if broken5:
        return 2 if contracts >= 3 else 1
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

def estimate_cost_from_entry_row(r: pd.Series) -> float:
    try:
        p = float(r.get("OptionLast", ""))
        if not np.isfinite(p) or p <= 0:
            return float("inf")
        return p * CONTRACT_MULTIPLIER
    except Exception:
        return float("inf")


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} not found")

    positions = pd.read_csv(CSV_FILE)
    if "contracts" not in positions.columns:
        positions["contracts"] = 1

    required = {"ticker", "option_name", "option_entry_price", "underlying_entry_price"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(f"{CSV_FILE} missing required columns: {sorted(missing)}")

    ticker_cache: Dict[str, pd.DataFrame] = {}
    report_lines: List[str] = []
    any_action = False

    # Mixed dtypes must be object to avoid pandas strict dtype issues
    enriched = positions.copy()
    enriched["option_mark"] = np.nan
    enriched["pos_value"] = 0.0
    enriched["sell_count_exit"] = 0
    enriched["sell_count_opt"] = 0
    enriched["sell_count_final"] = 0
    enriched["option_src"] = pd.Series([None] * len(enriched), dtype="object")
    enriched["recommendation_final"] = pd.Series(["HOLD"] * len(enriched), dtype="object")

    total_value = 0.0
    min_needed = max(ATR_PERIOD, STRUCTURE_10) + 10

    # ---- value + exit signals
    for i, row in positions.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        option_name = str(row["option_name"]).strip()
        entry_under = to_float(row["underlying_entry_price"])
        contracts = to_int(row.get("contracts", 1), 1)

        oq = fetch_option_quote_from_name(option_name)
        option_mark = None
        option_src = ""
        if oq.get("ok"):
            option_mark = float(oq["price"])
            option_src = str(oq.get("source", ""))
        else:
            option_mark = to_float(row.get("option_entry_price"))
            option_src = f"fallback_entry_price({oq.get('reason')})"

        pos_value = option_position_value(option_mark, contracts)
        total_value += pos_value

        enriched.at[i, "option_mark"] = option_mark if option_mark is not None else np.nan
        enriched.at[i, "option_src"] = option_src
        enriched.at[i, "pos_value"] = pos_value

        if contracts <= 0 or entry_under is None:
            enriched.at[i, "recommendation_final"] = "NO_POSITION"
            continue

        if ticker not in ticker_cache:
            df = yf.download(ticker, period="9mo", interval="1d", progress=False)
            df = flatten_columns(df).dropna() if df is not None else pd.DataFrame()
            ticker_cache[ticker] = df

        df = ticker_cache[ticker]
        if df.empty:
            enriched.at[i, "recommendation_final"] = "NO_DATA"
            continue

        close_price = float(df["Close"].iloc[-1])
        current_price, current_src = get_current_price(ticker, fallback=close_price)

        df_completed = remove_today_partial_bar(df).dropna()
        if len(df_completed) < min_needed:
            enriched.at[i, "recommendation_final"] = "NO_HISTORY"
            continue

        atr_series = calculate_atr(df_completed, ATR_PERIOD)
        atr_last = atr_series.iloc[-1]
        if pd.isna(atr_last):
            enriched.at[i, "recommendation_final"] = "NO_ATR"
            continue
        atr = float(atr_last)

        w10 = prior_window(df_completed, STRUCTURE_10)
        w5 = prior_window(df_completed, STRUCTURE_5)
        low10, high10 = float(w10["Low"].min()), float(w10["High"].max())
        low5, high5 = float(w5["Low"].min()), float(w5["High"].max())

        parsed = parse_option_name(option_name)
        if not parsed:
            enriched.at[i, "recommendation_final"] = "BAD_OPTION_NAME"
            continue
        _, _, opt_type, _ = parsed
        is_call = (opt_type == "CALL")
        direction = opt_type

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

        action10_close, trend10_close = action_and_trend(broken10_close)
        action5_close, trend5_close = action_and_trend(broken5_close)
        action10_cur, trend10_cur = action_and_trend(broken10_cur)
        action5_cur, trend5_cur = action_and_trend(broken5_cur)

        adv_c, dist_c, dist_c_atr, _ = atr_advice(is_call, close_price, atr_stop, atr)
        adv_u, dist_u, dist_u_atr, _ = atr_advice(is_call, current_price, atr_stop, atr)

        if RECO_MODE == "close":
            sell_exit = decide_sell_count(contracts, broken10_close, broken5_close)
            reco_basis = "CLOSE"
        else:
            sell_exit = decide_sell_count(contracts, broken10_cur, broken5_cur)
            reco_basis = "CURRENT"

        enriched.at[i, "sell_count_exit"] = sell_exit
        if sell_exit > 0:
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

EXIT RECOMMENDATION ({reco_basis}): {label_sell(sell_exit, contracts)}
"""
        )

    usable_value = total_value * (1.0 - CASH_BUFFER_PCT)
    slot_budget = (usable_value / max(MAX_POSITIONS, 1)) if total_value > 0 else 0.0

    # Oversize trim
    if TRIM_TO_SLOT and slot_budget > 0:
        for i, row in enriched.iterrows():
            contracts = to_int(row.get("contracts", 1), 1)
            if contracts <= 1:
                continue
            pos_value = float(row.get("pos_value", 0.0) or 0.0)
            option_mark = to_float(row.get("option_mark", np.nan))
            if option_mark is None or not np.isfinite(option_mark) or option_mark <= 0:
                continue
            s_exit = to_int(row.get("sell_count_exit", 0), 0)
            if s_exit >= contracts:
                continue
            if pos_value > slot_budget * OVERSIZE_MULT:
                cost1 = option_mark * CONTRACT_MULTIPLIER
                target_contracts = int(max(1, min(MAX_CONTRACTS_PER_POSITION, slot_budget // cost1)))
                sell_needed = max(0, contracts - target_contracts)
                if sell_needed > 0:
                    enriched.at[i, "sell_count_opt"] = sell_needed
                    any_action = True

    # Merge sells + freed cash
    freed_cash = 0.0
    for i, row in enriched.iterrows():
        contracts = to_int(row.get("contracts", 1), 1)
        s_exit = to_int(row.get("sell_count_exit", 0), 0)
        s_opt = to_int(row.get("sell_count_opt", 0), 0)
        s_final = min(contracts, max(s_exit, s_opt))
        enriched.at[i, "sell_count_final"] = s_final
        enriched.at[i, "recommendation_final"] = label_sell(s_final, contracts)

        option_mark = to_float(row.get("option_mark", np.nan))
        if option_mark is not None and np.isfinite(option_mark) and s_final > 0:
            freed_cash += option_mark * CONTRACT_MULTIPLIER * s_final

    def will_be_open(r: pd.Series) -> bool:
        c = to_int(r.get("contracts", 1), 1)
        s = to_int(r.get("sell_count_final", 0), 0)
        return (c - s) > 0

    open_positions = enriched[enriched.apply(will_be_open, axis=1)]
    open_tickers = set(open_positions["ticker"].astype(str).str.upper().str.strip().tolist())
    remaining_slots = max(MAX_POSITIONS - len(open_positions), 0)

    entries = run_entry_scan()
    if entries is None:
        entries = pd.DataFrame()

    buy_rows: List[Dict[str, Any]] = []
    if remaining_slots > 0 and freed_cash > 0 and not entries.empty:
        cand = entries.copy()
        cand["Ticker"] = cand["Ticker"].astype(str).str.upper().str.strip()
        cand = cand[~cand["Ticker"].isin(open_tickers)].copy()

        cand["EstCost1"] = cand.apply(estimate_cost_from_entry_row, axis=1)
        cand = cand.replace([np.inf, -np.inf], np.nan).dropna(subset=["EstCost1"])
        cand = cand.sort_values(["EstCost1", "Ticker"])

        cash_left = freed_cash
        adds = 0

        for _, r in cand.iterrows():
            if adds >= min(remaining_slots, MAX_NEW_PER_RUN):
                break

            est1 = float(r["EstCost1"])
            if not np.isfinite(est1) or est1 <= 0:
                continue

            max_by_cash = int(cash_left // est1)
            max_by_slot = int(slot_budget // est1) if slot_budget > 0 else max_by_cash
            buy_n = max(0, min(max_by_cash, max_by_slot, MAX_CONTRACTS_PER_POSITION))
            if buy_n <= 0:
                continue

            ticker = str(r["Ticker"]).upper()
            cash_left -= buy_n * est1
            adds += 1
            open_tickers.add(ticker)

            buy_rows.append({
                "Type": "BUY",
                "Ticker": ticker,
                "Strategy": str(r.get("Action", "")),
                "Expiry": str(r.get("Expiry", "")),
                "OptionSymbol": str(r.get("OptionSymbol", "")),
                "OptionLast": num(r.get("OptionLast", ""), 2),
                "BuyContracts": buy_n,
                "EstCostTotal": round(buy_n * est1, 2),
                "Reason": "New entry + budget available (from sells)",
            })

    # ---- Plan DF (saved)
    plan_rows: List[Dict[str, Any]] = []
    for _, r in enriched.iterrows():
        ticker = str(r.get("ticker", "")).strip().upper()
        option_name = str(r.get("option_name", "")).strip()
        held = to_int(r.get("contracts", 1), 1)
        sell_n = to_int(r.get("sell_count_final", 0), 0)
        if held <= 0:
            continue
        plan_rows.append({
            "Type": "SELL" if sell_n > 0 else "HOLD",
            "Ticker": ticker,
            "Option": option_name,
            "ContractsHeld": held,
            "SellContracts": sell_n,
            "Recommendation": str(r.get("recommendation_final", "HOLD")),
            "PositionValue": round(float(r.get("pos_value", 0.0) or 0.0), 2),
            "OptionMark": "" if pd.isna(r.get("option_mark")) else float(r.get("option_mark")),
            "Reason": ("Structure/Trim" if sell_n > 0 else "No action"),
        })
    plan_rows.extend(buy_rows)
    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(PLAN_FILE, index=False)

    # ---- Plain text body (fallback)
    header_txt = []
    header_txt.append(f"PORTFOLIO MANAGER — {datetime.now().strftime('%Y-%m-%d')}")
    header_txt.append(f"Total: {money2(total_value)} | Usable: {money2(usable_value)} | Slot: {money2(slot_budget)} | Freed: {money2(freed_cash)} | Slots: {remaining_slots}")
    header_txt.append(f"Scanner: TotalNarrow.py")
    header_txt.append(f"Plan saved: {PLAN_FILE}")
    header_txt.append("")
    body_details = "\n-------------------------\n".join(report_lines) if report_lines else "No details."
    text_body = "\n".join(header_txt) + "\nDETAILS\n=======\n" + body_details

    # ---- HTML body tables
    existing_df = plan_df[plan_df["Type"].isin(["SELL", "HOLD"])][
        ["Type", "Ticker", "Option", "ContractsHeld", "SellContracts", "Recommendation", "PositionValue", "OptionMark", "Reason"]
    ].copy()

    if not existing_df.empty:
        existing_df["PositionValue"] = existing_df["PositionValue"].apply(money0)
        existing_df["OptionMark"] = existing_df["OptionMark"].apply(lambda x: "" if x == "" else num(x, 2))

    buy_df = plan_df[plan_df["Type"].isin(["BUY"])].copy()
    if not buy_df.empty:
        cols = ["Type", "Ticker", "Strategy", "Expiry", "OptionSymbol", "OptionLast", "BuyContracts", "EstCostTotal", "Reason"]
        buy_df = buy_df[cols]
        buy_df["EstCostTotal"] = buy_df["EstCostTotal"].apply(money2)

    html = f"""
    <div style="font-family:Arial, sans-serif; color:#111;">
      <h2 style="margin:0 0 8px;">Portfolio Manager — {datetime.now().strftime('%Y-%m-%d')}</h2>
      <div style="margin:0 0 12px; font-size:14px;">
        <b>Total:</b> {money2(total_value)} &nbsp;·&nbsp;
        <b>Usable:</b> {money2(usable_value)} &nbsp;·&nbsp;
        <b>Slot:</b> {money2(slot_budget)} &nbsp;·&nbsp;
        <b>Freed:</b> {money2(freed_cash)} &nbsp;·&nbsp;
        <b>Slots:</b> {remaining_slots}
      </div>

      {df_to_html_table(existing_df, "Existing Positions — Action Plan")}
      {df_to_html_table(buy_df, "New Entries — Action Plan")}

      <details style="margin-top:14px;">
        <summary style="cursor:pointer; font-weight:bold;">Diagnostics (details)</summary>
        <pre style="white-space:pre-wrap; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; font-size:12px; background:#f7f7f7; padding:10px; border:1px solid #eee;">
{html_escape(body_details)}
        </pre>
      </details>

      <div style="margin-top:10px; font-size:12px; color:#666;">
        Plan file saved in runner as <b>{html_escape(PLAN_FILE)}</b> (upload as artifact to download).
      </div>
    </div>
    """

    subject = "🚨 Portfolio Plan – Action Needed" if (any_action or len(buy_rows) > 0) else "✅ Portfolio Plan – No Action"

    if smtp_ready():
        if EMAIL_MODE == "action_only" and subject.startswith("✅"):
            return
        send_email(subject, text_body=text_body, html_body=html)
        print(f"Email sent to {EMAIL_TO}. Plan saved: {PLAN_FILE}")
    else:
        print("SMTP secrets not set — printing report instead\n")
        print(subject)
        print(text_body)


if __name__ == "__main__":
    main()