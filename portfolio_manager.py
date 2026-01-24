#!/usr/bin/env python3
"""
portfolio_manager.py  (UPDATED WITH PYRAMIDING + HTML EMAIL TABLES)

What’s new:
- ✅ Pyramiding (ADD_1) on winners, gated by:
  - 10D + 5D structure intact (current price)
  - ATR advice == OK (not CLOSE / VERY CLOSE / STOP HIT)
  - Option return >= PYR_L1 / PYR_L2
  - Budget available (PYR_ADD_BUDGET_PCT of account value per run)
  - Max contracts cap (MAX_CONTRACTS_PER_POSITION)
- ✅ Adds "AddContracts" + "PyramidReason" columns to the plan + email tables
- ✅ Keeps your positions.csv format (no change required)
- ✅ Still uses TotalNarrow scanner (DataFrame or print-only parsing)
- ✅ pandas-safe HTML builder (no DataFrame.applymap)

Important notes:
- "Account value" here is estimated as sum(option_mark * 100 * contracts) from positions.csv.
- Pyramiding uses a per-run budget slice to avoid overleveraging.

Env vars (optional):
  PYRAMID_ON=true|false (default true)
  PYR_L1=0.60  (add eligible at +60%)
  PYR_L2=1.20  (add eligible at +120%)
  PYR_ADD_BUDGET_PCT=0.05 (5% of account value per run max spend on adds)
  PYR_REQUIRE_BOTH=true|false (default true)  # require 5D AND 10D intact
  PYR_MAX_ADDS_PER_RUN=99 (default 99)        # safety

Existing env vars still supported:
  MAX_NEW_PER_RUN, MAX_CONTRACTS_PER_POSITION, CONTRACT_MULTIPLIER, CASH_BUFFER_PCT, ATR_PERIOD, ATR_MULTIPLIER, STRUCTURE_10, STRUCTURE_5, RECO_MODE
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO, EMAIL_MODE
"""

import os
import re
import ssl
import smtplib
import io
import contextlib
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
# CONFIG
# =========================
CSV_FILE = env_str("CSV_FILE", "positions.csv")
PLAN_FILE = env_str("PLAN_FILE", "portfolio_plan.csv")

MAX_NEW_PER_RUN = env_int("MAX_NEW_PER_RUN", 2)
MAX_CONTRACTS_PER_POSITION = env_int("MAX_CONTRACTS_PER_POSITION", 6)
CONTRACT_MULTIPLIER = env_int("CONTRACT_MULTIPLIER", 100)

CASH_BUFFER_PCT = env_float("CASH_BUFFER_PCT", 0.05)

ATR_PERIOD = env_int("ATR_PERIOD", 14)
ATR_MULTIPLIER = env_float("ATR_MULTIPLIER", 1.5)
STRUCTURE_10 = env_int("STRUCTURE_10", 10)
STRUCTURE_5 = env_int("STRUCTURE_5", 5)
RECO_MODE = env_str("RECO_MODE", "current").lower()  # current | close

ATR_VERY_CLOSE = env_float("ATR_VERY_CLOSE", 0.25)
ATR_CLOSE = env_float("ATR_CLOSE", 0.50)

OVERSIZE_MULT = env_float("OVERSIZE_MULT", 1.40)
TRIM_TO_SLOT = env_str("TRIM_TO_SLOT", "true").lower() == "true"

# Pyramiding config
PYRAMID_ON = env_str("PYRAMID_ON", "true").lower() == "true"
PYR_L1 = env_float("PYR_L1", 0.60)   # +60% option return
PYR_L2 = env_float("PYR_L2", 1.20)   # +120% option return
PYR_ADD_BUDGET_PCT = env_float("PYR_ADD_BUDGET_PCT", 0.05)  # 5% of total value/run
PYR_REQUIRE_BOTH = env_str("PYR_REQUIRE_BOTH", "true").lower() == "true"
PYR_MAX_ADDS_PER_RUN = env_int("PYR_MAX_ADDS_PER_RUN", 99)

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
# HTML TABLE BUILDER (pandas-safe)
# =========================
def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def df_to_html_table(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h3 style='margin:16px 0 6px;'>{html_escape(title)}</h3><div>No rows.</div>"

    safe = df.copy().fillna("")
    safe = safe.astype(str)
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

        return {"ok": True, "price": price, "source": src}
    except Exception:
        return {"ok": False, "reason": "exception"}


# =========================
# ENTRY SCAN IMPORT (TotalNarrow): DF return OR print-only
# =========================
_BULLET_HEAD_RE = re.compile(r"^[•\-\*]\s*([A-Z]{1,6})\s+—\s+([A-Z_]+)", re.UNICODE)
_OPTION_LINE_RE = re.compile(
    r"Option:\s*(CALL|PUT)\s+([\d\.]+)\s+exp\s+(\d{4}-\d{2}-\d{2})\s+\[([A-Z0-9]+)\]\s+@\s+last\s+([\d\.]+)",
    re.IGNORECASE
)

def parse_scanner_stdout_to_df(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m1 = _BULLET_HEAD_RE.match(line)
        if m1:
            if cur.get("Ticker"):
                rows.append(cur)
            cur = {"Ticker": m1.group(1).upper(), "Action": m1.group(2).upper()}
            continue

        m2 = _OPTION_LINE_RE.search(line)
        if m2 and cur.get("Ticker"):
            cur["Expiry"] = m2.group(3)
            cur["OptionSymbol"] = m2.group(4)
            try:
                cur["OptionLast"] = float(m2.group(5))
            except Exception:
                cur["OptionLast"] = np.nan
            continue

    if cur.get("Ticker"):
        rows.append(cur)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ["Ticker", "Action", "Expiry", "OptionSymbol", "OptionLast"]:
        if c not in df.columns:
            df[c] = ""

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df

def run_entry_scan(report_lines: List[str]) -> pd.DataFrame:
    try:
        import TotalNarrow as scan
    except Exception as e:
        report_lines.append(f"DIAG: Failed to import TotalNarrow.py: {e}")
        return pd.DataFrame()

    if hasattr(scan, "generate_new_entries"):
        try:
            df = scan.generate_new_entries()
            if isinstance(df, pd.DataFrame) and not df.empty:
                report_lines.append(f"DIAG: Scanner returned DF rows={len(df)} cols={list(df.columns)}")
                return df
        except Exception as e:
            report_lines.append(f"DIAG: Scanner generate_new_entries() exception: {e}")

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            if hasattr(scan, "main") and callable(scan.main):
                scan.main()
            elif hasattr(scan, "run") and callable(scan.run):
                scan.run()
            elif hasattr(scan, "generate_new_entries") and callable(scan.generate_new_entries):
                scan.generate_new_entries()
    except Exception as e:
        report_lines.append(f"DIAG: Scanner stdout-capture run exception: {e}")

    out = buf.getvalue()
    if out.strip():
        parsed = parse_scanner_stdout_to_df(out)
        report_lines.append(f"DIAG: Parsed scanner stdout rows={len(parsed)}")
        return parsed

    report_lines.append("DIAG: Scanner produced no DF and no stdout.")
    return pd.DataFrame()


# =========================
# OPT / PYRAMID LOGIC
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
    for col in ("OptionLast", "OptionMid", "OptionPrice", "Mid", "Last"):
        try:
            p = float(r.get(col, ""))
            if np.isfinite(p) and p > 0:
                return p * CONTRACT_MULTIPLIER
        except Exception:
            continue
    return float("inf")

def compute_option_return(entry_price: Optional[float], mark: Optional[float]) -> Optional[float]:
    if entry_price is None or mark is None:
        return None
    if not np.isfinite(entry_price) or not np.isfinite(mark) or entry_price <= 0:
        return None
    return (mark / entry_price) - 1.0

def choose_add_contracts(
    contracts: int,
    opt_ret: Optional[float],
    trend10_intact: bool,
    trend5_intact: bool,
    atr_ok: bool,
    will_sell_any: bool,
) -> Tuple[int, str]:
    if not PYRAMID_ON:
        return 0, "Pyramiding off"
    if will_sell_any:
        return 0, "Selling/Trimming takes priority"
    if contracts <= 0:
        return 0, "No position"
    if contracts >= MAX_CONTRACTS_PER_POSITION:
        return 0, "At max contracts cap"
    if opt_ret is None:
        return 0, "No option return calc"
    if atr_ok is False:
        return 0, "ATR not OK"
    if PYR_REQUIRE_BOTH:
        if not (trend10_intact and trend5_intact):
            return 0, "Trend not intact (need 10D+5D)"
    else:
        if not trend10_intact:
            return 0, "Trend not intact (need 10D)"

    # Small-account friendly add sizing
    add_n = 1  # default add 1 contract

    # Trigger levels
    if opt_ret >= PYR_L2:
        return add_n, f"Winner {opt_ret*100:.0f}% ≥ {PYR_L2*100:.0f}% + trend intact + ATR OK"
    if opt_ret >= PYR_L1:
        return add_n, f"Winner {opt_ret*100:.0f}% ≥ {PYR_L1*100:.0f}% + trend intact + ATR OK"

    return 0, "Below pyramid thresholds"


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

    enriched = positions.copy()
    enriched["option_mark"] = np.nan
    enriched["pos_value"] = 0.0
    enriched["sell_count_exit"] = 0
    enriched["sell_count_opt"] = 0
    enriched["sell_count_final"] = 0
    enriched["add_contracts"] = 0
    enriched["pyramid_reason"] = pd.Series([""] * len(enriched), dtype="object")
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
        entry_opt = to_float(row.get("option_entry_price"))

        oq = fetch_option_quote_from_name(option_name)
        option_mark = None
        option_src = ""
        if oq.get("ok"):
            option_mark = float(oq["price"])
            option_src = str(oq.get("source", ""))
        else:
            option_mark = entry_opt
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
            broken10_cur = bool(current_price < low10)
            broken5_cur = bool(current_price < low5)
        else:
            atr_stop = float(entry_under + ATR_MULTIPLIER * atr)
            broken10_cur = bool(current_price > high10)
            broken5_cur = bool(current_price > high5)

        trend10_intact = not broken10_cur
        trend5_intact = not broken5_cur

        adv_u, dist_u, dist_u_atr, stop_hit = atr_advice(is_call, current_price, atr_stop, atr)
        atr_ok = (adv_u == "OK") and (stop_hit is False)

        # exits first
        sell_exit = decide_sell_count(contracts, broken10_cur, broken5_cur)
        enriched.at[i, "sell_count_exit"] = sell_exit
        if sell_exit > 0:
            any_action = True

        opt_ret = compute_option_return(entry_opt, option_mark)

        report_lines.append(
            f"""Ticker: {ticker} ({direction})
Option: {option_name}
Contracts: {contracts}
Option Entry: {entry_opt if entry_opt is not None else ''} | Mark: {option_mark if option_mark is not None else ''} (src: {option_src})
Option Return: {opt_ret*100:.1f}%""" + ("" if opt_ret is not None else "") + f"""

Underlying Entry: {entry_under:.2f}
Current: {current_price:.2f} (src: {current_src})

ATR({ATR_PERIOD}): {atr:.2f}
ATR Stop ({ATR_MULTIPLIER}x): {atr_stop:.2f}
ATR advice (CURRENT): {adv_u} dist {dist_u:+.2f} ({dist_u_atr:+.2f} ATR)

Trend (CURRENT): 10D={'INTACT' if trend10_intact else 'BROKEN'} | 5D={'INTACT' if trend5_intact else 'BROKEN'}
Exit recommendation: {label_sell(sell_exit, contracts)}
"""
        )

        # Store intermediate for pyramiding decision (we decide adds after we know total_value & budget)
        enriched.at[i, "_trend10_intact"] = trend10_intact
        enriched.at[i, "_trend5_intact"] = trend5_intact
        enriched.at[i, "_atr_ok"] = atr_ok
        enriched.at[i, "_opt_ret"] = opt_ret if opt_ret is not None else np.nan

    # Oversize trim (kept)
    usable_value = total_value * (1.0 - CASH_BUFFER_PCT)

    avg_pos_budget = usable_value / max(len(enriched), 1) if usable_value > 0 else 0.0
    if TRIM_TO_SLOT and avg_pos_budget > 0:
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
            if pos_value > avg_pos_budget * OVERSIZE_MULT:
                cost1 = option_mark * CONTRACT_MULTIPLIER
                target_contracts = int(max(1, min(MAX_CONTRACTS_PER_POSITION, avg_pos_budget // cost1)))
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

    # =========================
    # PYRAMIDING (adds) — after sells decided
    # =========================
    pyramid_budget = usable_value * PYR_ADD_BUDGET_PCT
    pyramid_spend = 0.0
    adds_used = 0

    for i, row in enriched.iterrows():
        if adds_used >= PYR_MAX_ADDS_PER_RUN:
            break

        contracts = to_int(row.get("contracts", 1), 1)
        sell_final = to_int(row.get("sell_count_final", 0), 0)
        will_sell_any = sell_final > 0

        option_mark = to_float(row.get("option_mark", np.nan))
        if option_mark is None or not np.isfinite(option_mark) or option_mark <= 0:
            enriched.at[i, "add_contracts"] = 0
            enriched.at[i, "pyramid_reason"] = "No mark price"
            continue

        trend10_intact = bool(row.get("_trend10_intact", True))
        trend5_intact = bool(row.get("_trend5_intact", True))
        atr_ok = bool(row.get("_atr_ok", False))

        opt_ret_val = row.get("_opt_ret", np.nan)
        opt_ret = None if (opt_ret_val is None or pd.isna(opt_ret_val)) else float(opt_ret_val)

        add_n, reason = choose_add_contracts(
            contracts=contracts,
            opt_ret=opt_ret,
            trend10_intact=trend10_intact,
            trend5_intact=trend5_intact,
            atr_ok=atr_ok,
            will_sell_any=will_sell_any,
        )

        if add_n <= 0:
            enriched.at[i, "add_contracts"] = 0
            enriched.at[i, "pyramid_reason"] = reason
            continue

        # Funding check (pyramid budget only)
        cost_add = option_mark * CONTRACT_MULTIPLIER * add_n
        if (pyramid_spend + cost_add) > pyramid_budget:
            enriched.at[i, "add_contracts"] = 0
            enriched.at[i, "pyramid_reason"] = f"No pyramid budget (need {money2(cost_add)}, left {money2(pyramid_budget - pyramid_spend)})"
            continue

        # Respect max contracts cap
        if contracts + add_n > MAX_CONTRACTS_PER_POSITION:
            add_n = max(0, MAX_CONTRACTS_PER_POSITION - contracts)

        if add_n <= 0:
            enriched.at[i, "add_contracts"] = 0
            enriched.at[i, "pyramid_reason"] = "At max contracts cap"
            continue

        pyramid_spend += option_mark * CONTRACT_MULTIPLIER * add_n
        adds_used += 1
        any_action = True

        enriched.at[i, "add_contracts"] = add_n
        enriched.at[i, "pyramid_reason"] = reason

    report_lines.append(f"DIAG: usable_value={usable_value:.2f} pyramid_budget={pyramid_budget:.2f} pyramid_spend={pyramid_spend:.2f} adds_used={adds_used}")

    # ---- Scanner + buys (still funded by freed cash only)
    report_lines.append(f"DIAG: freed_cash={freed_cash:.2f}")
    entries = run_entry_scan(report_lines)

    report_lines.append(f"DIAG: scanner_rows={len(entries)}")
    report_lines.append(f"DIAG: scanner_cols={list(entries.columns) if entries is not None else []}")

    open_positions = enriched[enriched.apply(lambda r: (to_int(r.get("contracts", 1), 1) - to_int(r.get("sell_count_final", 0), 0)) > 0, axis=1)]
    open_tickers = set(open_positions["ticker"].astype(str).str.upper().str.strip().tolist())

    buy_rows: List[Dict[str, Any]] = []
    if freed_cash > 0 and entries is not None and not entries.empty and "Ticker" in entries.columns:
        cand = entries.copy()
        cand["Ticker"] = cand["Ticker"].astype(str).str.upper().str.strip()
        cand = cand[~cand["Ticker"].isin(open_tickers)].copy()

        cand["EstCost1"] = cand.apply(estimate_cost_from_entry_row, axis=1)
        cand = cand.replace([np.inf, -np.inf], np.nan).dropna(subset=["EstCost1"])
        cand = cand.sort_values(["EstCost1", "Ticker"])

        cash_left = freed_cash
        adds = 0
        for _, r in cand.iterrows():
            if adds >= MAX_NEW_PER_RUN:
                break
            est1 = float(r["EstCost1"])
            if not np.isfinite(est1) or est1 <= 0:
                continue
            max_by_cash = int(cash_left // est1)
            buy_n = max(0, min(max_by_cash, MAX_CONTRACTS_PER_POSITION))
            if buy_n <= 0:
                continue
            ticker = str(r["Ticker"]).upper()
            cash_left -= buy_n * est1
            adds += 1
            buy_rows.append({
                "Type": "BUY",
                "Ticker": ticker,
                "Strategy": str(r.get("Action", "")),
                "Expiry": str(r.get("Expiry", "")),
                "OptionSymbol": str(r.get("OptionSymbol", "")),
                "OptionLast": num(r.get("OptionLast", ""), 2),
                "BuyContracts": buy_n,
                "EstCostTotal": round(buy_n * est1, 2),
                "Reason": "New entry funded by freed cash (from sells)",
            })

        report_lines.append(f"DIAG: buys_added={len(buy_rows)} cash_left={cash_left:.2f}")

    # ---- Plan DF (saved)
    plan_rows: List[Dict[str, Any]] = []

    for _, r in enriched.iterrows():
        ticker = str(r.get("ticker", "")).strip().upper()
        option_name = str(r.get("option_name", "")).strip()
        held = to_int(r.get("contracts", 1), 1)
        sell_n = to_int(r.get("sell_count_final", 0), 0)
        add_n = to_int(r.get("add_contracts", 0), 0)
        if held <= 0:
            continue

        action_type = "SELL" if sell_n > 0 else ("ADD" if add_n > 0 else "HOLD")
        reco = str(r.get("recommendation_final", "HOLD"))
        if add_n > 0 and sell_n == 0:
            reco = f"ADD_{add_n}"

        plan_rows.append({
            "Type": action_type,
            "Ticker": ticker,
            "Option": option_name,
            "ContractsHeld": held,
            "SellContracts": sell_n,
            "AddContracts": add_n,
            "Recommendation": reco,
            "PositionValue": round(float(r.get("pos_value", 0.0) or 0.0), 2),
            "OptionMark": "" if pd.isna(r.get("option_mark")) else float(r.get("option_mark")),
            "Reason": ("Structure/Trim" if sell_n > 0 else ("Pyramiding" if add_n > 0 else "No action")),
            "PyramidReason": str(r.get("pyramid_reason", "")),
        })

    plan_rows.extend(buy_rows)
    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(PLAN_FILE, index=False)

    # ---- Plain text body (fallback)
    header_txt = []
    header_txt.append(f"PORTFOLIO MANAGER — {datetime.now().strftime('%Y-%m-%d')}")
    header_txt.append(
        f"Total: {money2(total_value)} | Usable: {money2(usable_value)} | "
        f"Freed: {money2(freed_cash)} | PyramidBudget: {money2(pyramid_budget)} | PyramidSpend: {money2(pyramid_spend)} | "
        f"Adds: {adds_used} | NewBuys: {len(buy_rows)}"
    )
    header_txt.append("Scanner: TotalNarrow.py")
    header_txt.append(f"Plan saved: {PLAN_FILE}")
    header_txt.append("")
    body_details = "\n-------------------------\n".join(report_lines) if report_lines else "No details."
    text_body = "\n".join(header_txt) + "\nDETAILS\n=======\n" + body_details

    # ---- HTML body tables
    existing_df = plan_df[plan_df["Type"].isin(["SELL", "HOLD", "ADD"])][
        ["Type", "Ticker", "Option", "ContractsHeld", "SellContracts", "AddContracts",
         "Recommendation", "PositionValue", "OptionMark", "Reason", "PyramidReason"]
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
        <b>Freed:</b> {money2(freed_cash)} &nbsp;·&nbsp;
        <b>Pyramid budget:</b> {money2(pyramid_budget)} &nbsp;·&nbsp;
        <b>Pyramid spend:</b> {money2(pyramid_spend)} &nbsp;·&nbsp;
        <b>Adds:</b> {adds_used} &nbsp;·&nbsp;
        <b>New buys:</b> {len(buy_rows)}
      </div>

      {df_to_html_table(existing_df, "Existing Positions — Action Plan (includes pyramiding)")}
      {df_to_html_table(buy_df, "New Entries — Action Plan")}

      <details style="margin-top:14px;">
        <summary style="cursor:pointer; font-weight:bold;">Diagnostics (details)</summary>
        <pre style="white-space:pre-wrap; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace; font-size:12px; background:#f7f7f7; padding:10px; border:1px solid #eee;">
{html_escape(body_details)}
        </pre>
      </details>

      <div style="margin-top:10px; font-size:12px; color:#666;">
        Plan file saved in runner as <b>{html_escape(PLAN_FILE)}</b>.
      </div>
    </div>
    """

    subject = "🚨 Portfolio Plan – Action Needed" if (any_action or len(buy_rows) > 0 or adds_used > 0) else "✅ Portfolio Plan – No Action"

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