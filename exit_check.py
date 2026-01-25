#!/usr/bin/env python3
"""
exit_check.py

Now emails a clean HTML table (with a plain-text fallback).
"""

import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Tuple, List, Dict

import pandas as pd
import yfinance as yf

# =========================
# CONFIG
# =========================
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5

STRUCTURE_10 = 10
STRUCTURE_5 = 5

ATR_VERY_CLOSE = float(os.getenv("ATR_VERY_CLOSE", "0.25"))
ATR_CLOSE = float(os.getenv("ATR_CLOSE", "0.50"))

CSV_FILE = "positions.csv"

APPLY_RECOMMENDATIONS = os.getenv("APPLY_RECOMMENDATIONS", "false").strip().lower() == "true"
RECO_MODE = os.getenv("RECO_MODE", "current").strip().lower()

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


def send_email(subject: str, html_body: str, text_body: str) -> None:
    """
    Sends a multipart email with both HTML + plain text.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["To"] = EMAIL_TO
    msg["From"] = f"Scanner <{SMTP_USER}>"

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
    return df_completed.iloc[-(n + 1):-1]


def action_and_trend(structure_broken: bool) -> Tuple[str, str]:
    trend = "BROKEN" if structure_broken else "INTACT"
    action = "SELL" if structure_broken else "HOLD"
    return action, trend


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


def decide_sell_count(contracts: int, broken10: bool, broken5: bool) -> int:
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


def reco_badge_html(reco: str) -> str:
    """
    Returns an HTML <span> with a colored pill.
    """
    r = (reco or "").upper()
    if r == "HOLD":
        return '<span class="pill pill-hold">HOLD</span>'
    if r.startswith("SELL"):
        # sell all stronger than partial
        if r == "SELL_ALL":
            return '<span class="pill pill-sellall">SELL_ALL</span>'
        return f'<span class="pill pill-sell">{r}</span>'
    return f'<span class="pill">{r}</span>'


def advice_badge_html(advice: str) -> str:
    a = (advice or "").upper()
    if "STOP HIT" in a:
        return '<span class="pill pill-sellall">STOP HIT</span>'
    if "VERY CLOSE" in a:
        return '<span class="pill pill-sell">VERY CLOSE</span>'
    if a.startswith("CLOSE"):
        return '<span class="pill pill-warn">CLOSE</span>'
    if a.startswith("OK"):
        return '<span class="pill pill-hold">OK</span>'
    return f'<span class="pill">{advice}</span>'


def build_html(rows: List[Dict], any_action: bool, reco_mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = len(rows)
    action_count = sum(1 for r in rows if (r.get("recommendation") or "").upper() != "HOLD")

    # Sort: action rows first, then by ticker
    def sort_key(r: Dict):
        reco = (r.get("recommendation") or "").upper()
        return (0 if reco != "HOLD" else 1, str(r.get("ticker") or ""))

    rows_sorted = sorted(rows, key=sort_key)

    table_rows_html = []
    for r in rows_sorted:
        table_rows_html.append(f"""
          <tr>
            <td class="mono">{r.get("ticker","")}</td>
            <td>{r.get("direction","")}</td>
            <td class="wrap">{r.get("option_name","")}</td>
            <td class="num">{r.get("contracts","")}</td>

            <td class="num">{r.get("close_price","")}</td>
            <td class="num">{r.get("current_price","")}</td>
            <td class="muted wrap">{r.get("current_src","")}</td>

            <td class="num">{r.get("atr","")}</td>
            <td class="num">{r.get("atr_stop","")}</td>

            <td>{advice_badge_html(r.get("atr_adv_close",""))}<div class="muted small mono">{r.get("atr_dist_close","")}</div></td>
            <td>{advice_badge_html(r.get("atr_adv_cur",""))}<div class="muted small mono">{r.get("atr_dist_cur","")}</div></td>

            <td class="center">
              <div class="small"><b>10d</b> {r.get("trend10_close","")} · {r.get("action10_close","")}</div>
              <div class="small"><b>5d</b> {r.get("trend5_close","")} · {r.get("action5_close","")}</div>
            </td>

            <td class="center">
              <div class="small"><b>10d</b> {r.get("trend10_current","")} · {r.get("action10_current","")}</div>
              <div class="small"><b>5d</b> {r.get("trend5_current","")} · {r.get("action5_current","")}</div>
            </td>

            <td class="center">{r.get("reco_basis","")}</td>
            <td class="center">{reco_badge_html(r.get("recommendation",""))}</td>
          </tr>
        """)

    headline_class = "banner banner-warn" if any_action else "banner banner-ok"
    headline_text = "ACTION NEEDED" if any_action else "NO ACTION (HOLD)"

    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Exit Check</title>
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        background: #f6f8fb;
        margin: 0;
        padding: 24px;
        color: #111827;
      }}
      .card {{
        max-width: 1200px;
        margin: 0 auto;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(17,24,39,0.08);
        overflow: hidden;
        border: 1px solid #e5e7eb;
      }}
      .header {{
        padding: 18px 20px;
        border-bottom: 1px solid #e5e7eb;
        background: #ffffff;
      }}
      .title {{
        font-size: 16px;
        font-weight: 700;
        margin: 0 0 6px 0;
      }}
      .meta {{
        font-size: 12px;
        color: #6b7280;
        margin: 0;
      }}
      .banner {{
        padding: 10px 20px;
        font-weight: 700;
        font-size: 13px;
      }}
      .banner-ok {{
        background: #ecfdf5;
        color: #065f46;
        border-bottom: 1px solid #d1fae5;
      }}
      .banner-warn {{
        background: #fff7ed;
        color: #9a3412;
        border-bottom: 1px solid #fed7aa;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      thead th {{
        position: sticky;
        top: 0;
        background: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
        padding: 10px 10px;
        font-size: 12px;
        text-align: left;
        color: #374151;
        white-space: nowrap;
      }}
      tbody td {{
        border-bottom: 1px solid #f1f5f9;
        padding: 10px 10px;
        font-size: 12px;
        vertical-align: top;
      }}
      tbody tr:hover {{
        background: #f9fafb;
      }}
      .num {{
        text-align: right;
        font-variant-numeric: tabular-nums;
      }}
      .center {{
        text-align: center;
      }}
      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      }}
      .muted {{
        color: #6b7280;
      }}
      .small {{
        font-size: 11px;
        line-height: 1.2;
      }}
      .wrap {{
        word-break: break-word;
        max-width: 260px;
      }}
      .pill {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 11px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        color: #374151;
      }}
      .pill-hold {{
        background: #ecfdf5;
        border-color: #d1fae5;
        color: #065f46;
      }}
      .pill-warn {{
        background: #fffbeb;
        border-color: #fde68a;
        color: #92400e;
      }}
      .pill-sell {{
        background: #fff7ed;
        border-color: #fed7aa;
        color: #9a3412;
      }}
      .pill-sellall {{
        background: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
      }}
      .footer {{
        padding: 14px 20px;
        font-size: 11px;
        color: #6b7280;
        background: #ffffff;
      }}
      @media (max-width: 900px) {{
        body {{ padding: 12px; }}
        thead th, tbody td {{ padding: 8px 8px; }}
        .wrap {{ max-width: 180px; }}
      }}
    </style>
  </head>
  <body>
    <div class="card">
      <div class="header">
        <p class="title">Exit Check Report</p>
        <p class="meta">
          Timestamp: <span class="mono">{ts}</span> ·
          Reco mode: <b>{reco_mode.upper()}</b> ·
          Positions: <b>{total}</b> ·
          Action rows: <b>{action_count}</b>
        </p>
      </div>

      <div class="{headline_class}">{headline_text}</div>

      <div style="overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Dir</th>
              <th>Option</th>
              <th class="num">Contracts</th>

              <th class="num">Close</th>
              <th class="num">Current</th>
              <th>Current Src</th>

              <th class="num">ATR</th>
              <th class="num">ATR Stop</th>

              <th>ATR Advice (Close)</th>
              <th>ATR Advice (Current)</th>

              <th class="center">Close Structure</th>
              <th class="center">Current Structure</th>

              <th class="center">Reco Basis</th>
              <th class="center">Recommendation</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows_html) if table_rows_html else '<tr><td colspan="15">No valid positions found.</td></tr>'}
          </tbody>
        </table>
      </div>

      <div class="footer">
        Structure levels + ATR computed from completed daily bars only (today’s partial bar excluded).
        Optional paper execution: APPLY_RECOMMENDATIONS={str(APPLY_RECOMMENDATIONS).lower()}.
      </div>
    </div>
  </body>
</html>
"""
    return html


def build_text_fallback(rows: List[Dict], any_action: bool, reco_mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append(f"Exit Check Report — {ts} — RECO_MODE={reco_mode.upper()}")
    lines.append("ACTION NEEDED" if any_action else "NO ACTION (HOLD)")
    lines.append("")

    # Action rows first
    rows_sorted = sorted(rows, key=lambda r: (0 if (r.get("recommendation","").upper() != "HOLD") else 1, r.get("ticker","")))
    for r in rows_sorted:
        lines.append(f"{r.get('ticker','')} ({r.get('direction','')}) | {r.get('option_name','')}")
        lines.append(f"  Contracts: {r.get('contracts','')} | Close: {r.get('close_price','')} | Current: {r.get('current_price','')} ({r.get('current_src','')})")
        lines.append(f"  ATR: {r.get('atr','')} | ATR Stop: {r.get('atr_stop','')}")
        lines.append(f"  ATR Close: {r.get('atr_adv_close','')} {r.get('atr_dist_close','')}")
        lines.append(f"  ATR Cur  : {r.get('atr_adv_cur','')} {r.get('atr_dist_cur','')}")
        lines.append(f"  Close Structure: 10d {r.get('trend10_close','')}/{r.get('action10_close','')} | 5d {r.get('trend5_close','')}/{r.get('action5_close','')}")
        lines.append(f"  Cur   Structure: 10d {r.get('trend10_current','')}/{r.get('action10_current','')} | 5d {r.get('trend5_current','')}/{r.get('action5_current','')}")
        lines.append(f"  RECO ({r.get('reco_basis','')}): {r.get('recommendation','')}")
        lines.append("-" * 60)
    return "\n".join(lines)


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

    rows: List[Dict] = []
    any_action = False

    updated_positions = positions.copy()

    min_needed = max(ATR_PERIOD, STRUCTURE_10) + 10  # buffer

    for idx, row in positions.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        option_name = str(row.get("option_name", "")).strip()
        entry_underlying = to_float(row.get("underlying_entry_price"))
        contracts = to_int(row.get("contracts"), default=0)

        is_call, direction = infer_direction(option_name)

        # Default "skipped" row (optional: you can omit skipped rows entirely)
        if not ticker or entry_underlying is None or is_call is None:
            continue
        if contracts <= 0:
            continue

        df = yf.download(ticker, period=HISTORY_PERIOD, interval="1d", progress=False)
        if df is None or df.empty:
            continue

        df = flatten_columns(df).dropna()

        close_price = float(df["Close"].iloc[-1])
        current_price, current_src = get_current_price(ticker, fallback=close_price)

        df_completed = remove_today_partial_bar(df).dropna()
        if len(df_completed) < min_needed:
            continue

        atr_series = calculate_atr(df_completed, ATR_PERIOD)
        atr_last = atr_series.iloc[-1]
        if pd.isna(atr_last):
            continue
        atr = float(atr_last)

        w10 = prior_window(df_completed, STRUCTURE_10)
        w5 = prior_window(df_completed, STRUCTURE_5)
        if w10.empty or w5.empty:
            continue

        low10, high10 = float(w10["Low"].min()), float(w10["High"].max())
        low5, high5 = float(w5["Low"].min()), float(w5["High"].max())

        # ATR stop (from underlying entry)
        if is_call:
            atr_stop = float(entry_underlying - ATR_MULTIPLIER * atr)
            broken10_close = bool(close_price < low10)
            broken5_close = bool(close_price < low5)
            broken10_current = bool(current_price < low10)
            broken5_current = bool(current_price < low5)
        else:
            atr_stop = float(entry_underlying + ATR_MULTIPLIER * atr)
            broken10_close = bool(close_price > high10)
            broken5_close = bool(close_price > high5)
            broken10_current = bool(current_price > high10)
            broken5_current = bool(current_price > high5)

        action10_close, trend10_close = action_and_trend(broken10_close)
        action5_close, trend5_close = action_and_trend(broken5_close)

        action10_current, trend10_current = action_and_trend(broken10_current)
        action5_current, trend5_current = action_and_trend(broken5_current)

        atr_adv_close, atr_dist_close, atr_dist_close_atr, _ = atr_advice(is_call, close_price, atr_stop, atr)
        atr_adv_cur, atr_dist_cur, atr_dist_cur_atr, _ = atr_advice(is_call, current_price, atr_stop, atr)

        if RECO_MODE == "close":
            sell_count = decide_sell_count(contracts, broken10_close, broken5_close)
            reco_basis = "CLOSE"
        else:
            sell_count = decide_sell_count(contracts, broken10_current, broken5_current)
            reco_basis = "CURRENT"

        recommendation = label_action(sell_count, contracts)
        if recommendation != "HOLD":
            any_action = True

        if APPLY_RECOMMENDATIONS and sell_count > 0:
            new_contracts = max(contracts - sell_count, 0)
            updated_positions.at[idx, "contracts"] = new_contracts

        # Store a row for the HTML table
        rows.append({
            "ticker": ticker,
            "direction": direction,
            "option_name": option_name,
            "contracts": contracts,
            "close_price": f"{close_price:.2f}",
            "current_price": f"{current_price:.2f}",
            "current_src": current_src,
            "atr": f"{atr:.2f}",
            "atr_stop": f"{atr_stop:.2f}",
            "atr_adv_close": atr_adv_close,
            "atr_adv_cur": atr_adv_cur,
            "atr_dist_close": f"{atr_dist_close:+.2f} USD ({atr_dist_close_atr:+.2f} ATR)",
            "atr_dist_cur": f"{atr_dist_cur:+.2f} USD ({atr_dist_cur_atr:+.2f} ATR)",
            "action10_close": action10_close,
            "trend10_close": trend10_close,
            "action5_close": action5_close,
            "trend5_close": trend5_close,
            "action10_current": action10_current,
            "trend10_current": trend10_current,
            "action5_current": action5_current,
            "trend5_current": trend5_current,
            "reco_basis": reco_basis,
            "recommendation": recommendation,
        })

    subject = "🚨 ACTION NEEDED – Recommendations" if any_action else "✅ No Action – Hold"

    if APPLY_RECOMMENDATIONS and not updated_positions.equals(positions):
        updated_positions.to_csv(CSV_FILE, index=False)

    # If exits_only and no action, do nothing (as before)
    if EMAIL_MODE == "exits_only" and not any_action:
        return

    # Build email bodies
    html_body = build_html(rows, any_action=any_action, reco_mode=RECO_MODE)
    text_body = build_text_fallback(rows, any_action=any_action, reco_mode=RECO_MODE)

    if not smtp_ready():
        print("SMTP secrets not set — printing report instead\n")
        print(subject)
        print(text_body)
        return

    send_email(subject, html_body, text_body)


if __name__ == "__main__":
    main()