#!/usr/bin/env python3
"""
exit_check.py  (UPDATED: TURTLE-STYLE PRETTY HTML EMAIL)

Changes vs your version:
- ✅ Turtle-style email shell (dark header + summary cards + scroll table)
- ✅ Email-safe HTML (inline styles; minimal <style> usage)
- ✅ HTML-ONLY email (prevents clients from choosing text/plain)
- ✅ Keeps your existing logic and the plain-text fallback for stdout (when SMTP not set)
- ✅ Still respects EMAIL_MODE=exits_only

Env:
  ATR_VERY_CLOSE, ATR_CLOSE, APPLY_RECOMMENDATIONS, RECO_MODE
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO, EMAIL_MODE
  HISTORY_PERIOD
"""

import os
import re
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
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
def html_escape(s) -> str:
    return ("" if s is None else str(s)).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

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

def send_pretty_email(subject: str, html_body: str) -> None:
    """
    HTML-ONLY email to prevent clients from preferring text/plain.
    """
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = subject
    msg["To"] = EMAIL_TO
    msg["From"] = f"Exit Check <{SMTP_USER}>"
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.replace_header("Content-Type", 'text/html; charset="utf-8"')

    if SMTP_PORT == 465:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    else:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=ctx)
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
        return 2 if contracts >= 3 else 1
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
# PRETTY BADGES (INLINE)
# =========================
def pill(text: str, kind: str = "neutral") -> str:
    t = (text or "").strip()
    if t == "":
        return ""
    styles = {
        "good":   "background:#ecfdf5;color:#065f46;border:1px solid #d1fae5;",
        "warn":   "background:#fff7ed;color:#9a3412;border:1px solid #fed7aa;",
        "bad":    "background:#fef2f2;color:#991b1b;border:1px solid #fecaca;",
        "info":   "background:#eef2ff;color:#3730a3;border:1px solid #e0e7ff;",
        "neutral":"background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;",
    }
    st = styles.get(kind, styles["neutral"])
    return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;font-weight:800;font-size:11px;{st}white-space:nowrap;'>{html_escape(t)}</span>"

def reco_badge_html(reco: str) -> str:
    r = (reco or "").upper()
    if r == "HOLD":
        return pill("HOLD", "good")
    if r == "SELL_ALL":
        return pill("SELL_ALL", "bad")
    if r.startswith("SELL"):
        return pill(r, "warn")
    return pill(r, "neutral")

def advice_badge_html(advice: str) -> str:
    a = (advice or "").upper()
    if "STOP HIT" in a:
        return pill("STOP HIT", "bad")
    if "VERY CLOSE" in a:
        return pill("VERY CLOSE", "warn")
    if a.startswith("CLOSE"):
        return pill("CLOSE", "warn")
    if a.startswith("OK"):
        return pill("OK", "good")
    return pill(advice, "neutral")


# =========================
# PRETTY HTML EMAIL (TURTLE STYLE)
# =========================
def build_pretty_html(rows: List[Dict], any_action: bool, reco_mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    date_str = datetime.now().strftime("%Y-%m-%d")

    total = len(rows)
    action_rows = sum(1 for r in rows if (r.get("recommendation") or "").upper() != "HOLD")
    hold_rows = total - action_rows

    # Preheader controls inbox preview text
    preheader = html_escape(f"Exit Check · {date_str} · {action_rows} action rows, {hold_rows} holds.")

    # Sort: action first, then ticker
    def sort_key(r: Dict):
        reco = (r.get("recommendation") or "").upper()
        return (0 if reco != "HOLD" else 1, str(r.get("ticker") or ""))

    rows_sorted = sorted(rows, key=sort_key)

    # Build table rows
    numeric_right = {"contracts", "close_price", "current_price", "atr", "atr_stop"}
    zebra_a = "#ffffff"
    zebra_b = "#fcfcfd"

    tr_html = []
    for i, r in enumerate(rows_sorted):
        bg = zebra_a if (i % 2 == 0) else zebra_b

        def td(key: str, align: str = "left", mono: bool = False, nowrap: bool = True) -> str:
            style = f"padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:{align};"
            style += "font-variant-numeric:tabular-nums;" if align == "right" else ""
            style += "white-space:nowrap;" if nowrap else ""
            style += f"background:{bg};"
            if mono:
                style += "font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;"
            return f"<td style='{style}'>{html_escape(r.get(key,''))}</td>"

        # special cells
        reco_cell = f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:center;white-space:nowrap;background:{bg};'>{reco_badge_html(r.get('recommendation',''))}</td>"
        advice_close_cell = (
            f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;white-space:nowrap;background:{bg};'>"
            f"{advice_badge_html(r.get('atr_adv_close',''))}"
            f"<div style='margin-top:4px;font-size:11px;color:#6b7280;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;'>"
            f"{html_escape(r.get('atr_dist_close',''))}</div></td>"
        )
        advice_cur_cell = (
            f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;white-space:nowrap;background:{bg};'>"
            f"{advice_badge_html(r.get('atr_adv_cur',''))}"
            f"<div style='margin-top:4px;font-size:11px;color:#6b7280;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;'>"
            f"{html_escape(r.get('atr_dist_cur',''))}</div></td>"
        )

        close_struct = (
            f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:center;white-space:nowrap;background:{bg};'>"
            f"<div style='font-size:11px;'><b>10d</b> {html_escape(r.get('trend10_close',''))} · {html_escape(r.get('action10_close',''))}</div>"
            f"<div style='font-size:11px;'><b>5d</b> {html_escape(r.get('trend5_close',''))} · {html_escape(r.get('action5_close',''))}</div>"
            f"</td>"
        )
        cur_struct = (
            f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:center;white-space:nowrap;background:{bg};'>"
            f"<div style='font-size:11px;'><b>10d</b> {html_escape(r.get('trend10_current',''))} · {html_escape(r.get('action10_current',''))}</div>"
            f"<div style='font-size:11px;'><b>5d</b> {html_escape(r.get('trend5_current',''))} · {html_escape(r.get('action5_current',''))}</div>"
            f"</td>"
        )

        tr_html.append(
            "<tr>"
            + td("ticker", align="left", mono=True)
            + td("direction", align="center")
            + f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;white-space:nowrap;background:{bg};'>{html_escape(r.get('option_name',''))}</td>"
            + td("contracts", align="right")
            + td("close_price", align="right")
            + td("current_price", align="right")
            + f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#6b7280;white-space:nowrap;background:{bg};'>{html_escape(r.get('current_src',''))}</td>"
            + td("atr", align="right")
            + td("atr_stop", align="right")
            + advice_close_cell
            + advice_cur_cell
            + close_struct
            + cur_struct
            + f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;text-align:center;white-space:nowrap;background:{bg};'>{html_escape(r.get('reco_basis',''))}</td>"
            + reco_cell
            + "</tr>"
        )

    # Empty state
    if not tr_html:
        tr_html = ["<tr><td colspan='15' style='padding:12px;font-size:13px;'>No valid positions found.</td></tr>"]

    # summary cards
    headline = "🚨 ACTION NEEDED" if any_action else "✅ NO ACTION (HOLD)"
    headline_bar = (
        "background:#fff7ed;color:#9a3412;border-bottom:1px solid #fed7aa;"
        if any_action else
        "background:#ecfdf5;color:#065f46;border-bottom:1px solid #d1fae5;"
    )

    cards = f"""
      <table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;border-collapse:separate;border-spacing:12px 0;margin:0 0 12px 0;">
        <tr>
          <td style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:12px;">
            <div style="font-size:12px;color:#6b7280;font-weight:700;">Positions</div>
            <div style="font-size:18px;font-weight:900;color:#111827;margin-top:2px;">{total}</div>
          </td>
          <td style="background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:12px;">
            <div style="font-size:12px;color:#9a3412;font-weight:700;">Action rows</div>
            <div style="font-size:18px;font-weight:900;color:#9a3412;margin-top:2px;">{action_rows}</div>
          </td>
          <td style="background:#ecfdf5;border:1px solid #d1fae5;border-radius:12px;padding:12px;">
            <div style="font-size:12px;color:#065f46;font-weight:700;">HOLD rows</div>
            <div style="font-size:18px;font-weight:900;color:#065f46;margin-top:2px;">{hold_rows}</div>
          </td>
        </tr>
      </table>
    """

    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f6f7fb;">
  <div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent;">
    {preheader}
  </div>

  <div style="width:100%;padding:24px 12px;background:#f6f7fb;">
    <div style="max-width:1200px;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb;border-radius:16px;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;color:#111827;">

      <!-- Header -->
      <div style="background:#0b1220;color:#ffffff;padding:18px 22px;">
        <div style="font-size:18px;font-weight:900;line-height:1.25;">🧯 Exit Check — Report</div>
        <div style="font-size:12px;opacity:.9;margin-top:6px;">{html_escape(ts)} · RECO_MODE <b>{html_escape(reco_mode.upper())}</b></div>
      </div>

      <!-- Banner -->
      <div style="padding:10px 22px;font-weight:900;font-size:13px;{headline_bar}">
        {headline}
      </div>

      <!-- Content -->
      <div style="padding:18px 22px;">
        {cards}

        <!-- Table -->
        <div style="margin:0 0 16px 0;">
          <div style="font-size:13px;font-weight:900;margin:0 0 8px 0;">📋 Positions</div>
          <div style="border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;">
            <div style="overflow:auto;">
              <table style="width:100%;border-collapse:collapse;">
                <thead>
                  <tr>
                    <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Ticker</th>
                    <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Dir</th>
                    <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Option</th>
                    <th style="text-align:right;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Contracts</th>

                    <th style="text-align:right;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Close</th>
                    <th style="text-align:right;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Current</th>
                    <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Current Src</th>

                    <th style="text-align:right;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">ATR</th>
                    <th style="text-align:right;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">ATR Stop</th>

                    <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">ATR Advice (Close)</th>
                    <th style="text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">ATR Advice (Current)</th>

                    <th style="text-align:center;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Close Structure</th>
                    <th style="text-align:center;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Current Structure</th>

                    <th style="text-align:center;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Reco Basis</th>
                    <th style="text-align:center;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;">Recommendation</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(tr_html)}
                </tbody>
              </table>
            </div>
          </div>
          <div style="margin-top:8px;font-size:12px;color:#6b7280;">Tip: table scrolls horizontally on mobile.</div>
        </div>

        <!-- Notes -->
        <div style="padding:12px 14px;border:1px solid #e5e7eb;border-radius:12px;background:#fbfbfd;font-size:12px;color:#6b7280;">
          Structure levels + ATR computed from completed daily bars only (today’s partial bar excluded).<br/>
          APPLY_RECOMMENDATIONS=<b>{str(APPLY_RECOMMENDATIONS).lower()}</b>
        </div>

      </div>

      <!-- Footer -->
      <div style="padding:14px 22px;background:#fbfbfd;border-top:1px solid #eef0f6;font-size:12px;color:#6b7280;">
        Generated by exit_check.py · {html_escape(date_str)}
      </div>
    </div>
  </div>
</body>
</html>
"""

def build_text_fallback(rows: List[Dict], any_action: bool, reco_mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append(f"Exit Check Report — {ts} — RECO_MODE={reco_mode.upper()}")
    lines.append("ACTION NEEDED" if any_action else "NO ACTION (HOLD)")
    lines.append("")

    rows_sorted = sorted(
        rows,
        key=lambda r: (0 if (r.get("recommendation", "").upper() != "HOLD") else 1, r.get("ticker", "")),
    )
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

    # subject with counts
    date_str = datetime.now().strftime("%Y-%m-%d")
    action_rows = sum(1 for r in rows if (r.get("recommendation") or "").upper() != "HOLD")
    if any_action:
        subject = f"🚨 Exit Check — {date_str} ({action_rows} action)"
    else:
        subject = f"✅ Exit Check — {date_str} (no action)"

    if APPLY_RECOMMENDATIONS and not updated_positions.equals(positions):
        updated_positions.to_csv(CSV_FILE, index=False)

    if EMAIL_MODE == "exits_only" and not any_action:
        return

    html_body = build_pretty_html(rows, any_action=any_action, reco_mode=RECO_MODE)
    text_body = build_text_fallback(rows, any_action=any_action, reco_mode=RECO_MODE)

    if not smtp_ready():
        print("SMTP secrets not set — printing report instead\n")
        print(subject)
        print(text_body)
        return

    send_pretty_email(subject, html_body)


if __name__ == "__main__":
    main()