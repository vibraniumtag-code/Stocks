#!/usr/bin/env python3
"""
unified_turtle_entries_only.py

Nightly runner:
- outputs ONLY fresh entries for tomorrow (no ledger)
- prints a ready-to-trade checklist
- saves CSV + TXT checklist
- sends a PRETTY HTML EMAIL (HTML-only to avoid clients choosing text/plain)
- optionally saves the same HTML to a file

Usage:
  python unified_turtle_entries_only.py --system 1 --allow-shorts 1 --top 300 \
    --save entries_tomorrow.csv --emit-checklist 1 --emit-html 1 --send-email 1

GitHub Secrets / env vars (SMTP):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO

Optional env:
  EMAIL_MODE=always | entries_only   (default: always)

Notes:
- Email-safe HTML (inline styles).
- Entries table is horizontally scrollable on mobile.
"""

import os, re, argparse, html
from datetime import datetime, date
from typing import Optional, List

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

import smtplib
import ssl
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid


# ---------------------- Config defaults ----------------------
SYSTEM_DEFAULT          = 1        # 1 => 20/10, 2 => 55/20
ALLOW_SHORTS_DEFAULT    = True
ATR_PERIOD_DEFAULT      = 20
K_STOP_ATR_DEFAULT      = 2.0
K_TAKE_ATR_DEFAULT      = 3.0

OPT_MIN_DTE_DEFAULT     = 30
OPT_MAX_DTE_DEFAULT     = 60
OPT_TARGET_DTE_DEFAULT  = 45
OPT_SL_PCT_DEFAULT      = 0.50     # -50% stop
OPT_TP_PCT_DEFAULT      = 1.00     # +100% target
OPT_ITM_DELTA_CALL      = 0.60
OPT_ITM_DELTA_PUT       = -0.60

TOP_N_BY_DEFAULT        = 3000

FALLBACK_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","UNH",
    "XOM","JNJ","WMT","PG","MA","HD","CVX","MRK","ABBV","KO",
    "PEP","COST","AVGO","LLY","ORCL","NKE","ADBE","CRM","NFLX","INTC",
    "AMD","QCOM","TXN","CSCO","UPS","CAT","GE","HON","IBM","AXP"
]

# ---------------------- SMTP / EMAIL ----------------------
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT_RAW = (os.getenv("SMTP_PORT") or "").strip()
SMTP_PORT = int(SMTP_PORT_RAW) if SMTP_PORT_RAW.isdigit() else 0
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
EMAIL_TO = os.getenv("EMAIL_TO", "").strip()

EMAIL_MODE = os.getenv("EMAIL_MODE", "always").strip().lower()  # always | entries_only


# ---------------------- Helpers ----------------------
def looks_like_ticker(s: str) -> bool:
    return bool(re.match(r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$", (s or "").strip().upper()))


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup]).strip().lower() for tup in df.columns]
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]
    alias = {"open":"Open","high":"High","low":"Low","close":"Close","adj close":"Close","adjclose":"Close"}
    out = {}
    for raw, std in alias.items():
        for c in df.columns:
            if raw in c and std not in out:
                out[std] = c
    need = {"Open","High","Low","Close"}
    if need <= set(out):
        res = df.rename(columns={v:k for k,v in out.items()})[["Open","High","Low","Close"]].copy()
        for c in ["Open","High","Low","Close"]:
            res[c] = pd.to_numeric(res[c], errors="coerce")
        return res.dropna(subset=["Open","High","Low","Close"])
    return pd.DataFrame()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def breakout_signals(df: pd.DataFrame, entry_lb: int, exit_lb: int, confirm_on_close: bool = True):
    highs = df["High"].rolling(entry_lb).max()
    lows  = df["Low"].rolling(entry_lb).min()
    if confirm_on_close:
        highs = highs.shift(1); lows = lows.shift(1)
    long_entry  = df["Close"] >= highs
    short_entry = df["Close"] <= lows
    return long_entry, short_entry


def find_target_expiration(ticker: str, min_dte: int, max_dte: int, target_dte: int) -> Optional[str]:
    if yf is None:
        return None
    try:
        stock = yf.Ticker(ticker)
        exps = stock.options
        if not exps:
            return None
        today = date.today()
        candidates = []
        for exp in exps:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (d - today).days
            candidates.append((abs(dte - target_dte), dte, exp))
        candidates.sort()
        for _, dte, exp in candidates:
            if min_dte <= dte <= max_dte:
                return exp
        return candidates[0][2]
    except Exception:
        return None


def select_option(options_df: pd.DataFrame, spot: float, is_call: bool, target_delta: float) -> Optional[dict]:
    if options_df is None or options_df.empty:
        return None
    df = options_df.copy()

    # basic liquidity gates (soft)
    if "volume" in df.columns or "openInterest" in df.columns:
        vol = df.get("volume")
        oi = df.get("openInterest")
        mask = pd.Series(True, index=df.index)
        if vol is not None: mask &= (vol.fillna(0) >= 1)
        if oi is not None:  mask &= (oi.fillna(0) >= 10)
        df = df[mask] if not mask.empty else df
        if df.empty:
            df = options_df.copy()

    # prefer delta if present
    if "delta" in df.columns and df["delta"].notna().any():
        df["delta_diff"] = (df["delta"] - target_delta).abs()
        return df.nsmallest(1, "delta_diff").iloc[0].to_dict()

    # fallback: closest to ATM by moneyness
    if is_call:
        df["mny"] = (df["strike"] / spot - 1.0).abs()
    else:
        df["mny"] = (spot / df["strike"] - 1.0).abs()
    return df.nsmallest(1, "mny").iloc[0].to_dict()


def build_universe(top: int) -> List[str]:
    top = max(1, min(top, len(FALLBACK_UNIVERSE)))
    return FALLBACK_UNIVERSE[:top]


def smtp_ready() -> bool:
    return all([SMTP_HOST, SMTP_PORT > 0, SMTP_USER, SMTP_PASS, EMAIL_TO])


def send_pretty_email(subject: str, html_body: str) -> None:
    """
    HTML-ONLY email to prevent clients from preferring the text/plain part.
    """
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = subject
    msg["To"] = EMAIL_TO
    msg["From"] = f"Turtle Scanner <{SMTP_USER}>"
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()

    # Explicit Content-Type (extra safety through some gateways)
    msg.replace_header("Content-Type", 'text/html; charset="utf-8"')

    # Debug proof in logs
    print("SENDING MIME:", msg["Content-Type"])
    print("HTML PREVIEW (first 200 chars):", html_body[:200].replace("\n", " "))

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


# ---------------------- Main pipeline ----------------------
def generate_new_entries(top: int,
                         system: int = 1,
                         atr_period: int = 20,
                         k_stop_atr: float = 2.0,
                         k_take_atr: float = 3.0,
                         allow_shorts: bool = True,
                         opt_min_dte: int = 30,
                         opt_max_dte: int = 60,
                         opt_target_dte: int = 45,
                         opt_sl: float = 0.50,
                         opt_tp: float = 1.00) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is required. pip install yfinance")
    entry_lb, exit_lb = (20,10) if system==1 else (55,20)

    # ------------------ FILTER CONFIG ------------------
    SMA_TREND_PERIOD = 200
    MIN_DOLLAR_VOL_20D = 20_000_000   # $20M/day avg over last 20 bars
    ATRP_MIN = 0.01                  # 1% ATR of price
    ATRP_MAX = 0.08                  # 8% ATR of price
    BREAKOUT_BUFFER_ATR = 0.10       # 0.10 ATR beyond channel
    # ---------------------------------------------------

    tickers = build_universe(top)
    rows = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    for t in tickers:
        try:
            df = yf.download(t, period="12mo", interval="1d", progress=False, auto_adjust=False)
            ohlc = _normalize_ohlc(df)
            if ohlc.empty or len(ohlc) < max(entry_lb, atr_period, SMA_TREND_PERIOD):
                continue

            atr = compute_atr(ohlc, period=atr_period)
            long_entry, short_entry = breakout_signals(ohlc, entry_lb, exit_lb, confirm_on_close=True)

            px = float(ohlc["Close"].iloc[-1])
            atrv = float(atr.iloc[-1])
            if not np.isfinite(atrv) or atrv <= 0:
                continue

            # Trend filter: SMA200
            sma200 = ohlc["Close"].rolling(SMA_TREND_PERIOD).mean().iloc[-1]
            if not np.isfinite(sma200):
                continue

            # Liquidity filter (if volume present)
            if "Volume" in df.columns:
                vol = pd.to_numeric(df["Volume"], errors="coerce")
                dollar_vol_20d = (vol * ohlc["Close"]).rolling(20).mean().iloc[-1]
                if not np.isfinite(dollar_vol_20d) or dollar_vol_20d < MIN_DOLLAR_VOL_20D:
                    continue

            # ATR% sanity filter
            atrp = atrv / px
            if atrp < ATRP_MIN or atrp > ATRP_MAX:
                continue

            # Breakout strength buffer
            entry_high = ohlc["High"].rolling(entry_lb).max().shift(1).iloc[-1]
            entry_low  = ohlc["Low"].rolling(entry_lb).min().shift(1).iloc[-1]
            strong_long  = px >= (entry_high + BREAKOUT_BUFFER_ATR * atrv)
            strong_short = px <= (entry_low  - BREAKOUT_BUFFER_ATR * atrv)

            action = None
            if long_entry.iloc[-1] and px > sma200 and strong_long:
                action = "BUY_CALL"
                stop_under = round(px - k_stop_atr*atrv, 2)
                target_under = round(px + k_take_atr*atrv, 2)
                opt_type = "CALL"
                target_delta = OPT_ITM_DELTA_CALL
            elif allow_shorts and short_entry.iloc[-1] and px < sma200 and strong_short:
                action = "BUY_PUT"
                stop_under = round(px + k_stop_atr*atrv, 2)
                target_under = round(px - k_take_atr*atrv, 2)
                opt_type = "PUT"
                target_delta = OPT_ITM_DELTA_PUT
            else:
                continue

            # Option selection
            expiry = find_target_expiration(t, opt_min_dte, opt_max_dte, opt_target_dte)
            option_symbol = ""
            strike = ""
            last_prem = np.nan
            opt_stop = np.nan
            opt_tgt  = np.nan
            delta = ""
            iv = ""

            if expiry:
                chain = yf.Ticker(t).option_chain(expiry)
                tab = chain.calls if opt_type=="CALL" else chain.puts
                chosen = select_option(tab, px, is_call=(opt_type=="CALL"), target_delta=target_delta)
                if chosen:
                    option_symbol = str(chosen.get("contractSymbol") or "")
                    strike = float(chosen.get("strike")) if chosen.get("strike") is not None else ""
                    if chosen.get("lastPrice") is not None:
                        last_prem = float(chosen["lastPrice"])
                        opt_stop = round(last_prem*(1-opt_sl), 2)
                        opt_tgt  = round(last_prem*(1+opt_tp), 2)
                    if "delta" in chosen and chosen["delta"] == chosen["delta"]:
                        delta = float(chosen["delta"])
                    if "impliedVolatility" in chosen and chosen["impliedVolatility"] == chosen["impliedVolatility"]:
                        iv = float(chosen["impliedVolatility"])

            rows.append({
                "Date": today_str,
                "Ticker": t,
                "Action": action,
                "SpotClose": round(px,2),
                "ATR": round(atrv,3),
                "EntryPlan": "Next Open",
                "StopUnderlying": stop_under,
                "TargetUnderlying": target_under,
                "Expiry": expiry or "",
                "OptionType": opt_type,
                "OptionStrike": strike,
                "OptionLast": round(last_prem,2) if last_prem==last_prem else "",
                "OptionStop": opt_stop if opt_stop==opt_stop else "",
                "OptionTarget": opt_tgt if opt_tgt==opt_tgt else "",
                "OptionSymbol": option_symbol,
                "Delta": delta,
                "IV": iv,
            })
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if not out.empty:
        order = pd.CategoricalDtype(categories=["BUY_CALL","BUY_PUT"], ordered=True)
        out["ActionOrder"] = out["Action"].astype(order)
        out = out.sort_values(["ActionOrder","Ticker"]).drop(columns=["ActionOrder"])
    return out


# ---------------------- Checklist rendering ----------------------
def _fmt(x, n=2):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


def print_checklist(df: pd.DataFrame, date_str: str) -> str:
    title = f"Tomorrow's Entry Checklist — {date_str}"
    lines = [title, "=" * len(title), ""]
    if df.empty:
        lines.append("• No new entries for tomorrow.")
        text = "\n".join(lines)
        print(text)
        return text

    for _, r in df.iterrows():
        delta_txt = f", Δ {_fmt(r['Delta'],3)}" if str(r.get("Delta","")) not in ("", "nan") else ""
        iv_txt = f", IV {_fmt(r['IV'],3)}" if str(r.get("IV","")) not in ("", "nan") else ""
        lines.append(
            f"• {r['Ticker']} — {r['Action']} ({r['EntryPlan']})\n"
            f"  Spot {_fmt(r['SpotClose'])} | ATR {_fmt(r['ATR'],3)}\n"
            f"  Underlying SL {_fmt(r['StopUnderlying'])} / TP {_fmt(r['TargetUnderlying'])}\n"
            f"  Option: {r['OptionType']} {_fmt(r['OptionStrike'])} exp {r['Expiry']} "
            f"[{r['OptionSymbol']}] @ last {_fmt(r['OptionLast'])} | SL {_fmt(r['OptionStop'])} / TP {_fmt(r['OptionTarget'])}"
            f"{delta_txt}{iv_txt}\n"
        )
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------- Pretty HTML Email ----------------------
def build_pretty_html_email(df: pd.DataFrame, checklist_text: str, date_str: str, args: argparse.Namespace) -> str:
    safe_date = html.escape(date_str)
    safe_checklist = html.escape(checklist_text or "")

    total = 0 if df is None else len(df)
    calls = 0 if df is None or df.empty else int((df["Action"] == "BUY_CALL").sum()) if "Action" in df.columns else 0
    puts  = 0 if df is None or df.empty else int((df["Action"] == "BUY_PUT").sum())  if "Action" in df.columns else 0

    def badge(action: str) -> str:
        a = (action or "").upper()
        if a == "BUY_CALL":
            return "<span style='display:inline-block;padding:2px 8px;border-radius:999px;font-weight:800;font-size:11px;background:#ecfdf5;color:#065f46;border:1px solid #d1fae5;'>BUY_CALL</span>"
        if a == "BUY_PUT":
            return "<span style='display:inline-block;padding:2px 8px;border-radius:999px;font-weight:800;font-size:11px;background:#fff7ed;color:#9a3412;border:1px solid #fed7aa;'>BUY_PUT</span>"
        return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;font-weight:800;font-size:11px;background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;'>{html.escape(action or '')}</span>"

    def fmt(col, v):
        if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
            return ""
        try:
            if col in {"SpotClose","StopUnderlying","TargetUnderlying","OptionStrike","OptionLast","OptionStop","OptionTarget"}:
                return f"{float(v):.2f}"
            if col in {"ATR","Delta","IV"}:
                return f"{float(v):.3f}"
        except Exception:
            pass
        return str(v)

    cols = [
        "Ticker","Action","SpotClose","ATR","StopUnderlying","TargetUnderlying",
        "Expiry","OptionType","OptionStrike","OptionLast","OptionStop","OptionTarget","OptionSymbol","Delta","IV"
    ]

    # Build table
    if df is None or df.empty:
        table_html = """
          <div style="padding:12px 14px;border:1px solid #e5e7eb;border-radius:12px;background:#f9fafb;font-size:13px;">
            ✅ No new entries for tomorrow.
          </div>
        """
    else:
        cols = [c for c in cols if c in df.columns]
        head = "".join([
            f"<th style='text-align:left;padding:10px;border-bottom:1px solid #e5e7eb;background:#f9fafb;font-size:12px;white-space:nowrap;'>{html.escape(c)}</th>"
            for c in cols
        ])

        numeric_right = {"SpotClose","ATR","StopUnderlying","TargetUnderlying","OptionStrike","OptionLast","OptionStop","OptionTarget","Delta","IV"}

        rows_html = []
        for _, r in df[cols].iterrows():
            tds = []
            for c in cols:
                if c == "Action":
                    tds.append(f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;white-space:nowrap;text-align:center;'>{badge(str(r.get(c,'')))}</td>")
                else:
                    align = "right" if c in numeric_right else "left"
                    tds.append(f"<td style='padding:10px;border-bottom:1px solid #f1f5f9;font-size:12px;white-space:nowrap;text-align:{align};font-variant-numeric:tabular-nums;'>{html.escape(fmt(c, r.get(c,'')))}</td>")
            rows_html.append("<tr>" + "".join(tds) + "</tr>")

        table_html = f"""
          <div style="border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;">
            <div style="overflow:auto;">
              <table style="width:100%;border-collapse:collapse;">
                <thead><tr>{head}</tr></thead>
                <tbody>
                  {''.join(rows_html)}
                </tbody>
              </table>
            </div>
          </div>
        """

    # Summary cards
    cards = f"""
      <table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;border-collapse:separate;border-spacing:12px 0;margin:0 0 12px 0;">
        <tr>
          <td style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:12px;">
            <div style="font-size:12px;color:#6b7280;font-weight:700;">Total entries</div>
            <div style="font-size:18px;font-weight:900;color:#111827;margin-top:2px;">{total}</div>
          </td>
          <td style="background:#ecfdf5;border:1px solid #d1fae5;border-radius:12px;padding:12px;">
            <div style="font-size:12px;color:#065f46;font-weight:700;">BUY_CALL</div>
            <div style="font-size:18px;font-weight:900;color:#065f46;margin-top:2px;">{calls}</div>
          </td>
          <td style="background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;padding:12px;">
            <div style="font-size:12px;color:#9a3412;font-weight:700;">BUY_PUT</div>
            <div style="font-size:18px;font-weight:900;color:#9a3412;margin-top:2px;">{puts}</div>
          </td>
        </tr>
      </table>
    """

    # Run command
    usage = (
        f"python unified_turtle_entries_only.py --system {args.system} --allow-shorts {int(bool(args.allow_shorts))} "
        f"--top {args.top} --save {html.escape(args.save)} --emit-checklist {int(args.emit_checklist)} "
        f"--emit-html {int(args.emit_html)} --send-email {int(args.send_email)}"
    )

    html_out = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f6f7fb;">
  <div style="width:100%;padding:24px 12px;background:#f6f7fb;">
    <div style="max-width:980px;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb;border-radius:16px;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;color:#111827;">

      <!-- Header -->
      <div style="background:#0b1220;color:#ffffff;padding:18px 22px;">
        <div style="font-size:18px;font-weight:900;line-height:1.25;">🐢 Turtle Scanner — New Entries</div>
        <div style="font-size:12px;opacity:.9;margin-top:6px;">For next open · {safe_date}</div>
      </div>

      <!-- Content -->
      <div style="padding:18px 22px;">
        {cards}

        <!-- Run command -->
        <div style="margin:12px 0 16px 0;">
          <div style="font-size:13px;font-weight:900;margin:0 0 8px 0;">▶️ Run Command</div>
          <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:12px;font-size:12px;line-height:1.45;">
            <span style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;background:#eef0f6;padding:2px 6px;border-radius:6px;">{usage}</span>
          </div>
        </div>

        <!-- Entries table -->
        <div style="margin:0 0 16px 0;">
          <div style="font-size:13px;font-weight:900;margin:0 0 8px 0;">📈 Entries</div>
          {table_html}
          <div style="margin-top:8px;font-size:12px;color:#6b7280;">Tip: table scrolls horizontally on mobile.</div>
        </div>

        <!-- Checklist -->
        <div style="margin:0 0 6px 0;">
          <div style="font-size:13px;font-weight:900;margin:0 0 8px 0;">🧾 Checklist Output</div>
          <div style="background:#0b1220;color:#e5e7eb;border:1px solid #1f2a44;border-radius:12px;overflow:hidden;">
            <div style="padding:10px 12px;background:#101a2f;color:#cbd5e1;font-size:12px;border-bottom:1px solid #1f2a44;">
              stdout (copy/paste)
            </div>
            <pre style="margin:0;padding:12px;font-size:12px;line-height:1.5;white-space:pre;overflow:auto;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;">{safe_checklist}</pre>
          </div>
        </div>

      </div>

      <!-- Footer -->
      <div style="padding:14px 22px;background:#fbfbfd;border-top:1px solid #eef0f6;font-size:12px;color:#6b7280;">
        Generated by unified_turtle_entries_only.py · {safe_date}
      </div>
    </div>
  </div>
</body>
</html>
"""
    return html_out


# ---------------------- CLI ----------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Nightly Turtle new-entry scanner (no ledger) + checklist + pretty HTML email")
    ap.add_argument("--system", type=int, default=SYSTEM_DEFAULT, choices=[1,2])
    ap.add_argument("--allow-shorts", type=int, default=int(ALLOW_SHORTS_DEFAULT))
    ap.add_argument("--atr", type=int, default=ATR_PERIOD_DEFAULT)
    ap.add_argument("--k-stop-atr", type=float, default=K_STOP_ATR_DEFAULT)
    ap.add_argument("--k-take-atr", type=float, default=K_TAKE_ATR_DEFAULT)
    ap.add_argument("--top", type=int, default=TOP_N_BY_DEFAULT)
    ap.add_argument("--opt-min-dte", type=int, default=OPT_MIN_DTE_DEFAULT)
    ap.add_argument("--opt-max-dte", type=int, default=OPT_MAX_DTE_DEFAULT)
    ap.add_argument("--opt-target-dte", type=int, default=OPT_TARGET_DTE_DEFAULT)
    ap.add_argument("--opt-sl", type=float, default=OPT_SL_PCT_DEFAULT)
    ap.add_argument("--opt-tp", type=float, default=OPT_TP_PCT_DEFAULT)
    ap.add_argument("--save", type=str, default="entries_tomorrow.csv")
    ap.add_argument("--emit-checklist", type=int, default=1, help="1=also write a .txt checklist next to CSV, 0=print only")
    ap.add_argument("--emit-html", type=int, default=1, help="1=also write a .html email view next to CSV, 0=skip")
    ap.add_argument("--html-out", type=str, default="", help="Optional custom path for HTML output")
    ap.add_argument("--send-email", type=int, default=1, help="1=send email via SMTP env vars, 0=skip sending")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    df = generate_new_entries(
        top=args.top,
        system=args.system,
        atr_period=args.atr,
        k_stop_atr=args.k_stop_atr,
        k_take_atr=args.k_take_atr,
        allow_shorts=bool(int(args.allow_shorts)),
        opt_min_dte=args.opt_min_dte,
        opt_max_dte=args.opt_max_dte,
        opt_target_dte=args.opt_target_dte,
        opt_sl=args.opt_sl,
        opt_tp=args.opt_tp
    )

    # Save CSV
    df.to_csv(args.save, index=False)
    print(f"Saved {len(df)} new entries to {args.save}")

    # Checklist
    date_str = datetime.now().strftime("%Y-%m-%d")
    checklist = print_checklist(df, date_str)

    # Optionally save .txt checklist
    if int(args.emit_checklist) == 1:
        base, _ = os.path.splitext(args.save)
        txt_path = f"{base}_checklist_{date_str}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(checklist)
        print(f"Checklist saved to: {txt_path}")

    # Build pretty HTML
    html_email = build_pretty_html_email(df, checklist, date_str, args)

    # Optionally save HTML to disk
    if int(args.emit_html) == 1:
        if args.html_out.strip():
            html_path = args.html_out.strip()
        else:
            base, _ = os.path.splitext(args.save)
            html_path = f"{base}_email_{date_str}.html"

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_email)
        print(f"HTML email view saved to: {html_path}")

    # Optionally send email (HTML-only)
    if int(args.send_email) == 1:
        if EMAIL_MODE == "entries_only" and (df is None or df.empty):
            print("EMAIL_MODE=entries_only and no entries — skipping email.")
            return 0

        if not smtp_ready():
            print("SMTP not ready — set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO")
            return 0

        subject = f"🐢 Turtle Entries — {date_str}"
        send_pretty_email(subject, html_email)
        print(f"Email sent to: {EMAIL_TO}")

    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))