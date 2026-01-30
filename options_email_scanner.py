#!/usr/bin/env python3
"""
options_us_universe_scanner.py

US universe options scanner (free data via yfinance) that emails a shortlist.

✅ Uses SAME email env vars as your portfolio_manager scripts:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO

✅ Universe:
    - Downloads official symbol lists (NASDAQ + NYSE/AMEX) from NasdaqTrader
    - Ranks by recent $ volume and scans the top N (configurable)
    - Scans options chains for tradability filters + scores

NOTE:
- “Scan ALL US options for ALL tickers” is not feasible on GitHub Actions with free yfinance.
  This script uses a realistic approach: scan the most liquid tickers first.
"""

import os, math, time
from datetime import datetime, timezone
from io import StringIO
from urllib.request import urlopen, Request

import pandas as pd
import yfinance as yf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# -----------------------------
# Config (env overrides allowed)
# -----------------------------
MIN_DTE         = int(os.getenv("MIN_DTE", "14"))
MAX_DTE         = int(os.getenv("MAX_DTE", "45"))
MIN_OI          = int(os.getenv("MIN_OI", "300"))
MIN_OPT_VOL     = int(os.getenv("MIN_OPT_VOL", "50"))
MAX_SPREAD_PCT  = float(os.getenv("MAX_SPREAD_PCT", "0.06"))   # 6% of mid
TOP_PER_TICKER  = int(os.getenv("TOP_PER_TICKER", "2"))
MAX_TOTAL_ROWS  = int(os.getenv("MAX_TOTAL_ROWS", "40"))

# Universe controls (critical to make this runnable)
MAX_STOCKS      = int(os.getenv("MAX_STOCKS", "600"))          # tickers to scan after liquidity ranking
MIN_DVOL_USD    = float(os.getenv("MIN_DVOL_USD", "5e7"))      # min $ volume (e.g., 50,000,000)

# Calls-only or puts-only
CALLS_ONLY      = os.getenv("CALLS_ONLY", "0") == "1"
PUTS_ONLY       = os.getenv("PUTS_ONLY", "0") == "1"

# Performance / reliability knobs
SLEEP_BETWEEN_TICKERS = float(os.getenv("SLEEP_BETWEEN_TICKERS", "0.0"))
MAX_TICKER_ERRORS     = int(os.getenv("MAX_TICKER_ERRORS", "60"))

# Email transport knobs (match portfolio_manager conventions)
SMTP_TLS = os.getenv("SMTP_TLS", "").strip().lower()  # "", "starttls", "ssl"
SMTP_TIMEOUT = int(os.getenv("SMTP_TIMEOUT", "30"))

# Official symbol list sources (US)
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# -----------------------------
# Email helpers (PORTFOLIO MANAGER STYLE)
# -----------------------------
def parse_smtp_env() -> dict:
    """
    Same variables as your portfolio_manager scripts:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO
    """
    host = os.environ.get("SMTP_HOST", "").strip()
    port = os.environ.get("SMTP_PORT", "").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    pwd  = os.environ.get("SMTP_PASS", "").strip()
    to   = os.environ.get("EMAIL_TO", "").strip()

    missing = [k for k, v in {
        "SMTP_HOST": host,
        "SMTP_PORT": port,
        "SMTP_USER": user,
        "SMTP_PASS": pwd,
        "EMAIL_TO": to,
    }.items() if not v]

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    try:
        port_i = int(port)
    except ValueError as e:
        raise RuntimeError("SMTP_PORT must be an integer") from e

    return {"host": host, "port": port_i, "user": user, "pass": pwd, "to": to}

def send_email_smtp(cfg: dict, subject: str, html_body: str):
    """
    Sends HTML-only email. Supports:
      - SMTP_TLS=ssl      -> SMTP_SSL
      - SMTP_TLS=starttls -> SMTP + STARTTLS
      - SMTP_TLS=""       -> if port==465 uses SSL else STARTTLS (safe default)
    """
    msg = MIMEMultipart("alternative")
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    tls_mode = SMTP_TLS
    if not tls_mode:
        tls_mode = "ssl" if int(cfg["port"]) == 465 else "starttls"

    if tls_mode == "ssl":
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=SMTP_TIMEOUT) as server:
            server.login(cfg["user"], cfg["pass"])
            server.sendmail(cfg["user"], [cfg["to"]], msg.as_string())
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=SMTP_TIMEOUT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(cfg["user"], cfg["pass"])
            server.sendmail(cfg["user"], [cfg["to"]], msg.as_string())

# -----------------------------
# Market helpers
# -----------------------------
def _download_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")

def fetch_us_symbols() -> list[str]:
    """
    Pull NASDAQ + NYSE/AMEX symbols from NasdaqTrader official lists.
    Normalizes '.' -> '-' for yfinance (e.g., BRK.B => BRK-B).
    Handles NaNs (floats) safely.
    """
    nasdaq_txt = _download_text(NASDAQ_LISTED_URL)
    other_txt  = _download_text(OTHER_LISTED_URL)

    nasdaq_lines = [ln for ln in nasdaq_txt.splitlines() if ln and "File Creation Time" not in ln]
    other_lines  = [ln for ln in other_txt.splitlines() if ln and "File Creation Time" not in ln]

    nasdaq_df = pd.read_csv(StringIO("\n".join(nasdaq_lines)), sep="|", dtype=str)
    other_df  = pd.read_csv(StringIO("\n".join(other_lines)),  sep="|", dtype=str)

    # Filter test issues when column exists
    if "Test Issue" in nasdaq_df.columns:
        nasdaq_df = nasdaq_df[nasdaq_df["Test Issue"].fillna("") == "N"]
    if "Test Issue" in other_df.columns:
        other_df = other_df[other_df["Test Issue"].fillna("") == "N"]

    nasdaq_syms = nasdaq_df.get("Symbol", pd.Series([], dtype=str)).fillna("").astype(str).tolist()
    other_syms  = other_df.get("ACT Symbol", pd.Series([], dtype=str)).fillna("").astype(str).tolist()

    def normalize(sym: str) -> str:
        sym = (sym or "").strip()
        if not sym:
            return ""
        sym = sym.replace(".", "-")
        return sym

    out = set()
    for s in nasdaq_syms + other_syms:
        s = normalize(s)
        if not s:
            continue
        # keep only plain ASCII symbols yfinance can handle
        try:
            if not str(s).isascii():
                continue
        except Exception:
            continue

        # basic sanity filters
        if len(s) > 10:
            continue
        if "^" in s or "/" in s or " " in s:
            continue

        out.add(s)

    return sorted(out)

def pick_liquid_tickers(symbols: list[str]) -> pd.DataFrame:
    """
    Batch-download last close + volume and compute $ volume, then select top.
    """
    rows = []
    chunk = 200

    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            data = yf.download(
                tickers=batch,
                period="2d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False
            )
        except Exception:
            continue

        if data is None or data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            # Case: columns (Field, Ticker)
            if "Close" in data.columns.get_level_values(0):
                for t in batch:
                    try:
                        close = float(data["Close"][t].dropna().iloc[-1])
                        vol   = float(data["Volume"][t].dropna().iloc[-1])
                        dvol  = close * vol
                        if close > 0 and vol > 0:
                            rows.append((t, close, vol, dvol))
                    except Exception:
                        pass
            else:
                # columns (Ticker, Field)
                for t in batch:
                    try:
                        close = float(data[t]["Close"].dropna().iloc[-1])
                        vol   = float(data[t]["Volume"].dropna().iloc[-1])
                        dvol  = close * vol
                        if close > 0 and vol > 0:
                            rows.append((t, close, vol, dvol))
                    except Exception:
                        pass
        else:
            # single ticker fallback
            try:
                close = float(data["Close"].dropna().iloc[-1])
                vol   = float(data["Volume"].dropna().iloc[-1])
                dvol  = close * vol
                if close > 0 and vol > 0:
                    rows.append((batch[0], close, vol, dvol))
            except Exception:
                pass

    df = pd.DataFrame(rows, columns=["ticker", "close", "volume", "dvol_usd"])
    if df.empty:
        return df

    df = df[df["dvol_usd"] >= MIN_DVOL_USD].sort_values("dvol_usd", ascending=False)
    return df.head(MAX_STOCKS).reset_index(drop=True)

def dte(exp_str: str) -> int:
    exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (exp_dt - now).days

def safe_last_price(tk: yf.Ticker) -> float | None:
    try:
        px = tk.fast_info.get("last_price", None)
        if px and px > 0:
            return float(px)
    except Exception:
        pass
    try:
        px = tk.info.get("regularMarketPrice", None)
        if px and px > 0:
            return float(px)
    except Exception:
        pass
    return None

def score_contract(row, px: float, days: int) -> float:
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    mid = (bid + ask) / 2
    if mid <= 0:
        return -1

    spread_pct = (ask - bid) / mid if mid else 999
    oi = float(row.get("openInterest", 0) or 0)
    vol = float(row.get("volume", 0) or 0)
    strike = float(row.get("strike", 0) or 0)

    moneyness = abs(strike - px) / px if px else 999
    spread_component = 1 / (1 + spread_pct * 20)
    oi_component = min(oi, 5000) / 5000
    vol_component = min(vol, 2000) / 2000
    atm_component = 1 / (1 + moneyness * 25)
    dte_component = 1 / (1 + abs(days - 28) / 10)

    return (
        spread_component * 5 +
        oi_component * 3 +
        vol_component * 2 +
        atm_component * 4 +
        dte_component * 2
    )

def to_html_table(df: pd.DataFrame) -> str:
    styles = """
    <style>
      body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial; color:#111; }
      .meta { margin: 0 0 10px 0; color:#444; font-size: 13px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #e5e5e5; padding: 8px 10px; font-size: 13px; }
      th { background: #f6f6f6; text-align: left; }
      tr:nth-child(even) { background: #fcfcfc; }
      .num { text-align: right; font-variant-numeric: tabular-nums; }
      .badge { display:inline-block; padding:2px 8px; border-radius: 999px; font-size: 12px; background:#eef2ff; }
      .warn { background:#fff7ed; }
      .ok { background:#ecfdf5; }
      .small { font-size: 12px; color:#555; }
    </style>
    """

    fmt = df.copy()
    for c in ["strike", "mid", "spread_pct", "score", "dvol_usd"]:
        if c in fmt.columns:
            fmt[c] = pd.to_numeric(fmt[c], errors="coerce")

    if "spread_pct" in fmt.columns:
        fmt["spread_pct"] = (fmt["spread_pct"] * 100).round(2).astype(str) + "%"

    if "mid" in fmt.columns:
        fmt["mid"] = fmt["mid"].round(2)

    if "score" in fmt.columns:
        fmt["score"] = fmt["score"].round(2)

    if "dvol_usd" in fmt.columns:
        fmt["dvol_usd"] = (fmt["dvol_usd"] / 1e6).round(1).astype(str) + "M"

    cols = ["ticker", "side", "exp", "dte", "strike", "mid", "spread_pct", "OI", "vol", "score", "dvol_usd"]
    cols = [c for c in cols if c in fmt.columns]

    def td_class(col):
        return "num" if col in {"dte", "strike", "mid", "OI", "vol", "score", "spread_pct", "dvol_usd"} else ""

    header = "<tr>" + "".join([f"<th>{c.upper()}</th>" for c in cols]) + "</tr>"
    rows = []
    for _, r in fmt[cols].iterrows():
        rows.append("<tr>" + "".join([f"<td class='{td_class(c)}'>{r[c]}</td>" for c in cols]) + "</tr>")

    return styles + f"<table>{header}{''.join(rows)}</table>"

# -----------------------------
# Main
# -----------------------------
def main():
    smtp_cfg = parse_smtp_env()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()

    # 1) Build US ticker universe
    symbols = fetch_us_symbols()

    # 2) Pick liquid subset (makes it feasible on Actions)
    liquid = pick_liquid_tickers(symbols)
    if liquid.empty:
        html = f"""
        <html><body>
          <h2>Daily Options Shortlist (US Universe)</h2>
          <p style="color:#444;font-size:13px;">Run: {run_ts}</p>
          <p>Could not build a liquid ticker list today (data source issue).</p>
        </body></html>
        """
        send_email_smtp(smtp_cfg, f"Options Shortlist (US) — {datetime.now().strftime('%Y-%m-%d')}", html)
        print("Email sent (no liquid list).")
        return

    tickers = liquid["ticker"].tolist()
    dvol_map = dict(zip(liquid["ticker"], liquid["dvol_usd"]))

    all_rows = []
    errors = 0

    # 3) Scan options chains
    for t in tickers:
        if errors >= MAX_TICKER_ERRORS:
            break

        try:
            tk = yf.Ticker(t)
            px = safe_last_price(tk)
            if not px:
                continue

            expirations = list(tk.options)  # empty if no options
            if not expirations:
                continue

            for exp in expirations:
                days = dte(exp)
                if days < MIN_DTE or days > MAX_DTE:
                    continue

                try:
                    chain = tk.option_chain(exp)
                except Exception:
                    continue

                sides = []
                if not PUTS_ONLY:
                    sides.append(("call", chain.calls))
                if not CALLS_ONLY:
                    sides.append(("put", chain.puts))

                for side, df in sides:
                    if df is None or df.empty:
                        continue

                    df = df.copy()
                    df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
                    df["spread_pct"] = (df["ask"].fillna(0) - df["bid"].fillna(0)) / df["mid"].replace(0, math.nan)
                    df["dte"] = days
                    df["ticker"] = t
                    df["side"] = side
                    df["exp"] = exp

                    # Tradability filters
                    df = df[
                        (df["openInterest"].fillna(0) >= MIN_OI) &
                        (df["volume"].fillna(0) >= MIN_OPT_VOL) &
                        (df["mid"] > 0) &
                        (df["spread_pct"].fillna(999) <= MAX_SPREAD_PCT)
                    ].dropna(subset=["spread_pct"])

                    if df.empty:
                        continue

                    df["score"] = df.apply(lambda r: score_contract(r, px, days), axis=1)

                    top = df.sort_values("score", ascending=False).head(TOP_PER_TICKER)
                    for _, r in top.iterrows():
                        all_rows.append({
                            "ticker": t,
                            "side": side,
                            "exp": exp,
                            "dte": int(days),
                            "strike": float(r["strike"]),
                            "mid": float(r["mid"]),
                            "spread_pct": float(r["spread_pct"]),
                            "OI": int(r["openInterest"]),
                            "vol": int(r["volume"]),
                            "score": float(r["score"]),
                            "dvol_usd": float(dvol_map.get(t, 0.0)),
                        })

        except Exception:
            errors += 1

        if SLEEP_BETWEEN_TICKERS > 0:
            time.sleep(SLEEP_BETWEEN_TICKERS)

    out = pd.DataFrame(all_rows)
    out = out.sort_values("score", ascending=False).head(MAX_TOTAL_ROWS) if not out.empty else out

    title = "Daily Options Shortlist (US Universe)"
    subtitle = f"""
      <p class='meta'>
        <span class='badge ok'>US universe</span>
        <span class='badge'>liquidity-ranked</span>
        <span class='badge warn'>not financial advice</span><br/>
        Run: {run_ts}<br/>
        Universe: NASDAQ/NYSE/AMEX symbols → top {len(tickers)} by $ volume (≥ ${MIN_DVOL_USD:,.0f}/day)
      </p>
    """

    if out.empty:
        html = f"""
        <html><body>
          <h2>{title}</h2>
          {subtitle}
          <p>No contracts matched your filters today.</p>
          <p class="meta small">
            Filters: {MIN_DTE}-{MAX_DTE} DTE, OI≥{MIN_OI}, opt vol≥{MIN_OPT_VOL}, spread≤{int(MAX_SPREAD_PCT*100)}% of mid.
          </p>
        </body></html>
        """
    else:
        html = f"""
        <html><body>
          <h2>{title}</h2>
          {subtitle}
          <p class='meta small'>
            Filters: {MIN_DTE}-{MAX_DTE} DTE, OI≥{MIN_OI}, opt vol≥{MIN_OPT_VOL}, spread≤{int(MAX_SPREAD_PCT*100)}% of mid.
            Score ranks tradability (liquidity + fill quality + near-ATM + DTE preference).
          </p>
          {to_html_table(out)}
        </body></html>
        """

    subject = f"Options Shortlist (US) — {datetime.now().strftime('%Y-%m-%d')}"
    send_email_smtp(smtp_cfg, subject, html)
    print(f"Email sent OK. Rows={len(out)}  Tickers_scanned={len(tickers)}  Errors={errors}")

if __name__ == "__main__":
    main()