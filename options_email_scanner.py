#!/usr/bin/env python3
import os, json, math, time
from datetime import datetime, timezone
from io import StringIO

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
MAX_STOCKS      = int(os.getenv("MAX_STOCKS", "800"))          # how many tickers to scan after liquidity ranking
MIN_DVOL_USD    = float(os.getenv("MIN_DVOL_USD", "5e7"))      # min $ volume (e.g., 50,000,000)

# If you want calls-only or puts-only
CALLS_ONLY      = os.getenv("CALLS_ONLY", "0") == "1"
PUTS_ONLY       = os.getenv("PUTS_ONLY", "0") == "1"

# Performance / reliability knobs
SLEEP_BETWEEN_TICKERS = float(os.getenv("SLEEP_BETWEEN_TICKERS", "0.0"))  # seconds
MAX_TICKER_ERRORS     = int(os.getenv("MAX_TICKER_ERRORS", "50"))         # stop if too many failures

# Official symbol list sources (US)
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# -----------------------------
# Helpers
# -----------------------------
def parse_gmail_secret() -> dict:
    raw = os.environ.get("GMAIL", "").strip()
    if not raw:
        raise RuntimeError("Missing env var GMAIL (GitHub Secret).")
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError("GMAIL secret is not valid JSON.") from e
    for k in ("user", "app_password", "to"):
        if k not in cfg or not str(cfg[k]).strip():
            raise RuntimeError(f"GMAIL secret missing '{k}'.")
    return cfg

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
    for c in ["strike","mid","spread_pct","score","dvol_usd"]:
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

    cols = ["ticker","side","exp","dte","strike","mid","spread_pct","OI","vol","score","dvol_usd"]
    cols = [c for c in cols if c in fmt.columns]

    def td_class(col):
        return "num" if col in {"dte","strike","mid","OI","vol","score","spread_pct","dvol_usd"} else ""

    header = "<tr>" + "".join([f"<th>{c.upper()}</th>" for c in cols]) + "</tr>"
    rows = []
    for _, r in fmt[cols].iterrows():
        rows.append("<tr>" + "".join([f"<td class='{td_class(c)}'>{r[c]}</td>" for c in cols]) + "</tr>")

    return styles + f"<table>{header}{''.join(rows)}</table>"

def send_email(gmail_cfg: dict, subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["From"] = gmail_cfg["user"]
    msg["To"] = gmail_cfg["to"]
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_cfg["user"], gmail_cfg["app_password"])
        server.sendmail(gmail_cfg["user"], [gmail_cfg["to"]], msg.as_string())

def fetch_us_symbols() -> list[str]:
    """
    Pull NASDAQ + NYSE/AMEX symbols from NasdaqTrader official lists.
    Filters out test issues and non-common-stock types where possible.
    """
    def _download_text(url: str) -> str:
        # simple urllib to avoid extra deps
        from urllib.request import urlopen, Request
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    nasdaq_txt = _download_text(NASDAQ_LISTED_URL)
    other_txt  = _download_text(OTHER_LISTED_URL)

    # nasdaqlisted.txt is pipe-delimited with header line; ends with "File Creation Time"
    nasdaq_lines = [ln for ln in nasdaq_txt.splitlines() if ln and "File Creation Time" not in ln]
    nasdaq_df = pd.read_csv(StringIO("\n".join(nasdaq_lines)), sep="|")
    # Basic filters
    if "Test Issue" in nasdaq_df.columns:
        nasdaq_df = nasdaq_df[nasdaq_df["Test Issue"] == "N"]
    if "ETF" in nasdaq_df.columns:
        # keep ETFs too (SPY/QQQ etc are options-heavy)
        pass
    nasdaq_syms = nasdaq_df["Symbol"].astype(str).tolist()

    # otherlisted.txt: pipe-delimited; includes NYSE/AMEX + others; has "Test Issue" col
    other_lines = [ln for ln in other_txt.splitlines() if ln and "File Creation Time" not in ln]
    other_df = pd.read_csv(StringIO("\n".join(other_lines)), sep="|")
    if "Test Issue" in other_df.columns:
        other_df = other_df[other_df["Test Issue"] == "N"]
    # Exclude "NextShares" etc by filtering out symbols with weird chars
    other_syms = other_df["ACT Symbol"].astype(str).tolist()

    # Normalize: yfinance uses '-' instead of '.' for class shares (BRK.B -> BRK-B)
    def normalize(sym: str) -> str:
        sym = sym.strip()
        sym = sym.replace(".", "-")
        return sym

    syms = sorted({normalize(s) for s in (nasdaq_syms + other_syms) if s and s.isascii()})
    # Remove some known problematic symbols
    syms = [s for s in syms if s not in {"", "N/A"} and len(s) <= 6]
    return syms

def pick_liquid_tickers(symbols: list[str]) -> pd.DataFrame:
    """
    Batch-download last close + volume and compute $ volume, then select top.
    This step makes the full scan feasible.
    """
    rows = []
    chunk = 200

    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i+chunk]
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

        # yfinance returns:
        # - for multi tickers: columns are MultiIndex (Field, Ticker) or (Ticker, Field) depending
        # We'll handle both.
        if isinstance(data.columns, pd.MultiIndex):
            # Try to standardize to [ticker][field]
            # Case A: first level is fields
            if "Close" in data.columns.get_level_values(0):
                # columns: (Field, Ticker)
                for t in batch:
                    try:
                        close = float(data["Close"][t].dropna().iloc[-1])
                        vol   = float(data["Volume"][t].dropna().iloc[-1])
                        dvol  = close * vol
                        rows.append((t, close, vol, dvol))
                    except Exception:
                        pass
            else:
                # columns: (Ticker, Field)
                for t in batch:
                    try:
                        close = float(data[t]["Close"].dropna().iloc[-1])
                        vol   = float(data[t]["Volume"].dropna().iloc[-1])
                        dvol  = close * vol
                        rows.append((t, close, vol, dvol))
                    except Exception:
                        pass
        else:
            # single ticker fallback (unlikely in batch mode)
            try:
                close = float(data["Close"].dropna().iloc[-1])
                vol   = float(data["Volume"].dropna().iloc[-1])
                dvol  = close * vol
                rows.append((batch[0], close, vol, dvol))
            except Exception:
                pass

    df = pd.DataFrame(rows, columns=["ticker", "close", "volume", "dvol_usd"])
    if df.empty:
        return df

    df = df[df["dvol_usd"] >= MIN_DVOL_USD].sort_values("dvol_usd", ascending=False)
    return df.head(MAX_STOCKS).reset_index(drop=True)

# -----------------------------
# Main scan
# -----------------------------
def main():
    gmail_cfg = parse_gmail_secret()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()

    # 1) Build universe
    symbols = fetch_us_symbols()

    # 2) Pick liquid subset
    liquid = pick_liquid_tickers(symbols)
    if liquid.empty:
        html = f"""
        <html><body>
          <h2>Daily Options Shortlist (US Universe)</h2>
          <p style="color:#444;font-size:13px;">Run: {run_ts}</p>
          <p>Could not build a liquid ticker list today (data source issue). Try again later.</p>
        </body></html>
        """
        send_email(gmail_cfg, f"Options Shortlist — {datetime.now().strftime('%Y-%m-%d')}", html)
        print("Email sent (no liquid list).")
        return

    tickers = liquid["ticker"].tolist()
    dvol_map = dict(zip(liquid["ticker"], liquid["dvol_usd"]))

    all_rows = []
    errors = 0

    # 3) Options scan
    for idx, t in enumerate(tickers, start=1):
        if errors >= MAX_TICKER_ERRORS:
            break

        try:
            tk = yf.Ticker(t)
            px = safe_last_price(tk)
            if not px:
                continue

            expirations = list(tk.options)  # may be empty if no options
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
        Universe: official NASDAQ/NYSE/AMEX symbols → top {len(tickers)} by $ volume (≥ ${MIN_DVOL_USD:,.0f}/day)
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
        table_html = to_html_table(out)
        html = f"""
        <html><body>
          <h2>{title}</h2>
          {subtitle}
          <p class='meta small'>
            Filters: {MIN_DTE}-{MAX_DTE} DTE, OI≥{MIN_OI}, opt vol≥{MIN_OPT_VOL}, spread≤{int(MAX_SPREAD_PCT*100)}% of mid.
            Score ranks for tradability (liquidity + fill quality + near-ATM + DTE preference).
          </p>
          {table_html}
        </body></html>
        """

    subject = f"Options Shortlist (US) — {datetime.now().strftime('%Y-%m-%d')}"
    send_email(gmail_cfg, subject, html)
    print(f"Email sent OK. Rows={len(out)}  Tickers_scanned={len(tickers)}  Errors={errors}")

if __name__ == "__main__":
    main()