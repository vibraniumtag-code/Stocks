#!/usr/bin/env python3
"""
options_email_scanner.py

US options shortlist scanner (yfinance) + Portfolio-Manager-style email.

✅ Email env vars (same as your portfolio_manager scripts):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO
  Optional: SMTP_TLS=ssl|starttls  (default auto: 465->ssl else starttls)

✅ Universe:
  - Official NASDAQ/NYSE/AMEX symbol lists (NasdaqTrader)
  - Rank by recent $ volume and scan top MAX_STOCKS (configurable)

✅ Filters + scoring:
  - Pass 1: strict settings from env/defaults
  - If no matches -> Pass 2: relaxed settings automatically (so you don’t get “no matches” often)

✅ “From” display name:
  - Email shows as: Options daily Scanner <SMTP_USER>

Notes:
- Scanning every US option for every ticker is not feasible on Actions with free data.
  This scans the most liquid tickers first (best bang-for-buck).
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


# =============================================================================
# Config (env overrides allowed)
# =============================================================================
# Pass-1 (strict) defaults
MIN_DTE_1         = int(os.getenv("MIN_DTE", "14"))
MAX_DTE_1         = int(os.getenv("MAX_DTE", "45"))
MIN_OI_1          = int(os.getenv("MIN_OI", "300"))
MIN_OPT_VOL_1     = int(os.getenv("MIN_OPT_VOL", "50"))
MAX_SPREAD_PCT_1  = float(os.getenv("MAX_SPREAD_PCT", "0.06"))  # 6% of mid

TOP_PER_TICKER    = int(os.getenv("TOP_PER_TICKER", "3"))
MAX_TOTAL_ROWS    = int(os.getenv("MAX_TOTAL_ROWS", "30"))

# Universe controls
MAX_STOCKS        = int(os.getenv("MAX_STOCKS", "600"))         # liquid tickers to scan
MIN_DVOL_USD      = float(os.getenv("MIN_DVOL_USD", "5e7"))     # 50M/day by default

# Calls-only or puts-only
CALLS_ONLY        = os.getenv("CALLS_ONLY", "0") == "1"
PUTS_ONLY         = os.getenv("PUTS_ONLY", "0") == "1"

# Reliability knobs
SLEEP_BETWEEN_TICKERS = float(os.getenv("SLEEP_BETWEEN_TICKERS", "0.0"))
MAX_TICKER_ERRORS     = int(os.getenv("MAX_TICKER_ERRORS", "80"))

# Email knobs
SMTP_TLS         = os.getenv("SMTP_TLS", "").strip().lower()  # "", "ssl", "starttls"
SMTP_TIMEOUT     = int(os.getenv("SMTP_TIMEOUT", "30"))
FROM_NAME        = os.getenv("FROM_NAME", "Options daily Scanner")

# Symbol list sources
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


# =============================================================================
# Email (same env interface as portfolio_manager)
# =============================================================================
def parse_smtp_env() -> dict:
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
    msg = MIMEMultipart("alternative")

    # Display name (works in most clients; Gmail may still apply account-level name)
    msg["From"] = f'{FROM_NAME} <{cfg["user"]}>'
    msg["To"] = cfg["to"]
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    tls_mode = SMTP_TLS or ("ssl" if int(cfg["port"]) == 465 else "starttls")

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


# =============================================================================
# Universe helpers
# =============================================================================
def _download_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def fetch_us_symbols() -> list[str]:
    """
    Official NASDAQ + NYSE/AMEX lists. Normalizes '.' -> '-' for yfinance.
    Hardened against NaNs / odd rows.
    """
    nasdaq_txt = _download_text(NASDAQ_LISTED_URL)
    other_txt  = _download_text(OTHER_LISTED_URL)

    nasdaq_lines = [ln for ln in nasdaq_txt.splitlines() if ln and "File Creation Time" not in ln]
    other_lines  = [ln for ln in other_txt.splitlines() if ln and "File Creation Time" not in ln]

    nasdaq_df = pd.read_csv(StringIO("\n".join(nasdaq_lines)), sep="|", dtype=str)
    other_df  = pd.read_csv(StringIO("\n".join(other_lines)),  sep="|", dtype=str)

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
        return sym.replace(".", "-")

    out = set()
    for s in nasdaq_syms + other_syms:
        s = normalize(s)
        if not s:
            continue
        # yfinance-friendly sanity checks
        if not s.isascii():
            continue
        if "^" in s or "/" in s or " " in s:
            continue
        # Keep a bit looser length; class shares can be longer
        if len(s) > 12:
            continue
        out.add(s)

    return sorted(out)


def pick_liquid_tickers(symbols: list[str]) -> pd.DataFrame:
    """
    Rank by recent $ volume using yfinance batch download.
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
            # columns can be (Field, Ticker) or (Ticker, Field)
            if "Close" in data.columns.get_level_values(0):
                # (Field, Ticker)
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
                # (Ticker, Field)
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
            # single ticker case
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


# =============================================================================
# Option scan helpers
# =============================================================================
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
    """
    Tradability score:
      - tighter spreads better
      - higher OI/volume better
      - closer to ATM better
      - DTE closer to ~28 better
    """
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


# =============================================================================
# Portfolio-manager-like HTML theme (dark + cards + badges)
# =============================================================================
def build_email_html(title: str, run_ts: str, meta_lines: list[str], table_html: str | None, footer_note: str) -> str:
    meta = "".join([f"<div class='meta-row'>{ln}</div>" for ln in meta_lines])

    table_block = ""
    if table_html:
        table_block = f"""
          <div class="card">
            <div class="card-title">Top contracts</div>
            <div class="table-wrap">
              {table_html}
            </div>
          </div>
        """
    else:
        table_block = f"""
          <div class="card">
            <div class="card-title">No matches</div>
            <div class="empty">No contracts matched the filters today.</div>
          </div>
        """

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <style>
    :root {{
      --bg:#0b1220;
      --card:#0f172a;
      --muted:#94a3b8;
      --text:#e2e8f0;
      --line:#1f2a44;
      --good:#22c55e;
      --warn:#f59e0b;
      --bad:#ef4444;
      --chip:#111c34;
      --shadow: 0 10px 30px rgba(0,0,0,.35);
      --radius: 14px;
      --font: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial;
    }}
    *{{box-sizing:border-box}}
    body {{
      margin:0; padding:24px;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
    }}
    .wrap {{max-width: 980px; margin:0 auto;}}
    .header {{
      display:flex; align-items:flex-start; justify-content:space-between;
      gap:16px; margin-bottom: 14px;
    }}
    .hgroup h1 {{
      margin:0; font-size: 22px; letter-spacing: .2px;
    }}
    .sub {{
      margin-top:6px; color: var(--muted); font-size: 13px; line-height: 1.4;
    }}
    .chips {{display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end;}}
    .chip {{
      background: rgba(255,255,255,.07);
      border: 1px solid rgba(255,255,255,.08);
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      color: var(--text);
      white-space: nowrap;
    }}
    .chip.warn {{border-color: rgba(245,158,11,.35);}}
    .chip.ok {{border-color: rgba(34,197,94,.35);}}
    .card {{
      background: var(--card);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 14px 14px;
      margin: 14px 0;
    }}
    .card-title {{
      font-size: 14px; font-weight: 700; margin: 2px 0 10px 0;
      color: var(--text);
    }}
    .meta-row {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin: 4px 0;
    }}
    .table-wrap {{
      overflow-x:auto;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,.08);
      background: rgba(255,255,255,.03);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 860px;
    }}
    th, td {{
      padding: 10px 10px;
      border-bottom: 1px solid rgba(255,255,255,.08);
      font-size: 13px;
    }}
    th {{
      text-align:left;
      color: #cbd5e1;
      background: rgba(255,255,255,.04);
      position: sticky;
      top: 0;
    }}
    tr:hover td {{
      background: rgba(255,255,255,.03);
    }}
    .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
    .empty {{
      color: var(--muted);
      padding: 8px 2px;
      font-size: 13px;
    }}
    .footer {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 10px;
      line-height: 1.5;
    }}
    .hr {{
      height: 1px; background: rgba(255,255,255,.08); margin: 16px 0;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="hgroup">
        <h1>{title}</h1>
        <div class="sub">Run: {run_ts}</div>
      </div>
      <div class="chips">
        <div class="chip ok">US Universe</div>
        <div class="chip">Liquidity-ranked</div>
        <div class="chip warn">Not financial advice</div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Summary</div>
      {meta}
    </div>

    {table_block}

    <div class="hr"></div>
    <div class="footer">{footer_note}</div>
  </div>
</body>
</html>
"""


def df_to_table_html(df: pd.DataFrame) -> str:
    """
    Render a table matching the dark theme (no inline colors).
    """
    show = df.copy()

    # format
    if "spread_pct" in show.columns:
        show["spread_pct"] = (pd.to_numeric(show["spread_pct"], errors="coerce") * 100).round(2).astype(str) + "%"

    for c in ["mid", "strike", "score"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(2)

    if "dvol_usd" in show.columns:
        show["dvol_usd"] = (pd.to_numeric(show["dvol_usd"], errors="coerce") / 1e6).round(1).astype(str) + "M"

    cols = ["ticker", "side", "exp", "dte", "strike", "mid", "spread_pct", "OI", "vol", "score", "dvol_usd"]
    cols = [c for c in cols if c in show.columns]

    def td_class(col: str) -> str:
        return "num" if col in {"dte", "strike", "mid", "spread_pct", "OI", "vol", "score", "dvol_usd"} else ""

    thead = "<tr>" + "".join([f"<th>{c.upper()}</th>" for c in cols]) + "</tr>"
    body_rows = []
    for _, r in show[cols].iterrows():
        tds = "".join([f"<td class='{td_class(c)}'>{r[c]}</td>" for c in cols])
        body_rows.append(f"<tr>{tds}</tr>")

    return f"<table>{thead}{''.join(body_rows)}</table>"


# =============================================================================
# Scanning logic with auto-relax fallback
# =============================================================================
def scan_options_for_tickers(
    tickers: list[str],
    dvol_map: dict[str, float],
    min_dte: int,
    max_dte: int,
    min_oi: int,
    min_opt_vol: int,
    max_spread_pct: float,
) -> tuple[pd.DataFrame, int]:
    all_rows = []
    errors = 0

    for t in tickers:
        if errors >= MAX_TICKER_ERRORS:
            break

        try:
            tk = yf.Ticker(t)
            px = safe_last_price(tk)
            if not px:
                continue

            expirations = list(tk.options)
            if not expirations:
                continue

            for exp in expirations:
                days = dte(exp)
                if days < min_dte or days > max_dte:
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

                    # tradability filters
                    df = df[
                        (df["openInterest"].fillna(0) >= min_oi) &
                        (df["volume"].fillna(0) >= min_opt_vol) &
                        (df["mid"] > 0) &
                        (df["spread_pct"].fillna(999) <= max_spread_pct)
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
    if not out.empty:
        out = out.sort_values("score", ascending=False).head(MAX_TOTAL_ROWS)

    return out, errors


# =============================================================================
# Main
# =============================================================================
def main():
    smtp_cfg = parse_smtp_env()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    today = datetime.now().strftime("%Y-%m-%d")

    # 1) Universe + liquidity rank
    symbols = fetch_us_symbols()
    liquid = pick_liquid_tickers(symbols)

    if liquid.empty:
        html = build_email_html(
            title="Daily Options Shortlist",
            run_ts=run_ts,
            meta_lines=[
                "Universe build failed (no liquidity list). This is usually a data source hiccup.",
                "Try rerunning later."
            ],
            table_html=None,
            footer_note="Scanner uses free market data. Not financial advice."
        )
        send_email_smtp(smtp_cfg, f"Options Shortlist — {today}", html)
        print("Email sent (no liquidity list).")
        return

    tickers = liquid["ticker"].tolist()
    dvol_map = dict(zip(liquid["ticker"], liquid["dvol_usd"]))

    # 2) Pass 1 (strict)
    out1, errors1 = scan_options_for_tickers(
        tickers=tickers,
        dvol_map=dvol_map,
        min_dte=MIN_DTE_1,
        max_dte=MAX_DTE_1,
        min_oi=MIN_OI_1,
        min_opt_vol=MIN_OPT_VOL_1,
        max_spread_pct=MAX_SPREAD_PCT_1
    )

    used_pass = 1
    out = out1
    errors = errors1

    # 3) If no matches, auto-relax (Pass 2)
    if out.empty:
        # relaxed defaults (can still be overridden with env if you want, but these are safe fallbacks)
        MIN_DTE_2        = int(os.getenv("MIN_DTE_RELAX", "7"))
        MAX_DTE_2        = int(os.getenv("MAX_DTE_RELAX", "60"))
        MIN_OI_2         = int(os.getenv("MIN_OI_RELAX", "100"))
        MIN_OPT_VOL_2    = int(os.getenv("MIN_OPT_VOL_RELAX", "10"))
        MAX_SPREAD_PCT_2 = float(os.getenv("MAX_SPREAD_PCT_RELAX", "0.10"))

        out2, errors2 = scan_options_for_tickers(
            tickers=tickers,
            dvol_map=dvol_map,
            min_dte=MIN_DTE_2,
            max_dte=MAX_DTE_2,
            min_oi=MIN_OI_2,
            min_opt_vol=MIN_OPT_VOL_2,
            max_spread_pct=MAX_SPREAD_PCT_2
        )

        used_pass = 2
        out = out2
        errors = max(errors1, errors2)

        # Keep for meta display
        relax_meta = (MIN_DTE_2, MAX_DTE_2, MIN_OI_2, MIN_OPT_VOL_2, MAX_SPREAD_PCT_2)
    else:
        relax_meta = None

    # 4) Build email
    base_meta = [
        f"Tickers scanned: {len(tickers)} (top by $ volume ≥ ${MIN_DVOL_USD:,.0f}/day, cap {MAX_STOCKS})",
        f"Side: {'CALLS only' if CALLS_ONLY else ('PUTS only' if PUTS_ONLY else 'CALLS + PUTS')}",
        f"Errors (ticker-level): {errors} (stop cap {MAX_TICKER_ERRORS})",
    ]

    if used_pass == 1:
        filt_line = f"Filters (Pass 1): DTE {MIN_DTE_1}-{MAX_DTE_1}, OI ≥ {MIN_OI_1}, Vol ≥ {MIN_OPT_VOL_1}, Spread ≤ {int(MAX_SPREAD_PCT_1*100)}%"
        base_meta.append(filt_line)
    else:
        # show pass 1 + pass 2
        base_meta.append(
            f"Filters (Pass 1): DTE {MIN_DTE_1}-{MAX_DTE_1}, OI ≥ {MIN_OI_1}, Vol ≥ {MIN_OPT_VOL_1}, Spread ≤ {int(MAX_SPREAD_PCT_1*100)}%"
        )
        if relax_meta:
            md2, xd2, oi2, v2, sp2 = relax_meta
            base_meta.append(
                f"Auto-relaxed → Pass 2: DTE {md2}-{xd2}, OI ≥ {oi2}, Vol ≥ {v2}, Spread ≤ {int(sp2*100)}%"
            )

    if out.empty:
        html = build_email_html(
            title="Daily Options Shortlist",
            run_ts=run_ts,
            meta_lines=base_meta + ["Result: 0 matches."],
            table_html=None,
            footer_note="Ranks for tradability (liquidity + fill quality + near-ATM + DTE preference). Not financial advice."
        )
    else:
        table_html = df_to_table_html(out)
        html = build_email_html(
            title="Daily Options Shortlist",
            run_ts=run_ts,
            meta_lines=base_meta + [f"Result: {len(out)} contracts (top {MAX_TOTAL_ROWS})."],
            table_html=table_html,
            footer_note="Ranks for tradability (liquidity + fill quality + near-ATM + DTE preference). Not financial advice."
        )

    # 5) Send
    subject = f"Options Shortlist — {today}"
    send_email_smtp(smtp_cfg, subject, html)
    print(f"Email sent OK. pass={used_pass} rows={len(out)} tickers_scanned={len(tickers)} errors={errors}")


if __name__ == "__main__":
    main()