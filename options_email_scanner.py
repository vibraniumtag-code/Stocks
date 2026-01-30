import os, json, math
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# -----------------------------
# Config (edit as you like)
# -----------------------------
TICKERS = ["SPY","QQQ","AAPL","MSFT","NVDA","TSLA","AMD","META","AMZN","GOOGL"]
MIN_DTE = 14
MAX_DTE = 45
MIN_OI = 300
MIN_OPT_VOL = 50
MAX_SPREAD_PCT = 0.06     # 6% of mid
TOP_PER_TICKER = 3
MAX_TOTAL_ROWS = 30

# If you want calls-only or puts-only, set one of these True.
CALLS_ONLY = False
PUTS_ONLY = False

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
    for k in ("user","app_password","to"):
        if k not in cfg or not str(cfg[k]).strip():
            raise RuntimeError(f"GMAIL secret missing '{k}'.")
    return cfg

def dte(exp_str: str) -> int:
    # yfinance exp format: YYYY-MM-DD
    exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (exp_dt - now).days

def safe_last_price(tk: yf.Ticker) -> float | None:
    # try fast_info first, then info fallback
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
    A simple 'tradability' score:
    - tighter spreads better
    - higher OI/volume better
    - closer to ATM better
    - DTE closer to ~28 days slightly better
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
    # shape the components (all ~0..1-ish)
    spread_component = 1 / (1 + spread_pct * 20)            # penalize wide spreads strongly
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
    # Pretty-ish HTML table with minimal inline CSS (email-friendly)
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
    </style>
    """

    fmt = df.copy()
    # numeric formatting
    for c in ["strike","mid","spread_pct","score"]:
        if c in fmt.columns:
            fmt[c] = pd.to_numeric(fmt[c], errors="coerce")

    if "spread_pct" in fmt.columns:
        fmt["spread_pct"] = (fmt["spread_pct"] * 100).round(2).astype(str) + "%"

    if "mid" in fmt.columns:
        fmt["mid"] = fmt["mid"].round(2)

    if "score" in fmt.columns:
        fmt["score"] = fmt["score"].round(2)

    # build HTML
    cols = ["ticker","side","exp","dte","strike","mid","spread_pct","OI","vol","score"]
    cols = [c for c in cols if c in fmt.columns]

    # add numeric alignment
    def td_class(col): 
        return "num" if col in {"dte","strike","mid","OI","vol","score","spread_pct"} else ""

    header = "<tr>" + "".join([f"<th>{c.upper()}</th>" for c in cols]) + "</tr>"
    rows = []
    for _, r in fmt[cols].iterrows():
        rows.append(
            "<tr>" + "".join([f"<td class='{td_class(c)}'>{r[c]}</td>" for c in cols]) + "</tr>"
        )

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

# -----------------------------
# Main scan
# -----------------------------
def main():
    gmail_cfg = parse_gmail_secret()

    all_rows = []
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()

    for t in TICKERS:
        tk = yf.Ticker(t)
        px = safe_last_price(tk)
        if not px:
            continue

        try:
            expirations = list(tk.options)
        except Exception:
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

                # Filters: tradability
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
                    })

    out = pd.DataFrame(all_rows)
    out = out.sort_values("score", ascending=False).head(MAX_TOTAL_ROWS) if not out.empty else out

    title = "Daily Options Shortlist (Screen Only)"
    subtitle = f"<p class='meta'><span class='badge'>free data</span> <span class='badge warn'>not financial advice</span><br/>Run: {run_ts}</p>"

    if out.empty:
        html = f"""
        <html><body>
          <h2>{title}</h2>
          {subtitle}
          <p>No contracts matched your filters today.</p>
        </body></html>
        """
    else:
        table_html = to_html_table(out)
        html = f"""
        <html><body>
          <h2>{title}</h2>
          {subtitle}
          <p class='meta'>
            Filters: {MIN_DTE}-{MAX_DTE} DTE, OI≥{MIN_OI}, opt vol≥{MIN_OPT_VOL}, spread≤{int(MAX_SPREAD_PCT*100)}% of mid.
          </p>
          {table_html}
          <p class='meta'>
            Tip: This ranks for tradability (liquidity + fill quality) and “reasonable” expirations. It does not predict direction.
          </p>
        </body></html>
        """

    subject = f"Options Shortlist — {datetime.now().strftime('%Y-%m-%d')}"
    send_email(gmail_cfg, subject, html)
    print("Email sent OK.")

if __name__ == "__main__":
    main()