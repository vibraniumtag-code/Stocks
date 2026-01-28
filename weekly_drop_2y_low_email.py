#!/usr/bin/env python3
"""
weekly_drop_2y_low_email.py

Screen:
- Weekly drop >= DROP_PCT (default 15%) over last ~5 trading days
- Market cap >= MIN_MKTCAP (default $1B)
- Current close is near 2-year low: close <= (2y_low * (1 + NEAR_LOW_PCT))
  default NEAR_LOW_PCT = 0.05 (within 5% of the 2Y low)

Universe:
- By default reads tickers from docs/universe.txt (one ticker per line)
- Or pass --tickers "AAPL,MSFT,TSLA"
- Or pass --universe-file path

Email (SMTP env vars):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO
Optional:
  EMAIL_TO can be comma-separated
  EMAIL_MODE=always | nonempty  (default nonempty: only send if results found)

Outputs:
- saves CSV results if --save-csv provided
- saves HTML email body if --save-html provided
"""

import os
import re
import ssl
import smtplib
import argparse
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Missing dependency yfinance. Install with: pip install yfinance") from e


# -----------------------------
# Helpers
# -----------------------------
def read_universe(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if not t or t.startswith("#"):
                continue
            # allow "AAPL, MSFT" style too
            parts = re.split(r"[,\s]+", t)
            tickers.extend([p for p in parts if p])
    # dedupe while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def pct(x: float) -> str:
    return f"{x:.2f}%"


def money_billions(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x/1e9:,.1f}B"


def fmt_price(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.2f}"


def safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        v = float(v)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def build_email_html(df: pd.DataFrame, asof: str, drop_pct: float, near_low_pct: float, min_mktcap: float) -> str:
    # Email-safe, inline styles, mobile-friendly scroll table
    count = len(df)
    headline = f"📉 Weekly Drop Screener — {count} match(es)"
    subtitle = (
        f"As of {asof} • Drop ≥ {drop_pct:.0f}% (5 trading days) • "
        f"Market Cap ≥ ${min_mktcap/1e9:.0f}B • Near 2Y Low (within {near_low_pct*100:.0f}%)"
    )

    rows_html = ""
    if count == 0:
        rows_html = """
        <tr>
          <td style="padding:14px;border-top:1px solid rgba(255,255,255,.10);color:rgba(255,255,255,.85);">
            ✅ No matches today. (This is good — fewer falling knives.)
          </td>
        </tr>
        """
        table_html = f"""
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
               style="border-collapse:collapse;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.10);border-radius:12px;overflow:hidden;">
          {rows_html}
        </table>
        """
    else:
        # Build table header + rows
        def badge(text: str, bg: str) -> str:
            return f"""
            <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{bg};
                         color:white;font-size:12px;font-weight:700;letter-spacing:.2px;">
              {text}
            </span>
            """

        for _, r in df.iterrows():
            d5 = safe_float(r.get("drop_5d_pct"))
            dist = safe_float(r.get("dist_to_2y_low_pct"))
            # quick badge: deeper drops show redder
            if d5 is None:
                d_badge = badge("—", "rgba(148,163,184,.25)")
            elif d5 <= -25:
                d_badge = badge(pct(d5), "rgba(239,68,68,.85)")
            elif d5 <= -15:
                d_badge = badge(pct(d5), "rgba(245,158,11,.85)")
            else:
                d_badge = badge(pct(d5), "rgba(34,197,94,.75)")

            if dist is None:
                low_badge = badge("—", "rgba(148,163,184,.25)")
            elif dist <= 1.0:
                low_badge = badge(f"{dist:.2f}%", "rgba(239,68,68,.85)")
            elif dist <= 5.0:
                low_badge = badge(f"{dist:.2f}%", "rgba(245,158,11,.85)")
            else:
                low_badge = badge(f"{dist:.2f}%", "rgba(34,197,94,.75)")

            rows_html += f"""
            <tr>
              <td style="padding:12px 12px;border-top:1px solid rgba(255,255,255,.08);white-space:nowrap;">
                <div style="font-weight:800;color:white;font-size:14px;">{r.get('ticker','—')}</div>
                <div style="color:rgba(255,255,255,.60);font-size:12px;max-width:340px;overflow:hidden;text-overflow:ellipsis;">
                  {r.get('name','—')}
                </div>
              </td>
              <td style="padding:12px 12px;border-top:1px solid rgba(255,255,255,.08);white-space:nowrap;color:rgba(255,255,255,.90);text-align:right;">
                {money_billions(safe_float(r.get("market_cap")))}
              </td>
              <td style="padding:12px 12px;border-top:1px solid rgba(255,255,255,.08);white-space:nowrap;text-align:right;color:rgba(255,255,255,.90);">
                {fmt_price(safe_float(r.get("close")))}
              </td>
              <td style="padding:12px 12px;border-top:1px solid rgba(255,255,255,.08);white-space:nowrap;text-align:right;">
                {d_badge}
              </td>
              <td style="padding:12px 12px;border-top:1px solid rgba(255,255,255,.08);white-space:nowrap;text-align:right;color:rgba(255,255,255,.90);">
                {fmt_price(safe_float(r.get("low_2y")))}
              </td>
              <td style="padding:12px 12px;border-top:1px solid rgba(255,255,255,.08);white-space:nowrap;text-align:right;">
                {low_badge}
              </td>
            </tr>
            """

        table_html = f"""
        <div style="overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,.10);">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
                 style="border-collapse:collapse;min-width:780px;background:rgba(255,255,255,.03);">
            <thead>
              <tr style="background:rgba(255,255,255,.06);">
                <th align="left"  style="padding:12px 12px;color:rgba(255,255,255,.85);font-size:12px;letter-spacing:.3px;">Ticker</th>
                <th align="right" style="padding:12px 12px;color:rgba(255,255,255,.85);font-size:12px;letter-spacing:.3px;">Mkt Cap</th>
                <th align="right" style="padding:12px 12px;color:rgba(255,255,255,.85);font-size:12px;letter-spacing:.3px;">Close</th>
                <th align="right" style="padding:12px 12px;color:rgba(255,255,255,.85);font-size:12px;letter-spacing:.3px;">5D Drop</th>
                <th align="right" style="padding:12px 12px;color:rgba(255,255,255,.85);font-size:12px;letter-spacing:.3px;">2Y Low</th>
                <th align="right" style="padding:12px 12px;color:rgba(255,255,255,.85);font-size:12px;letter-spacing:.3px;">Dist to Low</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        """

    html = f"""\
<!doctype html>
<html>
  <body style="margin:0;background:#0b1220;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;">
    <div style="max-width:980px;margin:0 auto;padding:22px;">
      <div style="background:#0f172a;border:1px solid rgba(255,255,255,.10);border-radius:16px;overflow:hidden;box-shadow:0 14px 50px rgba(0,0,0,.45);">
        <div style="padding:18px 18px 12px 18px;border-bottom:1px solid rgba(255,255,255,.08);
                    background:linear-gradient(135deg, rgba(37,99,235,.22), rgba(15,23,42,.0));">
          <div style="font-size:18px;font-weight:900;color:white;letter-spacing:.2px;">{headline}</div>
          <div style="margin-top:6px;color:rgba(255,255,255,.70);font-size:12.5px;line-height:1.35;">{subtitle}</div>
        </div>

        <div style="padding:14px 18px 18px 18px;">
          {table_html}

          <div style="margin-top:14px;padding:12px 12px;border-radius:12px;background:rgba(255,255,255,.04);
                      border:1px solid rgba(255,255,255,.08);color:rgba(255,255,255,.70);font-size:12.5px;line-height:1.45;">
            🧠 <b style="color:rgba(255,255,255,.88);">Notes:</b>
            Weekly drop uses last 5 trading closes. “Near 2Y Low” = close within the configured % above the 2-year low.
            Always sanity-check liquidity, earnings/news, and spreads before touching anything that’s falling fast.
          </div>

          <div style="margin-top:12px;color:rgba(255,255,255,.45);font-size:11.5px;">
            Generated by GitHub Actions • {asof}
          </div>
        </div>
      </div>
    </div>
  </body>
</html>
"""
    return html


def send_email_html(subject: str, html_body: str) -> None:
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587").strip() or "587")
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASS", "").strip()
    to_raw = os.getenv("EMAIL_TO", "").strip()

    if not all([host, port, user, password, to_raw]):
        missing = [k for k in ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "EMAIL_TO"] if not os.getenv(k)]
        raise SystemExit(f"Missing SMTP env vars: {missing}")

    to_list = [x.strip() for x in re.split(r"[;,]", to_raw) if x.strip()]

    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to_list)

    # HTML only (same pattern you used in other scripts)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.ehlo()
        server.starttls(context=context)
        server.login(user, password)
        server.sendmail(user, to_list, msg.as_string())


# -----------------------------
# Core screening logic
# -----------------------------
def fetch_metrics(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """
    Returns a DataFrame:
      ticker, close, close_5d_ago, drop_5d_pct, low_2y, dist_to_2y_low_pct, market_cap, name
    """
    rows: List[Dict[str, Any]] = []

    # Pull historical in one call (faster) — yfinance supports multi-ticker download
    hist = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    # When multi ticker, columns are a MultiIndex; when single ticker, it's flat.
    is_multi = isinstance(hist.columns, pd.MultiIndex)

    for t in tickers:
        try:
            if is_multi:
                if t not in hist.columns.get_level_values(0):
                    continue
                h = hist[t].dropna()
                # Expect columns: Open High Low Close Adj Close Volume
                close_series = h.get("Close")
                low_series = h.get("Low")
            else:
                # single ticker case
                h = hist.dropna()
                close_series = h.get("Close")
                low_series = h.get("Low")

            if close_series is None or low_series is None or len(close_series) < 10:
                continue

            close = float(close_series.iloc[-1])
            # "last week" approx: 5 trading days
            if len(close_series) < 6:
                continue
            close_5d_ago = float(close_series.iloc[-6])
            drop_5d_pct = (close / close_5d_ago - 1.0) * 100.0

            low_2y = float(low_series.min())
            dist_to_2y_low_pct = ((close / low_2y) - 1.0) * 100.0

            # Market cap + name via fast_info / info (fast_info preferred)
            market_cap = None
            name = "—"
            try:
                yt = yf.Ticker(t)
                fi = getattr(yt, "fast_info", None)
                if fi and isinstance(fi, dict):
                    market_cap = fi.get("market_cap") or fi.get("marketCap")
                # fallback
                if market_cap is None:
                    info = yt.get_info()
                    market_cap = info.get("marketCap")
                    name = info.get("shortName") or info.get("longName") or name
                else:
                    # try to get name cheaply too
                    info = yt.get_info()
                    name = info.get("shortName") or info.get("longName") or name
            except Exception:
                pass

            rows.append(
                {
                    "ticker": t,
                    "name": name,
                    "market_cap": safe_float(market_cap),
                    "close": close,
                    "close_5d_ago": close_5d_ago,
                    "drop_5d_pct": drop_5d_pct,
                    "low_2y": low_2y,
                    "dist_to_2y_low_pct": dist_to_2y_low_pct,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Clean + sort
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["drop_5d_pct"] = pd.to_numeric(df["drop_5d_pct"], errors="coerce")
    df["dist_to_2y_low_pct"] = pd.to_numeric(df["dist_to_2y_low_pct"], errors="coerce")
    df = df.sort_values(["drop_5d_pct", "dist_to_2y_low_pct"], ascending=[True, True]).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="", help='Comma list, e.g. "AAPL,MSFT,TSLA"')
    ap.add_argument("--universe-file", default="docs/universe.txt", help="One ticker per line")
    ap.add_argument("--drop-pct", type=float, default=float(os.getenv("DROP_PCT", "15")))
    ap.add_argument("--near-low-pct", type=float, default=float(os.getenv("NEAR_LOW_PCT", "0.05")))
    ap.add_argument("--min-mktcap", type=float, default=float(os.getenv("MIN_MKTCAP", str(1e9))))
    ap.add_argument("--top", type=int, default=int(os.getenv("TOP_N", "50")))
    ap.add_argument("--save-csv", default="docs/weekly_drop_2y_low.csv")
    ap.add_argument("--save-html", default="docs/weekly_drop_2y_low_email.html")
    ap.add_argument("--send-email", type=int, default=int(os.getenv("SEND_EMAIL", "1")))
    args = ap.parse_args()

    # Build universe
    tickers: List[str] = []
    if args.tickers.strip():
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = read_universe(args.universe_file)

    if not tickers:
        raise SystemExit(
            "No tickers provided. Either pass --tickers 'AAPL,MSFT' or create docs/universe.txt"
        )

    # Pull metrics
    df = fetch_metrics(tickers, period="2y")
    if df.empty:
        filtered = df
    else:
        # Apply filters
        # drop pct is negative for drops (e.g. -18%), so require <= -drop_pct
        filtered = df[
            (df["drop_5d_pct"] <= (-abs(args.drop_pct)))
            & (df["market_cap"].fillna(0) >= args.min_mktcap)
            & (df["dist_to_2y_low_pct"].fillna(1e9) <= (args.near_low_pct * 100.0))
        ].copy()

        filtered = filtered.head(max(1, args.top))

    # Save outputs
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        filtered.to_csv(args.save_csv, index=False)

    asof = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_body = build_email_html(
        filtered,
        asof=asof,
        drop_pct=args.drop_pct,
        near_low_pct=args.near_low_pct,
        min_mktcap=args.min_mktcap,
    )

    if args.save_html:
        os.makedirs(os.path.dirname(args.save_html), exist_ok=True)
        with open(args.save_html, "w", encoding="utf-8") as f:
            f.write(html_body)

    # Send email?
    mode = os.getenv("EMAIL_MODE", "nonempty").strip().lower()
    should_send = (mode == "always") or (mode != "always" and len(filtered) > 0)

    subject = f"📉 Weekly Drop Screener ({len(filtered)} match{'es' if len(filtered)!=1 else ''}) — {asof}"
    if args.send_email and should_send:
        send_email_html(subject, html_body)
        print(f"Sent email: {subject}")
    else:
        print(f"Email skipped. mode={mode}, matches={len(filtered)}, send_email={args.send_email}")

    print("Done.")


if __name__ == "__main__":
    main()