#!/usr/bin/env python3
"""
weekly_drop_2y_low_email.py  (NOTE: logic is 1Y low; filenames kept for backward-compat)

Scans the *current US stock market* automatically (no universe file needed).

Universe source (Nasdaq Trader symbol directory):
- https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
- https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt

Filters:
- Weekly drop over last ~5 trading days <= -DROP_PCT (default 15)
- Market cap >= MIN_MKTCAP (default $1B)
- Close is near 1-year low:
    dist_to_1y_low_pct <= NEAR_LOW_PCT*100 (default within 5%)

Email (SMTP env vars):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO
Optional:
  EMAIL_TO can be comma-separated / semicolon separated
  EMAIL_MODE=always | nonempty (default nonempty)
  SEND_EMAIL=1/0

Performance knobs:
  STAGE1_CHUNK (default 200)
  STAGE2_CHUNK (default 100)
  MAX_STAGE1   (default 0 => scan all; set 4000 if you hit timeouts)

Outputs (kept as 2y names so your existing YAML upload step keeps working):
  docs/weekly_drop_2y_low.csv
  docs/weekly_drop_2y_low_email.html
"""

import os
import re
import ssl
import smtplib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Missing dependency yfinance. Install with: pip install pandas yfinance") from e


# -----------------------------
# Email
# -----------------------------
def send_email_html(subject: str, html_body: str) -> None:
    host = os.getenv("SMTP_HOST", "").strip()
    port = int((os.getenv("SMTP_PORT", "587") or "587").strip())
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASS", "").strip()
    to_raw = os.getenv("EMAIL_TO", "").strip()

    if not all([host, user, password, to_raw]):
        missing = [k for k in ["SMTP_HOST", "SMTP_USER", "SMTP_PASS", "EMAIL_TO"] if not os.getenv(k)]
        raise SystemExit(f"Missing SMTP env vars: {missing}")

    to_list = [x.strip() for x in re.split(r"[;,]", to_raw) if x.strip()]
    if not to_list:
        raise SystemExit("EMAIL_TO is empty after parsing.")

    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to_list)

    msg.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.ehlo()
        server.starttls(context=context)
        server.login(user, password)
        server.sendmail(user, to_list, msg.as_string())


# -----------------------------
# Formatting helpers
# -----------------------------
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


def fmt_price(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.2f}"


def money_billions(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x/1e9:,.1f}B"


def pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.2f}%"


def build_email_html(df: pd.DataFrame, asof: str, drop_pct: float, near_low_pct: float, min_mktcap: float) -> str:
    count = len(df)
    headline = f"📉 US Market Weekly Drop Screener — {count} match(es)"
    subtitle = (
        f"As of {asof} • Drop ≥ {drop_pct:.0f}% (5 trading days) • "
        f"Market Cap ≥ ${min_mktcap/1e9:.0f}B • Near 1Y Low (within {near_low_pct*100:.0f}%)"
    )

    if count == 0:
        table_html = """
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
               style="border-collapse:collapse;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.10);border-radius:12px;overflow:hidden;">
          <tr>
            <td style="padding:14px;color:rgba(255,255,255,.85);">
              ✅ No matches today.
            </td>
          </tr>
        </table>
        """
    else:
        rows_html = ""
        for _, r in df.iterrows():
            rows_html += f"""
            <tr>
              <td style="padding:12px;border-top:1px solid rgba(255,255,255,.08);white-space:nowrap;">
                <div style="font-weight:900;color:white;">{r.get('ticker','—')}</div>
                <div style="color:rgba(255,255,255,.55);font-size:12px;max-width:360px;overflow:hidden;text-overflow:ellipsis;">
                  {r.get('name','—')}
                </div>
              </td>
              <td style="padding:12px;border-top:1px solid rgba(255,255,255,.08);text-align:right;color:rgba(255,255,255,.9);white-space:nowrap;">
                {money_billions(safe_float(r.get('market_cap')))}
              </td>
              <td style="padding:12px;border-top:1px solid rgba(255,255,255,.08);text-align:right;color:rgba(255,255,255,.9);white-space:nowrap;">
                {fmt_price(safe_float(r.get('close')))}
              </td>
              <td style="padding:12px;border-top:1px solid rgba(255,255,255,.08);text-align:right;color:rgba(255,255,255,.9);white-space:nowrap;">
                {pct(safe_float(r.get('drop_5d_pct')))}
              </td>
              <td style="padding:12px;border-top:1px solid rgba(255,255,255,.08);text-align:right;color:rgba(255,255,255,.9);white-space:nowrap;">
                {fmt_price(safe_float(r.get('low_1y')))}
              </td>
              <td style="padding:12px;border-top:1px solid rgba(255,255,255,.08);text-align:right;color:rgba(255,255,255,.9);white-space:nowrap;">
                {pct(safe_float(r.get('dist_to_1y_low_pct')))}
              </td>
            </tr>
            """

        table_html = f"""
        <div style="overflow-x:auto;border-radius:12px;border:1px solid rgba(255,255,255,.10);">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0"
                 style="border-collapse:collapse;min-width:860px;background:rgba(255,255,255,.03);">
            <thead>
              <tr style="background:rgba(255,255,255,.06);">
                <th align="left"  style="padding:12px;color:rgba(255,255,255,.82);font-size:12px;">Ticker</th>
                <th align="right" style="padding:12px;color:rgba(255,255,255,.82);font-size:12px;">Mkt Cap</th>
                <th align="right" style="padding:12px;color:rgba(255,255,255,.82);font-size:12px;">Close</th>
                <th align="right" style="padding:12px;color:rgba(255,255,255,.82);font-size:12px;">5D Drop</th>
                <th align="right" style="padding:12px;color:rgba(255,255,255,.82);font-size:12px;">1Y Low</th>
                <th align="right" style="padding:12px;color:rgba(255,255,255,.82);font-size:12px;">Dist to Low</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        """

    return f"""<!doctype html>
<html>
  <body style="margin:0;background:#0b1220;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;">
    <div style="max-width:980px;margin:0 auto;padding:22px;">
      <div style="background:#0f172a;border:1px solid rgba(255,255,255,.10);border-radius:16px;overflow:hidden;box-shadow:0 14px 50px rgba(0,0,0,.45);">
        <div style="padding:18px;border-bottom:1px solid rgba(255,255,255,.08);
                    background:linear-gradient(135deg, rgba(239,68,68,.20), rgba(15,23,42,0));">
          <div style="font-size:18px;font-weight:950;color:white;">{headline}</div>
          <div style="margin-top:6px;color:rgba(255,255,255,.70);font-size:12.5px;line-height:1.35;">{subtitle}</div>
        </div>

        <div style="padding:14px 18px 18px 18px;">
          {table_html}

          <div style="margin-top:14px;padding:12px;border-radius:12px;background:rgba(255,255,255,.04);
                      border:1px solid rgba(255,255,255,.08);color:rgba(255,255,255,.70);font-size:12.5px;line-height:1.45;">
            🧠 <b style="color:rgba(255,255,255,.88);">Notes:</b>
            Stage-1 uses last 5 trading closes. Then we only fetch 1Y data + market cap for survivors.
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


# -----------------------------
# Universe: US listings (NasdaqTrader)
# -----------------------------
def load_us_listed_tickers() -> List[str]:
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    def read_pipe_txt(url: str) -> pd.DataFrame:
        df = pd.read_csv(url, sep="|", dtype=str, engine="python")
        first_col = df.columns[0]
        df = df[~df[first_col].astype(str).str.contains("File Creation Time", na=False)]
        return df

    nas = read_pipe_txt(nasdaq_url)
    oth = read_pipe_txt(other_url)

    tickers: List[str] = []

    # Nasdaq listed (Symbol)
    if "Symbol" in nas.columns:
        for _, r in nas.iterrows():
            sym = str(r.get("Symbol") or "").strip().upper()
            if sym in ("", "NAN"):
                continue

            test = str(r.get("Test Issue") or "").strip().upper()
            etf = str(r.get("ETF") or "").strip().upper()

            if test == "Y":
                continue
            # exclude ETFs by default
            if etf == "Y":
                continue

            tickers.append(sym)

    # Other listed (ACT Symbol)
    sym_col = "ACT Symbol" if "ACT Symbol" in oth.columns else ("Symbol" if "Symbol" in oth.columns else None)
    if sym_col:
        for _, r in oth.iterrows():
            sym = str(r.get(sym_col) or "").strip().upper()
            if sym in ("", "NAN"):
                continue

            test = str(r.get("Test Issue") or "").strip().upper()
            etf = str(r.get("ETF") or "").strip().upper()

            if test == "Y":
                continue
            if etf == "Y":
                continue

            tickers.append(sym)

    # Clean for yfinance
    cleaned: List[str] = []
    seen = set()
    for t in tickers:
        if not re.match(r"^[A-Z0-9\.\-]+$", t):
            continue
        if t not in seen:
            seen.add(t)
            cleaned.append(t)

    return cleaned


# -----------------------------
# Market data helpers
# -----------------------------
def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def download_closes_5d_drop(tickers: List[str], chunk_size: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for idx, batch in enumerate(chunked(tickers, chunk_size), start=1):
        print(f"[Stage1] Batch {idx}: {len(batch)} tickers")
        hist = yf.download(
            tickers=batch,
            period="7d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False,
        )
        if hist is None or hist.empty:
            continue

        is_multi = isinstance(hist.columns, pd.MultiIndex)

        for t in batch:
            try:
                if is_multi:
                    if t not in hist.columns.get_level_values(0):
                        continue
                    h = hist[t].dropna()
                    close_series = h.get("Close")
                else:
                    h = hist.dropna()
                    close_series = h.get("Close")

                if close_series is None or len(close_series) < 6:
                    continue

                close = float(close_series.iloc[-1])
                close_5d_ago = float(close_series.iloc[-6])
                drop_5d_pct = (close / close_5d_ago - 1.0) * 100.0

                rows.append(
                    {"ticker": t, "close": close, "close_5d_ago": close_5d_ago, "drop_5d_pct": drop_5d_pct}
                )
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["drop_5d_pct"] = pd.to_numeric(df["drop_5d_pct"], errors="coerce")
    return df


def add_1y_low_metrics(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    tickers = df["ticker"].tolist()
    out_rows: List[Dict[str, Any]] = []

    for idx, batch in enumerate(chunked(tickers, chunk_size), start=1):
        print(f"[Stage2] Batch {idx}: {len(batch)} tickers (1y)")
        hist = yf.download(
            tickers=batch,
            period="1y",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False,
        )
        if hist is None or hist.empty:
            continue

        is_multi = isinstance(hist.columns, pd.MultiIndex)

        for t in batch:
            try:
                if is_multi:
                    if t not in hist.columns.get_level_values(0):
                        continue
                    h = hist[t].dropna()
                    low_series = h.get("Low")
                    close_series = h.get("Close")
                else:
                    h = hist.dropna()
                    low_series = h.get("Low")
                    close_series = h.get("Close")

                if low_series is None or close_series is None or len(close_series) < 10:
                    continue

                close = float(close_series.iloc[-1])
                low_1y = float(low_series.min())
                dist_to_1y_low_pct = ((close / low_1y) - 1.0) * 100.0

                out_rows.append({"ticker": t, "low_1y": low_1y, "dist_to_1y_low_pct": dist_to_1y_low_pct})
            except Exception:
                continue

    m = pd.DataFrame(out_rows)
    if m.empty:
        df["low_1y"] = pd.NA
        df["dist_to_1y_low_pct"] = pd.NA
        return df

    merged = df.merge(m, on="ticker", how="left")
    merged["dist_to_1y_low_pct"] = pd.to_numeric(merged["dist_to_1y_low_pct"], errors="coerce")
    merged["low_1y"] = pd.to_numeric(merged["low_1y"], errors="coerce")
    return merged


def add_market_caps(df: pd.DataFrame) -> pd.DataFrame:
    tickers = df["ticker"].tolist()
    caps: Dict[str, Optional[float]] = {}
    names: Dict[str, str] = {}

    for i, t in enumerate(tickers, start=1):
        try:
            if i % 50 == 0:
                print(f"[Stage3] Market caps: {i}/{len(tickers)}")

            yt = yf.Ticker(t)

            mcap = None
            fi = getattr(yt, "fast_info", None)
            if fi and isinstance(fi, dict):
                mcap = fi.get("market_cap") or fi.get("marketCap")

            info = None
            if mcap is None:
                info = yt.get_info()
                mcap = info.get("marketCap") if info else None

            if info is None:
                try:
                    info = yt.get_info()
                except Exception:
                    info = None

            nm = "—"
            if info:
                nm = info.get("shortName") or info.get("longName") or "—"

            caps[t] = safe_float(mcap)
            names[t] = nm
        except Exception:
            caps[t] = None
            names[t] = "—"

    df["market_cap"] = df["ticker"].map(caps)
    df["name"] = df["ticker"].map(names)
    return df


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-pct", type=float, default=float(os.getenv("DROP_PCT", "15")))
    ap.add_argument("--near-low-pct", type=float, default=float(os.getenv("NEAR_LOW_PCT", "0.05")))
    ap.add_argument("--min-mktcap", type=float, default=float(os.getenv("MIN_MKTCAP", str(1e9))))
    ap.add_argument("--top", type=int, default=int(os.getenv("TOP_N", "50")))
    ap.add_argument("--stage1-chunk", type=int, default=int(os.getenv("STAGE1_CHUNK", "200")))
    ap.add_argument("--stage2-chunk", type=int, default=int(os.getenv("STAGE2_CHUNK", "100")))
    ap.add_argument("--max-stage1", type=int, default=int(os.getenv("MAX_STAGE1", "0")))  # 0 means no cap
    ap.add_argument("--save-csv", default="docs/weekly_drop_2y_low.csv")  # kept name
    ap.add_argument("--save-html", default="docs/weekly_drop_2y_low_email.html")  # kept name
    ap.add_argument("--send-email", type=int, default=int(os.getenv("SEND_EMAIL", "1")))
    args = ap.parse_args()

    print("[Universe] Loading US-listed tickers (NasdaqTrader)…")
    tickers = load_us_listed_tickers()
    print(f"[Universe] Total tickers (stocks, ETFs excluded): {len(tickers)}")

    if args.max_stage1 and args.max_stage1 > 0:
        tickers = tickers[: args.max_stage1]
        print(f"[Universe] MAX_STAGE1 applied: scanning first {len(tickers)} tickers")

    # Stage 1: weekly drop
    df1 = download_closes_5d_drop(tickers, chunk_size=args.stage1_chunk)

    if df1.empty:
        print("[Stage1] No data returned.")
        final = df1
    else:
        survivors = df1[df1["drop_5d_pct"] <= (-abs(args.drop_pct))].copy()
        print(f"[Stage1] Survivors after drop filter: {len(survivors)}")

        if survivors.empty:
            final = survivors
        else:
            # Stage 2: 1Y low metrics
            survivors = add_1y_low_metrics(survivors, chunk_size=args.stage2_chunk)

            survivors = survivors[
                survivors["dist_to_1y_low_pct"].fillna(1e9) <= (args.near_low_pct * 100.0)
            ].copy()
            print(f"[Stage2] Survivors near 1Y low: {len(survivors)}")

            if survivors.empty:
                final = survivors
            else:
                # Stage 3: market cap + name
                survivors = add_market_caps(survivors)
                final = survivors[survivors["market_cap"].fillna(0) >= args.min_mktcap].copy()

    if not final.empty:
        final = final.sort_values(["drop_5d_pct", "dist_to_1y_low_pct"], ascending=[True, True]).head(args.top)

    # Save outputs
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    final.to_csv(args.save_csv, index=False)

    asof = datetime.now().strftime("%Y-%m-%d %H:%M")
    html_body = build_email_html(final, asof, args.drop_pct, args.near_low_pct, args.min_mktcap)

    os.makedirs(os.path.dirname(args.save_html), exist_ok=True)
    with open(args.save_html, "w", encoding="utf-8") as f:
        f.write(html_body)

    # Email send logic
    mode = os.getenv("EMAIL_MODE", "nonempty").strip().lower()
    should_send = (mode == "always") or (mode != "always" and len(final) > 0)

    print(f"[EmailDebug] SEND_EMAIL={args.send_email} EMAIL_MODE={mode} matches={len(final)}")

    subject = f"📉 US Market Drops ≥{int(args.drop_pct)}% & Near 1Y Low ({len(final)} matches) — {asof}"

    if args.send_email and should_send:
        send_email_html(subject, html_body)
        print(f"Sent email: {subject}")
    else:
        print(f"Email skipped. mode={mode}, matches={len(final)}, send_email={args.send_email}")

    print("Done.")


if __name__ == "__main__":
    main()