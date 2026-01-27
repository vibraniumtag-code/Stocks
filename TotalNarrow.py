#!/usr/bin/env python3
"""
unified_turtle_entries_only.py

Nightly runner:
- outputs ONLY fresh entries for tomorrow (no ledger)
- prints a ready-to-trade checklist
- saves CSV + TXT checklist
- sends a PRETTY HTML EMAIL (HTML-only to avoid clients choosing text/plain)
- optionally saves the same HTML to a file

ENHANCED (optional):
- Adds NEWS overlay with 3 tiers:
  1) Ticker-specific news sentiment
  2) Industry/Sector news sentiment
  3) Geopolitical/Macro news sentiment
- Supports modes:
  - annotate: add columns only (no filtering)
  - gate: filter trades using thresholds + geo risk cap
  - rank: rank by NewsScore and optionally keep top N

Default news provider:
- GDELT (no API key)

Optional:
- Alpha Vantage for ticker headlines (fallback remains GDELT)

Usage:
  python unified_turtle_entries_only.py --system 1 --allow-shorts 1 --top 300 \
    --save entries_tomorrow.csv --emit-checklist 1 --emit-html 1 --send-email 1

GitHub Secrets / env vars (SMTP):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO

Optional env:
  EMAIL_MODE=always | entries_only   (default: always)

NEWS env vars (optional):
  NEWS_ENABLED=1
  NEWS_DEBUG=1
  NEWS_MODE=annotate | gate | rank
  NEWS_LOOKBACK_DAYS=3
  NEWS_MAX_HEADLINES=30
  NEWS_SLEEP_SEC=0.6
  SENT_LONG_MIN=0.05
  SENT_SHORT_MAX=-0.05
  GEO_RISK_MAX=0.30
  W_TICKER=0.55
  W_INDUSTRY=0.25
  W_GEO=0.20
  NEWS_RANK_KEEP=0   (0=keep all)
  GEO_QUERY="war OR sanctions OR tariff OR blockade OR ..."

AlphaVantage (optional):
  ALPHAVANTAGE_API_KEY=...
"""

import os, re, argparse, html, time
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List, Dict, Tuple

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

# Optional deps for NEWS overlay
try:
    import requests
except Exception:
    requests = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None


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


# ---------------------- NEWS / SENTIMENT (optional) ----------------------
NEWS_ENABLED = int(os.getenv("NEWS_ENABLED", "1"))  # 1=on, 0=off
NEWS_MODE = os.getenv("NEWS_MODE", "annotate").strip().lower()  # annotate | gate | rank

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()

NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "3"))
NEWS_MAX_HEADLINES = int(os.getenv("NEWS_MAX_HEADLINES", "30"))
NEWS_SLEEP_SEC = float(os.getenv("NEWS_SLEEP_SEC", "0.6"))

SENT_LONG_MIN = float(os.getenv("SENT_LONG_MIN", "0.05"))
SENT_SHORT_MAX = float(os.getenv("SENT_SHORT_MAX", "-0.05"))
GEO_RISK_MAX = float(os.getenv("GEO_RISK_MAX", "0.30"))

W_TICKER = float(os.getenv("W_TICKER", "0.55"))
W_INDUSTRY = float(os.getenv("W_INDUSTRY", "0.25"))
W_GEO = float(os.getenv("W_GEO", "0.20"))

NEWS_RANK_KEEP = int(os.getenv("NEWS_RANK_KEEP", "0"))
NEWS_DEBUG = int(os.getenv("NEWS_DEBUG", "1"))


      
GEO_QUERY = os.getenv(
    "GEO_QUERY",
    (
        "war OR missile OR sanctions OR tariff OR blockade OR oil supply OR OPEC OR "
        "shipping disruption OR Red Sea OR Strait of Hormuz OR cyberattack OR coup OR "
        "geopolitical tensions OR conflict escalation"
    )
).strip()


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

    msg.replace_header("Content-Type", 'text/html; charset="utf-8"')

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


# ---------------------- NEWS / SENTIMENT helpers ----------------------
_news_cache: Dict[Tuple[str, int], List[str]] = {}
_info_cache: Dict[str, Dict[str, str]] = {}
_sent_analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None


def _utc_startdatetime_yyyymmddhhmmss(lookback_days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))
    return dt.strftime("%Y%m%d%H%M%S")


def _weighted_headline_sentiment(headlines: List[str]) -> float:
    """
    Returns a single sentiment score in [-1, +1] from a list of headlines (VADER compound).
    Weighted slightly toward earlier items.
    """
    if not headlines or _sent_analyzer is None:
        return float("nan")
    scores = []
    n = min(len(headlines), 20)
    for i, h in enumerate(headlines[:n]):
        s = _sent_analyzer.polarity_scores(h)["compound"]
        w = 1.0 / (1.0 + 0.15 * i)
        scores.append(s * w)
    denom = sum(1.0 / (1.0 + 0.15 * i) for i in range(n))
    return (sum(scores) / denom) if denom else float("nan")


def fetch_headlines_gdelt(query: str, lookback_days: int, max_headlines: int) -> List[str]:
    """
    GDELT v2 DOC API. Returns titles.
    No key required. Keep queries small; rate-limit politely.
    """
    if requests is None:
        return []

    query = (query or "").strip()
    if not query:
        return []

    cache_key = (f"gdelt:{query}", lookback_days)
    if cache_key in _news_cache:
        return _news_cache[cache_key][:max_headlines]

    startdt = _utc_startdatetime_yyyymmddhhmmss(lookback_days)
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max(5, min(250, max_headlines))),
        "startdatetime": startdt,
        "sort": "HybridRel",
    }

    titles: List[str] = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TurtleScanner/1.0)"}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            _news_cache[cache_key] = []
            return []
        data = r.json() if r.text else {}
        arts = data.get("articles", []) or []
        for a in arts:
            t = (a.get("title") or "").strip()
            if t:
                titles.append(t)
        _news_cache[cache_key] = titles
        return titles[:max_headlines]
    except Exception:
        _news_cache[cache_key] = []
        return []
    finally:
        time.sleep(NEWS_SLEEP_SEC)


def fetch_headlines_alpha_vantage_ticker(ticker: str, lookback_days: int, max_headlines: int) -> List[str]:
    """
    Optional: Alpha Vantage NEWS_SENTIMENT endpoint for ticker headlines.
    If no key or request fails, return [] and caller can fallback to GDELT.
    """
    if requests is None or not ALPHAVANTAGE_API_KEY:
        return []

    cache_key = (f"av:{ticker}", lookback_days)
    if cache_key in _news_cache:
        return _news_cache[cache_key][:max_headlines]

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": ALPHAVANTAGE_API_KEY,
        "limit": str(max(10, min(200, max_headlines))),
    }

    titles: List[str] = []
    try:
        r = requests.get(url, params=params, timeout=25)
        if r.status_code != 200:
            _news_cache[cache_key] = []
            return []
        data = r.json() if r.text else {}
        feed = data.get("feed", []) or []
        for item in feed:
            title = (item.get("title") or "").strip()
            if title:
                titles.append(title)
        _news_cache[cache_key] = titles
        return titles[:max_headlines]
    except Exception:
        _news_cache[cache_key] = []
        return []
    finally:
        time.sleep(NEWS_SLEEP_SEC)


def get_sector_industry(ticker: str) -> Tuple[str, str]:
    """
    Uses yfinance info (cached). Returns (sector, industry) or ("","").
    """
    if ticker in _info_cache:
        sec = _info_cache[ticker].get("sector", "") or ""
        ind = _info_cache[ticker].get("industry", "") or ""
        return sec, ind

    if yf is None:
        _info_cache[ticker] = {"sector": "", "industry": ""}
        return "", ""

    try:
        info = yf.Ticker(ticker).info or {}
        sec = (info.get("sector") or "").strip()
        ind = (info.get("industry") or "").strip()
        _info_cache[ticker] = {"sector": sec, "industry": ind}
        return sec, ind
    except Exception:
        _info_cache[ticker] = {"sector": "", "industry": ""}
        return "", ""


def compute_news_overlay(ticker: str) -> Dict[str, object]:
    """
    Returns dict with:
      TickerSent, IndustrySent, GeoSent, GeoRisk, NewsScore,
      HeadlinesTicker, HeadlinesIndustry, HeadlinesGeo, Sector, Industry
    """
    # If news disabled or deps missing, return blanks
    if NEWS_ENABLED != 1 or requests is None or _sent_analyzer is None:
        sec, ind = get_sector_industry(ticker)
        return {
            "Sector": sec, "Industry": ind,
            "TickerSent": "", "IndustrySent": "", "GeoSent": "", "GeoRisk": "",
            "NewsScore": "", "HeadlinesTicker": "", "HeadlinesIndustry": "", "HeadlinesGeo": ""
        }

    sec, ind = get_sector_industry(ticker)

    # --- Ticker headlines: AlphaVantage preferred, else GDELT ---
    ticker_titles = fetch_headlines_alpha_vantage_ticker(ticker, NEWS_LOOKBACK_DAYS, NEWS_MAX_HEADLINES)
    if not ticker_titles:
        # make GDELT query robust: prefer $TICKER or company mentions
        ticker_titles = fetch_headlines_gdelt(f"(${ticker} OR {ticker}) AND (stock OR shares OR earnings OR guidance OR outlook)", NEWS_LOOKBACK_DAYS, NEWS_MAX_HEADLINES)

    # --- Industry headlines: use industry first, else sector ---
    industry_query = ""
    if ind:
        industry_query = f"({ind})"
    elif sec:
        industry_query = f"({sec}) industry"
    industry_titles = fetch_headlines_gdelt(industry_query, NEWS_LOOKBACK_DAYS, NEWS_MAX_HEADLINES) if industry_query else []

    # --- Geo / macro headlines ---
    geo_titles = fetch_headlines_gdelt(GEO_QUERY, NEWS_LOOKBACK_DAYS, NEWS_MAX_HEADLINES) if GEO_QUERY else []

    t_sent = _weighted_headline_sentiment(ticker_titles)
    i_sent = _weighted_headline_sentiment(industry_titles)
    g_sent = _weighted_headline_sentiment(geo_titles)

    # GeoRisk: treat NEGATIVE geo sentiment as "risk" (0..1-ish)
    geo_risk = max(0.0, -g_sent) if (g_sent == g_sent) else float("nan")

    # Combine: if a component is NaN, treat as 0 (neutral) so it doesn't wipe the score
    t0 = 0.0 if (t_sent != t_sent) else float(t_sent)
    i0 = 0.0 if (i_sent != i_sent) else float(i_sent)
    g0 = 0.0 if (g_sent != g_sent) else float(g_sent)

    news_score = (W_TICKER * t0) + (W_INDUSTRY * i0) + (W_GEO * g0)

    def _join3(xs: List[str]) -> str:
        return " | ".join([s for s in (xs or [])[:3] if s])

    return {
        "Sector": sec,
        "Industry": ind,
        "TickerSent": round(t_sent, 3) if (t_sent == t_sent) else "",
        "IndustrySent": round(i_sent, 3) if (i_sent == i_sent) else "",
        "GeoSent": round(g_sent, 3) if (g_sent == g_sent) else "",
        "GeoRisk": round(geo_risk, 3) if (geo_risk == geo_risk) else "",
        "NewsScore": round(news_score, 3),
        "HeadlinesTicker": _join3(ticker_titles),
        "HeadlinesIndustry": _join3(industry_titles),
        "HeadlinesGeo": _join3(geo_titles),
        "TickerHL": len(ticker_titles),
        "IndustryHL": len(industry_titles),
        "GeoHL": len(geo_titles)

    }


def news_gate_pass(action: str, news_score: float, geo_risk: float) -> bool:
    """
    Gating logic used when NEWS_MODE=gate.
    """
    # If geo risk is NaN, don't block on geo. If set, enforce cap.
    if geo_risk == geo_risk and geo_risk > GEO_RISK_MAX:
        return False

    if action == "BUY_CALL":
        return (news_score == news_score) and (news_score >= SENT_LONG_MIN)
    if action == "BUY_PUT":
        return (news_score == news_score) and (news_score <= SENT_SHORT_MAX)
    return True


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

    # If news requested but deps missing, fail loudly so you notice in Actions logs
    if NEWS_ENABLED == 1 and (requests is None or SentimentIntensityAnalyzer is None):
        raise RuntimeError("NEWS_ENABLED=1 requires: pip install requests vaderSentiment")

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
            opt_type = ""
            target_delta = 0.0

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

            # ---------- NEWS overlay (annotate/gate/rank) ----------
            news = {}
            news_score = float("nan")
            geo_risk = float("nan")
            try:
                news = compute_news_overlay(t) or {}
                print(f"NEWS {t}: ht={len(ticker_titles)} hi={len(industry_titles)} hg={len(geo_titles)} score={news_score}")

                ns = news.get("NewsScore", "")
                gr = news.get("GeoRisk", "")
                news_score = float(ns) if str(ns) not in ("", "nan") else float("nan")
                geo_risk = float(gr) if str(gr) not in ("", "nan") else float("nan")
            except Exception as e:
                # keep the trade, just skip news
                news = {"Sector":"","Industry":"","TickerSent":"","IndustrySent":"","GeoSent":"","GeoRisk":"","NewsScore":"",
                        "HeadlinesTicker":"","HeadlinesIndustry":"","HeadlinesGeo":""}


            if NEWS_ENABLED == 1 and NEWS_MODE == "gate":
                if not news_gate_pass(action, news_score, geo_risk):
                    continue
            # ------------------------------------------------------

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

                # NEWS fields
                "Sector": news.get("Sector",""),
                "Industry": news.get("Industry",""),
                "TickerSent": news.get("TickerSent",""),
                "IndustrySent": news.get("IndustrySent",""),
                "GeoSent": news.get("GeoSent",""),
                "GeoRisk": news.get("GeoRisk",""),
                "NewsScore": news.get("NewsScore",""),
                "HeadlinesTicker": news.get("HeadlinesTicker",""),
                "HeadlinesIndustry": news.get("HeadlinesIndustry",""),
                "HeadlinesGeo": news.get("HeadlinesGeo",""),
            })
        except Exception:
            continue

    out = pd.DataFrame(rows)

    # If NEWS_MODE=rank, reorder and optionally keep top N
    if not out.empty and NEWS_ENABLED == 1 and NEWS_MODE == "rank" and "NewsScore" in out.columns:
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        # Rank: Calls want higher NewsScore, Puts want lower NewsScore
        out["__ns"] = out["NewsScore"].map(_to_float)
        out["__rank_score"] = np.where(out["Action"]=="BUY_CALL", out["__ns"], -out["__ns"])

        # Keep your Action ordering first, then rank score desc
        order = pd.CategoricalDtype(categories=["BUY_CALL","BUY_PUT"], ordered=True)
        out["ActionOrder"] = out["Action"].astype(order)
        out = out.sort_values(["ActionOrder","__rank_score"], ascending=[True, False]).drop(columns=["ActionOrder"])

        if NEWS_RANK_KEEP and NEWS_RANK_KEEP > 0:
            out = out.head(NEWS_RANK_KEEP)

        out = out.drop(columns=["__ns","__rank_score"], errors="ignore")

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

        # News summary (short)
        ns = r.get("NewsScore", "")
        gr = r.get("GeoRisk", "")
        news_txt = ""
        if str(ns) not in ("", "nan"):
            news_txt = f"\n  NewsScore {_fmt(ns,3)} | GeoRisk {_fmt(gr,3)}"

        lines.append(
            f"• {r['Ticker']} — {r['Action']} ({r['EntryPlan']})\n"
            f"  Spot {_fmt(r['SpotClose'])} | ATR {_fmt(r['ATR'],3)}\n"
            f"  Underlying SL {_fmt(r['StopUnderlying'])} / TP {_fmt(r['TargetUnderlying'])}\n"
            f"  Option: {r['OptionType']} {_fmt(r['OptionStrike'])} exp {r['Expiry']} "
            f"[{r['OptionSymbol']}] @ last {_fmt(r['OptionLast'])} | SL {_fmt(r['OptionStop'])} / TP {_fmt(r['OptionTarget'])}"
            f"{delta_txt}{iv_txt}"
            f"{news_txt}\n"
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
            if col in {"ATR","Delta","IV","TickerSent","IndustrySent","GeoSent","GeoRisk","NewsScore"}:
                return f"{float(v):.3f}"
        except Exception:
            pass
        return str(v)

    cols = [
        "Ticker","Action","SpotClose","ATR","StopUnderlying","TargetUnderlying",
        "Expiry","OptionType","OptionStrike","OptionLast","OptionStop","OptionTarget","OptionSymbol","Delta","IV",
        "Sector","Industry","TickerSent","IndustrySent","GeoSent","GeoRisk","NewsScore"
    ]

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

        numeric_right = {
            "SpotClose","ATR","StopUnderlying","TargetUnderlying","OptionStrike","OptionLast","OptionStop","OptionTarget",
            "Delta","IV","TickerSent","IndustrySent","GeoSent","GeoRisk","NewsScore"
        }

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

      <div style="background:#0b1220;color:#ffffff;padding:18px 22px;">
        <div style="font-size:18px;font-weight:900;line-height:1.25;">🐢 Turtle Scanner — New Entries</div>
        <div style="font-size:12px;opacity:.9;margin-top:6px;">For next open · {safe_date}</div>
      </div>

      <div style="padding:18px 22px;">
        {cards}

        <div style="margin:12px 0 16px 0;">
          <div style="font-size:13px;font-weight:900;margin:0 0 8px 0;">▶️ Run Command</div>
          <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:12px;font-size:12px;line-height:1.45;">
            <span style="font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;background:#eef0f6;padding:2px 6px;border-radius:6px;">{usage}</span>
          </div>
        </div>

        <div style="margin:0 0 16px 0;">
          <div style="font-size:13px;font-weight:900;margin:0 0 8px 0;">📈 Entries (with News Overlay)</div>
          {table_html}
          <div style="margin-top:8px;font-size:12px;color:#6b7280;">Tip: table scrolls horizontally on mobile.</div>
        </div>

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
    
