# unified_turtle_entries_only.py
# Nightly runner: outputs ONLY fresh entries for tomorrow (no ledger) AND prints a ready-to-trade checklist.
#
# New in this version:
# - Prints a human-readable checklist to stdout.
# - Optional: saves a .txt checklist next to the CSV via --emit-checklist 1
# - NEW FILTERS to narrow Turtle signals:
#     1) Trend filter using SMA200
#     2) Liquidity filter using 20D avg dollar volume
#     3) ATR% sanity filter (avoid too dead / too wild)
#     4) Breakout strength buffer beyond channel
#
# Usage:
#   python unified_turtle_entries_only.py --system 1 --allow-shorts 1 --top 300 \
#     --save entries_tomorrow.csv --emit-checklist 1
#
# -----------------------------------------------------------------------------

import os, re, time, argparse
from datetime import datetime, date
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

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

TOP_N_BY_DEFAULT        = 300

FALLBACK_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","V","UNH",
    "XOM","JNJ","WMT","PG","MA","HD","CVX","MRK","ABBV","KO",
    "PEP","COST","AVGO","LLY","ORCL","NKE","ADBE","CRM","NFLX","INTC",
    "AMD","QCOM","TXN","CSCO","UPS","CAT","GE","HON","IBM","AXP"
]

# ---------------------- Helpers ----------------------
def looks_like_ticker(s: str) -> bool:
    return bool(re.match(r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$", (s or "").strip().upper()))

def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
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
        if not exps: return None
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
    # basic liquidity
    if "volume" in df.columns or "openInterest" in df.columns:
        vol = df.get("volume"); oi = df.get("openInterest")
        mask = pd.Series(True, index=df.index)
        if vol is not None: mask &= (vol.fillna(0) >= 1)
        if oi is not None:  mask &= (oi.fillna(0) >= 10)
        df = df[mask] if not mask.empty else df
        if df.empty: df = options_df.copy()
    # prefer delta if present
    if "delta" in df.columns and df["delta"].notna().any():
        df["delta_diff"] = (df["delta"] - target_delta).abs()
        return df.nsmallest(1, "delta_diff").iloc[0].to_dict()
    # fallback: ATM-ish by moneyness
    if is_call:
        df["mny"] = (df["strike"] / spot - 1.0).abs()
    else:
        df["mny"] = (spot / df["strike"] - 1.0).abs()
    return df.nsmallest(1, "mny").iloc[0].to_dict()

def build_universe(top: int) -> List[str]:
    # Keep it simple/static for reliability here (no scraping).
    top = max(1, min(top, len(FALLBACK_UNIVERSE)))
    return FALLBACK_UNIVERSE[:top]

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

    # ------------------ FILTER CONFIG (NEW) ------------------
    SMA_TREND_PERIOD = 200
    MIN_DOLLAR_VOL_20D = 20_000_000   # $20M/day avg over last 20 bars
    ATRP_MIN = 0.01                  # 1% ATR of price
    ATRP_MAX = 0.08                  # 8% ATR of price
    BREAKOUT_BUFFER_ATR = 0.10       # 0.10 ATR beyond channel
    # ---------------------------------------------------------

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

            # ----------- Trend filter: SMA200 -----------
            sma200 = ohlc["Close"].rolling(SMA_TREND_PERIOD).mean().iloc[-1]
            if not np.isfinite(sma200):
                continue

            # ----------- Liquidity filter --------------
            # Approx dollar volume = Close * Volume if available
            if "Volume" in df.columns:
                vol = pd.to_numeric(df["Volume"], errors="coerce")
                dollar_vol_20d = (vol * ohlc["Close"]).rolling(20).mean().iloc[-1]
                if not np.isfinite(dollar_vol_20d) or dollar_vol_20d < MIN_DOLLAR_VOL_20D:
                    continue
            # If no Volume column, skip liquidity filter silently

            # ----------- ATR% sanity filter -------------
            atrp = atrv / px
            if atrp < ATRP_MIN or atrp > ATRP_MAX:
                continue

            # ----------- Breakout strength filter -------
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
                target_delta = 0.60

            elif allow_shorts and short_entry.iloc[-1] and px < sma200 and strong_short:
                action = "BUY_PUT"
                stop_under = round(px + k_stop_atr*atrv, 2)
                target_under = round(px - k_take_atr*atrv, 2)
                opt_type = "PUT"
                target_delta = -0.60
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
                    if "delta" in chosen and chosen["delta"]==chosen["delta"]:
                        delta = float(chosen["delta"])
                    if "impliedVolatility" in chosen and chosen["impliedVolatility"]==chosen["impliedVolatility"]:
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

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Nightly Turtle new-entry scanner (no ledger) + checklist")
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

    # Print checklist
    date_str = datetime.now().strftime("%Y-%m-%d")
    checklist = print_checklist(df, date_str)

    # Optionally save .txt checklist
    if int(args.emit_checklist) == 1:
        base, _ = os.path.splitext(args.save)
        txt_path = f"{base}_checklist_{date_str}.txt"
        with open(txt_path, "w") as f:
            f.write(checklist)
        print(f"Checklist saved to: {txt_path}")

    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
