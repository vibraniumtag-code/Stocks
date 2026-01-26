def atr_pct_for_ticker(ticker: str, period: int, report_lines: List[str], ticker_cache: Dict[str, pd.DataFrame]) -> Optional[float]:
    """
    Returns ATR% = ATR / current_price using COMPLETED daily bars.
    """
    try:
        if ticker not in ticker_cache:
            df = yf.download(ticker, period="9mo", interval="1d", progress=False)
            df = flatten_columns(df).dropna() if df is not None else pd.DataFrame()
            ticker_cache[ticker] = df
        df = ticker_cache.get(ticker, pd.DataFrame())
        if df is None or df.empty:
            report_lines.append(f"DIAG: ATR% calc: no daily data for {ticker}")
            return None

        df_completed = remove_today_partial_bar(df).dropna()
        if len(df_completed) < max(period, 30):
            report_lines.append(f"DIAG: ATR% calc: insufficient history for {ticker}")
            return None

        atr_series = calculate_atr(df_completed, period)
        atr_last = atr_series.iloc[-1]
        if pd.isna(atr_last):
            report_lines.append(f"DIAG: ATR% calc: ATR NaN for {ticker}")
            return None

        close_price = float(df_completed["Close"].iloc[-1])
        cur_price, _src = get_current_price(ticker, fallback=close_price)
        if cur_price <= 0:
            return None

        atr_val = float(atr_last)
        return float(atr_val / cur_price)
    except Exception as e:
        report_lines.append(f"DIAG: ATR% calc exception for {ticker}: {e}")
        return None


def allocate_freed_cash_vol_adj(
    cand: pd.DataFrame,
    freed_cash: float,
    max_new_per_run: int,
    max_contracts_per_pos: int,
    report_lines: List[str]
) -> pd.DataFrame:
    """
    Adds columns:
      ATRpct, Weight, AllocCash, BuyContracts, EstCostTotal
    Uses weights = 1 / ATRpct, normalized across candidates.
    """
    if cand is None or cand.empty or freed_cash <= 0:
        return pd.DataFrame()

    work = cand.copy()

    # Keep only valid rows with finite EstCost1
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["EstCost1"])
    work = work[np.isfinite(work["EstCost1"]) & (work["EstCost1"] > 0)].copy()
    if work.empty:
        return pd.DataFrame()

    # Prefer lower ATR% (smoother). Drop unknown ATR% (fallback handled later).
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["ATRpct"])
    work = work[np.isfinite(work["ATRpct"]) & (work["ATRpct"] > 0)].copy()
    if work.empty:
        return pd.DataFrame()

    # If too many candidates, keep best ones by low ATR% (most stable) first
    work = work.sort_values(["ATRpct", "Ticker"]).head(max_new_per_run).copy()

    # Weights = 1 / ATR%
    work["Weight"] = 1.0 / work["ATRpct"].astype(float)
    wsum = float(work["Weight"].sum())
    if not np.isfinite(wsum) or wsum <= 0:
        return pd.DataFrame()

    work["AllocCash"] = (work["Weight"] / wsum) * float(freed_cash)

    # Contracts from allocation
    work["BuyContracts"] = (work["AllocCash"] / work["EstCost1"]).fillna(0).astype(float).apply(lambda x: int(np.floor(x)))
    work["BuyContracts"] = work["BuyContracts"].clip(lower=0, upper=max_contracts_per_pos).astype(int)

    # Enforce at least 1 contract if affordable (helps not “miss” a top-ranked stable name)
    for idx, r in work.iterrows():
        if int(r["BuyContracts"]) <= 0:
            est1 = float(r["EstCost1"])
            if freed_cash >= est1:
                work.at[idx, "BuyContracts"] = 1

    work["EstCostTotal"] = (work["BuyContracts"] * work["EstCost1"]).round(2)

    # If total exceeds freed_cash (because of min-1 rule), scale back with a simple trim loop:
    total_spend = float(work["EstCostTotal"].sum())
    if total_spend > freed_cash:
        # trim 1 contract at a time from highest ATR% (least preferred) until within budget
        work = work.sort_values(["ATRpct"], ascending=False).copy()
        guard = 10000
        while float(work["EstCostTotal"].sum()) > freed_cash and guard > 0:
            guard -= 1
            # pick first row that still has >0 contracts to trim
            trimmed = False
            for idx, r in work.iterrows():
                bc = int(r["BuyContracts"])
                if bc > 0:
                    work.at[idx, "BuyContracts"] = bc - 1
                    work.at[idx, "EstCostTotal"] = round((bc - 1) * float(r["EstCost1"]), 2)
                    trimmed = True
                    break
            if not trimmed:
                break

    # Return in preferred order (lowest ATR% first)
    work = work.sort_values(["ATRpct", "Ticker"]).copy()

    report_lines.append(
        "DIAG: Vol-adjusted allocation (1/ATR%) applied. "
        f"cands={len(work)} spend={work['EstCostTotal'].sum():.2f} freed={freed_cash:.2f}"
    )
    return work