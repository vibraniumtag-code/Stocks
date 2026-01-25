#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pandas as pd


PLAN_FILE = os.getenv("PLAN_FILE", "portfolio_plan.csv")
POSITIONS_FILE = os.getenv("CSV_FILE", "positions.csv")
LEDGER_FILE = os.getenv("LEDGER_FILE", "executions_ledger.csv")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm(s: Any) -> str:
    return "" if s is None else str(s).strip()


def build_move_id_from_row(r: Dict[str, Any]) -> str:
    t = norm(r.get("Type")).upper()
    ticker = norm(r.get("Ticker")).upper()
    if t in {"SELL", "ADD", "HOLD"}:
        # existing positions in your plan have "Option"
        opt = norm(r.get("Option"))
        return f"{t}|{ticker}|{opt}"
    if t == "BUY":
        sym = norm(r.get("OptionSymbol"))
        exp = norm(r.get("Expiry"))
        return f"BUY|{ticker}|{sym}|{exp}"
    # fallback
    return f"{t}|{ticker}"


def ledger_read() -> pd.DataFrame:
    if not os.path.exists(LEDGER_FILE):
        return pd.DataFrame(columns=["ts_utc", "move_id", "decision"])
    return pd.read_csv(LEDGER_FILE).fillna("")


def ledger_append(move_id: str, decision: str) -> None:
    exists = os.path.exists(LEDGER_FILE)
    with open(LEDGER_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts_utc", "move_id", "decision"])
        w.writerow([utc_now_iso(), move_id, decision])


def load_plan() -> pd.DataFrame:
    if not os.path.exists(PLAN_FILE):
        raise FileNotFoundError(f"{PLAN_FILE} not found. Ensure portfolio_manager created it in repo root.")
    df = pd.read_csv(PLAN_FILE).fillna("")
    return df


def load_positions() -> pd.DataFrame:
    if not os.path.exists(POSITIONS_FILE):
        raise FileNotFoundError(f"{POSITIONS_FILE} not found.")
    df = pd.read_csv(POSITIONS_FILE).fillna("")
    return df


def find_plan_row_by_move_id(plan: pd.DataFrame, move_id: str) -> Optional[Dict[str, Any]]:
    # Build ids for all rows and match
    records = plan.to_dict(orient="records")
    for r in records:
        if build_move_id_from_row(r) == move_id:
            return r
    return None


def apply_sell_add_to_positions(positions: pd.DataFrame, plan_row: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies SELL or ADD using:
      - Ticker match
      - Option match (positions.option_name == plan.Option)
    Updates positions.contracts accordingly.
    """
    t = norm(plan_row.get("Type")).upper()
    ticker = norm(plan_row.get("Ticker")).upper()
    option_name = norm(plan_row.get("Option"))

    if t not in {"SELL", "ADD"}:
        return positions

    if "ticker" not in positions.columns or "option_name" not in positions.columns:
        raise ValueError("positions.csv must include 'ticker' and 'option_name' columns.")

    if "contracts" not in positions.columns:
        raise ValueError("positions.csv must include 'contracts' column (int).")

    # Find exact row
    m = (
        positions["ticker"].astype(str).str.strip().str.upper().eq(ticker)
        & positions["option_name"].astype(str).str.strip().eq(option_name)
    )
    idxs = positions.index[m].tolist()
    if not idxs:
        raise ValueError(f"Could not find position row for ticker={ticker} option_name={option_name}")

    i = idxs[0]
    held = int(float(positions.at[i, "contracts"])) if str(positions.at[i, "contracts"]).strip() != "" else 0

    if t == "SELL":
        sell_n = int(float(plan_row.get("SellContracts", 0) or 0))
        new_held = max(held - sell_n, 0)
        positions.at[i, "contracts"] = new_held
    elif t == "ADD":
        add_n = int(float(plan_row.get("AddContracts", 0) or 0))
        positions.at[i, "contracts"] = max(held + add_n, 0)

    return positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move-id", required=True)
    ap.add_argument("--decision", required=True, choices=["executed", "skip", "revert"])
    args = ap.parse_args()

    move_id = args.move_id.strip()
    decision = args.decision.strip().lower()

    # prevent double-apply of executed
    ledger = ledger_read()
    if decision == "executed":
        already = (ledger["move_id"].astype(str) == move_id) & (ledger["decision"].astype(str) == "executed")
        if already.any():
            print(f"Move already executed in ledger: {move_id} — skipping re-apply.")
            return

    plan = load_plan()
    row = find_plan_row_by_move_id(plan, move_id)
    if row is None:
        raise ValueError(f"MoveId not found in {PLAN_FILE}: {move_id}")

    move_type = norm(row.get("Type")).upper()

    # Always record the click decision (even skip/revert)
    ledger_append(move_id, decision)

    if decision != "executed":
        print(f"Recorded decision={decision} for move_id={move_id}. No positions change.")
        return

    # Only apply SELL/ADD automatically (safe)
    if move_type in {"SELL", "ADD"}:
        positions = load_positions()
        positions2 = apply_sell_add_to_positions(positions, row)
        positions2.to_csv(POSITIONS_FILE, index=False)
        print(f"Applied {move_type} for {move_id} and updated {POSITIONS_FILE}.")
        return

    # HOLD / BUY: no mutation
    print(f"Recorded executed for {move_id} (type={move_type}). No positions.csv change for this type.")
    print("Tip: enable BUY auto-add only if your plan includes a full option_name format compatible with exit_check.")
    return


if __name__ == "__main__":
    main()