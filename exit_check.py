import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import numpy as np

# =========================
# CONFIG
# =========================
ATR_PERIOD = 14
STRUCTURE_DAYS = 10          # 10-day low / high
ATR_MULTIPLIER = 1.5

CSV_FILE = "positions.csv"

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO   = os.getenv("EMAIL_TO")

# =========================
# HELPERS
# =========================
def calculate_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr


def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)


# =========================
# MAIN LOGIC
# =========================
def main():
    positions = pd.read_csv(CSV_FILE)

    report_lines = []
    exit_found = False

    for _, row in positions.iterrows():
        ticker = row["ticker"]
        option_name = row["option_name"]
        entry_price = float(row["underlying_entry_price"])

        is_call = " C " in option_name
        is_put  = " P " in option_name

        # Pull data
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)

        if df.empty or len(df) < max(ATR_PERIOD, STRUCTURE_DAYS) + 2:
            report_lines.append(f"{ticker}: Not enough data")
            continue

        df.dropna(inplace=True)

        atr = calculate_atr(df, ATR_PERIOD).iloc[-1]
        close = df["Close"].iloc[-1]

        recent = df.iloc[-STRUCTURE_DAYS:]
        low_10 = recent["Low"].min()
        high_10 = recent["High"].max()

        # ATR stop
        if is_call:
            atr_stop = entry_price - ATR_MULTIPLIER * atr
            atr_hit = close <= atr_stop
            structure_hit = close < low_10
        elif is_put:
            atr_stop = entry_price + ATR_MULTIPLIER * atr
            atr_hit = close >= atr_stop
            structure_hit = close > high_10
        else:
            report_lines.append(f"{ticker}: Cannot infer CALL/PUT")
            continue

        # Decision
        if atr_hit:
            action = "EXIT"
            reason = "ATR stop hit"
            exit_found = True
        elif structure_hit:
            action = "EXIT"
            reason = f"{STRUCTURE_DAYS}-day structure break"
            exit_found = True
        else:
            action = "HOLD"
            reason = "Trend intact"

        report_lines.append(
            f"""
Ticker: {ticker}
Option: {option_name}
Action: {action}
Reason: {reason}
Close: {close:.2f}
ATR: {atr:.2f}
ATR Stop: {atr_stop:.2f}
10-Day Low/High: {low_10:.2f} / {high_10:.2f}
""".strip()
        )

    # =========================
    # EMAIL
    # =========================
    subject = "🚨 EXIT SIGNALS – Daily Check" if exit_found else "✅ Daily Trend Check – No Action"
    body = "\n\n--------------------\n\n".join(report_lines)

    send_email(subject, body)


if __name__ == "__main__":
    main()