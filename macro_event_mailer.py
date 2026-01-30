#!/usr/bin/env python3
"""
macro_event_mailer.py

Daily macro catalyst calendar email with:
- Event
- Expected announcement
- Date
- Historical bias
- MOST LIKELY MOVE (based on current macro narrative)
- Impacted assets

SMTP (same as portfolio_manager):
  SMTP_HOST
  SMTP_PORT
  SMTP_USER
  SMTP_PASS
  EMAIL_TO

Optional env:
  LOOKAHEAD_DAYS=30
  EMAIL_FROM_NAME="Options Daily Scanner"
"""

import os
import ssl
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from datetime import datetime, timedelta, date

LOOKAHEAD_DAYS = int(os.getenv("LOOKAHEAD_DAYS", "30"))

# ---------- SMTP ----------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_TO  = os.getenv("EMAIL_TO")
FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Options Daily Scanner")

today = datetime.utcnow().date()
cutoff = today + timedelta(days=LOOKAHEAD_DAYS)

# ---------- CURRENT MACRO NARRATIVE ----------
"""
Assumptions (can be tuned anytime):
- Inflation trending down but sticky
- Fed biased to CUT later, not hike
- Growth slowing but not recession
"""

def likely_move(event_type):
    if event_type in ["CPI", "PPI", "PCE"]:
        return "Gold: 🟢 Bullish | Stocks: 🟢 Bullish (disinflation narrative)"
    if event_type == "NFP":
        return "Gold: 🟢 Bullish | Stocks: 🟡 Mixed (soft labor favored)"
    if event_type == "FOMC":
        return "Gold: 🟢 Bullish | Stocks: 🟢 Bullish (dovish hold bias)"
    if event_type == "GDP":
        return "Gold: 🟢 Bullish | Stocks: 🟡 Cautious (slowdown risk)"
    if event_type == "JACKSON":
        return "Gold: 🟢 Bullish | Stocks: 🟡 Volatile"
    if event_type == "AI_CONF":
        return "Gold: 🔴 Bearish | Stocks: 🟢 Bullish (risk-on)"
    return "Neutral / Event-dependent"

# ---------- EVENT LIST ----------
EVENTS = [
    {
        "event": "CPI Inflation",
        "type": "CPI",
        "date": date(2026, 2, 11),
        "expected": "Headline & Core inflation trend",
        "bias": "Cooler CPI → Gold ↑ | Stocks ↑",
        "impacted": "GLD, GDX, QQQ, SPY, TLT"
    },
    {
        "event": "Non-Farm Payrolls",
        "type": "NFP",
        "date": date(2026, 2, 6),
        "expected": "Jobs growth & wage pressure",
        "bias": "Soft labor → Gold ↑ | Stocks ↑",
        "impacted": "GLD, SPY, QQQ, XLF"
    },
    {
        "event": "FOMC Rate Decision",
        "type": "FOMC",
        "date": date(2026, 3, 18),
        "expected": "Rates + Powell press conference",
        "bias": "Dovish hold → Gold ↑ | Stocks ↑",
        "impacted": "GLD, TLT, QQQ, SPY"
    },
    {
        "event": "Jackson Hole Symposium",
        "type": "JACKSON",
        "date": date(2026, 8, 20),
        "expected": "Policy signaling from Fed speakers",
        "bias": "Dovish rhetoric → Gold ↑",
        "impacted": "GLD, DXY, SPY, QQQ"
    },
    {
        "event": "NVIDIA GTC (AI Conference)",
        "type": "AI_CONF",
        "date": date(2026, 3, 16),
        "expected": "AI demand, chips, guidance",
        "bias": "Strong AI outlook → Tech ↑",
        "impacted": "NVDA, SMH, AMD, QQQ"
    },
]

rows = []
for e in EVENTS:
    if today <= e["date"] <= cutoff:
        rows.append({
            **e,
            "likely": likely_move(e["type"])
        })

if not rows:
    rows.append({
        "event": "No major scheduled catalysts",
        "expected": "—",
        "date": today,
        "bias": "—",
        "impacted": "—",
        "likely": "—"
    })

# ---------- HTML EMAIL ----------
def build_html(rows):
    tr = ""
    for r in rows:
        tr += f"""
        <tr>
          <td><b>{r['event']}</b></td>
          <td>{r['expected']}</td>
          <td>{r['date'].strftime('%b %d, %Y')}</td>
          <td>{r['bias']}</td>
          <td><b>{r['likely']}</b></td>
          <td>{r['impacted']}</td>
        </tr>
        """

    return f"""
    <html>
    <head>
    <style>
      body {{ background:#0b1220; color:#e5e7eb; font-family:system-ui; }}
      .card {{ background:#111827; padding:20px; border-radius:14px; }}
      h1 {{ color:#facc15; }}
      table {{ width:100%; border-collapse:collapse; margin-top:15px; }}
      th, td {{ padding:10px; border-bottom:1px solid #1f2937; text-align:left; }}
      th {{ color:#93c5fd; }}
      tr:hover {{ background:#1f2937; }}
      .foot {{ color:#9ca3af; font-size:12px; margin-top:15px; }}
    </style>
    </head>
    <body>
      <div class="card">
        <h1>📅 Macro Catalyst Watch</h1>
        <p>Events likely to move <b>Gold & US Markets</b> in the next {LOOKAHEAD_DAYS} days</p>
        <table>
          <tr>
            <th>Event</th>
            <th>Expected</th>
            <th>Date</th>
            <th>Historical Bias</th>
            <th>Most Likely Move</th>
            <th>Impacted</th>
          </tr>
          {tr}
        </table>
        <div class="foot">
          Direction reflects <b>current macro narrative</b>. Market reaction depends on surprise vs expectations.
        </div>
      </div>
    </body>
    </html>
    """

html_body = build_html(rows)

# ---------- SEND ----------
msg = MIMEText(html_body, "html")
msg["Subject"] = f"📊 Macro Catalyst Calendar — {today.strftime('%b %d')}"
msg["From"] = formataddr((FROM_NAME, SMTP_USER))
msg["To"] = EMAIL_TO

ctx = ssl.create_default_context()
with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
    s.login(SMTP_USER, SMTP_PASS)
    s.send_message(msg)

print("Macro catalyst email sent.")