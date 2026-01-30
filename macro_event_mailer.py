#!/usr/bin/env python3
"""
macro_event_mailer.py

Daily email of upcoming US macro events & conferences
that may move Gold and US equity markets.

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
from datetime import datetime, timedelta, date
from email.mime.text import MIMEText
from email.utils import formataddr

# =========================
# CONFIG
# =========================
LOOKAHEAD_DAYS = int(os.getenv("LOOKAHEAD_DAYS", "30"))

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_TO  = os.getenv("EMAIL_TO")
FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Options Daily Scanner")

if not SMTP_USER or not SMTP_PASS or not EMAIL_TO:
    raise RuntimeError("Missing SMTP credentials or EMAIL_TO")

today = datetime.utcnow().date()
cutoff = today + timedelta(days=LOOKAHEAD_DAYS)

# =========================
# CURRENT MACRO NARRATIVE
# =========================
"""
Baseline assumptions (can be edited anytime):
- Inflation trending lower but sticky
- Fed biased to HOLD / CUT later, not hike
- Growth slowing but not collapsing
"""

def most_likely_move(event_type: str) -> str:
    if event_type in ("CPI", "PPI", "PCE"):
        return "Gold: 🟢 Bullish | Stocks: 🟢 Bullish (disinflation favored)"
    if event_type == "NFP":
        return "Gold: 🟢 Bullish | Stocks: 🟡 Mixed (soft labor preferred)"
    if event_type == "FOMC":
        return "Gold: 🟢 Bullish | Stocks: 🟢 Bullish (dovish hold bias)"
    if event_type == "GDP":
        return "Gold: 🟢 Bullish | Stocks: 🟡 Cautious (slowdown risk)"
    if event_type == "JACKSON":
        return "Gold: 🟢 Bullish | Stocks: 🟡 Volatile"
    if event_type == "AI_CONF":
        return "Gold: 🔴 Bearish | Stocks: 🟢 Bullish (risk-on)"
    return "Event-dependent"

# =========================
# EVENT LIST (CURATED)
# =========================
EVENTS = [
    {
        "event": "Non-Farm Payrolls (NFP)",
        "type": "NFP",
        "date": date(2026, 2, 6),
        "expected": "Jobs growth & wage inflation",
        "bias": "Weaker labor → Gold ↑ | Stocks ↑",
        "impacted": "GLD, GDX, SPY, QQQ, XLF"
    },
    {
        "event": "CPI Inflation",
        "type": "CPI",
        "date": date(2026, 2, 11),
        "expected": "Headline & Core CPI trend",
        "bias": "Cooler CPI → Gold ↑ | Stocks ↑",
        "impacted": "GLD, GDX, QQQ, SPY, TLT"
    },
    {
        "event": "PCE Inflation",
        "type": "PCE",
        "date": date(2026, 2, 26),
        "expected": "Fed-preferred inflation gauge",
        "bias": "Lower PCE → Gold ↑ | Stocks ↑",
        "impacted": "GLD, QQQ, SPY, TLT"
    },
    {
        "event": "FOMC Rate Decision",
        "type": "FOMC",
        "date": date(2026, 3, 18),
        "expected": "Rates + Powell press conference",
        "bias": "Dovish hold → Gold ↑ | Stocks ↑",
        "impacted": "GLD, TLT, SPY, QQQ"
    },
    {
        "event": "NVIDIA GTC (AI Conference)",
        "type": "AI_CONF",
        "date": date(2026, 3, 16),
        "expected": "AI demand, chips, guidance",
        "bias": "Strong AI outlook → Tech ↑",
        "impacted": "NVDA, SMH, AMD, QQQ"
    },
    {
        "event": "Jackson Hole Symposium",
        "type": "JACKSON",
        "date": date(2026, 8, 20),
        "expected": "Fed policy signaling",
        "bias": "Dovish rhetoric → Gold ↑",
        "impacted": "GLD, DXY, SPY, QQQ"
    },
]

rows = []
for e in EVENTS:
    if today <= e["date"] <= cutoff:
        e = dict(e)
        e["likely"] = most_likely_move(e["type"])
        rows.append(e)

if not rows:
    rows.append({
        "event": "No major scheduled catalysts",
        "expected": "—",
        "date": today,
        "bias": "—",
        "impacted": "—",
        "likely": "—"
    })

# =========================
# HTML EMAIL (LIGHT THEME)
# =========================
def build_html(rows):
    body_rows = ""
    for r in rows:
        body_rows += f"""
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
      <meta charset="utf-8"/>
      <style>
        body {{
          background:#f5f7fb;
          font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif;
          color:#111827;
          padding:20px;
        }}
        .card {{
          max-width:980px;
          margin:auto;
          background:#ffffff;
          border-radius:14px;
          box-shadow:0 10px 30px rgba(0,0,0,.08);
          padding:24px;
        }}
        h1 {{
          margin:0 0 6px 0;
          font-size:22px;
          color:#1f2937;
        }}
        .subtitle {{
          color:#6b7280;
          font-size:14px;
          margin-bottom:18px;
        }}
        table {{
          width:100%;
          border-collapse:collapse;
          font-size:14px;
        }}
        th {{
          background:#f1f5f9;
          color:#334155;
          padding:10px;
          border-bottom:1px solid #e5e7eb;
          text-align:left;
        }}
        td {{
          padding:10px;
          border-bottom:1px solid #e5e7eb;
          vertical-align:top;
        }}
        tr:hover {{
          background:#f8fafc;
        }}
        .foot {{
          margin-top:16px;
          font-size:12px;
          color:#6b7280;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>📅 Macro Catalyst Watch</h1>
        <div class="subtitle">
          Upcoming events likely to move <b>Gold</b> and <b>US markets</b>
          (next {LOOKAHEAD_DAYS} days)
        </div>
        <table>
          <tr>
            <th>Event</th>
            <th>Expected</th>
            <th>Date</th>
            <th>Historical Bias</th>
            <th>Most Likely Move</th>
            <th>Impacted</th>
          </tr>
          {body_rows}
        </table>
        <div class="foot">
          Direction reflects the <b>current macro narrative</b>.
          Market reaction depends on surprise vs expectations.
        </div>
      </div>
    </body>
    </html>
    """

html_body = build_html(rows)

# =========================
# SEND EMAIL
# =========================
msg = MIMEText(html_body, "html")
msg["Subject"] = f"📊 Macro Catalyst Calendar — {today.strftime('%b %d')}"
msg["From"] = formataddr((FROM_NAME, SMTP_USER))
msg["To"] = EMAIL_TO

context = ssl.create_default_context()
with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as s:
    s.login(SMTP_USER, SMTP_PASS)
    s.send_message(msg)

print("Macro catalyst email sent successfully.")