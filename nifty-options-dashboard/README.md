# NIFTY Options Terminal Dashboard

A terminal-style NIFTY options dashboard with live data, OI analysis, PCR, VIX, max pain, IV skew, and plain-English market state explanations.

## Files

| File | Purpose |
|---|---|
| `nifty-options-terminal-dashboard.html` | The dashboard UI |
| `server.py` | Local Python backend proxy (fetches live NSE data) |
| `start-live-dashboard.bat` | One-click Windows launcher |

## Quick Start (Windows)

1. Make sure Python is installed
2. Double-click `start-live-dashboard.bat`
3. Keep the backend window open while using the dashboard

## Quick Start (Mac/Linux)

```bash
python server.py &
open nifty-options-terminal-dashboard.html
```

## How Live Data Works

- The backend runs on `http://127.0.0.1:8765`
- It fetches live NIFTY option-chain and India VIX data from NSE
- The dashboard polls the backend every 15 seconds
- If live fetch fails, falls back to embedded snapshot data
- Market hours: 9:15 AM – 3:30 PM IST, Mon–Fri (trading days)

## Dashboard Tabs

1. **01_overview** — Spot, PCR, VIX, Max Pain, OI wall map
2. **02_chain-map** — Strike concentration, support/resistance meter, key strikes table
3. **03_flow-signals** — Call/put writing scores, volatility stress, IV skew
4. **04_market-state** — State engine, regime matrix, trend/breakout analysis

## Color Convention

- 🔴 Red = Call OI (resistance overhead)
- 🟢 Green = Put OI (support below)
- 🟡 Amber = ATM strike
