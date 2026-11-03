"""Test an intraday strategy on real minute data BEFORE letting the bot trade it.

Run on your Mac (needs your Alpaca keys in the environment):
    export ALPACA_KEY=...  ALPACA_SECRET=...
    pip install alpaca-py pandas
    python research/intraday_test.py

Example strategy: opening-range breakout on UPRO.
  * Watch the first 30 minutes. Buy if price breaks above that range's high.
  * Exit at the close, or if price falls back below the range's low.
Results include a per-trade cost. Compare against just holding UPRO.
If it doesn't clearly beat holding after costs, don't turn it on.
Note: with under $25k in a margin account, US rules allow only 3 day trades per 5 business days.
"""
import os
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

ET = ZoneInfo("America/New_York")
SYMBOL, DAYS, COST = "UPRO", 365 * 2, 0.0005  # 5 bps per side for spread + slippage

client = StockHistoricalDataClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"])
req = StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame.Minute,
                       start=datetime.now(ET) - timedelta(days=DAYS), end=datetime.now(ET) - timedelta(minutes=20),
                       feed=DataFeed.IEX)
df = client.get_stock_bars(req).df.reset_index()
df["t"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
df = df[(df.t.dt.time >= time(9, 30)) & (df.t.dt.time < time(16, 0))]

rows = []
for day, g in df.groupby(df.t.dt.date):
    opening = g[g.t.dt.time < time(10, 0)]
    rest = g[g.t.dt.time >= time(10, 0)]
    if opening.empty or rest.empty:
        continue
    hi, lo = opening.high.max(), opening.low.min()
    day_ret = rest.close.iloc[-1] / opening.open.iloc[0] - 1
    entry = rest[rest.high > hi]
    if entry.empty:
        rows.append((day, 0.0, day_ret)); continue
    after = rest[rest.t >= entry.t.iloc[0]]
    stop = after[after.low < lo]
    exit_px = lo if not stop.empty else after.close.iloc[-1]
    rows.append((day, exit_px / hi - 1 - 2 * COST, day_ret))

r = pd.DataFrame(rows, columns=["day", "strategy", "hold_intraday"]).set_index("day")
eq = (1 + r).cumprod()
print(f"{len(r)} days, {int((r.strategy != 0).sum())} trades")
for c in r:
    dd = (eq[c] / eq[c].cummax() - 1).min()
    print(f"{c:15s} total {eq[c].iloc[-1] - 1:+.1%}   max drawdown {dd:.1%}   win days {(r[c] > 0).mean():.0%}")
