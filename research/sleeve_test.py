"""Active sleeve test: noise-band momentum day trading under the small-account day-trade limit.

Rules (as the live sleeve will trade them): checked every 30 minutes, at most one entry per day, flat by
the close, long via a 3x S&P ETF or short via an inverse 3x ETF, and no new entry once 3 day trades
have been made in the last 5 business days. Then the combined account: 75% Wick + 25% sleeve.
Also checks that the ETFs the sleeve trades are available on Alpaca.

Usage: python research/sleeve_test.py   (needs ALPACA_KEY / ALPACA_SECRET)
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "bot"))
import intraday_lab as IL   # noqa: E402
import strategy_lab as L    # noqa: E402
from strategy import ensemble  # noqa: E402


def sleeve_days(days, limit=3, window=5, max_per_day=1):
    moves = np.full((len(days), len(IL.CHECKPOINTS)), np.nan)
    for a, d in enumerate(days):
        for k, cp in enumerate(IL.CHECKPOINTS):
            p, _ = IL.price_at(d, cp)
            if p is not None:
                moves[a, k] = abs(p / d["o"][0] - 1)
    rets, traded, recent = [], [], []
    for a, d in enumerate(days):
        recent = [x for x in recent if x > a - window]          # day trades in the last 5 business days
        allowed = max_per_day if limit is None else max(0, min(max_per_day, limit - len(recent)))
        hist = {"sigma": np.nanmean(moves[a - 14:a], axis=0) if a >= 14 else None}
        if allowed == 0:
            rets.append(0.0); traded.append(0); continue
        r, t = IL.noise_area(d, hist, max_trades=allowed)
        rets.append(r); traded.append(t)
        if t:
            recent.append(a)
    idx = pd.DatetimeIndex([d["date"] for d in days])
    return pd.Series(rets, index=idx), pd.Series(traded, index=idx)


def main():
    out = {"generated": datetime.now(IL.ET).isoformat(timespec="minutes")}
    try:
        from alpaca.trading.client import TradingClient
        tc = TradingClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], paper=True)
        out["assets"] = {s: {k: getattr(a, k) for k in ("tradable", "fractionable", "shortable")}
                         for s in ("SPXL", "SPXS", "UPRO", "SPXU") for a in [tc.get_asset(s)]}
        print("assets:", out["assets"], flush=True)
    except Exception as e:
        print("asset check failed:", e)

    days = IL.to_days(IL.fetch())
    variants = {
        "Sleeve: 1 trade/day, 3 per 5 days (live rules)": sleeve_days(days),
        "Sleeve: 1 trade/day, no limit": sleeve_days(days, limit=None),
        "Sleeve: unlimited trades, no limit": sleeve_days(days, limit=None, max_per_day=99),
    }
    c = L.daily_closes()
    wick, wt = L.bt_daily(c, ensemble(c)["series"])
    res = {}
    for label, a in [("2018-now", "2018"), ("2022-now", "2022")]:
        res[label] = {}
        w = wick.loc[a:]
        res[label]["Wick 100%"] = L.metrics(w, wt.loc[a:])
        for n, (r, t) in variants.items():
            rr = r.loc[a:]
            res[label][n] = L.metrics(rr, t.loc[a:])
            res[label][n]["TradesPerWeek"] = float(t.loc[a:].mean() * 5)
        s, st = variants["Sleeve: 1 trade/day, 3 per 5 days (live rules)"]
        combo = (0.75 * w + 0.25 * s.reindex(w.index).fillna(0))
        res[label]["75% Wick + 25% sleeve (live rules)"] = L.metrics(combo, wt.loc[a:])
        L.table(res[label], label, cols=("CAGR", "Sharpe", "MaxDD", "WorstYear", "InMarket", "WinRate"))
    out["periods"] = res
    (ROOT / "results" / "sleeve_results.json").write_text(json.dumps(out, indent=1, default=str))
    print("saved")


if __name__ == "__main__":
    main()
