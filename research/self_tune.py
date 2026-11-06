"""Does a self-teaching Wick beat the fixed one?

Wick's Ensemble has four settings: trend length, momentum lookback, dip-buy RSI level, and the
volatility target. A self-tuning bot re-picks them from recent results. This script tests that idea
honestly with walk-forward selection: at each re-tune date it only looks at data up to that date,
picks the settings with the best trailing score, and trades them until the next re-tune.

Compared against the fixed defaults (200 / 252 / 10 / 15%), over 1970-now, 2000-now and 2016-now.
Usage: python research/self_tune.py [--synthetic]
"""
from __future__ import annotations
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "bot"))
import strategy_lab as L          # noqa: E402  (data, backtest, metrics)
import strategy as S              # noqa: E402

GRID = {"trend": [150, 200, 250], "mom": [126, 189, 252], "dip": [5, 10, 15], "vol": [0.10, 0.15, 0.20]}
DEFAULT = "200/252/10/0.15"
key = lambda combo: "/".join(str(x) for x in combo)


def component_cache(c):
    t = {n: S.trend(c, n) for n in GRID["trend"]}
    m = {n: S.momentum(c, n) for n in GRID["mom"]}
    d = {n: S.dip_buy(c, entry=n) for n in GRID["dip"]}
    v = {n: S.vol_scale(c, target=n) for n in GRID["vol"]}
    return t, m, d, v


def all_returns(c) -> pd.DataFrame:
    t, m, d, v = component_cache(c)
    cols = {}
    for combo in itertools.product(*GRID.values()):
        a, b, e, f = combo
        pos = ((t[a] + m[b] + d[e]) / 3 * v[f]).clip(upper=1.0)
        cols[key(combo)] = L.bt_daily(c, pos)[0]
    return pd.DataFrame(cols)


def score(r: np.ndarray, how: str) -> np.ndarray:
    if how == "sharpe":
        sd = r.std(axis=0)
        return np.where(sd > 0, r.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)
    eq = np.cumprod(1 + r, axis=0)
    dd = np.abs((eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0))
    cagr = eq[-1] ** (252 / len(r)) - 1
    return np.where(dd > 0, cagr / np.where(dd > 0, dd, 1), -np.inf)   # return per unit of worst drop


def walk_forward(R: pd.DataFrame, window_years, every: str, how: str, stickiness: float = 0.0):
    """Re-pick settings at each period start using only the trailing window; returns (returns, picks)."""
    A, cols = R.to_numpy(), list(R.columns)
    per = R.index.to_period(every) if every != "D" else R.index
    starts = np.flatnonzero(np.r_[True, per[1:] != per[:-1]])
    W = int(252 * window_years) if window_years else None
    need = 252 * min(window_years or 3, 3)
    out, picks, cur = np.zeros(len(R)), [], cols.index(DEFAULT)
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(R)
        lo = 0 if W is None else max(0, i - W)
        if i - lo >= need:
            sc = score(A[lo:i], how)
            best = int(np.argmax(sc))
            if sc[best] > sc[cur] * (1 + stickiness) if sc[cur] > 0 else sc[best] > sc[cur]:
                cur = best
        out[i:j] = A[i:j, cur]
        if every != "D" or k % 21 == 0:
            picks.append((str(R.index[i].date()), cols[cur]))
    return pd.Series(out, index=R.index), picks


def main():
    c = L.daily_closes()
    print(f"closes {c.index[0].date()} to {c.index[-1].date()}", flush=True)
    R = all_returns(c)
    print(f"{R.shape[1]} setting combinations backtested", flush=True)
    held = pd.Series(1, index=R.index)

    variants = {
        "Fixed (current Wick)": (R[DEFAULT], None),
        "Self-tune: 3-yr window, monthly": walk_forward(R, 3, "M", "sharpe"),
        "Self-tune: 5-yr window, monthly": walk_forward(R, 5, "M", "sharpe"),
        "Self-tune: 10-yr window, quarterly": walk_forward(R, 10, "Q", "sharpe"),
        "Self-tune: all history, yearly": walk_forward(R, None, "Y", "sharpe"),
        "Self-tune: 5-yr, monthly, sticky": walk_forward(R, 5, "M", "sharpe", stickiness=0.15),
        "Self-tune: 5-yr, monthly, by return/drop": walk_forward(R, 5, "M", "calmar"),
        "Daily self-tune: 1-yr window": walk_forward(R, 1, "D", "sharpe"),
    }
    out = {"generated": datetime.now(L.ET).isoformat(timespec="minutes"), "grid": GRID, "default": DEFAULT,
           "periods": {}, "recent_picks": {}}
    for label, a in [("1970-now", "1970"), ("2000-now", "2000"), ("2016-now", "2016")]:
        res = {}
        for n, (r, _) in variants.items():
            res[n] = L.metrics(r.loc[a:], held.loc[a:])
        sc = {k: L.metrics(R[k].loc[a:], held.loc[a:])["Sharpe"] for k in R.columns}
        hb = max(sc, key=sc.get)
        res[f"Hindsight best {hb} (not possible live)"] = L.metrics(R[hb].loc[a:], held.loc[a:])
        out["periods"][label] = res
        L.table(res, label, cols=("CAGR", "Sharpe", "MaxDD", "WorstYear"))
    for n, (_, picks) in variants.items():
        if picks:
            out["recent_picks"][n] = picks[-6:]
    # how often does a fixed setting sit in the top third of the grid, decade by decade?
    ranks = {}
    for dec in range(1970, 2030, 10):
        seg = R.loc[str(dec):str(dec + 9)]
        if len(seg) > 500:
            sh = seg.mean() / seg.std()
            ranks[f"{dec}s"] = int((sh > sh[DEFAULT]).sum()) + 1
    out["default_rank_by_decade"] = ranks
    print("\nCurrent Wick's rank among", R.shape[1], "settings, by decade:", ranks)
    name = "self_tune_synthetic.json" if L.SYN else "self_tune_results.json"
    (ROOT / "results" / name).write_text(json.dumps(out, indent=1, default=str))
    print("saved", name)


if __name__ == "__main__":
    main()
