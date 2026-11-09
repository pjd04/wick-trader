"""Intraday strategy lab: tests published day-trading strategies on 10 years of SPY 5-minute bars.

Runs inside GitHub Actions (research workflow) so it can use the repo's Alpaca keys.
Every strategy is traded as a 3x ETF: longs in UPRO, shorts by buying SPXU (inverse 3x),
so no short-selling is needed. All positions are closed by 4:00 pm. Costs: 3 bps per side.

Strategies
  hold_intraday   buy UPRO at the open, sell at the close, every day (baseline)
  orb30_long      30-minute opening-range breakout, long only, stop at the range low
  orb5            5-minute opening range (Zarattini & Aziz 2023): direction of the first
                  candle, stop at its other end, target 10R, else exit at the close
  intraday_mom    first-half-hour return predicts the last half hour (Gao, Han, Li & Zhou 2018)
  gap_fade        fade opening gaps larger than 0.5%
  noise_area      "Beat the Market" intraday momentum (Zarattini, Aziz & Barbon 2024):
                  trade breakouts of the day's noise band, trail with VWAP, checked every 30 min
  vwap_trend      long above the day's VWAP, short below (Zarattini & Aziz 2024, "VWAP: The Holy Grail
                  for Day Trading Systems"), here on 5-minute bars
  trend_pullback  EMA 9/30 trend with a pullback entry, the star worker in the Candle City demo
Baselines over the same dates
  wick_daily      the current Wick strategy (daily Ensemble, holds overnight)
  buy_hold_3x     hold UPRO through everything

Usage: python research/intraday_lab.py            (needs ALPACA_KEY / ALPACA_SECRET)
       python research/intraday_lab.py --synthetic (offline smoke test with fake prices)
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "bot"))
from strategy import ensemble  # noqa: E402

ET = ZoneInfo("America/New_York")
LEV = 3.0
COST = 0.0003          # per side, as a fraction of the position
DRAG = 0.009 / 252 + 2 * 0.03 / 252   # daily cost of holding a 3x ETF overnight (fees + embedded borrowing)
OOS_START = "2022-01-01"


# --------------------------------------------------------------------------- data
def fetch() -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment
    c = StockHistoricalDataClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"])
    start, end = datetime(2016, 1, 1, tzinfo=ET), datetime.now(ET) - timedelta(minutes=30)
    frames, y = [], start
    while y < end:
        z = min(datetime(y.year + 1, 1, 1, tzinfo=ET), end)
        req = StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                               start=y, end=z, feed=DataFeed.SIP, adjustment=Adjustment.ALL)
        df = c.get_stock_bars(req).df.reset_index()
        frames.append(df)
        print(f"{y.year}: {len(df):,} bars", flush=True)
        y = z
    df = pd.concat(frames)
    df["t"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
    return df[["t", "open", "high", "low", "close", "volume", "vwap"]]


def synthetic(days=900, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, p = [], 200.0
    d = pd.Timestamp("2019-01-02", tz=ET)
    while len(rows) < days * 78:
        if d.weekday() < 5:
            p *= 1 + rng.normal(0, 0.004)
            for k in range(78):
                t = d + pd.Timedelta(hours=9, minutes=30 + 5 * k)
                o = p
                p *= 1 + rng.normal(0.00002, 0.0012)
                rows.append((t, o, max(o, p) * 1.0004, min(o, p) * 0.9996, p, 1e5, (o + p) / 2))
        d += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "vwap"])


def to_days(df: pd.DataFrame) -> list[dict]:
    df = df[(df.t.dt.time >= dtime(9, 30)) & (df.t.dt.time < dtime(16, 0))].sort_values("t")
    days, prev_close = [], None
    for date, g in df.groupby(df.t.dt.date):
        mins = ((g.t.dt.hour - 9) * 60 + g.t.dt.minute - 30).to_numpy()   # bar start, minutes after 9:30
        if len(g) < 30 or mins[0] != 0:
            prev_close = float(g.close.iloc[-1]) if len(g) else prev_close
            continue
        d = {"date": pd.Timestamp(date), "m": mins, "o": g.open.to_numpy(float), "h": g.high.to_numpy(float),
             "l": g.low.to_numpy(float), "c": g.close.to_numpy(float), "v": g.volume.to_numpy(float),
             "vw": g.vwap.to_numpy(float), "prev": prev_close}
        prev_close = float(d["c"][-1])
        if d["prev"] is not None:
            days.append(d)
    return days


def price_at(d, minute):
    """Price at a clock time = close of the 5-minute bar that ends then."""
    idx = np.where(d["m"] == minute - 5)[0]
    return (float(d["c"][idx[0]]), int(idx[0])) if len(idx) else (None, None)


def trade(side, entry, exit_):
    return side * LEV * (exit_ / entry - 1) - 2 * COST


# --------------------------------------------------------------------------- strategies
def hold_intraday(d, _):
    return trade(1, d["o"][0], d["c"][-1]), 1


def orb30_long(d, _):
    first = d["m"] < 30
    hi, lo = d["h"][first].max(), d["l"][first].min()
    for i in np.where(~first)[0]:
        if d["h"][i] > hi:
            entry = max(hi, d["o"][i])
            for j in range(i + 1, len(d["c"])):
                if d["l"][j] <= lo:
                    return trade(1, entry, min(lo, d["o"][j])), 1
            return trade(1, entry, d["c"][-1]), 1
    return 0.0, 0


def orb5(d, _, long_only=False):
    o1, c1, h1, l1 = d["o"][0], d["c"][0], d["h"][0], d["l"][0]
    if c1 == o1 or (long_only and c1 < o1):
        return 0.0, 0
    side = 1 if c1 > o1 else -1
    entry = d["o"][1]
    stop = l1 if side == 1 else h1
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0, 0
    target = entry + side * 10 * risk
    for j in range(1, len(d["c"])):
        if side == 1:
            if d["l"][j] <= stop:
                return trade(1, entry, min(stop, d["o"][j])), 1
            if d["h"][j] >= target:
                return trade(1, entry, max(target, d["o"][j])), 1
        else:
            if d["h"][j] >= stop:
                return trade(-1, entry, max(stop, d["o"][j])), 1
            if d["l"][j] <= target:
                return trade(-1, entry, min(target, d["o"][j])), 1
    return trade(side, entry, d["c"][-1]), 1


def intraday_mom(d, _):
    p10, _i = price_at(d, 30)
    p1530, _j = price_at(d, 360)
    if p10 is None or p1530 is None or d["m"][-1] != 385:
        return 0.0, 0
    r1 = p10 / d["prev"] - 1
    if r1 == 0:
        return 0.0, 0
    return trade(1 if r1 > 0 else -1, p1530, d["c"][-1]), 1


def gap_fade(d, _, thresh=0.005):
    gap = d["o"][0] / d["prev"] - 1
    if abs(gap) < thresh:
        return 0.0, 0
    return trade(-1 if gap > 0 else 1, d["c"][0], d["c"][-1]), 1


CHECKPOINTS = list(range(30, 361, 30))   # 10:00, 10:30, ... 3:30


def noise_area(d, hist, long_only=False, max_trades=None):
    sig = hist.get("sigma")
    if sig is None:
        return 0.0, 0
    o, prev = d["o"][0], d["prev"]
    ub_base, lb_base = max(o, prev), min(o, prev)
    cum_pv, cum_v = np.cumsum(d["vw"] * d["v"]), np.cumsum(d["v"])
    pos, entry, eq, n = 0, 0.0, 1.0, 0
    for k, cp in enumerate(CHECKPOINTS):
        p, i = price_at(d, cp)
        if p is None or np.isnan(sig[k]):
            continue
        vwap = cum_pv[i] / cum_v[i] if cum_v[i] > 0 else p
        ub, lb = ub_base * (1 + sig[k]), lb_base * (1 - sig[k])
        if pos == 1 and p < max(ub, vwap):
            eq *= 1 + trade(1, entry, p); pos = 0
        elif pos == -1 and p > min(lb, vwap):
            eq *= 1 + trade(-1, entry, p); pos = 0
        if pos == 0 and (max_trades is None or n < max_trades):
            if p > ub and p > vwap:
                pos, entry, n = 1, p, n + 1
            elif not long_only and p < lb and p < vwap:
                pos, entry, n = -1, p, n + 1
    if pos:
        eq *= 1 + trade(pos, entry, d["c"][-1])
    return eq - 1, int(n > 0)


def _session_vwap(d):
    return np.cumsum(d["vw"] * d["v"]) / np.maximum(np.cumsum(d["v"]), 1e-9)


def vwap_trend(d, _, long_only=False):
    """Long while price is above the day's VWAP, short while below, flat at the close."""
    vw = _session_vwap(d)
    pos, entry, eq, n = 0, 0.0, 1.0, 0
    for i in range(len(d["c"]) - 1):
        p = d["c"][i]
        want = 1 if p > vw[i] else (-1 if p < vw[i] else pos)
        if long_only and want < 0:
            want = 0
        if want != pos:
            if pos:
                eq *= 1 + trade(pos, entry, p)
            pos, entry = want, p
            n += 1 if want else 0
    if pos:
        eq *= 1 + trade(pos, entry, d["c"][-1])
    return eq - 1, int(n > 0)


def trend_pullback(d, _, max_trades=4):
    """Candle City's star worker: with EMA 9 over EMA 30 and price over VWAP, buy the first bar that
    closes back above EMA 9 after dipping to it (mirror image for shorts). Stop under the dip,
    exit on a close through EMA 30, at most 4 trades a day, no entries after 3:30."""
    c, h, l, o = d["c"], d["h"], d["l"], d["o"]
    e9 = pd.Series(c).ewm(span=9, adjust=False).mean().to_numpy()
    e30 = pd.Series(c).ewm(span=30, adjust=False).mean().to_numpy()
    vw = _session_vwap(d)
    pos, entry, stop, eq, n = 0, 0.0, 0.0, 1.0, 0
    for i in range(6, len(c) - 1):
        if pos == 0:
            if n >= max_trades or d["m"][i] >= 360:
                continue
            if e9[i] > e30[i] and c[i] > vw[i] and l[i - 1] <= e9[i - 1] and c[i] > e9[i] and c[i] > c[i - 1]:
                pos, entry, stop, n = 1, c[i], min(l[i - 1], l[i]), n + 1
            elif e9[i] < e30[i] and c[i] < vw[i] and h[i - 1] >= e9[i - 1] and c[i] < e9[i] and c[i] < c[i - 1]:
                pos, entry, stop, n = -1, c[i], max(h[i - 1], h[i]), n + 1
        elif pos == 1:
            if l[i] <= stop:
                eq *= 1 + trade(1, entry, min(stop, o[i])); pos = 0
            elif c[i] < e30[i]:
                eq *= 1 + trade(1, entry, c[i]); pos = 0
        else:
            if h[i] >= stop:
                eq *= 1 + trade(-1, entry, max(stop, o[i])); pos = 0
            elif c[i] > e30[i]:
                eq *= 1 + trade(-1, entry, c[i]); pos = 0
    if pos:
        eq *= 1 + trade(pos, entry, c[-1])
    return eq - 1, int(n > 0)


STRATEGIES = {
    "Hold UPRO all day": hold_intraday,
    "Opening range 30-min (long)": orb30_long,
    "Opening range 5-min": orb5,
    "Opening range 5-min (long)": lambda d, h: orb5(d, h, long_only=True),
    "Last-half-hour momentum": intraday_mom,
    "Gap fade": gap_fade,
    "Noise-band momentum": noise_area,
    "Noise-band momentum (1 trade/day)": lambda d, h: noise_area(d, h, max_trades=1),
    "Noise-band momentum (long, 1/day)": lambda d, h: noise_area(d, h, long_only=True, max_trades=1),
    "VWAP trend": vwap_trend,
    "VWAP trend (long)": lambda d, h: vwap_trend(d, h, long_only=True),
    "EMA trend pullback (Candle City)": trend_pullback,
}


# --------------------------------------------------------------------------- engine
def run(days):
    moves = np.full((len(days), len(CHECKPOINTS)), np.nan)
    for a, d in enumerate(days):
        for k, cp in enumerate(CHECKPOINTS):
            p, _ = price_at(d, cp)
            if p is not None:
                moves[a, k] = abs(p / d["o"][0] - 1)
    out = {n: [] for n in STRATEGIES}
    traded = {n: [] for n in STRATEGIES}
    for a, d in enumerate(days):
        hist = {"sigma": np.nanmean(moves[a - 14:a], axis=0) if a >= 14 else None}
        for n, f in STRATEGIES.items():
            r, t = f(d, hist)
            out[n].append(r); traded[n].append(t)
    idx = pd.DatetimeIndex([d["date"] for d in days])
    rets = pd.DataFrame(out, index=idx)
    trades = pd.DataFrame(traded, index=idx)

    closes = pd.Series([d["c"][-1] for d in days], index=idx)
    r = closes.pct_change().fillna(0)
    lev = LEV * r - DRAG
    pos = ensemble(closes)["series"].shift(1).fillna(0)
    rets["Wick daily (current)"] = pos * lev - pos.diff().abs().fillna(0) * COST
    trades["Wick daily (current)"] = (pos.diff().abs() > 0.05).astype(int)
    rets["Buy & hold UPRO"] = lev
    trades["Buy & hold UPRO"] = 0
    return rets, trades


def metrics(r: pd.Series, t: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    vol = r.std() * np.sqrt(252)
    on = r[t > 0]
    return {
        "CAGR": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0,
        "Total": float(eq.iloc[-1] - 1),
        "Sharpe": float(r.mean() * 252 / vol) if vol else 0.0,
        "MaxDD": float((eq / eq.cummax() - 1).min()),
        "WorstDay": float(r.min()),
        "DaysTraded": float((t > 0).mean()),
        "WinRate": float((on > 0).mean()) if len(on) else 0.0,
        "AvgTrade": float(on.mean()) if len(on) else 0.0,
    }


def main():
    synthetic_mode = "--synthetic" in sys.argv
    df = synthetic() if synthetic_mode else fetch()
    days = to_days(df)
    print(f"{len(days):,} trading days, {days[0]['date'].date()} to {days[-1]['date'].date()}", flush=True)
    rets, trades = run(days)
    start = rets.index[0] + pd.Timedelta(days=400)   # warm-up for the daily strategy and noise band
    rets, trades = rets.loc[start:], trades.loc[start:]

    periods = {"Full": (None, None), "In-sample": (None, "2021-12-31"), "Out-of-sample": (OOS_START, None)}
    res = {"generated": datetime.now(ET).isoformat(timespec="minutes"),
           "range": [str(rets.index[0].date()), str(rets.index[-1].date())],
           "assumptions": {"leverage": LEV, "cost_per_side": COST, "bars": "SPY 5-minute, SIP feed"},
           "periods": {}, "yearly": {}, "equity": {}}
    for p, (a, b) in periods.items():
        res["periods"][p] = {n: metrics(rets[n].loc[a:b], trades[n].loc[a:b]) for n in rets}
    for n in rets:
        res["yearly"][n] = {str(y): float(v) for y, v in ((1 + rets[n]).groupby(rets.index.year).prod() - 1).items()}
        eq = (1 + rets[n]).cumprod().resample("ME").last()
        res["equity"][n] = {"dates": [d.strftime("%Y-%m") for d in eq.index], "values": [round(float(v), 4) for v in eq]}

    for p in periods:
        print(f"\n=== {p} ===")
        t = pd.DataFrame(res["periods"][p]).T
        print(t[["CAGR", "Sharpe", "MaxDD", "WorstDay", "DaysTraded", "WinRate"]].to_string(float_format=lambda x: f"{x:8.3f}"))
    out = ROOT / "results" / ("intraday_synthetic.json" if synthetic_mode else "intraday_results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("\nSaved", out)


if __name__ == "__main__":
    main()
