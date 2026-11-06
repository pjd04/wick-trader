"""Strategy lab v2: momentum, swing, mean reversion, scalping and machine learning, side by side.

Daily strategies run on 75 years of S&P closes (S&P 500 index 1950-2016, then SPY from Alpaca).
Intraday strategies run on SPY 5-minute bars from Alpaca, 2016 onward, and are flat by the close.
Everything is traded as 3x (UPRO long, SPXU for shorts) with costs. Machine-learning models are
trained walk-forward: each year is predicted by a model that has only seen earlier years.

Rules follow the quantvps.com "trading bot strategies" article where it gives them.
Usage: python research/strategy_lab.py            (needs ALPACA_KEY / ALPACA_SECRET)
       python research/strategy_lab.py --synthetic (offline smoke test)
"""
from __future__ import annotations
import json
import sys
import time as clock
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "bot"))
import intraday_lab as IL          # noqa: E402  (data loading, day structure, noise-band strategy)
from strategy import ensemble      # noqa: E402

ET = IL.ET
LEV, COST = 3.0, 0.0003
DRAG = 0.009 / 252 + 2 * 0.03 / 252
SYN = "--synthetic" in sys.argv


# =========================================================================== data
def daily_closes() -> pd.Series:
    old = pd.read_csv(ROOT / "data" / "spx_1950_2016.csv", parse_dates=["date"]).set_index("date")["close"]
    if SYN:
        return old
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment
    import os
    c = StockHistoricalDataClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"])
    df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
                                           start=datetime(2016, 1, 1, tzinfo=ET),
                                           end=datetime.now(ET) - timedelta(minutes=30),
                                           feed=DataFeed.SIP, adjustment=Adjustment.ALL)).df.reset_index()
    spy = pd.Series(df["close"].to_numpy(float),
                    index=pd.to_datetime(df["timestamp"]).dt.tz_convert(ET).dt.tz_localize(None).dt.normalize())
    spy = spy[~spy.index.duplicated(keep="last")]
    cut = old.index[-1]
    tail = spy[spy.index > cut]
    scale = old.iloc[-1] / spy[spy.index <= cut].iloc[-1]
    return pd.concat([old, tail * scale])


# =========================================================================== indicators
def sma(s, n): return s.rolling(n).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr_close(s, n=14):   # close-only ATR (the 1950-2016 file has no highs/lows)
    return s.diff().abs().rolling(n).mean()


# =========================================================================== daily strategies
def tsmom(c, L):
    return (c / c.shift(L) - 1 > 0).astype(float)


def momentum_blend(c, monthly=False):
    sig = sum(tsmom(c, L) for L in (63, 126, 189, 252)) / 4
    if monthly:
        me = c.groupby([c.index.year, c.index.month]).transform(lambda x: x.index == x.index[-1]).astype(bool)
        sig = sig.where(me).ffill().fillna(0)
    return sig


def ma_cross_trail(c):
    """Article #1: long when 50-day > 200-day; 3x ATR trailing stop; re-enter on a new 20-day high."""
    s50, s200, atr, hi20 = sma(c, 50), sma(c, 200), atr_close(c), c.rolling(20).max()
    pos, peak, out, stopped = 0, 0.0, [], False
    for i in range(len(c)):
        p = c.iat[i]
        up = s50.iat[i] > s200.iat[i]
        if pos:
            peak = max(peak, p)
            if not up or p < peak - 3 * atr.iat[i]:
                pos, stopped = 0, up
        elif up and (not stopped or p >= hi20.iat[i]):
            pos, peak, stopped = 1, p, False
        if not up:
            stopped = False
        out.append(pos)
    return pd.Series(out, index=c.index, dtype=float)


def swing(c):
    """Article #9: uptrend (above 50-day), RSI(14) turns up through 40; 2R target, 1.5 ATR stop, 20-day time stop."""
    s50, r, atr = sma(c, 50), rsi(c, 14), atr_close(c)
    pos, stop, tgt, age, out = 0, 0.0, 0.0, 0, []
    for i in range(1, len(c)):
        p = c.iat[i]
        if pos:
            age += 1
            if p <= stop or p >= tgt or age >= 20:
                pos = 0
        elif p > s50.iat[i] and r.iat[i - 1] < 40 <= r.iat[i] and atr.iat[i] > 0:
            risk = 1.5 * atr.iat[i]
            pos, stop, tgt, age = 1, p - risk, p + 2 * risk, 0
        out.append(pos)
    return pd.Series([0] + out, index=c.index, dtype=float)


def bollinger_mr(c):
    """Article #2: buy a close below the lower band (20, 2) or RSI(14) < 30; exit at the middle band; 1.5 ATR stop."""
    mid, sd, r, atr = sma(c, 20), c.rolling(20).std(), rsi(c, 14), atr_close(c)
    pos, stop, out = 0, 0.0, []
    for i in range(len(c)):
        p = c.iat[i]
        if pos and (p >= mid.iat[i] or p <= stop):
            pos = 0
        elif not pos and (p < mid.iat[i] - 2 * sd.iat[i] or r.iat[i] < 30):
            pos, stop = 1, p - 1.5 * atr.iat[i]
        out.append(pos)
    return pd.Series(out, index=c.index, dtype=float)


def vol_squeeze(c):
    """Article #17: band width at a 120-day low arms a 10-day window; buy a close above the upper band;
    exit below the middle band or on a 2 ATR stop."""
    mid, sd, atr = sma(c, 20), c.rolling(20).std(), atr_close(c)
    width = 4 * sd / mid
    low = width <= width.rolling(120).min() * 1.0001
    pos, stop, armed, out = 0, 0.0, 0, []
    for i in range(len(c)):
        p = c.iat[i]
        armed = 10 if low.iat[i] else max(0, armed - 1)
        if pos and (p < mid.iat[i] or p <= stop):
            pos = 0
        elif not pos and armed and p > mid.iat[i] + 2 * sd.iat[i]:
            pos, stop = 1, p - 2 * atr.iat[i]
        out.append(pos)
    return pd.Series(out, index=c.index, dtype=float)


# ---- machine learning (daily) -----------------------------------------------------------
def daily_features(c: pd.DataFrame) -> pd.DataFrame:
    r = c.pct_change()
    f = pd.DataFrame(index=c.index)
    for n in (1, 2, 5, 10, 21, 63, 126, 252):
        f[f"ret{n}"] = c / c.shift(n) - 1
    for n in (10, 50, 200):
        f[f"sma{n}"] = c / sma(c, n) - 1
    f["rsi2"], f["rsi14"] = rsi(c, 2), rsi(c, 14)
    mid, sd = sma(c, 20), c.rolling(20).std()
    f["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
    f["bb_width"] = 4 * sd / mid
    for n in (10, 21, 63):
        f[f"vol{n}"] = r.rolling(n).std()
    f["vol_ratio"] = f["vol10"] / f["vol63"]
    f["dd252"] = c / c.rolling(252).max() - 1
    f["dow"], f["month"] = c.index.dayofweek, c.index.month
    up = (r > 0).astype(int)
    f["streak"] = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    f["streak"] *= np.where(up == 1, 1, -1)
    return f


def walk_forward(X: pd.DataFrame, y: pd.Series, make_model, first_year: int, step_years: int = 1) -> pd.Series:
    """Predict each block of years with a model trained only on earlier data."""
    p = pd.Series(np.nan, index=X.index)
    years = sorted(set(X.index.year))
    ok = X.notna().all(axis=1) & y.notna()
    for y0 in range(first_year, years[-1] + 1, step_years):
        train = ok & (X.index.year < y0)
        test = X.notna().all(axis=1) & (X.index.year >= y0) & (X.index.year < y0 + step_years)
        train.iloc[np.where(train)[0][-1:]] = False      # its label falls in the test block
        if train.sum() < 500 or test.sum() == 0:
            continue
        m = make_model()
        m.fit(X[train], y[train])
        p[test] = m.predict_proba(X[test])[:, 1]
    return p


def boosted():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=100, l2_regularization=1.0, random_state=1)


def forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=50, max_features="sqrt",
                                  n_jobs=-1, random_state=1)


def bt_daily(c, pos, cost=COST):
    r = c.pct_change().fillna(0)
    lev = LEV * r - DRAG
    held = pos.shift(1).fillna(0)
    return held * lev - held.diff().abs().fillna(0) * cost, (held > 0).astype(int)


# =========================================================================== intraday strategies
def concat_days(days):
    """Indicators computed on the continuous 5-minute series, then read per day."""
    c = np.concatenate([d["c"] for d in days]); h = np.concatenate([d["h"] for d in days])
    l = np.concatenate([d["l"] for d in days])
    s = pd.Series(c)
    mid, sd = s.rolling(20).mean(), s.rolling(20).std()
    prev = s.shift(1).fillna(s)
    tr = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
    atr = pd.Series(tr).rolling(14).mean()
    r14 = rsi(s, 14)
    off = 0
    for d in days:
        n = len(d["c"])
        d["mid"], d["sd"] = mid.iloc[off:off + n].to_numpy(), sd.iloc[off:off + n].to_numpy()
        d["atr"], d["rsi14"] = atr.iloc[off:off + n].to_numpy(), r14.iloc[off:off + n].to_numpy()
        cum_pv, cum_v = np.cumsum(d["vw"] * d["v"]), np.cumsum(d["v"])
        d["vwap"] = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, 1e-9), d["c"])
        off += n


def tr(side, entry, exit_):
    return side * LEV * (exit_ / entry - 1) - 2 * COST


def scalp_vwap(d, _, both=True, max_trades=None, tp=0.0015, sl=0.0015, hold=6):
    """Article #7 adapted to 5-minute bars: in a trending day, buy a pullback that touches VWAP and
    closes back above it (mirror for shorts). Target and stop 0.15% on SPY (0.45% on the 3x ETF),
    30-minute time stop, flat by the close."""
    o, h, l, c, vw, m = d["o"], d["h"], d["l"], d["c"], d["vwap"], d["m"]
    n, eq, trades, i = len(c), 1.0, 0, 6
    while i < n - 2 and m[i] <= 375:
        side = 0
        if c[i] > o[0] and l[i] <= vw[i] < c[i] and c[i - 1] > vw[i - 1]:
            side = 1
        elif both and c[i] < o[0] and h[i] >= vw[i] > c[i] and c[i - 1] < vw[i - 1]:
            side = -1
        if side == 0 or (max_trades is not None and trades >= max_trades):
            i += 1
            continue
        e = o[i + 1]
        tgt, stp = e * (1 + side * tp), e * (1 - side * sl)
        x, j = None, i + 1
        while j < n:
            if side == 1:
                if l[j] <= stp: x = min(stp, o[j]); break
                if h[j] >= tgt: x = max(tgt, o[j]); break
            else:
                if h[j] >= stp: x = max(stp, o[j]); break
                if l[j] <= tgt: x = min(tgt, o[j]); break
            if j - i >= hold: x = c[j]; break
            j += 1
        if x is None:
            x, j = c[-1], n - 1
        eq *= 1 + tr(side, e, x)
        trades += 1
        i = j + 1
    return eq - 1, int(trades > 0)


def scalp_bollinger(d, _):
    """Article #2 on 5-minute bars: close below the lower band with RSI(14) < 30 -> buy next bar;
    exit at the middle band or a 1.5 ATR stop (mirror for shorts); flat by the close."""
    o, h, l, c, mid, sd, atr, r = d["o"], d["h"], d["l"], d["c"], d["mid"], d["sd"], d["atr"], d["rsi14"]
    n, eq, trades, i = len(c), 1.0, 0, 3
    while i < n - 2:
        if np.isnan(mid[i]) or np.isnan(atr[i]):
            i += 1; continue
        side = 1 if (c[i] < mid[i] - 2 * sd[i] and r[i] < 30) else -1 if (c[i] > mid[i] + 2 * sd[i] and r[i] > 70) else 0
        if not side:
            i += 1; continue
        e = o[i + 1]
        stp = e - side * 1.5 * atr[i]
        x, j = None, i + 1
        while j < n:
            if side == 1 and l[j] <= stp: x = min(stp, o[j]); break
            if side == -1 and h[j] >= stp: x = max(stp, o[j]); break
            if (side == 1 and c[j] >= mid[j]) or (side == -1 and c[j] <= mid[j]): x = c[j]; break
            j += 1
        if x is None:
            x, j = c[-1], n - 1
        eq *= 1 + tr(side, e, x)
        trades += 1
        i = j + 1
    return eq - 1, int(trades > 0)


def orb_article(d, _):
    """Article #10: 30-minute opening range; trade the first break; stop at the range midpoint,
    target 2x the range; flat by the close."""
    first = d["m"] < 30
    hi, lo = d["h"][first].max(), d["l"][first].min()
    rng, midp = hi - lo, (hi + lo) / 2
    if rng <= 0:
        return 0.0, 0
    for i in np.where(~first)[0]:
        side = 1 if d["h"][i] > hi else -1 if d["l"][i] < lo else 0
        if not side:
            continue
        e = max(hi, d["o"][i]) if side == 1 else min(lo, d["o"][i])
        tgt = e + side * 2 * rng
        for j in range(i + 1, len(d["c"])):
            if side == 1:
                if d["l"][j] <= midp: return tr(1, e, min(midp, d["o"][j])), 1
                if d["h"][j] >= tgt: return tr(1, e, max(tgt, d["o"][j])), 1
            else:
                if d["h"][j] >= midp: return tr(-1, e, max(midp, d["o"][j])), 1
                if d["l"][j] <= tgt: return tr(-1, e, min(tgt, d["o"][j])), 1
        return tr(side, e, d["c"][-1]), 1
    return 0.0, 0


# ---- machine learning (intraday): every 30 minutes, predict the next 30 minutes --------------
ML_CP = list(range(30, 331, 30))   # 10:00 ... 3:00 (each trade ends 30 minutes later)


def intraday_samples(days):
    rows = []
    for a, d in enumerate(days):
        o0, prev = d["o"][0], d["prev"]
        day_r = np.diff(d["c"]) / d["c"][:-1]
        prior = days[a - 1] if a else None
        for cp in ML_CP:
            p, i = IL.price_at(d, cp)
            p_next, _ = IL.price_at(d, cp + 30)
            p_back, _ = IL.price_at(d, cp - 30) if cp > 30 else (o0, None)
            if p is None or p_next is None or p_back is None:
                continue
            rows.append({
                "date": d["date"], "cp": cp, "price": p, "next": p_next,
                "r_open": p / o0 - 1, "r_prev": p / prev - 1, "r_30": p / p_back - 1,
                "gap": o0 / prev - 1, "rsi14": d["rsi14"][i], "bb": (p - d["mid"][i]) / (2 * d["sd"][i]) if d["sd"][i] else 0,
                "vwap": p / d["vwap"][i] - 1, "rvol": day_r[:i].std() if i > 2 else np.nan,
                "atr": d["atr"][i] / p, "prior_day": (prior["c"][-1] / prior["o"][0] - 1) if prior else np.nan,
                "dow": d["date"].dayofweek, "y": int(p_next > p),
            })
    return pd.DataFrame(rows)


def intraday_ml(days, lo=0.45, hi=0.55):
    s = intraday_samples(days)
    feats = ["cp", "r_open", "r_prev", "r_30", "gap", "rsi14", "bb", "vwap", "rvol", "atr", "prior_day", "dow"]
    X = s[feats].set_index(s["date"])
    y = s["y"].set_axis(X.index)
    first = sorted(set(X.index.year))[2]
    p = walk_forward(X, y, boosted, first)
    s["p"] = p.to_numpy()
    s = s.dropna(subset=["p"])
    acc = float(((s.p > 0.5) == (s.y == 1)).mean())
    out = {}
    for name, (l, h) in {"Intraday AI, boosted trees (55/45)": (lo, hi), "Intraday AI, boosted trees (60/40)": (0.4, 0.6)}.items():
        side = np.where(s.p > h, 1, np.where(s.p < l, -1, 0))
        r = np.where(side != 0, side * LEV * (s.next / s.price - 1) - 2 * COST, 0.0)
        g = pd.DataFrame({"r": r, "t": side != 0}, index=s.date).groupby(level=0)
        out[name] = (g.r.apply(lambda x: float(np.prod(1 + x) - 1)), g.t.max().astype(int))
    base = float(s.y.mean())
    return out, {"accuracy": acc, "base_rate_up": base, "samples": int(len(s))}


# =========================================================================== metrics
def metrics(r: pd.Series, t: pd.Series) -> dict:
    r, t = r.fillna(0), t.reindex(r.index).fillna(0)
    if len(r) < 2:
        return {k: 0.0 for k in ("CAGR", "Sharpe", "MaxDD", "WorstYear", "InMarket", "WinRate")}
    eq = (1 + r).cumprod()
    yrs = max(len(r) / 252, 1e-9)
    vol = r.std() * np.sqrt(252)
    on = r[t > 0]
    return {"CAGR": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0,
            "Sharpe": float(r.mean() * 252 / vol) if vol else 0.0,
            "MaxDD": float((eq / eq.cummax() - 1).min()),
            "WorstYear": float(((1 + r).groupby(r.index.year).prod() - 1).min()),
            "InMarket": float((t > 0).mean()), "WinRate": float((on > 0).mean()) if len(on) else 0.0}


def table(res, title, cols=("CAGR", "Sharpe", "MaxDD", "WorstYear", "InMarket")):
    print(f"\n=== {title} ===")
    print(pd.DataFrame(res).T[list(cols)].to_string(float_format=lambda x: f"{x:8.3f}"))


# =========================================================================== main
def main():
    t0 = clock.time()
    out = {"generated": datetime.now(ET).isoformat(timespec="minutes"),
           "assumptions": {"leverage": LEV, "cost_per_side": COST}, "daily": {}, "intraday": {}, "ml": {}}

    # ---------------- daily
    c = daily_closes()
    print(f"daily closes {c.index[0].date()} to {c.index[-1].date()} ({len(c):,})", flush=True)
    sigs = {
        "Wick now (Ensemble)": ensemble(c)["series"],
        "Momentum 3-month": tsmom(c, 63), "Momentum 6-month": tsmom(c, 126),
        "Momentum 9-month": tsmom(c, 189), "Momentum 12-month": tsmom(c, 252),
        "Momentum blend 3-12m": momentum_blend(c), "Momentum blend, monthly": momentum_blend(c, monthly=True),
        "50/200 cross + ATR trail": ma_cross_trail(c), "Swing pullback (RSI 40)": swing(c),
        "Bollinger mean reversion": bollinger_mr(c), "Volatility squeeze breakout": vol_squeeze(c),
    }
    X = daily_features(c)
    y = (c.pct_change().shift(-1) > 0).astype(float).where(c.pct_change().shift(-1).notna())
    first_ml = 1970
    for name, mk in {"AI boosted trees": boosted, "AI random forest": forest}.items():
        p = walk_forward(X, y, mk, first_ml)
        oos = p.notna() & y.notna()
        out["ml"][name] = {"accuracy": float(((p[oos] > 0.5) == (y[oos] == 1)).mean()),
                           "base_rate_up": float(y[oos].mean()), "days": int(oos.sum())}
        for th in (0.5, 0.55, 0.6):
            sigs[f"{name} (>{int(th * 100)}%)"] = (p > th).astype(float)
        sigs[f"{name} + Wick filter"] = ensemble(c)["series"] * (p > 0.5).astype(float)
        print(f"{name}: walk-forward done ({clock.time() - t0:.0f}s), "
              f"accuracy {out['ml'][name]['accuracy']:.3f} vs {out['ml'][name]['base_rate_up']:.3f} up-days", flush=True)
    rets = {n: bt_daily(c, s.reindex(c.index).fillna(0)) for n, s in sigs.items()}
    rets["Buy & hold 3x"] = (LEV * c.pct_change().fillna(0) - DRAG, pd.Series(1, index=c.index))
    for label, a, b in [("1970-now", "1970", None), ("2000-now", "2000", None), ("2016-now", "2016", None)]:
        res = {n: metrics(r.loc[a:b], t.loc[a:b]) for n, (r, t) in rets.items()}
        out["daily"][label] = res
        table(res, f"DAILY {label}")
    out["daily_yearly"] = {n: {str(k): float(v) for k, v in ((1 + r.loc["2016":]).groupby(r.loc["2016":].index.year).prod() - 1).items()}
                           for n, (r, t) in rets.items()}

    # ---------------- intraday
    df = IL.synthetic(days=700) if SYN else IL.fetch()
    days = IL.to_days(df)
    concat_days(days)
    print(f"intraday days {days[0]['date'].date()} to {days[-1]['date'].date()} ({len(days):,}) "
          f"[{clock.time() - t0:.0f}s]", flush=True)
    moves = np.full((len(days), len(IL.CHECKPOINTS)), np.nan)
    for a, d in enumerate(days):
        for k, cp in enumerate(IL.CHECKPOINTS):
            p, _ = IL.price_at(d, cp)
            if p is not None:
                moves[a, k] = abs(p / d["o"][0] - 1)
    strategies = {
        "Scalp VWAP bounce (long+short)": scalp_vwap,
        "Scalp VWAP bounce (long only)": lambda d, h: scalp_vwap(d, h, both=False),
        "Scalp VWAP bounce (1 trade/day)": lambda d, h: scalp_vwap(d, h, max_trades=1),
        "Scalp Bollinger reversion": scalp_bollinger,
        "Opening range breakout (article rules)": orb_article,
        "Noise-band momentum (previous best)": IL.noise_area,
    }
    ir, it = {n: [] for n in strategies}, {n: [] for n in strategies}
    for a, d in enumerate(days):
        hist = {"sigma": np.nanmean(moves[a - 14:a], axis=0) if a >= 14 else None}
        for n, f in strategies.items():
            r, t = f(d, hist)
            ir[n].append(r); it[n].append(t)
    idx = pd.DatetimeIndex([d["date"] for d in days])
    iret = {n: (pd.Series(ir[n], index=idx), pd.Series(it[n], index=idx)) for n in strategies}
    ml, ml_stats = intraday_ml(days)
    iret.update(ml)
    out["ml"]["Intraday AI, boosted trees"] = ml_stats
    print(f"intraday AI accuracy {ml_stats['accuracy']:.3f} vs {ml_stats['base_rate_up']:.3f} up-moves", flush=True)
    for label, a in [("2018-now", "2018"), ("2022-now", "2022")]:
        res = {n: metrics(r.loc[a:], t.loc[a:]) for n, (r, t) in iret.items()}
        res["Wick now (Ensemble, daily)"] = metrics(*[x.loc[a:] for x in rets["Wick now (Ensemble)"]])
        out["intraday"][label] = res
        table(res, f"INTRADAY {label}", cols=("CAGR", "Sharpe", "MaxDD", "WorstYear", "InMarket", "WinRate"))

    name = "lab_v2_synthetic.json" if SYN else "lab_v2_results.json"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / name).write_text(json.dumps(out, indent=1))
    print(f"\nSaved results/{name} in {clock.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
