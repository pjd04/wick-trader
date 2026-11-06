"""Wick's nightly review. Runs once after each close.

1. Report card: grades every decision by what the market did over the next 1, 5 and 20 trading days.
2. Shadow learner: a self-tuning copy of the strategy re-picks its four settings each quarter from the
   last 10 years (the best-behaved self-tuning variant in research/self_tune.py). It trades only on
   paper, inside this script. If, after at least 126 trading days (~6 months), the shadow beats the
   live settings by a clear margin, its settings are promoted to live (state/params.json).

Research note: in walk-forward tests on 1970-2026, self-tuning did NOT beat the fixed settings, so
promotion is meant to be rare. The bar is high on purpose.
"""
from __future__ import annotations
import itertools
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import strategy as S

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
LEARN = ROOT / "state" / "learning.json"
PARAMS = ROOT / "state" / "params.json"
REPORT = ROOT / "docs" / "learning.json"
OLD = ROOT / "research" / "data" / "spx_1950_2016.csv"

GRID = {"trend": [150, 200, 250], "mom": [126, 189, 252], "dip": [5, 10, 15], "vol": [0.10, 0.15, 0.20]}
LEV, COST, DRAG = 3.0, 0.0003, 0.009 / 252 + 2 * 0.03 / 252
WINDOW_YEARS, MIN_DAYS, SHARPE_MARGIN = 10, 126, 0.30


def key(p: dict) -> str:
    return f"{p['trend']}/{p['mom']}/{p['dip']}/{p['vol']}"


def unkey(k: str) -> dict:
    a, b, c, d = k.split("/")
    return {"trend": int(a), "mom": int(b), "dip": int(c), "vol": float(d)}


def live_params() -> dict:
    try:
        return {**S.DEFAULT_PARAMS, **json.loads(PARAMS.read_text())["params"]}
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return dict(S.DEFAULT_PARAMS)


def long_closes(broker) -> pd.Series:
    """S&P 500 closes 1950-2016 spliced onto full-market SPY closes 2016-now (as in the research)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment
    from datetime import timedelta
    old = pd.read_csv(OLD, parse_dates=["date"]).set_index("date")["close"]
    df = broker.data.get_stock_bars(StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame.Day, start=datetime(2016, 1, 1, tzinfo=ET),
        end=datetime.now(ET) - timedelta(minutes=16), feed=DataFeed.SIP, adjustment=Adjustment.ALL)).df.reset_index()
    spy = pd.Series(df["close"].to_numpy(float),
                    index=pd.to_datetime(df["timestamp"]).dt.tz_convert(ET).dt.tz_localize(None).dt.normalize())
    spy = spy[~spy.index.duplicated(keep="last")]
    cut = old.index[-1]
    return pd.concat([old, spy[spy.index > cut] * (old.iloc[-1] / spy[spy.index <= cut].iloc[-1])])


def grid_returns(c: pd.Series) -> pd.DataFrame:
    r = c.pct_change().fillna(0)
    lev = LEV * r - DRAG
    t = {n: S.trend(c, n) for n in GRID["trend"]}
    m = {n: S.momentum(c, n) for n in GRID["mom"]}
    d = {n: S.dip_buy(c, entry=n) for n in GRID["dip"]}
    v = {n: S.vol_scale(c, target=n) for n in GRID["vol"]}
    cols = {}
    for a, b, e, f in itertools.product(*GRID.values()):
        held = ((t[a] + m[b] + d[e]) / 3 * v[f]).clip(upper=1.0).shift(1).fillna(0)
        cols[f"{a}/{b}/{e}/{f}"] = held * lev - held.diff().abs().fillna(0) * COST
    return pd.DataFrame(cols)


def best_setting(R: pd.DataFrame, asof: pd.Timestamp) -> str:
    hist = R.loc[asof - pd.Timedelta(days=int(365.25 * WINDOW_YEARS)):asof]
    sh = hist.mean() / hist.std().replace(0, np.nan)
    return str(sh.idxmax())


def stats(r: pd.Series) -> dict:
    if len(r) < 2:
        return {"total": 0.0, "sharpe": 0.0}
    sd = r.std()
    return {"total": float((1 + r).prod() - 1), "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd else 0.0}


def report_card(actions: list, c: pd.Series) -> list:
    """Grade each buy or sell by the 3x S&P move over the next 1, 5 and 20 trading days.
    A buy is a good call if the market rose afterwards; a sell is good if it fell. Holds aren't graded."""
    r3 = 1 + LEV * c.pct_change().fillna(0)
    rows = []
    for a in actions:
        act = a.get("action", "")
        side = 1 if "buy" in act else -1 if "sell" in act else 0
        row = {"date": a["date"], "action": act, "amount": a.get("amount", 0), "exposure": a.get("exposure")}
        d = pd.Timestamp(a["date"])
        if d in c.index:
            i = c.index.get_loc(d)
            for k in (1, 5, 20):
                if i + k < len(c):
                    fwd = float(r3.iloc[i + 1:i + 1 + k].prod() - 1)
                    row[f"after_{k}d"] = round(fwd, 4)
                    row[f"good_{k}d"] = bool(side * fwd > 0) if side else None
        rows.append(row)
    return rows


def nightly(broker, state: dict, notify=lambda *a, **k: None, now: datetime | None = None) -> dict | None:
    now = now or datetime.now(ET)
    today = now.date().isoformat()
    L = json.loads(LEARN.read_text()) if LEARN.exists() else {"since": None, "picks": [], "last_review": None, "promotions": []}
    if L.get("last_review") == today:
        return None

    c = long_closes(broker)
    R = grid_returns(c)
    last = c.index[-1]
    live = key(live_params())

    # the shadow re-picks once per quarter, using only data up to that day
    q = f"{last.year}Q{(last.month - 1) // 3 + 1}"
    if not L["picks"] or L["picks"][-1][2] != q:
        L["picks"].append([str(last.date()), best_setting(R, last), q])
    L["since"] = L["since"] or str(last.date())

    since = pd.Timestamp(L["since"])
    span = R.index[R.index > since]
    shadow = pd.Series(0.0, index=span)
    for d0, k, _ in L["picks"]:
        shadow.loc[shadow.index > pd.Timestamp(d0)] = R.loc[span[span > pd.Timestamp(d0)], k].to_numpy()
    live_r = R.loc[span, live]
    s_live, s_shadow = stats(live_r), stats(shadow)
    current_pick = L["picks"][-1][1]

    promoted = None
    if (len(span) >= MIN_DAYS and current_pick != live
            and s_shadow["sharpe"] > s_live["sharpe"] + SHARPE_MARGIN and s_shadow["total"] > s_live["total"]):
        promoted = current_pick
        PARAMS.write_text(json.dumps({"params": unkey(promoted), "promoted_on": today,
                                      "evidence": {"days": len(span), "live": s_live, "shadow": s_shadow}}, indent=1))
        L["promotions"].append({"date": today, "from": live, "to": promoted})
        L["since"], L["picks"] = str(last.date()), [[str(last.date()), promoted, q]]
        notify("Wick learned something", f"New settings {promoted} beat {live} over {len(span)} days. Using them from tomorrow.")

    L["last_review"] = today
    LEARN.write_text(json.dumps(L, indent=1))

    card = report_card(state.get("actions", []), c)
    graded = [r for r in card if r.get("good_5d") is not None]
    report = {
        "updated": now.isoformat(timespec="minutes"),
        "live_settings": live, "shadow_settings": current_pick,
        "shadow_since": L["since"], "days": len(span), "days_needed": MIN_DAYS,
        "live": s_live, "shadow": s_shadow, "promotions": L["promotions"], "just_promoted": promoted,
        "report_card": card[-12:],
        "hit_rate_5d": (sum(r["good_5d"] for r in graded) / len(graded)) if graded else None,
        "graded": len(graded),
        "research": {"fixed_2016_now": 0.194, "best_self_tune_2016_now": 0.189, "daily_self_tune_2016_now": 0.152},
    }
    REPORT.write_text(json.dumps(report, indent=1))
    return report
