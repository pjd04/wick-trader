"""Multi-market Wick: does spreading the strategy across other markets beat S&P-only?

Markets are traded through ETFs (Alpaca has no futures): US large caps, Nasdaq, small caps,
international and emerging stocks, long and medium Treasuries, gold, silver, broad commodities,
oil, and the US dollar. Daily full-market closes from Alpaca, 2016-now.

Each market gets its own signal; positions are sized so every market contributes similar risk
(inverse volatility), the whole portfolio is scaled to a target volatility, and gross exposure is
capped at 3x (what 3x ETFs allow). Long or flat only. Costs: 3 bps per side on every change, plus
borrowing on exposure above 1x and ETF fees.

Usage: python research/multi_asset.py   (needs ALPACA_KEY / ALPACA_SECRET)
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "bot"))
import strategy as S  # noqa: E402

ET = S.__dict__.get("ET") or __import__("zoneinfo").ZoneInfo("America/New_York")
COST, BORROW, FEE = 0.0003, 0.03, 0.009
UNIVERSE = {
    "SPY": "US large caps", "QQQ": "Nasdaq 100", "IWM": "US small caps", "EFA": "Intl developed",
    "EEM": "Emerging markets", "TLT": "20+yr Treasuries", "IEF": "7-10yr Treasuries", "GLD": "Gold",
    "SLV": "Silver", "DBC": "Broad commodities", "USO": "Crude oil", "UUP": "US dollar",
}
EQUITY = {"SPY", "QQQ", "IWM", "EFA", "EEM"}


def fetch() -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment
    c = StockHistoricalDataClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"])
    df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=list(UNIVERSE), timeframe=TimeFrame.Day,
                                           start=datetime(2015, 1, 1, tzinfo=ET),
                                           end=datetime.now(ET) - timedelta(minutes=30),
                                           feed=DataFeed.SIP, adjustment=Adjustment.ALL)).df.reset_index()
    df["d"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    px = df.pivot_table(index="d", columns="symbol", values="close").sort_index()
    return px.ffill()


# ---------------------------------------------------------------- per-market signals (0..1)
def sig_trend_mom(p):          # 200-day trend + 12-month momentum (classic managed-futures pair)
    return (S.trend(p) + S.momentum(p)) / 2


def sig_tsmom_blend(p):        # momentum averaged over 3, 6, 9, 12 months
    return sum(S.momentum(p, L) for L in (63, 126, 189, 252)) / 4


def sig_wick(p, sym):          # the live Wick Ensemble; dip-buying only on stock markets
    if sym in EQUITY:
        return (S.trend(p) + S.momentum(p) + S.dip_buy(p)) / 3
    return (S.trend(p) + S.momentum(p)) / 2


# ---------------------------------------------------------------- portfolio engine
def portfolio(px: pd.DataFrame, signals: pd.DataFrame, target_vol: float, cap: float = 3.0, universe=None):
    cols = list(universe or px.columns)
    r = px[cols].pct_change().fillna(0)
    vol = r.rolling(63).std() * np.sqrt(252)
    raw = signals[cols] / vol.replace(0, np.nan) / len(cols)            # equal risk per market
    raw = raw.fillna(0)
    est = (raw.shift(1) * r).sum(axis=1).rolling(63).std() * np.sqrt(252)   # realized vol of the unscaled book
    scale = (target_vol / est.replace(0, np.nan)).clip(upper=50).fillna(0)
    w = raw.mul(scale, axis=0)
    gross = w.sum(axis=1)
    w = w.div(np.maximum(gross / cap, 1.0), axis=0)                    # cap total exposure at 3x
    held = w.shift(1).fillna(0)
    g = held.sum(axis=1)
    ret = (held * r).sum(axis=1) - held.diff().abs().sum(axis=1).fillna(0) * COST \
        - (g - 1).clip(lower=0) * BORROW / 252 - g * FEE / 3 / 252
    return ret, held


def single(px, sym, sig, lev=3.0):
    r = px[sym].pct_change().fillna(0)
    held = sig.shift(1).fillna(0) * lev
    ret = held * r - held.diff().abs().fillna(0) * COST - (held - 1).clip(lower=0) * BORROW / 252 - held * FEE / 3 / 252
    return ret, held.to_frame(sym)


def metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    vol = r.std() * np.sqrt(252)
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    return {"CAGR": float(eq.iloc[-1] ** (1 / yrs) - 1), "Vol": float(vol),
            "Sharpe": float(r.mean() * 252 / vol) if vol else 0.0,
            "MaxDD": float((eq / eq.cummax() - 1).min()), "WorstYear": float(yearly.min()),
            "PositiveYears": float((yearly > 0).mean())}


def main():
    px = fetch()
    print(f"{px.shape[1]} markets, {px.index[0].date()} to {px.index[-1].date()}", flush=True)
    start = px.index[0] + pd.Timedelta(days=400)            # warm-up for 200-day / 12-month signals
    print("first full date per market:", {c: str(px[c].first_valid_index().date()) for c in px}, flush=True)

    spy_wick = S.ensemble(px["SPY"])["series"]
    variants = {"Wick now: S&P only (UPRO)": single(px, "SPY", spy_wick)}
    variants["Wick on Nasdaq only (TQQQ)"] = single(px, "QQQ", S.ensemble(px["QQQ"])["series"])
    sw = pd.DataFrame({c: sig_wick(px[c], c) for c in px})
    tm = pd.DataFrame({c: sig_trend_mom(px[c]) for c in px})
    tb = pd.DataFrame({c: sig_tsmom_blend(px[c]) for c in px})
    stocks = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
    for tv in (0.20, 0.30):
        variants[f"All 12 markets, Wick signal, {int(tv*100)}% vol"] = portfolio(px, sw, tv)
        variants[f"All 12 markets, trend+momentum, {int(tv*100)}% vol"] = portfolio(px, tm, tv)
        variants[f"All 12 markets, momentum blend, {int(tv*100)}% vol"] = portfolio(px, tb, tv)
        variants[f"Stocks only (5 markets), Wick signal, {int(tv*100)}% vol"] = portfolio(px, sw, tv, universe=stocks)
    variants["Stocks + Treasuries + gold, Wick signal, 30% vol"] = portfolio(px, sw, 0.30, universe=stocks + ["TLT", "IEF", "GLD"])
    hold = px["SPY"].pct_change().fillna(0)
    variants["Buy & hold S&P (1x)"] = (hold, None)
    variants["Buy & hold 3x S&P"] = (3 * hold - 2 * BORROW / 252 - FEE / 252, None)

    out = {"generated": datetime.now(ET).isoformat(timespec="minutes"), "universe": UNIVERSE, "periods": {},
           "equal_risk": {}, "avg_exposure": {}, "yearly": {}}
    wick_vol = metrics(variants["Wick now: S&P only (UPRO)"][0].loc[start:])["Vol"]
    for label, a in [("Full (warm-up excluded)", start), ("2020-now", "2020"), ("2022-now", "2022")]:
        res = {n: metrics(r.loc[a:]) for n, (r, _) in variants.items()}
        out["periods"][label] = res
        print(f"\n=== {label} ===")
        print(pd.DataFrame(res).T.to_string(float_format=lambda x: f"{x:7.3f}"))
    # equal risk: scale each variant to the live Wick's volatility, to compare return per unit of risk
    for n, (r, _) in variants.items():
        rr = r.loc[start:]
        k = wick_vol / (rr.std() * np.sqrt(252))
        out["equal_risk"][n] = {"scale": float(k), **metrics(rr * k)}
    print("\n=== Equal risk (scaled to S&P-only Wick's volatility) ===")
    print(pd.DataFrame(out["equal_risk"]).T[["scale", "CAGR", "Sharpe", "MaxDD", "WorstYear"]].to_string(float_format=lambda x: f"{x:7.3f}"))
    for n, (r, held) in variants.items():
        rr = r.loc[start:]
        out["yearly"][n] = {str(y): float(v) for y, v in ((1 + rr).groupby(rr.index.year).prod() - 1).items()}
        if held is not None:
            out["avg_exposure"][n] = {c: float(v) for c, v in held.loc[start:].mean().items() if v > 0.005}
    print("\nAverage exposure by market (best multi-market variant shown in JSON)")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "multi_asset_results.json").write_text(json.dumps(out, indent=1))
    print("saved results/multi_asset_results.json")


if __name__ == "__main__":
    main()
