"""The Ensemble signal, computed on the S&P 500 (SPY daily closes).

Three independent, published ideas, blended and volatility-targeted:
  * 200-day trend:        long when price is above its 200-day average
  * 12-month momentum:    long when the past year's return is positive
  * RSI(2) dip-buying:    buy sharp pullbacks inside an uptrend
Output is the target exposure from 0 (all cash) to 1 (fully invested in the 3x ETF).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(s, n):
    return s.rolling(n).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def trend(px, n=200):
    return (px > sma(px, n)).astype(float)


def momentum(px, lookback=252):
    return (px / px.shift(lookback) - 1 > 0).astype(float)


def dip_buy(px, entry=10, exit_sma=5, trend_n=200):
    r, up, ex = rsi(px, 2), px > sma(px, trend_n), px > sma(px, exit_sma)
    pos, out = 0.0, []
    for i in range(len(px)):
        if pos == 0 and up.iat[i] and r.iat[i] < entry:
            pos = 1.0
        elif pos == 1 and (ex.iat[i] or not up.iat[i]):
            pos = 0.0
        out.append(pos)
    return pd.Series(out, index=px.index)


def vol_scale(px, target=0.15, n=20, cap=1.5):
    vol = px.pct_change().rolling(n).std() * np.sqrt(252)
    return (target / vol).clip(upper=cap).fillna(0)


DEFAULT_PARAMS = {"trend": 200, "mom": 252, "dip": 10, "vol": 0.15}


def ensemble(px: pd.Series, params: dict | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    t, m, d = trend(px, p["trend"]), momentum(px, p["mom"]), dip_buy(px, entry=p["dip"])
    exposure = ((t + m + d) / 3 * vol_scale(px, target=p["vol"])).clip(upper=1.0)
    return {
        "params": p,
        "exposure": round(float(exposure.iloc[-1]), 3),
        "trend": int(t.iloc[-1]),
        "momentum": int(m.iloc[-1]),
        "dip_buy": int(d.iloc[-1]),
        "series": exposure,
    }
