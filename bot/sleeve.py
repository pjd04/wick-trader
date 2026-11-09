"""Active sleeve: day-trades a slice of the account with noise-band momentum
(Zarattini, Aziz & Barbon 2024, "Beat the Market"), the best intraday strategy in research/.

* Signal on SPY. Each day the "noise band" is how far SPY usually moves from its open by each time
  of day (average of the last 14 days). A break above the band (and above VWAP) is a long; a break
  below it (and below VWAP) is a short. Exit when price falls back inside the band or crosses VWAP.
* Long = buy SPXL (3x S&P). Short = buy SPXS (inverse 3x S&P), so no short selling is needed.
  Separate funds from Wick's UPRO, so the two never mix.
* Checked every 30 minutes, 10:00 to 3:30 ET. At most one entry a day. Always flat by ~3:50 ET.
* Small-account rule: no new entry once 3 day trades were made in the last 5 business days.
"""
from __future__ import annotations
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as cfg

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state" / "sleeve.json"
CHECKPOINTS = list(range(30, 361, 30))     # minutes after 9:30 -> 10:00 ... 3:30
FLATTEN_AT = 378                           # 3:48 pm ET
CHECK_WINDOW = 25                          # act on a checkpoint only if we're within 25 min of it


def load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"day": None, "pos": None, "day_trades": [], "trades": [], "pnl_total": 0.0}


def save(s: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def day_trades_recent(s: dict, today) -> int:
    """Day trades in the last 5 business days, today included."""
    start = (pd.Timestamp(today) - pd.offsets.BDay(4)).date().isoformat()
    return sum(1 for d in s["day_trades"] if d >= start)


# ------------------------------------------------------------------ market data
def bars(broker, timeframe_min: int, start, end=None, sip=True) -> pd.DataFrame:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    req = StockBarsRequest(symbol_or_symbols="SPY", timeframe=TimeFrame(timeframe_min, TimeFrameUnit.Minute),
                           start=start, end=end, feed=DataFeed.SIP if sip else DataFeed.IEX)
    df = broker.data.get_stock_bars(req).df
    if not len(df):
        return pd.DataFrame()
    df = df.reset_index()
    df["t"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
    df["m"] = (df.t.dt.hour - 9) * 60 + df.t.dt.minute - 30
    return df[(df.m >= 0) & (df.m < 390)]


def prepare_day(broker, now: datetime) -> dict:
    """Noise band (sigma per checkpoint) from the last 14 sessions, and yesterday's close."""
    hist = bars(broker, 5, now - timedelta(days=35), now - timedelta(minutes=16))
    hist = hist[hist.t.dt.date < now.date()]
    sig_rows, prev = [], None
    for _, g in hist.groupby(hist.t.dt.date):
        if g.m.iloc[0] != 0:
            continue
        o = float(g.open.iloc[0])
        row = []
        for cp in CHECKPOINTS:
            hit = g[g.m == cp - 5]
            row.append(abs(float(hit.close.iloc[0]) / o - 1) if len(hit) else np.nan)
        sig_rows.append(row)
        prev = float(g.close.iloc[-1])
    sigma = np.nanmean(np.array(sig_rows[-14:]), axis=0) if len(sig_rows) >= 14 else None
    return {"sigma": None if sigma is None else [round(float(x), 6) for x in sigma], "prev": prev}


def today_open_vwap(broker, now: datetime):
    """Today's official open (full-market data, available ~16 min after the open) and running VWAP."""
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    open_px = None
    try:
        sip = bars(broker, 1, start, now - timedelta(minutes=16))
        if len(sip) and sip.m.iloc[0] == 0:
            open_px = float(sip.open.iloc[0])
    except Exception:
        pass
    iex = bars(broker, 1, start, None, sip=False)
    if open_px is None and len(iex):
        open_px = float(iex.open.iloc[0])
    vwap = float((iex.vwap * iex.volume).sum() / iex.volume.sum()) if len(iex) and iex.volume.sum() > 0 else None
    return open_px, vwap


# ------------------------------------------------------------------ orders
def market(broker, symbol: str, qty: int, buy: bool):
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    o = broker.trade.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY if buy else OrderSide.SELL,
                                                     time_in_force=TimeInForce.DAY))
    for _ in range(20):
        time.sleep(1)
        o = broker.trade.get_order_by_id(o.id)
        if o.filled_avg_price:
            return float(o.filled_avg_price)
    return None


def close_position(broker, s: dict, today: str, reason: str, notify, log) -> None:
    pos = s["pos"]
    held = broker.position(pos["symbol"])["qty"]
    qty = int(min(pos["qty"], held)) if held else 0
    px = market(broker, pos["symbol"], qty, buy=False) if qty > 0 else None
    px = px or broker.last_price(pos["symbol"])
    pnl = (px - pos["entry"]) * pos["qty"]
    s["pnl_total"] = round(s["pnl_total"] + pnl, 2)
    if pos["date"] == today:
        s["day_trades"] = (s["day_trades"] + [today])[-20:]
    s["trades"] = (s["trades"] + [{**pos, "exit": round(px, 4), "exit_time": datetime.now(ET).strftime("%H:%M"),
                                   "pnl": round(pnl, 2), "reason": reason}])[-60:]
    s["pos"] = None
    log.append(f"Active sleeve: sold {pos['qty']} {pos['symbol']} at ${px:.2f} ({reason}), P&L ${pnl:+,.2f}")
    notify(f"Sleeve {'+' if pnl >= 0 else '−'}${abs(pnl):,.0f}", f"Closed {pos['side_name']} ({pos['symbol']}): {reason}", up=pnl >= 0)


# ------------------------------------------------------------------ one pass
def step(broker, now: datetime, equity: float, is_open: bool, log: list, notify=lambda *a, **k: None) -> dict:
    s = load()
    today = now.date().isoformat()
    m = (now.hour - 9) * 60 + now.minute - 30

    if is_open and s.get("day") != today:
        prep = prepare_day(broker, now)
        s.update({"day": today, "sigma": prep["sigma"], "prev": prep["prev"], "open": None,
                  "done": [], "entries_today": 0, "pnl_day_start": s.get("pnl_total", 0.0), "bands": None})

    # reconcile with the broker (e.g. a position closed by hand, or an entry whose record was lost)
    if s.get("pos") and broker.position(s["pos"]["symbol"])["qty"] <= 0:
        log.append("Active sleeve: position no longer at the broker; cleared.")
        s["pos"] = None

    if is_open and cfg.SLEEVE_ENABLED:
        if s.get("pos") and m >= FLATTEN_AT:
            close_position(broker, s, today, "end of day", notify, log)
        due = [cp for cp in CHECKPOINTS if cp <= m and cp not in s["done"]]
        if due and m < FLATTEN_AT:
            s["done"] = sorted(set(s["done"]) | set(due))
            cp = due[-1]
            if m - cp <= CHECK_WINDOW and s.get("sigma") and s.get("prev"):
                if not s.get("open"):
                    s["open"], _ = today_open_vwap(broker, now)
                _, vwap = today_open_vwap(broker, now)
                spy = broker.last_price("SPY")
                k = CHECKPOINTS.index(cp)
                sig = s["sigma"][k]
                ub = max(s["open"], s["prev"]) * (1 + sig)
                lb = min(s["open"], s["prev"]) * (1 - sig)
                vwap = vwap or spy
                s["bands"] = {"time": now.strftime("%H:%M"), "spy": round(spy, 2), "upper": round(ub, 2),
                              "lower": round(lb, 2), "vwap": round(vwap, 2)}
                pos = s.get("pos")
                if pos and pos["side"] == 1 and spy < max(ub, vwap):
                    close_position(broker, s, today, "fell back into the band", notify, log)
                elif pos and pos["side"] == -1 and spy > min(lb, vwap):
                    close_position(broker, s, today, "rose back into the band", notify, log)
                used = day_trades_recent(s, today)
                if not s.get("pos") and s["entries_today"] < 1 and cp <= 360:
                    side = 1 if (spy > ub and spy > vwap) else -1 if (spy < lb and spy < vwap) else 0
                    if side and used >= cfg.DAY_TRADE_LIMIT:
                        log.append(f"Active sleeve: signal to go {'long' if side == 1 else 'short'}, "
                                   f"but {used} day trades used in 5 days (limit {cfg.DAY_TRADE_LIMIT}). Skipped.")
                    elif side:
                        sym = cfg.SLEEVE_LONG if side == 1 else cfg.SLEEVE_SHORT
                        etf_px = broker.last_price(sym)
                        qty = math.floor(equity * cfg.SLEEVE_SHARE * 0.98 / etf_px) if etf_px else 0
                        if qty > 0:
                            fill = market(broker, sym, qty, buy=True) or etf_px
                            name = "bet up" if side == 1 else "bet down"
                            s["pos"] = {"date": today, "time": now.strftime("%H:%M"), "side": side, "side_name": name,
                                        "symbol": sym, "qty": qty, "entry": round(fill, 4), "spy_at_entry": round(spy, 2)}
                            s["entries_today"] += 1
                            log.append(f"Active sleeve: {name}, bought {qty} {sym} at ${fill:.2f} "
                                       f"(SPY {spy:.2f} broke {'above' if side == 1 else 'below'} the band)")
                            notify(f"Sleeve: {name}", f"Bought {qty} {sym} at ${fill:.2f}", up=side == 1)
                    elif cp == CHECKPOINTS[-1] and s["entries_today"] == 0:
                        log.append("Active sleeve: no breakout today; stayed in cash.")
    save(s)

    pos = s.get("pos")
    pos_value = (broker.position(pos["symbol"])["value"] if pos else 0.0)
    open_pnl = (pos_value - pos["entry"] * pos["qty"]) if pos else 0.0
    return {
        "enabled": cfg.SLEEVE_ENABLED, "share": cfg.SLEEVE_SHARE,
        "position": {**pos, "value": round(pos_value, 2), "open_pnl": round(open_pnl, 2)} if pos else None,
        "pnl_total": round(s.get("pnl_total", 0.0) + open_pnl, 2),
        "pnl_today": round(s.get("pnl_total", 0.0) - s.get("pnl_day_start", 0.0) + open_pnl, 2),
        "day_trades_used": day_trades_recent(s, today), "day_trade_limit": cfg.DAY_TRADE_LIMIT,
        "bands": s.get("bands"), "trades": s.get("trades", [])[-8:],
        "next_check": next((f"{9 + (30 + cp) // 60}:{(30 + cp) % 60:02d}" for cp in CHECKPOINTS if cp > m), None),
    }
