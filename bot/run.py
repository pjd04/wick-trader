"""One pass of the bot. GitHub Actions runs this every 15 minutes on market days.

Each pass:
  * reads the account and the market
  * if it's past DECIDE_AFTER_ET and today's decision hasn't happened, rebalances to the signal
  * writes docs/status.json, which the phone page reads
  * sends a phone notification after the daily decision (if NTFY_TOPIC is set)
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from strategy import ensemble

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state" / "state.json"
STATUS = ROOT / "docs" / "status.json"
ET = ZoneInfo("America/New_York")


# ------------------------------------------------------------------ broker
class AlpacaBroker:
    def __init__(self):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        key, secret = os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"]
        self.trade = TradingClient(key, secret, paper=cfg.PAPER)
        self.data = StockHistoricalDataClient(key, secret)

    def market_open(self) -> bool:
        return bool(self.trade.get_clock().is_open)

    def account(self) -> dict:
        a = self.trade.get_account()
        return {"equity": float(a.equity), "last_equity": float(a.last_equity), "cash": float(a.cash),
                "account": str(a.account_number)}

    def position(self, symbol) -> dict:
        for p in self.trade.get_all_positions():
            if p.symbol == symbol:
                return {"qty": float(p.qty), "value": float(p.market_value), "avg": float(p.avg_entry_price)}
        return {"qty": 0.0, "value": 0.0, "avg": 0.0}

    def daily_closes(self, symbol) -> pd.Series:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                               start=datetime.now(ET) - timedelta(days=800), feed=DataFeed.IEX)
        df = self.data.get_stock_bars(req).df.reset_index()
        s = df.set_index(pd.to_datetime(df["timestamp"]).dt.tz_convert(ET).dt.normalize())["close"]
        return s[~s.index.duplicated(keep="last")].astype(float)

    def last_price(self, symbol) -> float:
        from alpaca.data.requests import StockLatestTradeRequest
        from alpaca.data.enums import DataFeed
        t = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX))
        return float(t[symbol].price)

    def candles(self, symbol) -> dict:
        """5-minute candles for the latest session and daily candles for ~3 months, for the phone chart."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
        now = datetime.now(ET)

        def bars(tf, start):
            df = self.data.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                                                           start=start, feed=DataFeed.IEX)).df
            if not len(df):
                return pd.DataFrame()
            df = df.reset_index()
            df["t"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
            return df

        out = {"intraday": self.intraday_candles(symbol), "daily": []}
        daily = bars(TimeFrame.Day, now - timedelta(days=130))
        if len(daily):
            out["daily"] = [[r.t.strftime("%Y-%m-%d"), r.open, r.high, r.low, r.close] for r in daily.itertuples()]
        return out

    def intraday_candles(self, symbol) -> list:
        """5-minute candles for the latest session, built from 1-minute bars so the current
        candle is still forming, with the newest trade price as its close."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        df = self.data.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                                                       start=datetime.now(ET) - timedelta(days=5),
                                                       feed=DataFeed.IEX)).df
        if not len(df):
            return []
        df = df.reset_index()
        df["t"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(ET)
        mins = df.t.dt.hour * 60 + df.t.dt.minute
        df = df[(mins >= 570) & (mins < 960)]
        df = df[df.t.dt.date == df.t.dt.date.max()].set_index("t")
        five = df.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        rows = [[int(t.timestamp()), r.open, r.high, r.low, r.close] for t, r in five.iterrows()]
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            tr = self.data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX))[symbol]
            p, ts = float(tr.price), tr.timestamp.astimezone(ET)
            bucket = int(ts.replace(minute=ts.minute - ts.minute % 5, second=0, microsecond=0).timestamp())
            if rows and ts.date() == five.index[-1].date() and 570 <= ts.hour * 60 + ts.minute < 960:
                if bucket > rows[-1][0]:            # a newer candle has started: open it from the last close
                    o = rows[-1][4]
                    rows.append([bucket, o, max(o, p), min(o, p), p])
                elif bucket == rows[-1][0]:
                    rows[-1][4] = p
                    rows[-1][2], rows[-1][3] = max(rows[-1][2], p), min(rows[-1][3], p)
        except Exception:
            pass
        return [[t, round(o, 4), round(h, 4), round(l, 4), round(c, 4)] for t, o, h, l, c in rows]

    def fills(self, symbol, days=130) -> list:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=200,
                               after=datetime.now(ET) - timedelta(days=days))
        out = []
        for o in self.trade.get_orders(filter=req):
            if o.filled_at and o.filled_avg_price and float(o.filled_qty or 0) > 0:
                out.append({"t": int(o.filled_at.timestamp()), "side": getattr(o.side, "value", str(o.side)),
                            "price": float(o.filled_avg_price), "qty": float(o.filled_qty)})
        return sorted(out, key=lambda f: f["t"])

    def buy_dollars(self, symbol, dollars):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        self.trade.submit_order(MarketOrderRequest(symbol=symbol, notional=round(dollars, 2),
                                                   side=OrderSide.BUY, time_in_force=TimeInForce.DAY))

    def sell_qty(self, symbol, qty):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        self.trade.submit_order(MarketOrderRequest(symbol=symbol, qty=round(qty, 6),
                                                   side=OrderSide.SELL, time_in_force=TimeInForce.DAY))


# ------------------------------------------------------------------ helpers
def load(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def notify(title, msg, up=True):
    if not cfg.NTFY_TOPIC:
        return
    try:
        import urllib.request
        req = urllib.request.Request(f"https://ntfy.sh/{cfg.NTFY_TOPIC}", data=msg.encode(),
                                     headers={"Title": title, "Tags": "green_circle" if up else "red_circle"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:  # a failed notification must never block trading
        print("notify failed:", e)


def rebalance(broker, exposure, equity, pos, price, log):
    target = max(0.0, exposure) * equity * (1 - cfg.CASH_BUFFER)
    diff = target - pos["value"]
    if abs(diff) < max(cfg.MIN_TRADE_DOLLARS, cfg.MIN_TRADE_PCT * equity):
        log.append(f"Holding: target ${target:,.2f} is close to current ${pos['value']:,.2f}")
        return "hold", 0.0
    if diff > 0:
        broker.buy_dollars(cfg.SYMBOL, diff)
        log.append(f"BUY ${diff:,.2f} of {cfg.SYMBOL}")
        return "buy", diff
    qty = pos["qty"] if target < 1 else min(pos["qty"], -diff / price)
    broker.sell_qty(cfg.SYMBOL, qty)
    log.append(f"SELL {qty:.4f} {cfg.SYMBOL} (~${qty * price:,.2f})")
    return "sell", -qty * price


# ------------------------------------------------------------------ main pass
def run(broker, now=None):
    now = now or datetime.now(ET)
    today = now.date().isoformat()
    state = load(STATE, {"start_equity": None, "peak": 0, "last_decision": None,
                         "halted": False, "history": [], "actions": []})
    log = []

    acct = broker.account()
    pos = broker.position(cfg.SYMBOL)
    equity = acct["equity"]
    acct_id = acct.get("account", "")
    if state.get("account") != acct_id or (ROOT / "RESET").exists():
        # new account, paper reset, or switch to live: start the history fresh
        state = {"account": acct_id, "start_equity": None, "peak": 0, "last_decision": None,
                 "halted": False, "history": [], "actions": []}
        log.append("Started fresh tracking for this account.")
    state["start_equity"] = state["start_equity"] or equity
    state["peak"] = max(state["peak"], equity)

    closes = broker.daily_closes(cfg.SIGNAL_SYMBOL)
    is_open = broker.market_open()
    spy_now = broker.last_price(cfg.SIGNAL_SYMBOL)
    if closes.index[-1].date() < now.date():
        closes = pd.concat([closes, pd.Series([spy_now], index=[pd.Timestamp(now.date(), tz=ET)])])
    else:
        closes.iloc[-1] = spy_now
    sig = ensemble(closes)
    spy_day = spy_now / closes.iloc[-2] - 1
    price = broker.last_price(cfg.SYMBOL) if is_open or pos["qty"] else 0.0

    action = None
    stop_file = (ROOT / "STOP").exists()
    hh, mm = map(int, cfg.DECIDE_AFTER_ET.split(":"))

    if stop_file:
        log.append("STOP file present: no trading.")
    elif equity <= state["peak"] * (1 - cfg.MAX_DRAWDOWN_HALT) and not state["halted"]:
        if pos["qty"] > 0:
            broker.sell_qty(cfg.SYMBOL, pos["qty"])
        state["halted"] = True
        log.append("Drawdown halt: sold everything. Delete the halt in state.json to resume.")
        notify("Bot halted", f"Account fell {cfg.MAX_DRAWDOWN_HALT:.0%} from peak. Moved to cash.", up=False)
    elif state["halted"]:
        log.append("Halted after a large drawdown. Not trading.")
    elif is_open and cfg.CRASH_GUARD is not None and spy_day <= cfg.CRASH_GUARD and pos["qty"] > 0:
        broker.sell_qty(cfg.SYMBOL, pos["qty"])
        action = "crash-guard sell"
        state["last_decision"] = today
        log.append(f"Crash guard: S&P {spy_day:.1%} today. Sold.")
    elif is_open and (now.hour, now.minute) >= (hh, mm) and state["last_decision"] != today:
        action, amt = rebalance(broker, sig["exposure"], equity, pos, price, log)
        state["last_decision"] = today
        state["actions"] = (state["actions"] + [{"date": today, "action": action, "amount": round(amt, 2),
                                                  "exposure": sig["exposure"]}])[-60:]
        total = equity / state["start_equity"] - 1
        notify(f"{'Up' if total >= 0 else 'Down'} {total:+.1%} overall",
               f"Today {equity / acct['last_equity'] - 1:+.2%} · ${equity:,.2f}\n"
               f"Decision: {action} · target {sig['exposure']:.0%} in {cfg.SYMBOL}", up=total >= 0)
    elif not is_open:
        log.append("Market closed.")
    else:
        log.append(f"Watching. Decision after {cfg.DECIDE_AFTER_ET} ET.")

    # keep one equity point per day (latest value of the day wins)
    hist = [h for h in state["history"] if h["date"] != today] + [{"date": today, "equity": round(equity, 2)}]
    state["history"] = hist[-400:]

    pos = broker.position(cfg.SYMBOL) if action in ("buy", "sell", "crash-guard sell") else pos
    status = {
        "updated": now.isoformat(timespec="minutes"),
        "mode": "paper" if cfg.PAPER else "live",
        "symbol": cfg.SYMBOL,
        "equity": round(equity, 2),
        "start_equity": round(state["start_equity"], 2),
        "total_pl": round(equity - state["start_equity"], 2),
        "total_pl_pct": round(equity / state["start_equity"] - 1, 4),
        "day_pl": round(equity - acct["last_equity"], 2),
        "day_pl_pct": round(equity / acct["last_equity"] - 1, 4) if acct["last_equity"] else 0,
        "position_value": round(pos["value"], 2),
        "avg_entry": round(pos.get("avg", 0.0), 4),
        "invested_pct": round(pos["value"] / equity, 3) if equity else 0,
        "signal": {k: sig[k] for k in ("exposure", "trend", "momentum", "dip_buy")},
        "spy_day": round(spy_day, 4),
        "market_open": is_open,
        "halted": state["halted"] or stop_file,
        "history": state["history"],
        "actions": state["actions"][-10:],
        "log": log,
    }
    STATE.parent.mkdir(exist_ok=True)
    STATUS.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))
    # candle chart for the phone page; a failure here must never affect trading
    try:
        if hasattr(broker, "candles"):
            status["chart"] = {"symbol": cfg.SYMBOL, **broker.candles(cfg.SYMBOL), "fills": broker.fills(cfg.SYMBOL)}
    except Exception as e:
        print("chart update skipped:", e, flush=True)
        status["chart"] = load(STATUS, {}).get("chart")
    STATUS.write_text(json.dumps(status, indent=1))
    print("\n".join(log))
    return status


if __name__ == "__main__":
    run(AlpacaBroker())
