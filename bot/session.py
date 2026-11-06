"""Keeps Wick alive through the trading day.

GitHub's timed triggers are often late or skipped, so instead of one short run every 15 minutes,
a single session starts before the open, runs a bot pass every few minutes, and before GitHub's
6-hour job limit it starts a fresh copy of itself to carry on until the close.

Alpaca is the source of truth for positions, so a session can stop and restart at any time.
"""
from __future__ import annotations
import importlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
import run as bot  # noqa: E402

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
PASS_EVERY = int(os.environ.get("PASS_EVERY_SECONDS", "240"))      # one bot pass every 4 minutes
PUSH_EVERY = int(os.environ.get("PUSH_EVERY_SECONDS", "600"))      # publish the phone page every 10 minutes
SEGMENT = timedelta(hours=float(os.environ.get("SEGMENT_HOURS", "5.25")))


def sh(*args) -> int:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).returncode


def publish(note: str) -> None:
    sh("git", "add", "-A", "state", "docs")
    if sh("git", "diff", "--cached", "--quiet") == 0:
        return
    sh("git", "commit", "-q", "-m", f"bot: {note}")
    for _ in range(3):
        if sh("git", "pull", "--rebase", "-q") == 0 and sh("git", "push", "-q") == 0:
            return
        time.sleep(5)
    print("publish failed; will retry next pass", flush=True)


def relay() -> None:
    """Start a fresh run of this workflow so the session outlives GitHub's job time limit."""
    repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("relay skipped: not running on GitHub", flush=True)
        return
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/session.yml/dispatches",
        data=json.dumps({"ref": "main"}).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    urllib.request.urlopen(req, timeout=20)
    print("relay: started the next session segment", flush=True)


def tick_live(broker) -> None:
    """Publish price, candles and account value to the `live` branch (one commit, overwritten each
    minute). The phone page reads it straight from GitHub, so it doesn't wait for a Pages rebuild."""
    try:
        import config as cfg
        state = json.loads((ROOT / "state" / "state.json").read_text())
        acct, pos = broker.account(), broker.position(cfg.SYMBOL)
        start = state.get("start_equity") or acct["equity"]
        data = {
            "updated": datetime.now(ET).isoformat(timespec="seconds"),
            "symbol": cfg.SYMBOL,
            "equity": round(acct["equity"], 2),
            "day_pl": round(acct["equity"] - acct["last_equity"], 2),
            "day_pl_pct": round(acct["equity"] / acct["last_equity"] - 1, 4) if acct["last_equity"] else 0,
            "total_pl": round(acct["equity"] - start, 2),
            "total_pl_pct": round(acct["equity"] / start - 1, 4),
            "position_value": round(pos["value"], 2),
            "invested_pct": round(pos["value"] / acct["equity"], 3) if acct["equity"] else 0,
            "intraday": broker.intraday_candles(cfg.SYMBOL),
        }
        tmp = ROOT / ".live.json"
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        git = lambda *a, inp=None: subprocess.run(["git", *a], cwd=ROOT, input=inp, capture_output=True, text=True)
        blob = git("hash-object", "-w", str(tmp)).stdout.strip()
        tree = git("mktree", inp=f"100644 blob {blob}\tlive.json\n").stdout.strip()
        commit = git("commit-tree", tree, "-m", "live price feed").stdout.strip()
        if git("push", "-q", "-f", "origin", f"{commit}:refs/heads/live").returncode != 0:
            print("live feed push failed", flush=True)
    except Exception as e:  # the live feed is cosmetic; never let it disturb trading
        print(f"live feed skipped: {e!r}", flush=True)


_head = None


def fresh_broker(broker):
    """If new code was pulled in (publish does a git pull), reload it so updates apply mid-day."""
    global _head, bot
    head = subprocess.run(["git", "rev-parse", "HEAD:bot"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if _head is None:
        _head = head
    elif head and head != _head:
        import config, strategy
        importlib.reload(config); importlib.reload(strategy); bot = importlib.reload(bot)
        _head = head
        print("loaded updated bot code", flush=True)
        return bot.AlpacaBroker()
    return broker


def main() -> None:
    broker = bot.AlpacaBroker()
    started = datetime.now(ET)
    last_push = 0.0
    print(f"session start {started:%Y-%m-%d %H:%M} ET", flush=True)

    while True:
        now = datetime.now(ET)
        clock = broker.trade.get_clock()
        next_open = clock.next_open.astimezone(ET)
        open_today = clock.is_open or next_open.date() == now.date()

        if not open_today:
            # market is done for the day (or a holiday): one last update, then stop
            status = bot.run(broker)
            try:
                import learn
                rep = learn.nightly(broker, json.loads((ROOT / "state" / "state.json").read_text()), notify=bot.notify)
                if rep:
                    print(f"nightly review: shadow {rep['shadow_settings']} vs live {rep['live_settings']}, "
                          f"{rep['days']}/{rep['days_needed']} days", flush=True)
            except Exception as e:  # the review must never block the closing update
                print(f"nightly review skipped: {e!r}", flush=True)
            publish("closing update")
            print("market closed for the day; session over", flush=True)
            return

        if not clock.is_open:
            wait = (next_open - now).total_seconds() + 30
            if now + timedelta(seconds=wait) - started > SEGMENT:
                relay()
                return
            print(f"waiting {wait / 60:.0f} min for the open", flush=True)
            time.sleep(max(30, wait))
            continue

        try:
            broker = fresh_broker(broker)
            status = bot.run(broker)
        except Exception as e:  # one bad pass must not end the trading day
            print(f"pass failed, retrying next pass: {e!r}", flush=True)
            time.sleep(PASS_EVERY)
            continue
        traded = any(a.get("date") == now.date().isoformat() and a.get("action") not in ("hold",)
                     for a in status.get("actions", []))
        if traded or time.time() - last_push >= PUSH_EVERY:
            publish(f"{now:%H:%M} ET update")
            last_push = time.time()

        if datetime.now(ET) - started > SEGMENT:
            relay()
            return
        # between trading passes, publish the live price feed every minute
        for _ in range(max(1, PASS_EVERY // 60)):
            tick_live(broker)
            time.sleep(60)


if __name__ == "__main__":
    main()
