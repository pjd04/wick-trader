# Wick — automated 3x S&P trader

Runs in the cloud on GitHub Actions. Your MacBook can stay closed.

- **Strategy:** the Ensemble signal (200-day trend + 12-month momentum + RSI(2) dip-buying,
  volatility-targeted), read from SPY and traded in **UPRO** (3x S&P 500).
- **Schedule:** one long session per market day (`session.yml`). It starts before the open, checks
  every 4 minutes, trades once after 3:30 pm ET, and hands off to a fresh run before GitHub's 6-hour
  limit. Several backup start times cover GitHub's habit of delaying or skipping timed jobs.
- **Reset tracking:** add an empty file named `RESET` to start Wick's profit tracking over
  (it also resets automatically when the Alpaca account changes). Delete the file afterwards.
- **Phone page:** `docs/index.html`, published with GitHub Pages. Wick the candle turns green
  when you're up and red when you're down. Refreshes every 5 minutes.
- **Alerts:** optional daily push notification through the free ntfy app.

Backtest (S&P data 1951–2016, price-only, 3x with costs): about 9–12% a year, worst drop −47%,
worst year −38%. Past results don't guarantee future ones. Not financial advice.

## One-time setup
1. **Alpaca account** (alpaca.markets). Start with the free **paper** account. Under API Keys,
   generate a key and secret.
2. **Add secrets to this repo:** Settings → Secrets and variables → Actions → New repository secret
   - `ALPACA_KEY`, `ALPACA_SECRET`
   - `NTFY_TOPIC` (optional): any hard-to-guess name, e.g. `wick-7f3k2q`. Install the ntfy app
     on your phone and subscribe to the same name.
3. **Turn on the phone page:** Settings → Pages → Deploy from branch → `main` / `docs`.
   (Pages on a private repo needs GitHub Pro, free with the GitHub Student Developer Pack.)
   The page link is unlisted but not password-protected.
4. **Run it once now:** Actions → trade → Run workflow. Then open your Pages link.

## Reliable start (cron-job.org)
GitHub's own timer can run hours late. A free cron-job.org job starts the session every weekday at
8:20 am Central (with a 1:50 pm backup) by calling the GitHub API with a fine-grained token that can
only run this repo's workflows. If a close is ever missed anyway, the bot catches up at the next open,
and Wick shows a red "no check-in" warning when the bot goes quiet during market hours.

## How Wick learns (nightly review)
After each close, `bot/learn.py`:
- **Grades every buy and sell** by what the 3x market did over the next 1, 5 and 20 trading days
  (the Report card on the phone page).
- **Runs a shadow Wick** that re-tunes its four settings every quarter from the last 10 years and trades
  only on paper. If, after 126+ trading days, the shadow beats the live settings by a Sharpe margin of 0.3
  and on total return, its settings are promoted to live (`state/params.json`) and you get a notification.
  Delete `state/params.json` to go back to the defaults.

Why so strict: in walk-forward tests (research/self_tune.py) every self-tuning variant did worse than the
fixed settings from 2000 and from 2016 on (daily re-tuning: 15.2%/yr vs 19.4%), because the best settings
in one period rarely stay best in the next.

## Going live
1. Fund your Alpaca live account and generate **live** API keys.
2. Replace the two secrets with the live keys.
3. Settings → Secrets and variables → Actions → Variables → `ALPACA_PAPER` = `false`.

## Controls
- **Pause:** add an empty file named `STOP` to the repo (Add file → Create new file on github.com,
  works from your phone). Delete it to resume.
- **Turn off completely:** Actions → trade → ⋯ → Disable workflow.
- **2x instead of 3x:** change `SYMBOL = "UPRO"` to `"SSO"` in `bot/config.py`.
- **Drawdown halt:** the bot sells everything and stops if the account falls 50% from its peak.

## Day trading
`research/intraday_test.py` tests an intraday strategy on real minute data. Run it on your Mac
first; only wire an intraday strategy into the bot if it clearly beats holding after costs.
The old rule (3 day trades per 5 business days under $25k) was removed by FINRA on June 4, 2026,
but brokers are switching over at their own pace. Check Alpaca's current day-trading terms before
raising `DAY_TRADE_LIMIT` in `bot/config.py`.

`research/intraday_lab.py` (Actions -> research -> intraday_lab) compares every intraday strategy
on 10 years of SPY 5-minute bars, traded as 3x ETFs with costs. Latest run, full period:
noise-band momentum +5.7%/yr (what the sleeve trades), VWAP trend -66%/yr, EMA trend pullback
-33%/yr, 5-minute opening range -10%/yr, against +21.6%/yr for Wick's daily strategy.

## Candle City
`docs/city/index.html` is a 3D view of the same account: one tower for Wick, one for the sleeve,
beams while a position is open, and a vault with the total. It reads `docs/status.json`, so it
shows whatever account the bot is connected to (paper or live). Open `<your Pages link>/city/`.
