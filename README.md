# Wick — automated 3x S&P trader

Runs in the cloud on GitHub Actions. Your MacBook can stay closed.

- **Strategy:** the Ensemble signal (200-day trend + 12-month momentum + RSI(2) dip-buying,
  volatility-targeted), read from SPY and traded in **UPRO** (3x S&P 500).
- **Schedule:** checks every 15 minutes on market days and trades once a day after 3:30 pm ET.
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
With under $25k in a margin account, US rules allow 3 day trades per 5 business days.
