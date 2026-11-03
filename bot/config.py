"""Bot settings. API keys never go in this file: they come from GitHub Secrets."""
import os

PAPER = os.environ.get("ALPACA_PAPER", "true").lower() != "false"  # set the repo variable to "false" for real money

SYMBOL = "UPRO"        # 3x S&P 500. Use "SSO" for 2x (steadier) or "SPY" for 1x.
SIGNAL_SYMBOL = "SPY"  # the strategy reads the S&P itself, then trades the leveraged ETF

DECIDE_AFTER_ET = "15:30"    # trade once a day in the last 30 minutes before the close
CASH_BUFFER = 0.02           # keep 2% in cash so orders never bounce
MIN_TRADE_DOLLARS = 5.0      # ignore tiny rebalances
MIN_TRADE_PCT = 0.05         # ...and anything under 5% of the account, to avoid churn

# Crash guard: sell everything if the S&P is down this much on the day (e.g. -0.05).
# OFF by default: in testing, stop-losses on this strategy lowered returns.
CRASH_GUARD = None

MAX_DRAWDOWN_HALT = 0.50     # stop trading and go to cash if the account falls 50% from its peak
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")  # optional phone notifications via the free ntfy app
