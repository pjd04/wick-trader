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

# ---- active sleeve (bot/sleeve.py): day trading on part of the account ----
SLEEVE_ENABLED = True
SLEEVE_SHARE = 0.25          # share of the account the sleeve trades; Wick sizes to the rest
SLEEVE_LONG = "SPXL"         # 3x S&P (separate fund from Wick's UPRO)
SLEEVE_SHORT = "SPXS"        # inverse 3x S&P, bought to bet on a drop
DAY_TRADE_LIMIT = 3          # max day trades per 5 business days (US rule for accounts under $25k)
WICK_SHARE = 1 - SLEEVE_SHARE if SLEEVE_ENABLED else 1.0
