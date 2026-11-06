"""Dry run of the nightly review against real Alpaca data, in a throwaway checkout (nothing is committed
except this report). Usage: python research/learn_check.py"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "bot"))
import learn, run  # noqa: E402

for p in (learn.LEARN, learn.REPORT):
    p.unlink(missing_ok=True)
state = json.loads((ROOT.parent / "state" / "state.json").read_text())
rep = learn.nightly(run.AlpacaBroker(), state, notify=lambda *a, **k: print("would notify:", a))
print(json.dumps({k: rep[k] for k in ("live_settings", "shadow_settings", "shadow_since", "days", "hit_rate_5d", "graded")}, indent=1))
for r in rep["report_card"]:
    print(r)
(ROOT / "results" / "learn_check.json").write_text(json.dumps(rep, indent=1))
