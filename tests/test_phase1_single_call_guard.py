"""phases/phase1_grid.py phase1_grid_search_and_train writes the selected settings
into cfg in place, so a second call with the same cfg would start round 0 from
the selected values and treat cached results as stale. The function therefore
refuses a second call in the same session:
  1. the first call marks cfg before doing any work;
  2. a second call with the same cfg raises RuntimeError before it creates,
     trains or writes anything;
  3. a fresh copy of the configuration is not refused;
  4. config.py's own CONFIG never carries the marker.

Usage:  python3 tests/test_phase1_single_call_guard.py
"""
import os, sys, copy, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from src.config import CONFIG
import phases.phase1_grid as PG

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

class Stop(Exception):
    pass

check("config.py's CONFIG starts without the marker", PG._PHASE1_CALLED_KEY not in CONFIG)

orig = PG.detect_anomaly_value
def _stop(*a, **k):
    raise Stop()
# Stop at the first real work of the pipeline (detecting the anomaly label).
PG.detect_anomaly_value = _stop
try:
    with tempfile.TemporaryDirectory() as d:
        cfg = copy.deepcopy(CONFIG); cfg["OUTPUT_DIR"] = d
        try:
            PG.phase1_grid_search_and_train(cfg, "cpu", run_idx=0); first = "returned"
        except Stop:
            first = "reached Step 1"
        check(f"first call proceeds to Step 1 ({first}) and marks cfg",
              first == "reached Step 1" and cfg.get(PG._PHASE1_CALLED_KEY) is True)

        d2 = Path(d) / "second"
        cfg["OUTPUT_DIR"] = str(d2)
        try:
            PG.phase1_grid_search_and_train(cfg, "cpu", run_idx=0); second = None
        except RuntimeError as e:
            second = str(e)
        except Stop:
            second = "reached Step 1"
        check("second call with the same cfg is refused with a RuntimeError",
              second is not None and second != "reached Step 1" and "Restart session" in (second or ""))
        check("the refused call created nothing on disk", not d2.exists())

        cfg_fresh = copy.deepcopy(CONFIG); cfg_fresh["OUTPUT_DIR"] = d
        try:
            PG.phase1_grid_search_and_train(cfg_fresh, "cpu", run_idx=0); third = "returned"
        except Stop:
            third = "reached Step 1"
        except RuntimeError:
            third = "refused"
        check(f"a fresh copy of the config is not refused ({third})", third == "reached Step 1")
finally:
    PG.detect_anomaly_value = orig

check("config.py's CONFIG still has no marker after the calls on copies", PG._PHASE1_CALLED_KEY not in CONFIG)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All Phase I single-call guard checks passed.")
