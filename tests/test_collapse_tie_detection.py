"""Headline elbow of the collapse analysis (src/collapse.py _modal_elbow): the most
frequent elbow SNR across the percentile settings, with no headline on a tie.
  1. a two-way tie gives no headline, tie_detected and the sorted tied values;
  2. a clear majority gives that value and no tie;
  3. a three-way tie lists all three values;
  4. six agreeing settings give agreement 1.0;
  5. a setting without a candidate counts as the value None, and None sorts last
     among tied values.

Usage:  python3 tests/test_collapse_tie_detection.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.collapse import _modal_elbow

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

h, a, tie, vals = _modal_elbow([-4, -4, -4, -2, -2, -2])
check(f"two-way tie: no headline, tied values [-4, -2] ({h}, {tie}, {vals})",
      h is None and tie is True and vals == [-4, -2] and a == 0.5)

h, a, tie, vals = _modal_elbow([-4, -4, -4, -4, -2, 0])
check(f"clear majority: headline -4, no tie ({h}, {tie}, {vals})", h == -4 and tie is False and vals is None)

h, a, tie, vals = _modal_elbow([-4, -2, 0, -4, -2, 0])
check(f"three-way tie: all three values ({vals})", h is None and tie is True and vals == [-4, -2, 0])

h, a, tie, vals = _modal_elbow([-6] * 6)
check(f"all six agree: agreement {a}", h == -6 and tie is False and a == 1.0)

h, a, tie, vals = _modal_elbow([None, None, -8, -8, -6, -4])
check(f"no-candidate settings count as None and sort last among tied values ({vals})",
      h is None and tie is True and vals == [-8, None])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All collapse tie-detection checks passed.")
