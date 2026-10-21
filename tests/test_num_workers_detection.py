"""src/config.py _detect_num_workers sets the number of DataLoader workers from the
CPU cores available to the process:
  1. src.config imports without torch (so tests that stub torch still work);
  2. the result is the available cores minus the reserved cores, at least 1;
  3. an explicit override is used as given;
  4. reserve_cores is respected.

Usage:  python3 tests/test_num_workers_detection.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import types
# 1. src.config imports with torch unavailable (any use of torch would fail).
_fake_torch_before = sys.modules.get('torch')
sys.modules['torch'] = None
try:
    if 'src.config' in sys.modules:
        del sys.modules['src.config']
    if 'src' in sys.modules:
        del sys.modules['src']
    from src.config import CONFIG, _detect_num_workers
    print("[1] PASS: src.config imports with torch unavailable")
finally:
    if _fake_torch_before is not None:
        sys.modules['torch'] = _fake_torch_before
    else:
        sys.modules.pop('torch', None)

try:
    # 2. Detection against this machine's available cores.
    real_available = len(os.sched_getaffinity(0))
except (AttributeError, NotImplementedError):
    real_available = os.cpu_count()

result = _detect_num_workers(reserve_cores=1)
expected = max(1, real_available - 1)
print(f"\n[2] Real available cores: {real_available}, expected result: {expected}, "
      f"got: {result}")
assert result == expected, f"Expected {expected}, got {result}"
assert result >= 1, "Must never return less than 1 (0 workers changes how DataLoader runs)"
print("PASS: correctly detects real core count and reserves the requested amount")

# 3. An override is used as given.
result_override = _detect_num_workers(override=5)
print(f"\n[3] With override=5: {result_override}")
assert result_override == 5, f"Expected exactly 5, got {result_override}"
print("PASS: explicit override bypasses detection and is used exactly as given")

# 4. reserve_cores is respected.
result_reserve2 = _detect_num_workers(reserve_cores=2)
expected_reserve2 = max(1, real_available - 2)
print(f"\n[4] With reserve_cores=2: {result_reserve2} (expected {expected_reserve2})")
assert result_reserve2 == expected_reserve2
print("PASS: reserve_cores parameter is correctly respected")

print("\n" + "="*70)
print("ALL NUM_WORKERS DETECTION TESTS PASSED")
print("="*70)
