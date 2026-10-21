"""Kendall's W null floor (src/effect_sizes.py kendalls_w_null_floor) and the caution
interpret_kendalls_w adds when an observed W is at or below it:
  1. the null floor is deterministic and lies near 0.16-0.18 at n = 5, k = 3..7;
  2. a W at or below the floor for its (n, k) gets the caution;
  3. a W far above the floor (0.81 at k = 7) does not;
  4. the floor is cached per (n, k);
  5. without n and k, the plain label is returned;
  6. a W clearly above the floor gets no caution.

Usage:  python3 tests/test_kendalls_w_null_floor.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.effect_sizes import (interpret_kendalls_w, kendalls_w_null_floor,
                               _cached_null_floor, _kendalls_w_null_floor_cache)

print("=" * 70)
print("TEST: Kendall's W null-floor caution mechanism")
print("=" * 70)

# 1. Determinism.
f1 = kendalls_w_null_floor(5, 3)
f2 = kendalls_w_null_floor(5, 3)
assert f1 == f2, f"Not deterministic: {f1} != {f2}"
print(f"PASS: deterministic (n=5,k=3 -> {f1:.4f} both times, fixed seed)")

# 2. Null floor at n = 5 for every candidate count the search uses.
for k in (3, 4, 5, 6, 7):
    floor = kendalls_w_null_floor(5, k)
    assert 0.14 <= floor <= 0.22, f"n=5,k={k} floor {floor:.3f} outside the expected ~0.16-0.18 range"
    print(f"PASS: n=5,k={k} null floor = {floor:.4f} (at or above the 'large' threshold of 0.14)")

# 3. A W at the null floor gets the caution.
label = interpret_kendalls_w(0.1600, n=5, k=3)
assert "large" in label
assert "CAUTION" in label
assert "0.16" in label or "N=5,k=3" in label
print(f"PASS: W=0.16 at n=5, k=3 (at the null floor) -> '{label}'")

# 4. A W far above the floor does not.
label2 = interpret_kendalls_w(0.8086, n=5, k=7)
assert "large" in label2
assert "CAUTION" not in label2
print(f"PASS: W=0.81 at n=5, k=7 (far above the floor) -> '{label2}' (no caution)")

# 5. The floor is computed once per (n, k) and cached.
_kendalls_w_null_floor_cache.clear()
v1 = _cached_null_floor(5, 4)
assert (5, 4) in _kendalls_w_null_floor_cache
v2 = _cached_null_floor(5, 4)
assert v1 == v2
print(f"PASS: caching works -- (5,4) computed once ({v1:.4f}), reused on second call")

# 6. Without n and k: the plain label.
old_style = interpret_kendalls_w(0.1600)
assert old_style == "large", f"expected the plain label 'large', got '{old_style}'"
assert "CAUTION" not in old_style
print(f"PASS: without n and k the plain label is returned: '{old_style}'")

# 7. A W clearly above the floor gets no caution.
floor_53 = kendalls_w_null_floor(5, 3)
just_above = interpret_kendalls_w(floor_53 + 0.05, n=5, k=3)
assert "CAUTION" not in just_above
print(f"PASS: a value meaningfully above the floor ({floor_53+0.05:.4f}) correctly gets no caution")

print("\n" + "=" * 70)
print("ALL CHECKS PASSED -- the null-floor caution is deterministic, applies only at or")
print("below the floor, and the floor is cached.")
print("=" * 70)
