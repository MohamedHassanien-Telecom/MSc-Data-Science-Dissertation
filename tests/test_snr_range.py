"""SNR settings in src/config.py:
  1. the 40 dB reference point is part of SNR_RANGE_DB;
  2. DEFAULT_SNR_DB (the SNR at which the search scores candidates) is the
     midpoint of the swept range without the 40 dB point;
  3. CLS_FIT_SNR (training and calibration SNR) is 20 dB above the highest swept
     SNR, i.e. 40 dB;
  4. the two are distinct.

Usage:  python3 tests/test_snr_range.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import CONFIG

print("=" * 70)
print("TEST: SNR range, 40 dB reference point and SNR defaults")
print("=" * 70)

# 1. The 40 dB reference point is in the sweep.
assert 40 in CONFIG["SNR_RANGE_DB"], \
    f"40dB reference point missing from SNR_RANGE_DB: {CONFIG['SNR_RANGE_DB']}"
print(f"PASS: 40 in SNR_RANGE_DB (full range: {CONFIG['SNR_RANGE_DB']})")

# 2. DEFAULT_SNR_DB is the midpoint of the swept range without 40 dB.
range_without_40 = [s for s in CONFIG["SNR_RANGE_DB"] if s != 40]
expected_default = (min(range_without_40) + max(range_without_40)) // 2
assert CONFIG["DEFAULT_SNR_DB"] == expected_default, \
    (f"DEFAULT_SNR_DB={CONFIG['DEFAULT_SNR_DB']} is not the midpoint ({expected_default}) "
     f"of the swept range without the 40 dB point.")
print(f"PASS: DEFAULT_SNR_DB={CONFIG['DEFAULT_SNR_DB']} is the midpoint of the swept range")

# 3. CLS_FIT_SNR is 20 dB above the highest swept SNR.
expected_cls_fit = max(range_without_40) + 20
assert CONFIG["CLS_FIT_SNR"] == float(expected_cls_fit), \
    (f"CLS_FIT_SNR={CONFIG['CLS_FIT_SNR']} is not 20 dB above the highest swept SNR "
     f"({expected_cls_fit}).")
print(f"PASS: CLS_FIT_SNR={CONFIG['CLS_FIT_SNR']} is 20 dB above the highest swept SNR")

# 4. The two are distinct.
assert CONFIG["DEFAULT_SNR_DB"] != CONFIG["CLS_FIT_SNR"]
print(f"PASS: DEFAULT_SNR_DB ({CONFIG['DEFAULT_SNR_DB']}) and CLS_FIT_SNR "
      f"({CONFIG['CLS_FIT_SNR']}) are distinct")

print("\n" + "=" * 70)
print("ALL CHECKS PASSED -- the 40 dB reference point is in the sweep and both SNR")
print("defaults are derived from the swept range without it.")
print("=" * 70)
