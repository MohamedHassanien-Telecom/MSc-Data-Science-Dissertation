"""src/dataset.py detect_ram_cache_budget_gb sets the image-cache budget as a
fraction of the currently available RAM (MemAvailable in /proc/meminfo):
  1. against this machine's /proc/meminfo, the budget is that fraction of
     MemAvailable;
  2. the budget scales with the fraction;
  3. an explicit override is used as given;
  4. without /proc/meminfo, caching is disabled (0.0).

Usage:  python3 tests/test_ram_budget_detection.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load src/dataset.py with minimal stubs for torch, torchvision and PIL;
# only detect_ram_cache_budget_gb is used.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_dataset_for_test", os.path.join(os.path.dirname(__file__), "..", "src", "dataset.py"))
import types
for mod_name in ["torch", "torchvision", "torchvision.transforms", "PIL", "PIL.Image"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["torch"].utils = types.ModuleType("torch.utils")
sys.modules["torch"].utils.data = types.ModuleType("torch.utils.data")
sys.modules["torch"].utils.data.Dataset = object
sys.modules["torch"].utils.data.DataLoader = object
sys.modules["torch"].utils.data.Subset = object
sys.modules["torch"].utils.data.ConcatDataset = object
sys.modules["torch"].utils.data.RandomSampler = object
sys.modules["torch"].Tensor = object
sys.modules["torch.utils"] = sys.modules["torch"].utils
sys.modules["torch.utils.data"] = sys.modules["torch"].utils.data
sys.modules["PIL"].Image = object

_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
detect_ram_cache_budget_gb = _mod.detect_ram_cache_budget_gb

# 1. Against this machine's /proc/meminfo.
with open("/proc/meminfo") as f:
    real_meminfo = f.read()
real_available_kb = None
for line in real_meminfo.splitlines():
    if line.startswith("MemAvailable:"):
        real_available_kb = int(line.split()[1])
        break

assert real_available_kb is not None, "This container's own /proc/meminfo has no MemAvailable -- can't verify"
expected_gb = (real_available_kb / 1e6) * 0.25

result = detect_ram_cache_budget_gb(fraction=0.25)
print(f"[1] Real /proc/meminfo MemAvailable: {real_available_kb/1e6:.2f} GB")
print(f"[1] detect_ram_cache_budget_gb(0.25) returned: {result:.4f} GB")
print(f"[1] Expected (25% of real available): {expected_gb:.4f} GB")
assert abs(result - expected_gb) < 0.01, \
    f"Mismatch: got {result}, expected {expected_gb} -- parsing or fraction logic is wrong"
print("PASS: correctly parses real /proc/meminfo and applies the fraction accurately")

# 2. The budget scales with the fraction.
result_10pct = detect_ram_cache_budget_gb(fraction=0.10)
result_50pct = detect_ram_cache_budget_gb(fraction=0.50)
print(f"\n[2] At 10%: {result_10pct:.4f} GB, at 50%: {result_50pct:.4f} GB")
assert abs(result_50pct - result_10pct * 5) < 0.01, \
    "Fraction scaling isn't proportional -- something other than a simple multiply is happening"
print("PASS: budget scales proportionally with the fraction parameter")

# 3. An override is used as given.
result_override = detect_ram_cache_budget_gb(fraction=0.25, override_gb=3.7)
print(f"\n[3] With override_gb=3.7: {result_override}")
assert result_override == 3.7, f"Override was not used exactly, got {result_override}"
print("PASS: explicit override bypasses detection and is used exactly as given")

# 4. Without /proc/meminfo, caching is disabled.
_real_open = open
def _open_that_fails_for_meminfo(path, *a, **kw):
    if path == "/proc/meminfo":
        raise FileNotFoundError("simulated: no /proc/meminfo")
    return _real_open(path, *a, **kw)

import builtins
builtins.open = _open_that_fails_for_meminfo
try:
    result_missing = detect_ram_cache_budget_gb(fraction=0.25)
finally:
    builtins.open = _real_open

print(f"\n[4] With /proc/meminfo missing: {result_missing}")
assert result_missing == 0.0, \
    f"Expected 0.0 (caching disabled) when /proc/meminfo is missing, got {result_missing}"
print("PASS: without /proc/meminfo caching is disabled (0.0)")

print("\n" + "="*70)
print("ALL RAM BUDGET DETECTION TESTS PASSED")
print("="*70)
