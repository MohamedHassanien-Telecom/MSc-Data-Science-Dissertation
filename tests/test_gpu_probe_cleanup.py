"""src/gpu_utils.py _try_forward_backward, the single forward/backward attempt used
by the batch-size probe, releases every object it creates (models, tensors) on
every exit path, so that memory held by a failed attempt cannot make the next
attempt fail. Object lifetimes are tracked with __del__ (immediate in CPython for
objects without reference cycles).
  1. each of several failed (out-of-memory) attempts leaves no object alive;
  2. a successful attempt leaves no object alive;
  3. eight consecutive failures accumulate nothing.

Usage:  python3 tests/test_gpu_probe_cleanup.py
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_alive = set()
_next_id = [0]


class TrackedObj:
    """Stands in for models and tensors; records whether it is still alive."""
    def __init__(self, bs):
        self.bs = bs
        self.id = _next_id[0]
        _next_id[0] += 1
        _alive.add(self.id)

    def to(self, d): return self
    def pow(self, p): return self
    def mean(self): return self
    def float(self): return self
    def backward(self): pass
    def __add__(self, other): return self
    def __del__(self):
        _alive.discard(self.id)


class TrackedJSCC(TrackedObj):
    """jscc(x) returns (x_hat, z, z_prime), like SwinJSCCSystem.forward."""
    def __call__(self, x):
        return TrackedObj(x.bs), TrackedObj(x.bs), TrackedObj(x.bs)


class TrackedDetector(TrackedObj):
    """detector(z_prime) returns one value (the logits)."""
    def __call__(self, x):
        return TrackedObj(x.bs)


class FakeOOM(Exception):
    pass


class FakeCudaModule:
    OutOfMemoryError = FakeOOM
    @staticmethod
    def is_available(): return True
    @staticmethod
    def empty_cache(): pass
    @staticmethod
    def reset_peak_memory_stats(): pass
    @staticmethod
    def max_memory_allocated(): return 1e9


# 1. Several failed attempts: each must release its objects before the next.
_call_count = [0]
FAIL_UNTIL = 3

def fake_randn(bs, *a, **kw):
    _call_count[0] += 1
    t = TrackedObj(bs)
    if _call_count[0] <= FAIL_UNTIL:
        raise FakeOOM("simulated OOM")
    return t

class FakeAutocast:
    """No-op stand-in for torch.autocast."""
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False

class FakeDevice:
    """Stands in for torch.device: the helper reads device.type."""
    def __init__(self, type_str): self.type = type_str

FAKE_CPU_DEVICE = FakeDevice("cpu")

sys.modules['torch'] = types.SimpleNamespace(cuda=FakeCudaModule, randn=fake_randn,
                                               autocast=FakeAutocast, bfloat16="bf16", float16="fp16")
if 'src.gpu_utils' in sys.modules:
    del sys.modules['src.gpu_utils']
from src.gpu_utils import _try_forward_backward

def model_builder():
    return TrackedJSCC(0), TrackedDetector(0)

print(f"[1] Alive objects before any attempt: {len(_alive)}")
assert len(_alive) == 0

for i in range(FAIL_UNTIL):
    ok, peak = _try_forward_backward(model_builder, FAKE_CPU_DEVICE, 224, 10, None)
    print(f"[1] Attempt {i+1}: ok={ok}, alive objects immediately after = {len(_alive)}")
    assert ok is False, f"Expected this attempt to fail (OOM), got ok={ok}"
    assert len(_alive) == 0, \
        f"LEAK: {len(_alive)} object(s) still alive after a failed attempt -- " \
        f"cleanup is not happening before the next attempt can begin"

print("[1] PASS: every failed attempt released all its objects completely, immediately")

# 2. A successful attempt also releases its objects.
_call_count[0] = 999
ok, peak = _try_forward_backward(model_builder, FAKE_CPU_DEVICE, 224, 10, None)
print(f"\n[2] Successful attempt: ok={ok}, alive objects immediately after = {len(_alive)}")
assert ok is True
assert len(_alive) == 0, f"LEAK: {len(_alive)} object(s) alive after a SUCCESSFUL attempt too"
print("[2] PASS: successful attempts also fully clean up their objects")

# 3. Eight consecutive failures, as in the probe's backoff loop.
_call_count[0] = 0
FAIL_UNTIL_LONG = 8
def fake_randn_long(bs, *a, **kw):
    _call_count[0] += 1
    t = TrackedObj(bs)
    if _call_count[0] <= FAIL_UNTIL_LONG:
        raise FakeOOM("simulated OOM")
    return t
sys.modules['torch'].randn = fake_randn_long

for i in range(FAIL_UNTIL_LONG):
    _try_forward_backward(model_builder, FAKE_CPU_DEVICE, 224, 10, None)
print(f"\n[3] Alive objects after {FAIL_UNTIL_LONG} consecutive failed attempts: {len(_alive)}")
assert len(_alive) == 0, \
    f"LEAK after {FAIL_UNTIL_LONG} consecutive failures: {len(_alive)} objects still alive"
print(f"[3] PASS: no accumulation across {FAIL_UNTIL_LONG} consecutive failures")

print("\n" + "="*70)
print("ALL GPU MEMORY CLEANUP TESTS PASSED")
print("="*70)
