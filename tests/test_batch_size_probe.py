"""src/gpu_utils.py detect_max_batch_size, the GPU batch-size probe used when
CONFIG["BATCH_SIZE_OVERRIDE"] is None. The probe doubles the batch size until an
out-of-memory error, refines the limit by binary search between the last
success and the first failure, applies a safety factor and verifies the result
with one more pass, backing off in 10% steps if that pass fails.
  1. on a large simulated GPU the probe reaches the out-of-memory boundary (no
     fixed ceiling) and the binary search refines beyond the doubling result;
  2. warn_above only prints a warning; it does not limit the result;
  3. a small true limit is also found;
  4. the returned batch size is one that was actually run;
  5. one transient failure of the verification pass costs one 10% step, not half
     of the batch size;
  6. persistent failure of every verification attempt ends after several
     attempts with a usable value instead of looping or crashing.
torch.cuda is simulated; no GPU is needed.

Usage:  python3 tests/test_batch_size_probe.py
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeOOM(Exception):
    pass


class FakeDevice:
    """Stands in for torch.device: the probe reads device.type."""
    def __init__(self, type_str): self.type = type_str
    def __eq__(self, other): return str(self) == str(other)
    def __str__(self): return self.type

FAKE_CUDA_DEVICE = FakeDevice("cuda")

class FakeAutocast:
    """No-op stand-in for torch.autocast, which wraps every probe pass."""
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _build_fake_torch(oom_at_batch, total_mem=80e9, free_mem=60e9):
    """A simulated torch whose GPU runs out of memory at batch sizes >= oom_at_batch
    (so oom_at_batch - 1 is the true limit). Returns (fake_torch, call_sizes, FakeTensor).
    """
    call_sizes = []

    class FakeCudaModule:
        OutOfMemoryError = FakeOOM
        @staticmethod
        def is_available(): return True
        @staticmethod
        def is_bf16_supported(): return True
        @staticmethod
        def get_device_name(i): return "Fake GPU"
        @staticmethod
        def get_device_properties(i):
            return types.SimpleNamespace(total_memory=total_mem)
        @staticmethod
        def mem_get_info(): return (free_mem, total_mem)
        @staticmethod
        def empty_cache(): pass
        @staticmethod
        def reset_peak_memory_stats(): pass
        @staticmethod
        def max_memory_allocated(): return 1e9

    class FakeTensor:
        def __init__(self, bs): self.bs = bs
        def pow(self, p): return self
        def mean(self): return self
        def float(self): return self
        def backward(self): pass
        def __add__(self, other): return self

    def fake_randn(bs, *a, **kw):
        call_sizes.append(bs)
        if bs >= oom_at_batch:
            raise FakeOOM(f"simulated OOM at {oom_at_batch}")
        return FakeTensor(bs)

    fake_torch = types.SimpleNamespace(cuda=FakeCudaModule, randn=fake_randn, autocast=FakeAutocast, bfloat16="bf16", float16="fp16")
    return fake_torch, call_sizes, FakeTensor


def _fake_model_builder(FakeTensor):
    class J:
        def to(self, d): return self
        def __call__(self, x): return FakeTensor(x.bs), FakeTensor(x.bs), FakeTensor(x.bs)
    class D:
        def to(self, d): return self
        def __call__(self, x): return FakeTensor(x.bs)
    return lambda: (J(), D())


def _run_detect(oom_at_batch, total_mem=80e9, free_mem=60e9, warn_above=256):
    """Run detect_max_batch_size against the simulated GPU; returns (result, call_sizes)."""
    fake_torch, call_sizes, FakeTensor = _build_fake_torch(oom_at_batch, total_mem, free_mem)
    sys.modules['torch'] = fake_torch
    if 'src.gpu_utils' in sys.modules:
        del sys.modules['src.gpu_utils']
    from src.gpu_utils import detect_max_batch_size
    model_builder = _fake_model_builder(FakeTensor)
    result = detect_max_batch_size(model_builder, FAKE_CUDA_DEVICE, img_size=224, warn_above=warn_above)
    return result, call_sizes


# 1. Large simulated GPU: out of memory at 2048, so the binary search must close
# the gap between 1024 and 2048.
result, call_sizes = _run_detect(oom_at_batch=2048)
print(f"\n[1] Batch sizes attempted: {call_sizes}")
print(f"[1] Returned batch size: {result}")

assert max(call_sizes) >= 1024, \
    f"Search stopped early: never attempted a batch size above {max(call_sizes)}."
assert 2048 in call_sizes, "Probe should have attempted 2048 and received the simulated OOM"
# Result without refinement: last doubling success x safety factor 0.75.
naive_doubling_result = int(1024 * 0.75)
assert result > naive_doubling_result, \
    f"Binary search found nothing better than the doubling-only result ({naive_doubling_result})"
assert result >= int(2000 * 0.75), \
    f"Refined result ({result}) is far short of the true ~2047 ceiling -- binary search " \
    f"isn't converging close enough"
print(f"PASS: reached the simulated out-of-memory boundary (2048), and the binary search "
      f"refined the result ({result}) beyond the doubling-only value ({naive_doubling_result})")

# 2. warn_above is informational only.
assert result > 256, "Result should be well above warn_above=256 on this simulated large GPU"
print(f"\n[2] PASS: warn_above=256 did not limit the result (returned {result})")

# 3. Small simulated GPU: out of memory at 32.
result2, call_sizes2 = _run_detect(oom_at_batch=32, total_mem=24e9, free_mem=20e9)
print(f"\n[3] Small-GPU batch sizes attempted: {call_sizes2}")
print(f"[3] Returned batch size: {result2}")
assert 32 in call_sizes2, "Probe should have attempted and received the simulated OOM at 32"
assert result2 < 256, "Small-GPU result should be well under warn_above -- no warning expected"
naive_small = int(16 * 0.75)
assert result2 >= naive_small, \
    f"Refined small-GPU result ({result2}) is below the doubling-only value ({naive_small})"
print(f"PASS: a small true limit is found too ({result2})")

# 4. The returned batch size was actually run.
assert result in call_sizes, \
    f"Returned batch size {result} was never run"
assert result2 in call_sizes2, \
    f"Returned batch size {result2} was never run"
print(f"\n[4] PASS: both returned batch sizes ({result}, {result2}) were run and verified")

# 5. One transient failure of the verification pass at the selected batch size.
DISCOVERY_OOM_AT = 64
clean_result, clean_sizes = _run_detect(oom_at_batch=DISCOVERY_OOM_AT, total_mem=24e9, free_mem=20e9)
print(f"\n[5] Discovery run (no fragmentation): safe_bs={clean_result} for a GPU whose true "
      f"ceiling is {DISCOVERY_OOM_AT - 1}")

_seen_sizes = []
def fake_randn_edge(bs, *a, **kw):
    _seen_sizes.append(bs)
    if bs >= DISCOVERY_OOM_AT:
        raise FakeOOM("main probe / binary-search OOM boundary")
    # Only the first attempt at the selected batch size fails.
    if bs == clean_result and _seen_sizes.count(clean_result) == 1:
        raise FakeOOM("simulated transient fragmentation, first attempt only")
    class T:
        def __init__(self, bs): self.bs = bs
        def pow(self, p): return self
        def mean(self): return self
        def float(self): return self
        def backward(self): pass
        def __add__(self, other): return self
    return T(bs)

class FakeCudaModuleEdgeCase:
    OutOfMemoryError = FakeOOM
    @staticmethod
    def is_available(): return True
    @staticmethod
    def is_bf16_supported(): return True
    @staticmethod
    def get_device_name(i): return "Fake Fragmented GPU"
    @staticmethod
    def get_device_properties(i): return types.SimpleNamespace(total_memory=24e9)
    @staticmethod
    def mem_get_info(): return (2e9, 24e9)
    @staticmethod
    def empty_cache(): pass
    @staticmethod
    def reset_peak_memory_stats(): pass
    @staticmethod
    def max_memory_allocated(): return 1e9

fake_torch_edge = types.SimpleNamespace(cuda=FakeCudaModuleEdgeCase, randn=fake_randn_edge, autocast=FakeAutocast, bfloat16="bf16", float16="fp16")
sys.modules['torch'] = fake_torch_edge
del sys.modules['src.gpu_utils']
from src.gpu_utils import detect_max_batch_size as detect_edge

class J2:
    def to(self, d): return self
    def __call__(self, x): return type(x)(x.bs), type(x)(x.bs), type(x)(x.bs)
class D2:
    def to(self, d): return self
    def __call__(self, x): return type(x)(x.bs)

result_edge = detect_edge(lambda: (J2(), D2()), FAKE_CUDA_DEVICE, img_size=224, warn_above=256)
print(f"[5] Sizes attempted (edge case): {_seen_sizes}")
print(f"[5] Returned batch size after ONE transient OOM: {result_edge}")

assert _seen_sizes.count(clean_result) >= 1, \
    f"The verification pass at safe_bs={clean_result} was never attempted -- test setup is wrong"
blind_halving_result = clean_result // 2
assert result_edge > blind_halving_result, \
    f"Backoff result ({result_edge}) is no better than halving ({blind_halving_result})"
assert result_edge >= int(clean_result * 0.85), \
    f"One transient failure should cost about one 10% step ({clean_result} -> {result_edge})"
print(f"PASS: one transient failure gives {result_edge} (one 10% step down from "
      f"{clean_result}; halving would give {blind_halving_result})")

# 6. Persistent failure of every verification attempt. The doubling and binary-search
# calls behave as in the clean run (identified by call count); every later call fails.
main_phase_call_count = len(clean_sizes) - 1
_seen_sizes_persistent = []
def fake_randn_persistent(bs, *a, **kw):
    _seen_sizes_persistent.append(bs)
    call_num = len(_seen_sizes_persistent)
    if call_num <= main_phase_call_count:
        if bs >= DISCOVERY_OOM_AT:
            raise FakeOOM("main probe / binary-search OOM boundary")
    else:
        raise FakeOOM("simulated persistent GPU pressure -- never recovers")
    class T:
        def __init__(self, bs): self.bs = bs
        def pow(self, p): return self
        def mean(self): return self
        def float(self): return self
        def backward(self): pass
        def __add__(self, other): return self
    return T(bs)

class FakeCudaModuleEdgeCase:
    OutOfMemoryError = FakeOOM
    @staticmethod
    def is_available(): return True
    @staticmethod
    def is_bf16_supported(): return True
    @staticmethod
    def get_device_name(i): return "Fake Fragmented GPU"
    @staticmethod
    def get_device_properties(i): return types.SimpleNamespace(total_memory=24e9)
    @staticmethod
    def mem_get_info(): return (2e9, 24e9)
    @staticmethod
    def empty_cache(): pass
    @staticmethod
    def reset_peak_memory_stats(): pass
    @staticmethod
    def max_memory_allocated(): return 1e9

class J2:
    def to(self, d): return self
    def __call__(self, x): return type(x)(x.bs), type(x)(x.bs), type(x)(x.bs)
class D2:
    def to(self, d): return self
    def __call__(self, x): return type(x)(x.bs)

fake_torch_persistent = types.SimpleNamespace(cuda=FakeCudaModuleEdgeCase, randn=fake_randn_persistent, autocast=FakeAutocast, bfloat16="bf16", float16="fp16")
sys.modules['torch'] = fake_torch_persistent
del sys.modules['src.gpu_utils']
from src.gpu_utils import detect_max_batch_size as detect_persistent

result_persistent = detect_persistent(lambda: (J2(), D2()), FAKE_CUDA_DEVICE, img_size=224, warn_above=256)
print(f"\n[6] Sizes attempted (persistent pressure): {_seen_sizes_persistent}")
print(f"[6] Returned batch size after exhausting backoff budget: {result_persistent}")
assert result_persistent >= 1, "Should still return SOMETHING (at least 1), not crash"
verification_phase_attempts = _seen_sizes_persistent[main_phase_call_count:]
assert len(verification_phase_attempts) >= 3, \
    f"Expected multiple backoff attempts before giving up, only saw " \
    f"{len(verification_phase_attempts)}: {verification_phase_attempts}"
print(f"PASS: after {len(verification_phase_attempts)} failed verification attempts the probe "
      f"stopped and returned a usable value")

print("\n" + "="*70)
print("ALL BATCH-SIZE PROBE TESTS PASSED")
print("="*70)
