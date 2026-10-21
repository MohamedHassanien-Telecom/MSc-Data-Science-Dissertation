"""src/train.py _unwrap_for_saving: a checkpoint saved from a torch.compile()
wrapper has the same state_dict keys as one saved from the plain model, so it
loads into a freshly built, uncompiled model.
  1. an uncompiled model is returned unchanged;
  2. a compiled-style wrapper (with _orig_mod) is unwrapped to the original model;
  3. the saved keys equal those a fresh, uncompiled model expects.
torch is stubbed; no GPU and no real torch.compile are needed.

Usage:  python3 tests/test_checkpoint_unwrap.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub the heavy imports of src/train.py; only _unwrap_for_saving is used.
import types
for mod_name in ["torch", "torchvision", "torchvision.transforms", "PIL", "PIL.Image"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["torch"].Tensor = object
sys.modules["torch"].nn = types.ModuleType("torch.nn")
sys.modules["torch"].nn.functional = types.ModuleType("torch.nn.functional")
sys.modules["torch"].optim = types.ModuleType("torch.optim")
sys.modules["torch"].autocast = lambda *a, **kw: None
sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"].compile = None
sys.modules["torch.nn"] = sys.modules["torch"].nn
sys.modules["torch.nn.functional"] = sys.modules["torch"].nn.functional
sys.modules["torch.optim"] = sys.modules["torch"].optim
sys.modules["torch"].utils = types.ModuleType("torch.utils")
sys.modules["torch"].utils.data = types.ModuleType("torch.utils.data")
sys.modules["torch"].utils.data.Dataset = object
sys.modules["torch"].utils.data.DataLoader = object
sys.modules["torch"].utils.data.Subset = object
sys.modules["torch"].utils.data.ConcatDataset = object
sys.modules["torch"].utils.data.RandomSampler = object
sys.modules["torch.utils"] = sys.modules["torch"].utils
sys.modules["torch.utils.data"] = sys.modules["torch"].utils.data
sys.modules["PIL"].Image = object

import importlib.util
spec = importlib.util.spec_from_file_location(
    "_train_for_test", os.path.join(os.path.dirname(__file__), "..", "src", "train.py"))
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
_unwrap_for_saving = _mod._unwrap_for_saving


# 1. An uncompiled model is returned unchanged.
class PlainModel:
    def __init__(self):
        self.weight = "real_weights"
    def state_dict(self):
        return {"layer.weight": self.weight}

plain = PlainModel()
result = _unwrap_for_saving(plain)
print(f"[1] Uncompiled model: unwrap returned the same object = {result is plain}")
assert result is plain, "An uncompiled model should be returned completely unchanged"
print("PASS: uncompiled models pass through untouched")

class FakeOptimizedModuleWrapper:
    """Stands in for torch.compile()'s wrapper: _orig_mod points to the original
    model, and its own state_dict uses prefixed keys.
    """
    def __init__(self, original):
        self._orig_mod = original
    def state_dict(self):
        return {"_orig_mod.layer.weight": "WRONG_KEY_STRUCTURE"}

# 2. A compiled-style wrapper is unwrapped to the original model.
original_model = PlainModel()
wrapped = FakeOptimizedModuleWrapper(original_model)

result2 = _unwrap_for_saving(wrapped)
print(f"\n[2] Compiled-style wrapper: unwrap returned the ORIGINAL model = {result2 is original_model}")
assert result2 is original_model, \
    "Should unwrap to the original model, not stay as the wrapper"

saved_keys = result2.state_dict()
print(f"[2] state_dict() from the unwrapped result: {saved_keys}")
assert "layer.weight" in saved_keys, \
    "Should get the ORIGINAL model's clean keys, not the wrapper's own (possibly " \
    "differently-prefixed) state_dict"
assert saved_keys["layer.weight"] == "real_weights"
print("PASS: compiled-style wrapper correctly unwraps to the original model's "
      "clean, portable state_dict keys -- not the wrapper's own")

# 3. The saved keys match those of a fresh, uncompiled model.
fresh_uncompiled_model = PlainModel()
fresh_expected_keys = set(fresh_uncompiled_model.state_dict().keys())
actually_saved_keys = set(_unwrap_for_saving(wrapped).state_dict().keys())
print(f"\n[3] Fresh, never-compiled model's expected keys: {fresh_expected_keys}")
print(f"[3] Keys actually saved (via the compiled wrapper, but unwrapped for saving): "
      f"{actually_saved_keys}")
assert fresh_expected_keys == actually_saved_keys, \
    "Keys saved from a compiled model must match those a fresh, uncompiled model expects"
print("PASS: a checkpoint saved from a compiled model has IDENTICAL keys to what a "
      "fresh, uncompiled model expects -- portable, not broken by compilation")

print("\n" + "="*70)
print("ALL TORCH.COMPILE() CHECKPOINT SAFETY TESTS PASSED")
print("="*70)
