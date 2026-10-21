"""Step A of the search (phases/phase1_grid.py _verify_patch_config) uses six seeds
(PATCH_CONFIG_VERIFY_N = 6), because with five seeds the exact two-sided
Wilcoxon test cannot reach p < 0.05. Resumed from a stored state with five seeds:
  1. exactly 4 new trainings (the sixth seed x 2 candidates x 2 detectors);
  2. the 10 stored (candidate, seed) pairs are skipped;
  3. the trained seed is 6006;
  4. the statistics are reported at N = 6, still with the Wilcoxon test;
  5. the smallest two-sided p at N = 6 (0.0312) is reached for Max Logit;
  6. the first five selection seeds and the other settings are unchanged.
The per-seed values are illustrative.

Usage:  python3 tests/test_patch_config_six_seeds.py
"""
import sys, json, types, random, os, io, contextlib, tempfile
for mod in list(sys.modules.keys()):
    if mod.startswith('phases') or mod.startswith('src'):
        del sys.modules[mod]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path

_calls = {"train": 0, "trained_seeds": []}


class FakeModel:
    def __init__(self, tag): self.tag = tag
    def eval(self): pass
    def to(self, d): return self
    def parameters(self): return iter([types.SimpleNamespace(device="cpu")])


def fake_swin(nsym, **kw): return FakeModel("jscc")
def fake_maxlogit(*a, **kw): return FakeModel("max_logit")
def fake_prototype(*a, **kw): return FakeModel("prototype")
def fake_train_joint(jscc, det, *a, **kw):
    _calls["train"] += 1
    return {"loss": 0.5, "epochs_run": 20, "epochs_since_best_at_stop": 10, "best_epoch_idx": 10}
def fake_calibrate_tau(*a, **kw): return 0.5
def fake_calibrate_sdop(*a, **kw): return 0.1
def fake_set_seed(s): random.seed(s)

# The sixth seed favours (14, 8) for Max Logit, like the first five.
_SIXTH_SEED_RESULT = {"max_logit": {"14x14_8": 0.7750, "8x8_16": 0.7440},
                       "prototype": {"14x14_8": 0.5700, "8x8_16": 0.5730}}

def fake_eval_capturing(jscc, det, tau, sdop, ldr, snr, cfg, device, channel):
    grid, dim = (cfg["PATCH_GRID_ML"], cfg["PATCH_DIM_ML"]) if det.tag == "max_logit" else \
                (cfg["PATCH_GRID_PR"], cfg["PATCH_DIM_PR"])
    key = f"{grid}x{grid}_{dim}"
    _calls["trained_seeds"].append((det.tag, key))
    return {"auc_roc_pixel": _SIXTH_SEED_RESULT[det.tag][key]}

# torch and the src modules are replaced by stubs before phases.phase1_grid is imported.
sys.modules['torch'] = types.SimpleNamespace(load=lambda *a, **kw: {}, save=lambda *a, **kw: None, Tensor=type('T', (), {}))
sys.modules['src.config'] = types.SimpleNamespace(set_seed=fake_set_seed, CONFIG={}, LOADER_GENERATORS=[])
sys.modules['src.dataset'] = types.SimpleNamespace(
    build_dataloaders=lambda cfg: {}, detect_anomaly_value=lambda root: 255,
    discover_known_classes=lambda root, av: (list(range(5)), None))
sys.modules['src.gpu_utils'] = types.SimpleNamespace(detect_max_batch_size=lambda *a, **kw: 16)
sys.modules['src.models'] = types.SimpleNamespace(SwinJSCCSystem=fake_swin, MaxLogitDetector=fake_maxlogit, PrototypeDetector=fake_prototype)
sys.modules['src.train'] = types.SimpleNamespace(train_joint_system=fake_train_joint)
sys.modules['src.metrics'] = types.SimpleNamespace(calibrate_tau=fake_calibrate_tau, calibrate_sdop_threshold=fake_calibrate_sdop)
sys.modules['src.evaluate'] = types.SimpleNamespace(
    eval_both_systems=lambda *a, **kw: {}, _eval_one_system=fake_eval_capturing)

from phases.phase1_grid import _verify_patch_config

tmpdir = Path(tempfile.mkdtemp())
verify_dir = tmpdir / "patch"
verify_dir.mkdir(parents=True)
state_path = verify_dir.parent / "patch_config_verify_state_round0.json"

# Stored results of the first five seeds (illustrative values).
_seeds5 = ["1001", "2002", "3003", "4004", "5005"]
_ml_14 = [0.7823, 0.7691, 0.7756, 0.7712, 0.7633]
_pr_14 = [0.5644, 0.6012, 0.5401, 0.5891, 0.5832]
_ml_8  = [0.7401, 0.7523, 0.7489, 0.7502, 0.7460]
_pr_8  = [0.5612, 0.5834, 0.5701, 0.5722, 0.5756]

patch_state = {
    "14x14_8": {s: {"max_logit": ml, "prototype": pr, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0}
                for s, ml, pr in zip(_seeds5, _ml_14, _pr_14)},
    "8x8_16":  {s: {"max_logit": ml, "prototype": pr, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0}
                for s, ml, pr in zip(_seeds5, _ml_8, _pr_8)},
}
with open(state_path, "w") as f:
    json.dump(patch_state, f)

cfg = {
    "PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8, "PATCH_GRID_PR": 14, "PATCH_DIM_PR": 8,
    "JOINT_LR_ML": 0.007, "JOINT_LR_PR": 0.02,
    "JOINT_MOMENTUM_ML": 0.9, "JOINT_MOMENTUM_PR": 0.95,
    "JOINT_LR_POWER_ML": 0.9, "JOINT_LR_POWER_PR": 1.0,
    "WEIGHT_DECAY_ML": 1e-5, "WEIGHT_DECAY_PR": 1e-3,
    "MIN_REL_IMPROVE_ML": 0.01, "MIN_REL_IMPROVE_PR": 0.01,
    "PROTOTYPE_SCALE": 32.0,
    # Six seeds for Step A.
    "PATCH_CONFIG_CANDIDATES": [(14, 8), (8, 16)], "PATCH_CONFIG_VERIFY_N": 6,
    "SELECTION_SEEDS": [1001, 2002, 3003, 4004, 5005, 6006],
    "GRID_SEARCH_PATIENCE": 15, "JOINT_MAX_EPOCHS": 100,
    "LAMBDA_SEM_GRID_MAXLOGIT": [1.0, 1.5, 2.0], "LAMBDA_SEM_GRID_PROTOTYPE": [1.5, 2.0, 2.5], "PROVISIONAL_LAMBDA_SEM_ML": 1.5, "PROVISIONAL_LAMBDA_SEM_PR": 2.0, "LR_POWER_WD_VERIFY_N": 5,
    "TARGET_TPR": 0.95, "CLS_FIT_SNR": 40.0, "SDOP_CALIBRATION_PERCENTILE": 90,
    "DEFAULT_SNR_DB": 0, "LAMBDA_REC_ML": 1.0, "LAMBDA_REC_PR": 1.0,
}
_dummy_loaders = {"cls_train": None, "cls_val": None, "val_mixed": None, "test": None}

print("=" * 70)
print("TEST: Step A resumes from five stored seeds with the six-seed configuration")
print("=" * 70)
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    _verify_patch_config(cfg, "cpu", _dummy_loaders, 5, 2048, verify_dir, round_idx=0)
out = captured.getvalue()
print(out)

# 1. Only the sixth seed is trained.
assert _calls["train"] == 4, f"Expected exactly 4 new training calls, got {_calls['train']}"
print(f"PASS: exactly 4 new training calls made -- the 6th seed, for both candidates and both "
      f"detectors, nothing else retrained")

# 2. The stored pairs are skipped.
skip_count = out.count("SKIP, already done (resumed)")
assert skip_count == 10, f"Expected 10 skipped (candidate,seed) pairs (5 seeds x 2 candidates), got {skip_count}"
print(f"PASS: all 10 already-complete (candidate, seed) pairs correctly skipped")

# 3. The trained seed is 6006.
assert "seed=6006: training both systems" in out
print("PASS: seed=6006 specifically identified as the new, trained seed")

# 4. Statistics at N = 6, Wilcoxon test.
assert "N=6" in out
print("PASS: statistics correctly recomputed and reported at the new N=6")

assert "Wilcoxon" in out
print("PASS: Wilcoxon signed-rank test used (two candidates)")

# 5. The six-seed minimum p is reached.
assert "p=0.0312" in out, "Expected the six-seed Wilcoxon minimum (0.0312) for Max Logit"
print("PASS: Max Logit reaches p = 0.0312 (< 0.05) with the sixth seed")

# 6. Other settings and the first five seeds are unchanged.
assert cfg["JOINT_LR_ML"] == 0.007 and cfg["JOINT_LR_PR"] == 0.02, "Other config values were mutated unexpectedly"
assert cfg["SELECTION_SEEDS"][:5] == [1001, 2002, 3003, 4004, 5005], \
    "First 5 SELECTION_SEEDS must be unchanged for every other step's own [:5] slice"
print("PASS: SELECTION_SEEDS[:5] unchanged -- every other step's own seed slice is unaffected")

print("\n" + "=" * 70)
print("ALL CHECKS PASSED -- the five stored seeds are reused, only the sixth seed is")
print("trained, and the statistics are recomputed at N = 6.")
print("=" * 70)
