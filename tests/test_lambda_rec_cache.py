"""Step F of the search (phases/phase1_grid.py _verify_lambda_rec): its cached
results depend on lambda_sem, unlike the other optimiser steps, and running it
does not invalidate their caches.
  1. a fresh run trains every (candidate, seed, detector);
  2. a resume with the same lambda_sem trains nothing;
  3. a different lambda_sem retrains every candidate;
  4. running Step F does not invalidate the momentum step's cache.

Usage:  python3 tests/test_lambda_rec_cache.py
"""
import sys, os, json, random, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_calls = {"train": 0}
class FakeModel:
    def __init__(self, tag): self.tag = tag
    def eval(self): pass
    def to(self, d): return self
    def parameters(self): return iter([types.SimpleNamespace(device="cpu")])

def fake_swin(nsym, **kw): return FakeModel(f"jscc_{kw.get('patch_grid')}_{kw.get('patch_dim')}")
def fake_maxlogit(*a, **kw): return FakeModel("max_logit")
def fake_prototype(*a, **kw): return FakeModel("prototype")
def fake_train_joint(jscc, det, *a, **kw):
    _calls["train"] += 1
    return {"loss": 0.5, "epochs_run": 30, "epochs_since_best_at_stop": 15, "best_epoch_idx": 20}
def fake_calibrate_tau(*a, **kw): return 0.5
def fake_calibrate_sdop(*a, **kw): return 0.1
def fake_set_seed(s): random.seed(s)
def fake_eval_one(jscc, det, tau, sdop, ldr, snr, cfg, device, channel):
    return {"auc_roc_pixel": 0.75}

# torch and the src modules are replaced by stubs before phases.phase1_grid is imported.
sys.modules['torch'] = types.SimpleNamespace(load=lambda *a,**kw: {}, save=lambda *a,**kw: None, Tensor=type('T',(),{}))
sys.modules['src.config'] = types.SimpleNamespace(set_seed=fake_set_seed, CONFIG={}, LOADER_GENERATORS=[])
sys.modules['src.dataset'] = types.SimpleNamespace(
    build_dataloaders=lambda cfg: {}, detect_anomaly_value=lambda root: 255,
    discover_known_classes=lambda root, av: (list(range(5)), None))
sys.modules['src.gpu_utils'] = types.SimpleNamespace(detect_max_batch_size=lambda *a, **kw: 16)
sys.modules['src.models'] = types.SimpleNamespace(SwinJSCCSystem=fake_swin, MaxLogitDetector=fake_maxlogit, PrototypeDetector=fake_prototype)
sys.modules['src.evaluate'] = types.SimpleNamespace(eval_both_systems=lambda *a,**kw: {}, _eval_one_system=fake_eval_one)
sys.modules['src.train'] = types.SimpleNamespace(train_joint_system=fake_train_joint)
sys.modules['src.metrics'] = types.SimpleNamespace(calibrate_tau=fake_calibrate_tau, calibrate_sdop_threshold=fake_calibrate_sdop)

from phases.phase1_grid import _verify_lambda_rec, _verify_one_optimizer_param

import tempfile
from pathlib import Path

def fresh_cfg():
    return {
        "PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8, "PATCH_GRID_PR": 14, "PATCH_DIM_PR": 8,
        "JOINT_LR_ML": 0.003, "JOINT_LR_PR": 0.003,
        "JOINT_MOMENTUM_ML": 0.9, "JOINT_MOMENTUM_PR": 0.9, "JOINT_MOMENTUM_CANDIDATES": [0.8, 0.9, 0.95],
        "JOINT_LR_POWER_ML": 0.9, "JOINT_LR_POWER_PR": 0.9, "JOINT_LR_POWER_CANDIDATES": [0.5, 0.9, 1.0],
        "WEIGHT_DECAY_ML": 1e-4, "WEIGHT_DECAY_PR": 1e-4, "WEIGHT_DECAY_CANDIDATES": [1e-5, 1e-4, 1e-3],
        "MIN_REL_IMPROVE_ML": 0.01, "MIN_REL_IMPROVE_PR": 0.01, "MIN_REL_IMPROVE_CANDIDATES": [0.005, 0.01, 0.02],
        "OPTIMIZER_HPARAM_VERIFY_N": 5, "PROTOTYPE_SCALE": 32.0,
        "LAMBDA_REC_ML": 1.0, "LAMBDA_REC_PR": 1.0, "LAMBDA_REC_SANITY_CANDIDATES": [0.5, 1.0, 2.0],
        "GRID_SEARCH_PATIENCE": 15, "SEEDS": [42, 123, 456, 789, 999],
        "SELECTION_SEEDS": [1001, 2002, 3003, 4004, 5005],
        "BATCH_SIZE_WARN_ABOVE": 256, "SDOP_CALIBRATION_PERCENTILE": 90, "ANOMALY_VALUE_SAMPLE_SIZE": 300,
        "TARGET_TPR": 0.95, "JOINT_MAX_EPOCHS": 5, "CLS_FIT_SNR": 10,
        "DEFAULT_SNR_DB": 0,
    }

loaders = {"test": None, "val_mixed": None, "cls_train": None, "cls_val": None}
# 3 candidates x 5 seeds x 2 detectors.
EXPECTED_PER_PARAM = 3 * 5 * 2

# 1. Fresh run.
tmpdir1 = Path(tempfile.mkdtemp())
verify_dir1 = tmpdir1 / "lambda_rec"; verify_dir1.mkdir(parents=True)
cfg1 = fresh_cfg()
lambda_sem_round0 = {"max_logit": 1.5, "prototype": 2.0}
_calls["train"] = 0
_verify_lambda_rec(cfg1, "cpu", loaders, 5, 2048, verify_dir1, lambda_sem_round0, round_idx=0)
print(f"[1] Fresh run: {_calls['train']} trainings (expected {EXPECTED_PER_PARAM})")
assert _calls["train"] == EXPECTED_PER_PARAM

# 2. Resume with the same lambda_sem: fully cached.
cfg2 = fresh_cfg()
_calls["train"] = 0
_verify_lambda_rec(cfg2, "cpu", loaders, 5, 2048, verify_dir1, lambda_sem_round0, round_idx=0)
print(f"[2] Resume, same lambda_sem: {_calls['train']} trainings (expected 0)")
assert _calls["train"] == 0

# 3. A different lambda_sem invalidates Step F's cache.
cfg3 = fresh_cfg()
_calls["train"] = 0
lambda_sem_different = {"max_logit": 2.0, "prototype": 2.5}
_verify_lambda_rec(cfg3, "cpu", loaders, 5, 2048, verify_dir1, lambda_sem_different, round_idx=0)
print(f"[3] Same round, DIFFERENT lambda_sem: {_calls['train']} trainings (expected {EXPECTED_PER_PARAM})")
assert _calls["train"] == EXPECTED_PER_PARAM, \
    f"LAMBDA_REC's cache did not correctly invalidate when lambda_sem changed, got {_calls['train']}"
print("    PASS: LAMBDA_REC's cache correctly invalidates on a lambda_sem change alone")

# 4. Step F does not invalidate the momentum step's cache.
tmpdir4 = Path(tempfile.mkdtemp())
verify_dir4 = tmpdir4 / "optim"; verify_dir4.mkdir(parents=True)
lambda_rec_dir4 = tmpdir4 / "lambda_rec"; lambda_rec_dir4.mkdir(parents=True)
cfg4 = fresh_cfg()

_calls["train"] = 0
_verify_one_optimizer_param(cfg4, "cpu", loaders, 5, 2048, verify_dir4,
                              "JOINT_MOMENTUM", cfg4["JOINT_MOMENTUM_CANDIDATES"],
                              lambda_sem_round0, "round0", 5, round_idx=0)
momentum_trainings = _calls["train"]
print(f"\n[4] Momentum verified first: {momentum_trainings} trainings")

_calls["train"] = 0
_verify_lambda_rec(cfg4, "cpu", loaders, 5, 2048, lambda_rec_dir4, lambda_sem_round0, round_idx=0)
print(f"[4] LAMBDA_REC verified: {_calls['train']} trainings")

_calls["train"] = 0
_verify_one_optimizer_param(cfg4, "cpu", loaders, 5, 2048, verify_dir4,
                              "JOINT_MOMENTUM", cfg4["JOINT_MOMENTUM_CANDIDATES"],
                              lambda_sem_round0, "round0", 5, round_idx=0)
print(f"[4] Momentum re-checked after LAMBDA_REC ran: {_calls['train']} trainings (expected 0)")
assert _calls["train"] == 0, \
    "Running Step F invalidated the momentum step's cache"
print("    PASS: Step F does not invalidate the cache of the momentum step (or, by the same "
      "construction, of the other optimiser steps)")

print("\n" + "="*70)
print("ALL LAMBDA_REC VERIFICATION TESTS PASSED")
print("="*70)
