"""Step B of the search (phases/phase1_grid.py _verify_joint_lr) selects the
learning rate separately for each detector:
  1. when Max Logit's AUROC peaks at the highest candidate and Prototype's at the
     lowest, each detector gets its own optimum, and the summary file records the
     per-detector results;
  2. when both peak at the same candidate, both get it.
Training and evaluation are stubbed; the evaluation stub reads the candidate
being tested from cfg.

Usage:  python3 tests/test_joint_lr_per_detector.py
"""
import sys, json, types, random
import os
for mod in list(sys.modules.keys()):
    if mod.startswith('phases') or mod.startswith('src'):
        del sys.modules[mod]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_calls = {"train": 0}

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

# 1. Max Logit peaks at the highest candidate, Prototype at the lowest.
CANDIDATES_1 = [0.01, 0.005, 0.001]
# Small per-seed jitter, so the paired test is not degenerate; never large
# enough to change either detector's best candidate.
_JITTER = {1001: 0.000, 2002: 0.004, 3003: -0.003, 4004: 0.002, 5005: -0.001}

def fake_eval_one_conflict(jscc, det, tau, sdop, ldr, snr, cfg, device, channel):
    lr = cfg["JOINT_LR_ML"]
    seed = cfg.get("_current_test_seed", 1001)
    jitter = _JITTER.get(seed, 0.0)
    if det.tag == "max_logit":
        base = {0.01: 0.90, 0.005: 0.80, 0.001: 0.70}[lr]
    else:
        base = {0.01: 0.60, 0.005: 0.75, 0.001: 0.88}[lr]
    return {"auc_roc_pixel": base + jitter}

# torch and the src modules are replaced by stubs before phases.phase1_grid is imported.
sys.modules['torch'] = types.SimpleNamespace(load=lambda *a,**kw: {}, save=lambda *a,**kw: None, Tensor=type('T',(),{}))
sys.modules['src.config'] = types.SimpleNamespace(set_seed=fake_set_seed, CONFIG={}, LOADER_GENERATORS=[])
sys.modules['src.dataset'] = types.SimpleNamespace(
    build_dataloaders=lambda cfg: {}, detect_anomaly_value=lambda root: 255,
    discover_known_classes=lambda root, av: (list(range(5)), None))
sys.modules['src.gpu_utils'] = types.SimpleNamespace(detect_max_batch_size=lambda *a, **kw: 16)
sys.modules['src.models'] = types.SimpleNamespace(SwinJSCCSystem=fake_swin, MaxLogitDetector=fake_maxlogit, PrototypeDetector=fake_prototype)
sys.modules['src.train'] = types.SimpleNamespace(train_joint_system=fake_train_joint)
sys.modules['src.metrics'] = types.SimpleNamespace(calibrate_tau=fake_calibrate_tau, calibrate_sdop_threshold=fake_calibrate_sdop)
sys.modules['src.evaluate'] = types.SimpleNamespace(
    eval_both_systems=lambda *a, **kw: {}, _eval_one_system=fake_eval_one_conflict)

from phases.phase1_grid import _verify_joint_lr

# set_seed is called with each seed: record it so the evaluation stub can add
# that seed's jitter.
_orig_set_seed = fake_set_seed
def _capturing_set_seed(s):
    _orig_set_seed(s)
    cfg1["_current_test_seed"] = s
sys.modules['src.config'].set_seed = _capturing_set_seed

import tempfile
from pathlib import Path

tmpdir = Path(tempfile.mkdtemp())
verify_dir = tmpdir / "lr"; verify_dir.mkdir(parents=True)

_dummy_loaders = {"cls_train": None, "cls_val": None, "val_mixed": None, "test": None}

cfg1 = {
    "PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8, "PATCH_GRID_PR": 14, "PATCH_DIM_PR": 8,
    "JOINT_LR_ML": 0.005, "JOINT_LR_PR": 0.005,
    "JOINT_MOMENTUM_ML": 0.9, "JOINT_MOMENTUM_PR": 0.9,
    "JOINT_LR_POWER_ML": 0.9, "JOINT_LR_POWER_PR": 0.9,
    "WEIGHT_DECAY_ML": 1e-4, "WEIGHT_DECAY_PR": 1e-4,
    "JOINT_LR_CANDIDATES": CANDIDATES_1, "JOINT_LR_VERIFY_N": 5,
    "SELECTION_SEEDS": [1001, 2002, 3003, 4004, 5005],
    "GRID_SEARCH_PATIENCE": 15, "MIN_REL_IMPROVE_ML": 0.01, "MIN_REL_IMPROVE_PR": 0.01, "JOINT_MAX_EPOCHS": 100,
    "LAMBDA_SEM_GRID_MAXLOGIT": [1.0, 1.5, 2.0], "LAMBDA_SEM_GRID_PROTOTYPE": [1.5, 2.0, 2.5], "PROVISIONAL_LAMBDA_SEM_ML": 1.5, "PROVISIONAL_LAMBDA_SEM_PR": 2.0, "LR_POWER_WD_VERIFY_N": 5,
    "TARGET_TPR": 0.95, "CLS_FIT_SNR": 40.0, "SDOP_CALIBRATION_PERCENTILE": 90,
    "DEFAULT_SNR_DB": 10, "LAMBDA_REC_ML": 1.0, "LAMBDA_REC_PR": 1.0, "PROTOTYPE_SCALE": 32.0,
}

print("=" * 70)
print("SCENARIO 1: deliberate ML/PR conflict")
print("=" * 70)
_verify_joint_lr(cfg1, "cpu", _dummy_loaders, 5, 2048, verify_dir, round_idx=0)

print(f"\nResult: JOINT_LR_ML={cfg1['JOINT_LR_ML']}, JOINT_LR_PR={cfg1['JOINT_LR_PR']}")
assert cfg1["JOINT_LR_ML"] == 0.01, f"Expected ML to independently pick 0.01, got {cfg1['JOINT_LR_ML']}"
assert cfg1["JOINT_LR_PR"] == 0.001, f"Expected PR to independently pick 0.001, got {cfg1['JOINT_LR_PR']}"
assert cfg1["JOINT_LR_ML"] != cfg1["JOINT_LR_PR"], "The two detectors must be genuinely independent here"
print("PASS: Max Logit and Prototype each selected their own optimum")

summary_path = verify_dir.parent / "joint_lr_verify_summary_round0.json"
assert summary_path.exists(), "Summary file must be written"
summary = json.load(open(summary_path))
assert "per_detector" in summary, "Summary must report per-detector results"
assert summary["per_detector"]["max_logit"]["best_lr"] == 0.01
assert summary["per_detector"]["prototype"]["best_lr"] == 0.001
print("PASS: summary file correctly records per-detector results independently")

print("\n" + "=" * 70)
print("SCENARIO 2: ML and PR agree on the same winner")
print("=" * 70)

# 2. Both detectors peak at 0.01.
def fake_eval_one_agree(jscc, det, tau, sdop, ldr, snr, cfg, device, channel):
    lr = cfg["JOINT_LR_ML"]
    seed = cfg.get("_current_test_seed", 1001)
    jitter = _JITTER.get(seed, 0.0)
    if det.tag == "max_logit":
        base = {0.01: 0.90, 0.005: 0.80, 0.001: 0.70}[lr]
    else:
        base = {0.01: 0.85, 0.005: 0.75, 0.001: 0.65}[lr]
    return {"auc_roc_pixel": base + jitter}

sys.modules['src.evaluate'].evaluate = None
sys.modules['src.evaluate']._eval_one_system = fake_eval_one_agree
import importlib
import phases.phase1_grid as pg
# Reload so phase1_grid picks up the new evaluation stub.
importlib.reload(pg)

cfg2 = dict(cfg1)
cfg2["JOINT_LR_ML"] = 0.005
cfg2["JOINT_LR_PR"] = 0.005
tmpdir2 = Path(tempfile.mkdtemp())
verify_dir2 = tmpdir2 / "lr"; verify_dir2.mkdir(parents=True)
pg._verify_joint_lr(cfg2, "cpu", _dummy_loaders, 5, 2048, verify_dir2, round_idx=0)

print(f"\nResult: JOINT_LR_ML={cfg2['JOINT_LR_ML']}, JOINT_LR_PR={cfg2['JOINT_LR_PR']}")
assert cfg2["JOINT_LR_ML"] == 0.01
assert cfg2["JOINT_LR_PR"] == 0.01
assert cfg2["JOINT_LR_ML"] == cfg2["JOINT_LR_PR"], "Independent selection agreeing is a valid, common outcome"
print("PASS: both detectors selected the common optimum")

print("\n" + "=" * 70)
print("ALL INDEPENDENT PER-DETECTOR LR SELECTION TESTS PASSED")
print("=" * 70)
