"""Step C of the search (phases/phase1_grid.py _verify_one_optimizer_param) on a
complete stored state (momentum, three candidates x five seeds; per-seed values
recorded from a completed momentum step):
  1. nothing is retrained (the evaluation stub raises if called);
  2. the winner is tested with the paired Friedman test and Kendall's W, not the
     unpaired Kruskal-Wallis test;
  3. each detector gets the candidate with the highest mean (Max Logit 0.9,
     Prototype 0.95);
  4. the summary file records the test name and effect-size type per detector.

Usage:  python3 tests/test_optimizer_step_paired_test.py
"""
import sys, json, types, random, os
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
def fake_eval_never_called(*a, **kw):
    raise AssertionError("eval was called -- this means training was NOT correctly "
                          "skipped for an already-complete (candidate, seed) pair")

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
    eval_both_systems=lambda *a, **kw: {}, _eval_one_system=fake_eval_never_called)

from phases.phase1_grid import _verify_one_optimizer_param

import tempfile
from pathlib import Path

tmpdir = Path(tempfile.mkdtemp())
verify_dir = tmpdir / "momentum"
verify_dir.mkdir(parents=True)
# _verify_one_optimizer_param reads its state file from verify_dir.parent.
state_path = verify_dir.parent / "joint_momentum_verify_state_round0.json"

# Stored per-seed AUROCs of a completed momentum step.
_real_momentum_state = {
    "0.8": {
        "1001": {"max_logit": 0.7718, "prototype": 0.7344, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "2002": {"max_logit": 0.7755, "prototype": 0.5747, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "3003": {"max_logit": 0.7948, "prototype": 0.7660, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "4004": {"max_logit": 0.7759, "prototype": 0.6427, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "5005": {"max_logit": 0.7752, "prototype": 0.6793, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
    },
    "0.9": {
        "1001": {"max_logit": 0.7860, "prototype": 0.7317, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "2002": {"max_logit": 0.7755, "prototype": 0.5747, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "3003": {"max_logit": 0.7948, "prototype": 0.7660, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "4004": {"max_logit": 0.7810, "prototype": 0.6702, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "5005": {"max_logit": 0.7793, "prototype": 0.6883, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
    },
    "0.95": {
        "1001": {"max_logit": 0.7762, "prototype": 0.7472, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "2002": {"max_logit": 0.7820, "prototype": 0.6432, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "3003": {"max_logit": 0.7949, "prototype": 0.7752, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "4004": {"max_logit": 0.7872, "prototype": 0.6324, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
        "5005": {"max_logit": 0.7711, "prototype": 0.6557, "_patch_grid_ml": 14, "_patch_dim_ml": 8,
                  "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
                  "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9, "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4, "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0},
    },
}
with open(state_path, "w") as f:
    json.dump(_real_momentum_state, f)

cfg = {
    "PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8, "PATCH_GRID_PR": 14, "PATCH_DIM_PR": 8,
    "JOINT_LR_ML": 0.007, "JOINT_LR_PR": 0.02,
    "JOINT_MOMENTUM_ML": 0.9, "JOINT_MOMENTUM_PR": 0.9,
    "JOINT_LR_POWER_ML": 0.9, "JOINT_LR_POWER_PR": 0.9,
    "WEIGHT_DECAY_ML": 1e-4, "WEIGHT_DECAY_PR": 1e-4,
    "MIN_REL_IMPROVE_ML": 0.01, "MIN_REL_IMPROVE_PR": 0.01,
    "PROTOTYPE_SCALE": 32.0,
    "JOINT_MOMENTUM_CANDIDATES": [0.8, 0.9, 0.95], "OPTIMIZER_HPARAM_VERIFY_N": 5,
    "SELECTION_SEEDS": [1001, 2002, 3003, 4004, 5005],
    "GRID_SEARCH_PATIENCE": 15, "JOINT_MAX_EPOCHS": 100,
    "LAMBDA_SEM_GRID_MAXLOGIT": [1.0, 1.5, 2.0], "LAMBDA_SEM_GRID_PROTOTYPE": [1.5, 2.0, 2.5], "PROVISIONAL_LAMBDA_SEM_ML": 1.5, "PROVISIONAL_LAMBDA_SEM_PR": 2.0, "LR_POWER_WD_VERIFY_N": 5,
    "TARGET_TPR": 0.95, "CLS_FIT_SNR": 40.0, "SDOP_CALIBRATION_PERCENTILE": 90,
    "DEFAULT_SNR_DB": 0, "LAMBDA_REC_ML": 1.0, "LAMBDA_REC_PR": 1.0,
}
_dummy_loaders = {"cls_train": None, "cls_val": None, "val_mixed": None, "test": None}

print("=" * 70)
print("TEST: resume from a complete state -> no retraining, winner by the paired test")
print("=" * 70)

import io, contextlib
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    _verify_one_optimizer_param(
        cfg, "cpu", _dummy_loaders, 5, 2048, verify_dir,
        param_name="JOINT_MOMENTUM", candidates=[0.8, 0.9, 0.95],
        lambda_sem_for_run={"max_logit": 1.5, "prototype": 2.0},
        round_tag="round0", verify_n=5, round_idx=0)
output = captured.getvalue()
print(output)

assert _calls["train"] == 0, (
    f"Expected 0 new training calls (all 15 (candidate, seed) pairs stored), got {_calls['train']}")
print("PASS: zero retraining triggered -- fully resumed from existing state, as required")

assert "Friedman" in output, "Expected 'Friedman' in output -- paired test was not used"
assert "Kendall's W" in output, "Expected \"Kendall's W\" in output -- paired effect size was not used"
assert "Kruskal-Wallis" not in output, "The unpaired Kruskal-Wallis test was reported"
print("PASS: Friedman test and Kendall's W used")

assert cfg["JOINT_MOMENTUM_ML"] == 0.9, f"Expected ML to confirm 0.9, got {cfg['JOINT_MOMENTUM_ML']}"
assert cfg["JOINT_MOMENTUM_PR"] == 0.95, f"Expected PR to move to 0.95, got {cfg['JOINT_MOMENTUM_PR']}"
print("PASS: each detector selected its highest-mean candidate (Max Logit 0.9, Prototype 0.95)")

summary_path = verify_dir.parent / "joint_momentum_verify_summary_round0.json"
summary = json.load(open(summary_path))
for det in ("max_logit", "prototype"):
    assert "test_name" in summary["per_detector"][det]
    assert "Friedman" in summary["per_detector"][det]["test_name"]
    assert summary["per_detector"][det]["effect_size_type"] == "Kendall's W"
print("PASS: the summary file records the test name and effect-size type")

print("\n" + "=" * 70)
print("ALL CHECKS PASSED -- a complete stored state is reused without retraining and")
print("the winner is decided with the paired test.")
print("=" * 70)
