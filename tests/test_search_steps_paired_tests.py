"""Paired tests in the other search steps (phases/phase1_grid.py), each resumed from
a complete stored state (the evaluation stub raises if called):
  1. Step A, _verify_patch_config (two candidates): Wilcoxon signed-rank test,
     nothing retrained, (14, 8) kept for both detectors;
  2. Step D, _verify_prototype_scale: Friedman test, highest-mean scale selected;
  3. Step F, _verify_lambda_rec: Friedman test per detector, highest-mean value;
  4. Step E, _analyze_grid_results: highest-AUROC lambda_sem per detector, with
     the Wilcoxon p-value of best against second best.
The per-seed values are illustrative.

Usage:  python3 tests/test_search_steps_paired_tests.py
"""
import sys, json, types, random, os, io, contextlib, tempfile
for mod in list(sys.modules.keys()):
    if mod.startswith('phases') or mod.startswith('src'):
        del sys.modules[mod]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path

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
    raise AssertionError("eval was called -- an already-complete (candidate, seed) "
                          "pair was retrained instead of skipped")

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

from phases.phase1_grid import _verify_patch_config, _verify_prototype_scale, _verify_lambda_rec, _analyze_grid_results

_dummy_loaders = {"cls_train": None, "cls_val": None, "val_mixed": None, "test": None}
_BASE_CFG = {
    "PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8, "PATCH_GRID_PR": 14, "PATCH_DIM_PR": 8,
    "JOINT_LR_ML": 0.007, "JOINT_LR_PR": 0.02,
    "JOINT_MOMENTUM_ML": 0.9, "JOINT_MOMENTUM_PR": 0.95,
    "JOINT_LR_POWER_ML": 0.9, "JOINT_LR_POWER_PR": 1.0,
    "WEIGHT_DECAY_ML": 1e-5, "WEIGHT_DECAY_PR": 1e-3,
    "MIN_REL_IMPROVE_ML": 0.01, "MIN_REL_IMPROVE_PR": 0.01,
    "PROTOTYPE_SCALE": 32.0,
    "PATCH_CONFIG_CANDIDATES": [(14, 8), (8, 16)], "PATCH_CONFIG_VERIFY_N": 5,
    "PROTOTYPE_SCALE_GRID": [16.0, 32.0, 64.0], "PROTOTYPE_SCALE_VERIFY_N": 5,
    "LAMBDA_REC_SANITY_CANDIDATES": [0.5, 1.0, 2.0], "OPTIMIZER_HPARAM_VERIFY_N": 5,
    "SELECTION_SEEDS": [1001, 2002, 3003, 4004, 5005],
    "GRID_SEARCH_PATIENCE": 15, "JOINT_MAX_EPOCHS": 100,
    "LAMBDA_SEM_GRID_MAXLOGIT": [1.0, 1.5, 2.0], "LAMBDA_SEM_GRID_PROTOTYPE": [1.5, 2.0, 2.5], "PROVISIONAL_LAMBDA_SEM_ML": 1.5, "PROVISIONAL_LAMBDA_SEM_PR": 2.0, "LR_POWER_WD_VERIFY_N": 5,
    "TARGET_TPR": 0.95, "CLS_FIT_SNR": 40.0, "SDOP_CALIBRATION_PERCENTILE": 90,
    "DEFAULT_SNR_DB": 0, "LAMBDA_REC_ML": 1.0, "LAMBDA_REC_PR": 1.0,
}

n_pass = 0

print("=" * 70)
print("STEP A: _verify_patch_config")
print("=" * 70)
# 1. Step A: two candidates, Wilcoxon signed-rank test.
tmpdir1 = Path(tempfile.mkdtemp())
verify_dir1 = tmpdir1 / "patch"
verify_dir1.mkdir(parents=True)
state_path1 = verify_dir1.parent / "patch_config_verify_state_round0.json"

_seeds = ["1001", "2002", "3003", "4004", "5005"]
_ml_14 = [0.7823, 0.7691, 0.7756, 0.7712, 0.7633]
_pr_14 = [0.5644, 0.6012, 0.5401, 0.5891, 0.5832]
_ml_8  = [0.7401, 0.7523, 0.7489, 0.7502, 0.7460]
_pr_8  = [0.5612, 0.5834, 0.5701, 0.5722, 0.5756]

patch_state = {
    "14x14_8": {s: {"max_logit": ml, "prototype": pr, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0}
                for s, ml, pr in zip(_seeds, _ml_14, _pr_14)},
    "8x8_16":  {s: {"max_logit": ml, "prototype": pr, "_lambda_sem_ml": 1.5, "_lambda_sem_pr": 2.0}
                for s, ml, pr in zip(_seeds, _ml_8, _pr_8)},
}
with open(state_path1, "w") as f:
    json.dump(patch_state, f)

cfg1 = dict(_BASE_CFG)
_calls["train"] = 0
captured1 = io.StringIO()
with contextlib.redirect_stdout(captured1):
    _verify_patch_config(cfg1, "cpu", _dummy_loaders, 5, 2048, verify_dir1, round_idx=0)
out1 = captured1.getvalue()
print(out1)

assert _calls["train"] == 0, f"Expected 0 retraining calls, got {_calls['train']}"
assert "Wilcoxon" in out1, "Expected Wilcoxon in output"
assert "Mann-Whitney" not in out1, "The unpaired Mann-Whitney test was reported"
assert cfg1["PATCH_GRID_ML"] == 14 and cfg1["PATCH_DIM_ML"] == 8, "ML should confirm (14,8)"
assert cfg1["PATCH_GRID_PR"] == 14 and cfg1["PATCH_DIM_PR"] == 8, "PR should confirm (14,8)"
print("PASS: Step A (patch config) -- Wilcoxon used, no retraining, (14, 8) kept for both detectors\n")
n_pass += 1

print("=" * 70)
print("STEP D: _verify_prototype_scale")
print("=" * 70)
# 2. Step D: Prototype only, Friedman test.
tmpdir2 = Path(tempfile.mkdtemp())
verify_dir2 = tmpdir2 / "scale"
verify_dir2.mkdir(parents=True)
state_path2 = verify_dir2.parent / "prototype_scale_verify_state_round0.json"

_jitter2 = {"1001": 0.00, "2002": 0.01, "3003": -0.01, "4004": 0.02, "5005": -0.02}
scale_state = {}
for scale, base in [("16.0", 0.65), ("32.0", 0.69), ("64.0", 0.71)]:
    scale_state[scale] = {s: {"auc": base + _jitter2[s],
                                "_patch_grid_pr": 14, "_patch_dim_pr": 8, "_joint_lr_pr": 0.02,
                                "_joint_momentum_pr": 0.95, "_joint_lr_power_pr": 1.0,
                                "_weight_decay_pr": 1e-3, "_min_rel_improve_pr": 0.01,
                                "_lambda_sem_pr": 2.0,
                                # Stored results are stamped with the settings they were trained under.
                                "_backbone_stage_pr": 2}
                            for s in _seeds}
with open(state_path2, "w") as f:
    json.dump(scale_state, f)

cfg2 = dict(_BASE_CFG)
_calls["train"] = 0
captured2 = io.StringIO()
with contextlib.redirect_stdout(captured2):
    _verify_prototype_scale(cfg2, "cpu", _dummy_loaders, 5, 2048, verify_dir2, round_idx=0)
out2 = captured2.getvalue()
print(out2)

assert _calls["train"] == 0, f"Expected 0 retraining calls, got {_calls['train']}"
assert "Friedman" in out2, "Expected Friedman in output"
assert "Kruskal-Wallis" not in out2, "The unpaired Kruskal-Wallis test was reported"
assert cfg2["PROTOTYPE_SCALE"] == 64.0, f"Expected winner 64.0 (highest mean), got {cfg2['PROTOTYPE_SCALE']}"
print("PASS: Step D (PROTOTYPE_SCALE) -- Friedman used, no retraining, highest-mean candidate selected\n")
n_pass += 1

print("=" * 70)
print("STEP F: _verify_lambda_rec")
print("=" * 70)
# 3. Step F: Friedman test per detector.
tmpdir3 = Path(tempfile.mkdtemp())
verify_dir3 = tmpdir3 / "lrec"
verify_dir3.mkdir(parents=True)
state_path3 = verify_dir3.parent / "lambda_rec_verify_state_round0.json"

_meta = {"_patch_grid_ml": 14, "_patch_dim_ml": 8, "_patch_grid_pr": 14, "_patch_dim_pr": 8,
         "_joint_lr_ml": 0.007, "_joint_lr_pr": 0.02,
         "_joint_momentum_ml": 0.9, "_joint_momentum_pr": 0.95,
         "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 1.0,
         "_weight_decay_ml": 1e-5, "_weight_decay_pr": 1e-3,
         "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01,
         "_prototype_scale": 32.0, "_lambda_sem_max_logit": 1.5, "_lambda_sem_prototype": 2.0}
lrec_state = {}
for lrec, (base_ml, base_pr) in [("0.5", (0.78, 0.68)), ("1.0", (0.79, 0.70)), ("2.0", (0.77, 0.66))]:
    lrec_state[lrec] = {s: {"max_logit": base_ml + _jitter2[s], "prototype": base_pr + _jitter2[s], **_meta}
                          for s in _seeds}
with open(state_path3, "w") as f:
    json.dump(lrec_state, f)

cfg3 = dict(_BASE_CFG)
_calls["train"] = 0
captured3 = io.StringIO()
with contextlib.redirect_stdout(captured3):
    _verify_lambda_rec(cfg3, "cpu", _dummy_loaders, 5, 2048, verify_dir3,
                        lambda_sem_for_run={"max_logit": 1.5, "prototype": 2.0}, round_idx=0)
out3 = captured3.getvalue()
print(out3)

assert _calls["train"] == 0, f"Expected 0 retraining calls, got {_calls['train']}"
assert "Friedman" in out3, "Expected Friedman in output"
assert "Kruskal-Wallis" not in out3, "The unpaired Kruskal-Wallis test was reported"
assert cfg3["LAMBDA_REC_ML"] == 1.0, f"Expected ML winner 1.0, got {cfg3['LAMBDA_REC_ML']}"
assert cfg3["LAMBDA_REC_PR"] == 1.0, f"Expected PR winner 1.0, got {cfg3['LAMBDA_REC_PR']}"
print("PASS: Step F (LAMBDA_REC) -- Friedman used, no retraining, highest-mean candidate selected\n")
n_pass += 1

print("=" * 70)
print("STEP E: _analyze_grid_results (lambda_sem, best against second best)")
print("=" * 70)
# 4. Step E: lambda_sem grid, analysed directly from a results list.
# AUROC = 1 - loss, so the lowest-loss value (1.5 for Max Logit, 2.0 for Prototype) wins.
grid_results = []
for lam, base in [(1.0, 0.55), (1.5, 0.48), (2.0, 0.52)]:
    for i, s in enumerate([1001, 2002, 3003, 4004, 5005]):
        grid_results.append({"detector_type": "max_logit", "lambda_sem": lam,
                              "loss": base + _jitter2[str(s)],
                              "auc_roc_pixel": 1.0 - (base + _jitter2[str(s)]),
                              "best_epoch_idx": 10, "epochs_run": 20,
                              "epochs_since_best_at_stop": 10})
for lam, base in [(1.5, 0.60), (2.0, 0.50), (2.5, 0.53)]:
    for i, s in enumerate([1001, 2002, 3003, 4004, 5005]):
        grid_results.append({"detector_type": "prototype", "lambda_sem": lam,
                              "loss": base + _jitter2[str(s)],
                              "auc_roc_pixel": 1.0 - (base + _jitter2[str(s)]),
                              "best_epoch_idx": 10, "epochs_run": 20,
                              "epochs_since_best_at_stop": 10})

captured4 = io.StringIO()
with contextlib.redirect_stdout(captured4):
    analysis = _analyze_grid_results(grid_results, grid_search_patience=15)
out4 = captured4.getvalue()
winning, margins = analysis["winning_lambda_sem"], analysis["lambda_sem_margins"]

assert winning["max_logit"] == 1.5, f"Expected ML winner 1.5, got {winning['max_logit']}"
assert winning["prototype"] == 2.0, f"Expected PR winner 2.0, got {winning['prototype']}"
# The p-value is not printed; it must be computed (not NaN).
assert margins["max_logit"]["wilcoxon_p"] == margins["max_logit"]["wilcoxon_p"], "wilcoxon_p is NaN, Wilcoxon did not run"
print(f"ML winner: {winning['max_logit']}, margin p={margins['max_logit']['wilcoxon_p']:.4f}")
print(f"PR winner: {winning['prototype']}, margin p={margins['prototype']['wilcoxon_p']:.4f}")
print("PASS: Step E (lambda_sem) -- highest-AUROC winners selected, Wilcoxon p-value computed\n")
n_pass += 1

print("=" * 70)
print(f"ALL {n_pass}/4 SEARCH STEPS PASSED")
print("=" * 70)
