"""Step C of the search (phases/phase1_grid.py _verify_optimizer_hparams) selects
momentum, learning-rate decay power, weight decay and MIN_REL_IMPROVE in that
order, each under the values already selected before it. Its cached results are
reused or retrained accordingly:
  1. a fresh run trains every (parameter, candidate, seed, detector);
  2. a resume with valid caches trains nothing;
  3. a different momentum winner retrains the three later parameters only;
  4. a different decay-power winner retrains the two later parameters only;
  5. a resume part-way through the momentum sweep trains only the missing
     candidate, then the later parameters.

Usage:  python3 tests/test_optimizer_step_cache.py
"""
import sys, json, random, types
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

from phases.phase1_grid import _verify_optimizer_hparams, _verify_one_optimizer_param

import tempfile
from pathlib import Path

def fresh_cfg():
    return {
        "PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8, "PATCH_GRID_PR": 14, "PATCH_DIM_PR": 8,
        "JOINT_LR_ML": 0.003, "JOINT_LR_PR": 0.003,
        "JOINT_MOMENTUM_ML": 0.9, "JOINT_MOMENTUM_PR": 0.9, "JOINT_MOMENTUM_CANDIDATES": [0.8, 0.9, 0.95],
        "JOINT_LR_POWER_ML": 0.9, "JOINT_LR_POWER_PR": 0.9, "JOINT_LR_POWER_CANDIDATES": [0.5, 0.9, 1.0],
        "WEIGHT_DECAY_ML": 1e-4, "WEIGHT_DECAY_PR": 1e-4, "WEIGHT_DECAY_CANDIDATES": [1e-5, 1e-4, 1e-3],
        "OPTIMIZER_HPARAM_VERIFY_N": 5, "PROTOTYPE_SCALE": 32.0,
        "LAMBDA_SEM_GRID_MAXLOGIT": [1.0, 1.5, 2.0], "LAMBDA_SEM_GRID_PROTOTYPE": [1.5, 2.0, 2.5], "PROVISIONAL_LAMBDA_SEM_ML": 1.5, "PROVISIONAL_LAMBDA_SEM_PR": 2.0, "LR_POWER_WD_VERIFY_N": 5,
        "GRID_SEARCH_PATIENCE": 15, "SEEDS": [42, 123, 456, 789, 999],
        "SELECTION_SEEDS": [1001, 2002, 3003, 4004, 5005],
        "BATCH_SIZE_WARN_ABOVE": 256, "SDOP_CALIBRATION_PERCENTILE": 90, "ANOMALY_VALUE_SAMPLE_SIZE": 300,
        "TARGET_TPR": 0.95,
        "LAMBDA_REC_ML": 1.0, "LAMBDA_REC_PR": 1.0,
        "MIN_REL_IMPROVE_ML": 0.01, "MIN_REL_IMPROVE_PR": 0.01, "MIN_REL_IMPROVE_CANDIDATES": [0.005, 0.01, 0.02],
        "JOINT_MAX_EPOCHS": 5, "CLS_FIT_SNR": 10,
        "DEFAULT_SNR_DB": 0,
    }

loaders = {"test": None, "val_mixed": None, "cls_train": None, "cls_val": None}
lambda_sem = {"max_logit": 1.5, "prototype": 2.0}
# 3 candidates x 5 seeds x 2 detectors per parameter.
EXPECTED_PER_PARAM = 3 * 5 * 2

results = {}

# 1. Fresh run.
tmpdir1 = Path(tempfile.mkdtemp())
verify_dir1 = tmpdir1 / "optim"; verify_dir1.mkdir(parents=True)
cfg = fresh_cfg()
_calls["train"] = 0
_verify_optimizer_hparams(cfg, "cpu", loaders, 5, 2048, verify_dir1, lambda_sem_override=None)
results["1_fresh_run"] = _calls["train"]
print(f"[1] Fresh run: {_calls['train']} trainings (expected {4*EXPECTED_PER_PARAM})")
assert _calls["train"] == 4 * EXPECTED_PER_PARAM

# 2. Resume with valid caches.
cfg2 = fresh_cfg()
_calls["train"] = 0
_verify_optimizer_hparams(cfg2, "cpu", loaders, 5, 2048, verify_dir1, lambda_sem_override=None)
results["2_resume_valid"] = _calls["train"]
print(f"[2] Resume, fully valid caches: {_calls['train']} trainings (expected 0)")
assert _calls["train"] == 0

# 3. A different momentum winner.
tmpdir3 = Path(tempfile.mkdtemp())
verify_dir3 = tmpdir3 / "optim"; verify_dir3.mkdir(parents=True)
cfg3a = fresh_cfg()
_calls["train"] = 0
_verify_optimizer_hparams(cfg3a, "cpu", loaders, 5, 2048, verify_dir3, lambda_sem_override=None)
first_momentum = cfg3a["JOINT_MOMENTUM_ML"]
print(f"[3] Initial run settled momentum_ml={first_momentum}, momentum_pr={cfg3a['JOINT_MOMENTUM_PR']}, "
      f"lr_power_ml={cfg3a['JOINT_LR_POWER_ML']}, wd_ml={cfg3a['WEIGHT_DECAY_ML']}")

# Raise one non-winning momentum candidate's stored scores (both detectors) so
# that it wins on the re-run; the later parameters were cached under the first winner's momentum.
momentum_state_path = verify_dir3.parent / "joint_momentum_verify_state_round0.json"
momentum_state = json.load(open(momentum_state_path))
other_candidates = [c for c in cfg3a["JOINT_MOMENTUM_CANDIDATES"] if c != first_momentum]
force_winner = str(other_candidates[0])
for seed_key in momentum_state[force_winner]:
    momentum_state[force_winner][seed_key]["max_logit"] = 0.99
    momentum_state[force_winner][seed_key]["prototype"] = 0.99
json.dump(momentum_state, open(momentum_state_path, "w"))

cfg3b = fresh_cfg()
_calls["train"] = 0
_verify_optimizer_hparams(cfg3b, "cpu", loaders, 5, 2048, verify_dir3, lambda_sem_override=None)
print(f"[3] After forcing a different momentum winner: new momentum_ml={cfg3b['JOINT_MOMENTUM_ML']}, "
      f"momentum_pr={cfg3b['JOINT_MOMENTUM_PR']}, trainings={_calls['train']}")
results["3_after_momentum_change"] = _calls["train"]
assert cfg3b["JOINT_MOMENTUM_ML"] == float(force_winner), "ML momentum should have changed to the forced winner"
assert cfg3b["JOINT_MOMENTUM_PR"] == float(force_winner), "PR momentum should have changed to the forced winner"
# Momentum reuses its results; the three later parameters retrain.
assert _calls["train"] == 3 * EXPECTED_PER_PARAM, \
    f"expected exactly {3*EXPECTED_PER_PARAM} (LR-power + weight-decay + min_rel_improve " \
    f"retrained, momentum reused), got {_calls['train']}"
print(f"    PASS: momentum reused (0 retrains), LR-power, weight-decay, AND "
      f"min_rel_improve all correctly invalidated and retrained "
      f"({_calls['train']} = 3 x {EXPECTED_PER_PARAM})")

# 4. A different decay-power winner: only the two later parameters retrain.
tmpdir4 = Path(tempfile.mkdtemp())
verify_dir4 = tmpdir4 / "optim"; verify_dir4.mkdir(parents=True)
cfg4a = fresh_cfg()
_calls["train"] = 0
_verify_optimizer_hparams(cfg4a, "cpu", loaders, 5, 2048, verify_dir4, lambda_sem_override=None)
first_lrpower = cfg4a["JOINT_LR_POWER_ML"]
print(f"\n[4] Initial run settled lr_power_ml={first_lrpower}, lr_power_pr={cfg4a['JOINT_LR_POWER_PR']}")

lrpower_state_path = verify_dir4.parent / "joint_lr_power_verify_state_round0.json"
lrpower_state = json.load(open(lrpower_state_path))
other_lrpower = [c for c in cfg4a["JOINT_LR_POWER_CANDIDATES"] if c != first_lrpower]
force_winner_lrp = str(other_lrpower[0])
for seed_key in lrpower_state[force_winner_lrp]:
    lrpower_state[force_winner_lrp][seed_key]["max_logit"] = 0.99
    lrpower_state[force_winner_lrp][seed_key]["prototype"] = 0.99
json.dump(lrpower_state, open(lrpower_state_path, "w"))

cfg4b = fresh_cfg()
_calls["train"] = 0
_verify_optimizer_hparams(cfg4b, "cpu", loaders, 5, 2048, verify_dir4, lambda_sem_override=None)
print(f"[4] After forcing a different LR-power winner: new lr_power_ml={cfg4b['JOINT_LR_POWER_ML']}, "
      f"lr_power_pr={cfg4b['JOINT_LR_POWER_PR']}, trainings={_calls['train']}")
results["4_after_lrpower_change"] = _calls["train"]
assert cfg4b["JOINT_LR_POWER_ML"] == float(force_winner_lrp)
assert cfg4b["JOINT_LR_POWER_PR"] == float(force_winner_lrp)
assert _calls["train"] == 2 * EXPECTED_PER_PARAM, \
    f"expected exactly {2*EXPECTED_PER_PARAM} (weight-decay + min_rel_improve retrained), " \
    f"got {_calls['train']}"
print(f"    PASS: momentum untouched (upstream, 0 retrains), weight-decay AND "
      f"min_rel_improve correctly invalidated and retrained "
      f"({_calls['train']} = 2 x {EXPECTED_PER_PARAM}), momentum itself NOT retrained "
      f"(correctly not downstream of LR-power)")

print("\n" + "="*70)
print("SCENARIOS 1-4 PASSED")
print("="*70)
for k, v in results.items():
    print(f"  {k}: {v} trainings")

print("\n[5] Testing mid-sweep resume (disconnect partway through momentum's own loop)...")
# 5. Resume part-way through the momentum sweep: 2 of 3 candidates stored.
tmpdir5 = Path(tempfile.mkdtemp())
verify_dir5 = tmpdir5 / "optim"; verify_dir5.mkdir(parents=True)
cfg5 = fresh_cfg()

state_path_5 = verify_dir5.parent / "joint_momentum_verify_state_round0.json"
partial_state = {}
lambda_sem_for_run = {"max_logit": 1.5, "prototype": 2.0}
for cand in [0.8, 0.9]:
    partial_state[str(cand)] = {}
    for seed in [1001, 2002, 3003, 4004, 5005]:
        partial_state[str(cand)][str(seed)] = {
            "max_logit": 0.75, "prototype": 0.60,
            "_patch_grid_ml": 14, "_patch_dim_ml": 8, "_patch_grid_pr": 14, "_patch_dim_pr": 8,
            "_joint_lr_ml": 0.003, "_joint_lr_pr": 0.003,
            "_joint_momentum_ml": cand, "_joint_momentum_pr": cand,
            "_joint_lr_power_ml": 0.9, "_joint_lr_power_pr": 0.9,
            "_weight_decay_ml": 1e-4, "_weight_decay_pr": 1e-4,
            "_min_rel_improve_ml": 0.01, "_min_rel_improve_pr": 0.01,
            "_lambda_sem_ml": lambda_sem_for_run["max_logit"],
            "_lambda_sem_pr": lambda_sem_for_run["prototype"],
        }
json.dump(partial_state, open(state_path_5, "w"))

_calls["train"] = 0
_verify_optimizer_hparams(cfg5, "cpu", loaders, 5, 2048, verify_dir5, lambda_sem_override=None)
print(f"[5] Trainings after mid-sweep resume: {_calls['train']}")
# The missing momentum candidate (10) plus the three later parameters (30 each).
expected_5 = (1 * 5 * 2) + EXPECTED_PER_PARAM + EXPECTED_PER_PARAM + EXPECTED_PER_PARAM
print(f"    Expected: momentum's remaining 1 candidate ({1*5*2}) + lr_power full "
      f"({EXPECTED_PER_PARAM}) + weight_decay full ({EXPECTED_PER_PARAM}) + "
      f"min_rel_improve full ({EXPECTED_PER_PARAM}) = {expected_5}")
assert _calls["train"] == expected_5, f"expected {expected_5}, got {_calls['train']}"
print(f"    PASS: momentum correctly resumed mid-sweep, trained only its 1 remaining "
      f"candidate ({1*5*2} runs), not all 3 candidates again")

print("\n" + "="*70)
print("ALL 5 OPTIMISER-STEP CACHE SCENARIOS PASSED")
print("="*70)
