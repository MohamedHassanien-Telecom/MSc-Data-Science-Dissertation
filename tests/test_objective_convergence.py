"""Stopping rule of the Phase I search (phases/phase1_grid.py
_run_hyperparameter_search_to_convergence with phases/objective_convergence.py).

After every round r >= 1, each detector's Step F per-seed AUROCs are compared with
the previous round's (paired Wilcoxon test with the small-N tie rule). The search
stops when neither detector improved significantly, and each detector keeps the
configuration with the higher mean. The search steps are replaced by scripted
stubs that set the round's configuration and write its Step F results.
  1. with the per-seed AUROCs of round 0 and round 1 of the reported search, the
     search stops after round 1; Max Logit returns to its round-0 configuration
     (higher mean) and Prototype keeps round 1's;
  2. a significant improvement at round 1 continues the search, which stops at
     round 2 when the gain disappears;
  3. one detector still improving keeps the search going;
  4. a search still improving at MAX_CONVERGENCE_ROUNDS reports
     "max_rounds_exceeded".

Usage:  python3 tests/test_objective_convergence.py
"""
import sys, json, types
import os
for mod in list(sys.modules.keys()):
    if mod.startswith('phases') or mod.startswith('src'):
        del sys.modules[mod]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# torch and the src modules are replaced by stubs before phases.phase1_grid is imported.
sys.modules['torch'] = types.SimpleNamespace(load=lambda *a,**kw: {}, save=lambda *a,**kw: None, Tensor=type('T',(),{}))
sys.modules['torch.nn'] = types.SimpleNamespace(Module=object, functional=types.SimpleNamespace())
sys.modules['torch.nn.functional'] = types.SimpleNamespace()
sys.modules['src.config'] = types.SimpleNamespace(set_seed=lambda s: None, CONFIG={})
sys.modules['src.dataset'] = types.SimpleNamespace(
    build_dataloaders=lambda cfg: {"known_class_ids": list(range(5))},
    detect_anomaly_value=lambda root: 255, discover_known_classes=lambda root, av: (list(range(5)), None))
sys.modules['src.gpu_utils'] = types.SimpleNamespace(detect_max_batch_size=lambda *a, **kw: 16)
sys.modules['src.models'] = types.SimpleNamespace(
    SwinJSCCSystem=lambda *a, **kw: None, MaxLogitDetector=lambda *a, **kw: None, PrototypeDetector=lambda *a, **kw: None)
sys.modules['src.train'] = types.SimpleNamespace(train_joint_system=lambda *a, **kw: None)
sys.modules['src.evaluate'] = types.SimpleNamespace(eval_both_systems=lambda *a,**kw: {}, _eval_one_system=lambda *a,**kw: {"auc_roc_pixel": 0.75})
sys.modules['src.metrics'] = types.SimpleNamespace(calibrate_tau=lambda *a,**kw: 0.5, calibrate_sdop_threshold=lambda *a,**kw: 0.1)
sys.modules['phases.phase1_train'] = types.SimpleNamespace(
    train_one_system=lambda *a, **kw: {"loss": 0.5, "epochs_run": 30, "epochs_since_best_at_stop": 15,
                                         "best_epoch_idx": 20, "jscc": None, "detector": None},
    phase1=lambda *a, **kw: None, load_phase1=lambda *a, **kw: None)

import phases.phase1_grid as pg
import tempfile, json
from pathlib import Path

SEEDS = [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]

def state_for(ml, pr, lam="0.5"):
    """A Step F state file: per-seed AUROCs of the selected LAMBDA_REC."""
    return {lam: {str(s): {"max_logit": m, "prototype": p} for s, m, p in zip(SEEDS, ml, pr)}}

# Per-seed Step F AUROCs of rounds 0 and 1 of the reported search (five seeds).
R0 = state_for([0.7712, 0.7756, 0.7897, 0.7822, 0.7796], [0.7702, 0.7679, 0.7692, 0.7623, 0.7674])
R1 = state_for([0.7689, 0.7714, 0.7868, 0.7753, 0.7803], [0.7999, 0.7347, 0.7786, 0.7838, 0.7820])
# Round-end settings of the same two rounds.
CFG0 = {"JOINT_LR_ML": 0.02, "JOINT_LR_PR": 0.02, "JOINT_LR_POWER_ML": 1.0, "JOINT_LR_POWER_PR": 0.5}
CFG1 = {"JOINT_LR_ML": 0.03, "JOINT_LR_PR": 0.003, "JOINT_LR_POWER_ML": 1.5, "JOINT_LR_POWER_PR": 2.0}

SCRIPT = {}
tmpdir = None
_log = []

def mk_cfg(**kw):
    cfg = {"PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8, "PATCH_GRID_PR": 7, "PATCH_DIM_PR": 32,
           "JOINT_LR_ML": 0.02, "JOINT_LR_PR": 0.02, "JOINT_MOMENTUM_ML": 0.9, "JOINT_MOMENTUM_PR": 0.97,
           "JOINT_LR_POWER_ML": 1.0, "JOINT_LR_POWER_PR": 0.5, "WEIGHT_DECAY_ML": 1e-3, "WEIGHT_DECAY_PR": 1e-4,
           "MIN_REL_IMPROVE_ML": 0.04, "MIN_REL_IMPROVE_PR": 0.02, "PROTOTYPE_SCALE": 4.0,
           "MAX_CONVERGENCE_ROUNDS": 5, "IMG_H": 64, "GRID_SEARCH_PATIENCE": 15,
           "LAMBDA_REC_ML": 0.5, "LAMBDA_REC_PR": 0.5, "BATCH_SIZE_WARN_ABOVE": 256, "BATCH_SIZE_SAFETY_FACTOR": 0.75,
           "CONVERGENCE_TIE_ALPHA": 0.1, "SELECTION_SEEDS": SEEDS}
    cfg.update(kw); return cfg

# Scripted search steps: Step B and Step C set the round's settings; Step F writes
# the round's per-seed results.
def noop(*a, **k): _log.append("noop")
def mock_lr(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_override=None, round_idx=0):
    for k in ("JOINT_LR_ML", "JOINT_LR_PR"): cfg[k] = SCRIPT[round_idx]["cfg"][k]
def mock_optim(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_override=None, round_idx=0):
    for k in ("JOINT_LR_POWER_ML", "JOINT_LR_POWER_PR"): cfg[k] = SCRIPT[round_idx]["cfg"][k]
def mock_grid(cfg, device, loaders, num_classes, nsym, grid_dir, round_idx=0): return []
def mock_analyze(g, p): return {"winning_lambda_sem": {"max_logit": 2.0, "prototype": 2.0}, "lambda_sem_margins": {}, "derived_patience": 15, "lr_mismatch_note": ""}
def mock_lambda_rec(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_for_run, round_idx=0):
    json.dump(SCRIPT[round_idx]["state"], open(tmpdir / f"lambda_rec_verify_state_round{round_idx}.json", "w"))
    cfg["LAMBDA_REC_ML"] = cfg["LAMBDA_REC_PR"] = 0.5
    _log.append(("lambda_rec", round_idx))

pg._verify_backbone_stage = noop; pg._verify_patch_config = noop; pg._verify_prototype_scale = noop
pg._verify_joint_lr = mock_lr; pg._verify_optimizer_hparams = mock_optim
pg._run_grid_search = mock_grid; pg._analyze_grid_results = mock_analyze; pg._verify_lambda_rec = mock_lambda_rec
pg.build_dataloaders = lambda cfg: {"known_class_ids": list(range(5))}
loaders = {"known_class_ids": list(range(5))}

def run(cfg):
    global tmpdir
    tmpdir = Path(tempfile.mkdtemp()); _log.clear()
    return pg._run_hyperparameter_search_to_convergence(cfg, "cpu", loaders, 5, 2048, tmpdir, "cpu")

print("=" * 70); print("TEST 1: rounds 0 and 1 of the reported search -> stop at round 1, higher mean kept"); print("=" * 70)
SCRIPT.clear(); SCRIPT.update({0: {"cfg": CFG0, "state": R0}, 1: {"cfg": CFG1, "state": R1}})
cfg = mk_cfg()
analysis, _, _, snaps, outcome = run(cfg)
summ = json.load(open(tmpdir / "convergence_summary.json"))
assert outcome == "converged_objective", outcome
assert len(snaps) == 2, len(snaps)
d = summ["objective_decisions"][0]["per_detector"]
assert d["max_logit"]["keep"] == "prev" and not d["max_logit"]["improved"]
assert d["prototype"]["keep"] == "this" and d["prototype"]["tied"] and not d["prototype"]["improved"]
# Max Logit is restored to round 0's settings; Prototype keeps round 1's.
assert (cfg["JOINT_LR_ML"], cfg["JOINT_LR_POWER_ML"]) == (0.02, 1.0), (cfg["JOINT_LR_ML"], cfg["JOINT_LR_POWER_ML"])
assert (cfg["JOINT_LR_PR"], cfg["JOINT_LR_POWER_PR"]) == (0.003, 2.0)
print("PASS: stops at round 1; Max Logit returns to its higher-mean round-0 configuration; Prototype keeps round 1's")

print("=" * 70); print("TEST 2: a significant improvement at round 1 -> continue to round 2"); print("=" * 70)
R1_better = state_for([x + 0.02 for x in [0.7712, 0.7756, 0.7897, 0.7822, 0.7796]], [x + 0.02 for x in [0.7702, 0.7679, 0.7692, 0.7623, 0.7674]])
R2_same = R1_better
SCRIPT.clear(); SCRIPT.update({0: {"cfg": CFG0, "state": R0}, 1: {"cfg": CFG1, "state": R1_better}, 2: {"cfg": CFG1, "state": R2_same}})
cfg = mk_cfg()
analysis, _, _, snaps, outcome = run(cfg)
assert outcome == "converged_objective" and len(snaps) == 3, (outcome, len(snaps))
summ = json.load(open(tmpdir / "convergence_summary.json"))
assert summ["objective_decisions"][0]["stop"] is False and summ["objective_decisions"][1]["stop"] is True
assert cfg["JOINT_LR_ML"] == 0.03
print("PASS: continued after a +0.02 AUROC gain on every seed; stopped at round 2 when the gain disappeared")

print("=" * 70); print("TEST 3: one detector still improving -> do not stop"); print("=" * 70)
R1_mixed = state_for([0.7712, 0.7756, 0.7897, 0.7822, 0.7796], [x + 0.02 for x in [0.7702, 0.7679, 0.7692, 0.7623, 0.7674]])
SCRIPT.clear(); SCRIPT.update({0: {"cfg": CFG0, "state": R0}, 1: {"cfg": CFG1, "state": R1_mixed}, 2: {"cfg": CFG1, "state": R1_mixed}})
cfg = mk_cfg()
analysis, _, _, snaps, outcome = run(cfg)
assert len(snaps) == 3, len(snaps)
print("PASS: Prototype's significant gain kept the search going although Max Logit did not change")

print("=" * 70); print("TEST 4: still improving at the round limit -> max_rounds_exceeded"); print("=" * 70)
SCRIPT.clear()
for r in range(5):
    SCRIPT[r] = {"cfg": CFG1, "state": state_for([0.70 + 0.02 * r + 0.0001 * i for i in range(5)], [0.70 + 0.02 * r + 0.0001 * i for i in range(5)])}
cfg = mk_cfg()
analysis, _, _, snaps, outcome = run(cfg)
assert outcome == "max_rounds_exceeded", outcome
print("PASS: max_rounds_exceeded reported when the AUROC improves in every round")

print("\n" + "=" * 70); print("ALL STOPPING-RULE TESTS PASSED"); print("=" * 70)
