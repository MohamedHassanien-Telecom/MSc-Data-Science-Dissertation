"""Structure of one round of the Phase I search (phases/phase1_grid.py
_run_hyperparameter_search_to_convergence). The search steps are replaced by
scripted stubs that set the round's configuration and write its Step F results.
  1. the steps of a round run in the order 0, A, B, C, D, E, F;
  2. each round's Steps 0-D use the previous round's lambda_sem winners
     (none in round 0);
  3. the batch size is set again exactly when Step A changes a patch
     configuration (round 1 here), and not when nothing changes;
  4. with identical results in two rounds, the search stops after round 1;
  5. with BATCH_SIZE_OVERRIDE set, a patch change uses the override and the
     batch-size probe is not run.

Usage:  python3 tests/test_search_round_structure.py
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
import tempfile
from pathlib import Path

SEEDS = [1001, 2002, 3003, 4004, 5005]
STEP_F_AUC = {str(s): {"max_logit": 0.77 + 0.001 * i, "prototype": 0.76 + 0.001 * i} for i, s in enumerate(SEEDS)}
LAMBDA_SEM = {"max_logit": 1.5, "prototype": 2.0}

_log = []          # (step, round, lambda_sem_override)
SCRIPT = {}        # round -> (patch_grid, patch_dim) set by Step A
OUT = {"dir": None}

def _step(name):
    def f(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_override=None, round_idx=0):
        _log.append((name, round_idx, lambda_sem_override))
        if name == "patch_config":
            g, dm = SCRIPT[round_idx]
            cfg["PATCH_GRID_ML"] = cfg["PATCH_GRID_PR"] = g
            cfg["PATCH_DIM_ML"] = cfg["PATCH_DIM_PR"] = dm
    return f

def mock_grid(cfg, device, loaders, num_classes, nsym, grid_dir, round_idx=0):
    _log.append(("grid_search", round_idx, None)); return []

def mock_analyze(grid_results, patience):
    return {"winning_lambda_sem": dict(LAMBDA_SEM), "lambda_sem_margins": {}, "derived_patience": 15,
            "lr_mismatch_note": ""}

def mock_lambda_rec(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_for_run, round_idx=0):
    # The same per-seed Step F results in every round: no improvement.
    _log.append(("lambda_rec", round_idx, None))
    json.dump({str(cfg["LAMBDA_REC_ML"]): STEP_F_AUC},
              open(OUT["dir"] / f"lambda_rec_verify_state_round{round_idx}.json", "w"))

def mock_batch_detect(*a, **kw):
    _log.append(("batch_detect", None, None)); return 16

pg._verify_backbone_stage = _step("backbone_stage")
pg._verify_patch_config = _step("patch_config")
pg._verify_joint_lr = _step("joint_lr")
pg._verify_optimizer_hparams = _step("optimizer_hparams")
pg._verify_prototype_scale = _step("prototype_scale")
pg._run_grid_search = mock_grid
pg._analyze_grid_results = mock_analyze
pg._verify_lambda_rec = mock_lambda_rec
pg.detect_max_batch_size = mock_batch_detect
pg.build_dataloaders = lambda cfg: {"known_class_ids": list(range(5))}

def mk_cfg(**kw):
    cfg = {"PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8, "PATCH_GRID_PR": 14, "PATCH_DIM_PR": 8,
           "JOINT_LR_ML": 0.003, "JOINT_LR_PR": 0.003, "JOINT_MOMENTUM_ML": 0.9, "JOINT_MOMENTUM_PR": 0.9,
           "JOINT_LR_POWER_ML": 0.9, "JOINT_LR_POWER_PR": 0.9, "WEIGHT_DECAY_ML": 1e-4, "WEIGHT_DECAY_PR": 1e-4,
           "PROTOTYPE_SCALE": 32.0, "MAX_CONVERGENCE_ROUNDS": 5, "IMG_H": 64, "GRID_SEARCH_PATIENCE": 15,
           "LAMBDA_REC_ML": 1.0, "LAMBDA_REC_PR": 1.0, "MIN_REL_IMPROVE_ML": 0.01, "MIN_REL_IMPROVE_PR": 0.01,
           "BATCH_SIZE_WARN_ABOVE": 256, "BATCH_SIZE_SAFETY_FACTOR": 0.75, "SELECTION_SEEDS": SEEDS}
    cfg.update(kw); return cfg

def run(cfg):
    OUT["dir"] = Path(tempfile.mkdtemp()); _log.clear()
    return pg._run_hyperparameter_search_to_convergence(cfg, "cpu", {"known_class_ids": list(range(5))},
                                                         5, 2048, OUT["dir"], "cpu")

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("=" * 70); print("TEST A: Step A changes the patch configuration in round 1"); print("=" * 70)
SCRIPT.clear(); SCRIPT.update({0: (14, 8), 1: (8, 16)})
cfg = mk_cfg()
analysis, _, _, snaps, outcome = run(cfg)
order0 = [n for n, r, _ in _log if r == 0]
check(f"round 0 runs the steps in order ({order0})",
      order0 == ["backbone_stage", "patch_config", "joint_lr", "optimizer_hparams", "prototype_scale",
                 "grid_search", "lambda_rec"])
lam = {r: {n: l for n, rr, l in _log if rr == r} for r in (0, 1)}
check("round 0 Steps 0-D run without a lambda_sem override; round 1 uses round 0's winners",
      all(lam[0][s] is None for s in ("backbone_stage", "patch_config", "joint_lr", "optimizer_hparams", "prototype_scale"))
      and all(lam[1][s] == LAMBDA_SEM for s in ("backbone_stage", "patch_config", "joint_lr", "optimizer_hparams", "prototype_scale")))
n_probe = sum(1 for n, _, _ in _log if n == "batch_detect")
check(f"batch size set again once, after the round-1 patch change ({n_probe} probe run)", n_probe == 1)
check(f"identical results in rounds 0 and 1 -> stop after round 1 ({outcome}, {len(snaps)} rounds)",
      outcome == "converged_objective" and len(snaps) == 2)
check("round 1 records the new patch configuration", (snaps[1]["patch_grid_ml"], snaps[1]["patch_dim_ml"]) == (8, 16))

print("=" * 70); print("TEST B: nothing changes"); print("=" * 70)
SCRIPT.clear(); SCRIPT.update({0: (14, 8), 1: (14, 8)})
cfg = mk_cfg()
analysis, _, _, snaps, outcome = run(cfg)
check(f"stops after round 1 ({outcome}, {len(snaps)} rounds) without running the probe",
      outcome == "converged_objective" and len(snaps) == 2 and not any(n == "batch_detect" for n, _, _ in _log))

print("=" * 70); print("TEST C: BATCH_SIZE_OVERRIDE with a patch change"); print("=" * 70)
SCRIPT.clear(); SCRIPT.update({0: (8, 16)})
cfg = mk_cfg(BATCH_SIZE_OVERRIDE=90, MAX_CONVERGENCE_ROUNDS=1)
run(cfg)
check(f"the override is used ({cfg.get('BATCH_SIZE')}) and the probe is not run",
      cfg.get("BATCH_SIZE") == 90 and not any(n == "batch_detect" for n, _, _ in _log))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All search-round structure checks passed.")
