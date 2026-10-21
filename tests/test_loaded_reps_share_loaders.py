"""Production repetitions loaded from disk (phases/phase1_train.py load_phase1) use
the same data-loader set as the repetitions trained in the session, so that the
later phases do not keep a separate set of worker processes per repetition. The
real phase1_grid_search_and_train runs twice with training stubbed: the first
run trains every repetition; phase1_meta.json files matching the final
configuration are then written, so the second run loads every repetition.

Usage:  python3 tests/test_loaded_reps_share_loaders.py
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
    def train(self): pass
    def to(self, d): return self
    def parameters(self): return iter([types.SimpleNamespace(device="cpu")])
    def set_channel(self, *a, **kw): pass
    def state_dict(self): return {}
    def load_state_dict(self, sd): pass

def fake_swin(nsym, **kw): return FakeModel(f"jscc_{kw.get('patch_grid')}_{kw.get('patch_dim')}")
def fake_maxlogit(*a, **kw): return FakeModel("max_logit")
def fake_prototype(*a, **kw): return FakeModel("prototype")
def fake_train_joint(jscc, det, *a, **kw):
    _calls["train"] += 1
    lam = kw.get("lambda_sem", 0)
    r = random.Random(hash((str(jscc.tag), str(det.tag), lam)) & 0xffffffff)
    return {"loss": 0.5 + r.uniform(-0.02, 0.02), "epochs_run": 30,
            "epochs_since_best_at_stop": 15, "best_epoch_idx": 20}
def fake_calibrate_tau(*a, **kw): return 0.5
def fake_calibrate_sdop(*a, **kw): return 0.1
def fake_set_seed(s): random.seed(s)
# Returns the two metrics the search reads (pixel AUROC and FPR95).
def fake_eval_one(jscc, det, tau, sdop, ldr, snr, cfg, device, channel):
    r = random.Random(hash((str(jscc.tag), str(det.tag))) & 0xffffffff)
    return {"auc_roc_pixel": 0.75 + r.uniform(-0.02, 0.03),
            "fpr95": 0.4 + r.uniform(-0.05, 0.05)}
def fake_eval_both(*a, **kw): return {}
def fake_detect_batch_size(*a, **kw): return 32
def fake_build_dataloaders(cfg):
    return {"test": None, "val_mixed": None, "cls_train": None, "cls_val": None, "known_class_ids": list(range(5))}

# torch and the src modules are replaced by stubs before phases.phase1_grid is imported.
sys.modules['torch'] = types.SimpleNamespace(load=lambda *a,**kw: {}, save=lambda *a,**kw: None, Tensor=type('T',(),{}))
sys.modules['src.config'] = types.SimpleNamespace(set_seed=fake_set_seed, CONFIG={})
sys.modules['src.dataset'] = types.SimpleNamespace(
    build_dataloaders=fake_build_dataloaders, detect_anomaly_value=lambda root, sample_size=300: 255,
    discover_known_classes=lambda root, av: (list(range(5)), None))
sys.modules['src.gpu_utils'] = types.SimpleNamespace(detect_max_batch_size=fake_detect_batch_size)
sys.modules['src.models'] = types.SimpleNamespace(SwinJSCCSystem=fake_swin, MaxLogitDetector=fake_maxlogit, PrototypeDetector=fake_prototype)
sys.modules['src.evaluate'] = types.SimpleNamespace(eval_both_systems=fake_eval_both, _eval_one_system=fake_eval_one)
sys.modules['src.train'] = types.SimpleNamespace(train_joint_system=fake_train_joint)
sys.modules['src.metrics'] = types.SimpleNamespace(calibrate_tau=fake_calibrate_tau, calibrate_sdop_threshold=fake_calibrate_sdop)
sys.modules['phases.phase1_train'] = types.SimpleNamespace(
    train_one_system=lambda cfg, device, loaders, num_classes, nsym, det_name, lam, patience, save_dir, tag: {
        **fake_train_joint(fake_swin(nsym, patch_grid=cfg["PATCH_GRID_ML"] if det_name=="max_logit" else cfg["PATCH_GRID_PR"],
                                      patch_dim=cfg["PATCH_DIM_ML"] if det_name=="max_logit" else cfg["PATCH_DIM_PR"]),
                            fake_maxlogit() if det_name=="max_logit" else fake_prototype(), lambda_sem=lam),
        "jscc": fake_swin(nsym, patch_grid=cfg["PATCH_GRID_ML"] if det_name=="max_logit" else cfg["PATCH_GRID_PR"],
                           patch_dim=cfg["PATCH_DIM_ML"] if det_name=="max_logit" else cfg["PATCH_DIM_PR"]),
        "detector": fake_maxlogit() if det_name=="max_logit" else fake_prototype(),
    },
    phase1=lambda *a, **kw: {"jscc_a": FakeModel("a"), "jscc_b": FakeModel("b"),
                              "max_logit_det": FakeModel("ml"), "prototype_det": FakeModel("pr"),
                              "tau_max_logit": 0.5, "tau_prototype": 0.5,
                              "sdop_threshold_a": 0.1, "sdop_threshold_b": 0.1,
                              "num_classes": 5, "loaders": {}, "out": None},
    load_phase1=lambda *a, **kw: {"jscc_a": FakeModel("a"), "jscc_b": FakeModel("b"), "loaders": {"own": True}},
)

from phases.phase1_grid import phase1_grid_search_and_train

import tempfile
from pathlib import Path

def fresh_cfg(tmpdir):
    return {
        "OUTPUT_DIR": str(tmpdir), "DEFAULT_CR": "medium", "CHANNEL_SYMBOLS": {"low": 1024, "medium": 2048, "high": 4096},
        "STREETHAZARDS_ROOT": "/fake", "IMG_H": 64, "IMG_W": 64,
        "PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8, "PATCH_GRID_PR": 14, "PATCH_DIM_PR": 8,
        "PATCH_CONFIG_CANDIDATES": [(14, 8), (8, 16)], "PATCH_CONFIG_VERIFY_N": 5,
        "BACKBONE_STAGE_ML": 2, "BACKBONE_STAGE_PR": 2,
        # One stage candidate keeps the test fast.
        "STAGE_CANDIDATES": [2], "STAGE_VERIFY_N": 5,
        "JOINT_LR_ML": 0.003, "JOINT_LR_PR": 0.003, "JOINT_LR_CANDIDATES": [0.007, 0.003, 0.001, 0.0005], "JOINT_LR_VERIFY_N": 5,
        "JOINT_MOMENTUM_ML": 0.9, "JOINT_MOMENTUM_PR": 0.9, "JOINT_MOMENTUM_CANDIDATES": [0.8, 0.9, 0.95],
        "JOINT_LR_POWER_ML": 0.9, "JOINT_LR_POWER_PR": 0.9, "JOINT_LR_POWER_CANDIDATES": [0.5, 0.9, 1.0],
        "WEIGHT_DECAY_ML": 1e-4, "WEIGHT_DECAY_PR": 1e-4, "WEIGHT_DECAY_CANDIDATES": [1e-5, 1e-4, 1e-3],
        "OPTIMIZER_HPARAM_VERIFY_N": 5,
        "PROTOTYPE_SCALE": 32.0, "PROTOTYPE_SCALE_GRID": [16.0, 32.0, 64.0], "PROTOTYPE_SCALE_VERIFY_N": 5,
        "LAMBDA_SEM_GRID_MAXLOGIT": [1.0, 1.5, 2.0], "LAMBDA_SEM_GRID_PROTOTYPE": [1.5, 2.0, 2.5], "PROVISIONAL_LAMBDA_SEM_ML": 1.5, "PROVISIONAL_LAMBDA_SEM_PR": 2.0, "LR_POWER_WD_VERIFY_N": 5,
        "GRID_SEARCH_PATIENCE": 15, "SEED": 42, "SEEDS": [42, 123, 456, 789, 999],
        "SELECTION_SEEDS": [1001, 2002, 3003, 4004, 5005],
        "BATCH_SIZE_WARN_ABOVE": 256, "BATCH_SIZE_SAFETY_FACTOR": 0.75, "SDOP_CALIBRATION_PERCENTILE": 90, "ANOMALY_VALUE_SAMPLE_SIZE": 300,
        "TARGET_TPR": 0.95,
        "LAMBDA_REC_ML": 1.0, "LAMBDA_REC_PR": 1.0, "LAMBDA_REC_SANITY_CANDIDATES": [0.5, 1.0, 2.0],
        "MIN_REL_IMPROVE_ML": 0.01, "MIN_REL_IMPROVE_PR": 0.01, "MIN_REL_IMPROVE_CANDIDATES": [0.005, 0.01, 0.02], "JOINT_MAX_EPOCHS": 5, "CLS_FIT_SNR": 10,
        "DEFAULT_SNR_DB": 0, "N_RUNS": 5, "MAX_CONVERGENCE_ROUNDS": 5,
    }

tmpdir = Path(tempfile.mkdtemp())
cfg1 = fresh_cfg(tmpdir)
r1 = phase1_grid_search_and_train(cfg1, "cpu", run_idx=0)
# Write phase1_meta.json files that match the final configuration, so the
# second run loads every repetition instead of training it.
win = r1["grid_analysis"]["winning_lambda_sem"]
for rep in range(cfg1["N_RUNS"]):
    d = tmpdir / f"run_0_rep{rep}"; d.mkdir(parents=True, exist_ok=True)
    meta = {"lambda_sem_max_logit": win["max_logit"], "lambda_sem_prototype": win["prototype"],
            "prototype_scale": cfg1["PROTOTYPE_SCALE"]}
    for k in ("JOINT_LR", "PATCH_GRID", "PATCH_DIM", "JOINT_MOMENTUM", "WEIGHT_DECAY",
              "JOINT_LR_POWER", "MIN_REL_IMPROVE", "LAMBDA_REC"):
        for s in ("ML", "PR"):
            meta[f"{k.lower()}_{s.lower()}"] = cfg1[f"{k}_{s}"]
    json.dump(meta, open(d / "phase1_meta.json", "w"))

cfg2 = fresh_cfg(tmpdir)
_calls["train"] = 0
r2 = phase1_grid_search_and_train(cfg2, "cpu", run_idx=0)
ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
check("second run trained nothing (all repetitions loaded)", _calls["train"] == 0)
check(f"all {cfg2['N_RUNS']} repetitions returned", len(r2["final_runs"]) == cfg2["N_RUNS"])
check("every loaded repetition uses the shared loader set",
      all(rep["loaders"] is r2["loaders"] for rep in r2["final_runs"]))
check("the rest of load_phase1's result is kept", all(rep["jscc_a"].tag == "a" for rep in r2["final_runs"]))
print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All shared-loader checks passed.")
