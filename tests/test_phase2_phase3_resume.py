"""Resuming and stale results in Phase II (phases/phase2_sweep.py phase2) and
Phase III (phases/phase3_compression.py phase3 with its per-ratio lambda_sem
grid), with training and evaluation replaced by counters (no GPU, no data).

Phase II
  1. the first run evaluates every SNR point with eval_noise_seed(cfg, snr, ch);
  2. a second run with the same settings evaluates nothing and returns the same;
  3. a changed training setting, evaluation setting or threshold re-evaluates;
  4. a result file without its stamp (interrupted run) is re-evaluated.
Phase III
  5. first run: lambda_sem grid and production training at the low and high
     ratios, the production models reused at the medium ratio;
  6. second run: nothing trained, nothing evaluated;
  7. a changed Prototype setting retrains the Prototype grid and every production
     run, reuses the Max Logit grid and re-evaluates the medium ratio;
  8. one more seed trains only that seed;
  9. the stored default-SNR result is the sweep's own point;
 10. temporary training checkpoints are written outside OUTPUT_DIR.

Usage:  python3 tests/test_phase2_phase3_resume.py
"""
import os, sys, copy, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from src.config import CONFIG
import src.evaluate as E
import phases.phase2_sweep as P2
import phases.phase3_compression as P3

calls = {"eval": [], "train": 0, "grid_eval": 0, "ckpt_paths": []}

def fake_eval_both(*a, **kw):
    snr, ch = a[9], a[12]
    calls["eval"].append((snr, ch, kw.get("noise_seed")))
    r = {"snr_db": snr, "channel": ch}
    for d in ("max_logit", "prototype"):
        r.update({f"{d}_auc_roc_pixel": 0.7 + 0.001 * snr, f"{d}_fpr95": 0.5, f"{d}_dsem": 0.1,
                  f"{d}_sri_auc": 7.0, f"{d}_sri_op": 5.0})
    return r

class FakeModel:
    def __init__(self, *a, **k): pass
    def to(self, d): return self
    def eval(self): return self

def fake_train(*a, **kw):
    calls["train"] += 1
    calls["ckpt_paths"] += [str(kw["jscc_save_path"]), str(kw["detector_save_path"])]
    return {"loss": 0.5}

def fake_eval_one(*a, **kw):
    calls["grid_eval"] += 1
    return {"auc_roc_pixel": 0.7}

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

P2.eval_both_systems = fake_eval_both
P3.eval_both_systems = fake_eval_both
P3.SwinJSCCSystem = P3.MaxLogitDetector = P3.PrototypeDetector = FakeModel
P3.train_joint_system = fake_train
P3.calibrate_tau = lambda *a, **k: 0.3
P3.calibrate_sdop_threshold = lambda *a, **k: 0.1
E._eval_one_system = fake_eval_one

with tempfile.TemporaryDirectory() as d:
    cfg = copy.deepcopy(CONFIG)
    cfg["USE_RAYLEIGH"] = False
    n_snr = len(cfg["SNR_RANGE_DB"])

    # Phase II.
    rep = Path(d) / "run_0_rep0"; rep.mkdir()
    p1 = {"out": rep, "jscc_a": None, "jscc_b": None, "max_logit_det": None, "prototype_det": None,
          "tau_max_logit": 0.1, "tau_prototype": 0.2, "sdop_threshold_a": 0.0, "sdop_threshold_b": 0.0,
          "loaders": {"test": None}}
    calls["eval"].clear(); r1 = P2.phase2(p1, cfg, "cpu")
    check(f"Phase II first run: {n_snr} evaluations", len(calls["eval"]) == n_snr)
    check("Phase II noise seed = eval_noise_seed(cfg, snr, 'awgn')",
          all(s == E.eval_noise_seed(cfg, snr, ch) for snr, ch, s in calls["eval"]))
    calls["eval"].clear(); r2 = P2.phase2(p1, cfg, "cpu")
    check("Phase II same settings: 0 evaluations, same result", not calls["eval"] and r1 == r2)
    for desc, mutate in [("JOINT_LR_ML changed", lambda c, p: c.__setitem__("JOINT_LR_ML", 0.03)),
                         ("EVAL_NOISE_SEED changed", lambda c, p: c.__setitem__("EVAL_NOISE_SEED", 1)),
                         ("tau changed", lambda c, p: p.__setitem__("tau_prototype", 0.25))]:
        c2, p1b = copy.deepcopy(cfg), dict(p1); mutate(c2, p1b)
        calls["eval"].clear(); P2.phase2(p1b, c2, "cpu")
        check(f"Phase II {desc}: re-evaluated", len(calls["eval"]) == n_snr)
        # Restore the stamp for the current settings.
        P2.phase2(p1, cfg, "cpu")
    (rep / "phase2_stamp.json").unlink()
    calls["eval"].clear(); P2.phase2(p1, cfg, "cpu")
    check("Phase II result without stamp (interrupted): re-evaluated", len(calls["eval"]) == n_snr)

    # Phase III.
    cfg3 = copy.deepcopy(cfg)
    cfg3["SEEDS"] = [11, 22]
    out = Path(d) / "run_0"; out.mkdir()
    reps = [dict(p1, out=Path(d) / f"run_0_rep{i}", num_classes=13,
                 loaders={"test": None, "cls_train": None, "cls_val": None, "val_mixed": None}) for i in range(2)]
    p1_grid = {"out": out, "final_runs": reps,
               "grid_analysis": {"winning_lambda_sem": {"max_logit": 2.0, "prototype": 2.0}, "derived_patience": 15}}
    n_grid = (len(cfg3["LAMBDA_SEM_GRID_MAXLOGIT"]) + len(cfg3["LAMBDA_SEM_GRID_PROTOTYPE"])) * 5
    n_nondef = len(cfg3["CHANNEL_SYMBOLS"]) - 1

    def run(c):
        calls["eval"].clear(); calls["train"] = 0; calls["grid_eval"] = 0
        return P3.phase3(p1_grid, c, "cpu")

    res = run(cfg3)
    check(f"Phase III first run: {n_nondef * n_grid} grid + {n_nondef * 2 * 2} production trainings",
          calls["train"] == n_nondef * n_grid + n_nondef * 2 * 2 and calls["grid_eval"] == n_nondef * n_grid)
    check(f"Phase III first run: every CR x seed swept ({3 * 2 * n_snr} evaluations)",
          len(calls["eval"]) == 3 * 2 * n_snr)
    check(f"Phase III temporary checkpoints written outside OUTPUT_DIR (local disk), "
          f"{len(calls['ckpt_paths'])} paths",
          calls["ckpt_paths"] and not any(x.startswith(str(d)) for x in calls["ckpt_paths"]))
    st = json.load(open(out / "phase3_multiseed_state.json"))
    d0 = str(cfg3["DEFAULT_SNR_DB"])
    check("Phase III r0 is the sweep's own DEFAULT_SNR_DB point",
          all(st[cr][s]["r0"] == st[cr][s]["sweep"][d0] for cr in st for s in st[cr]))
    run(cfg3)
    check("Phase III same settings: 0 trainings, 0 evaluations",
          calls["train"] == 0 and not calls["eval"] and calls["grid_eval"] == 0)
    c_pr = copy.deepcopy(cfg3); c_pr["WEIGHT_DECAY_PR"] = 1e-2
    run(c_pr)
    # The Max Logit grid entries do not depend on Prototype settings.
    n_grid_pr = len(cfg3["LAMBDA_SEM_GRID_PROTOTYPE"]) * 5
    check("Phase III Prototype setting changed: Prototype grid + production retrained at low/high, "
          "Max Logit grid reused, medium re-evaluated",
          calls["train"] == n_nondef * n_grid_pr + n_nondef * 2 * 2
          and calls["grid_eval"] == n_nondef * n_grid_pr and len(calls["eval"]) == 3 * 2 * n_snr)
    c_more = copy.deepcopy(c_pr); c_more["SEEDS"] = [11, 22, 33]
    try:
        res = run(c_more); crashed = None
    except Exception as e:
        crashed = repr(e)
    check(f"Phase III one more seed: only that seed trained, at all 3 CRs ({crashed})",
          crashed is None and calls["train"] == 3 * 2 and len(calls["eval"]) == 3 * n_snr
          and res["n_runs"] == 3)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All Phase II/III resume checks passed.")
