"""Controls of the dissertation (Appendix F) that use the Phase I models:
  1. location audit (phases/location_prior_check.py): partner images come from
     the other town; the shuffled labels and the location map equal brute-force
     computations; the matched AUROC equals src/evaluate.py's pixel AUROC on a
     real model with seeded noise; an image-independent score map gives matched
     about equal to shuffled, an image-dependent one does not; the run (scoring
     mocked) reproduces the Phase II values, writes its outputs and resumes;
  2. training over a range of SNRs, warm start (phases/multisnr_check.py): the
     collapse guard flags flat or input-independent models only; _train_pair
     loads each system's production weights before training (without
     downloading pretrained ones); run_multisnr_check(warm_start=True) fine-tunes
     from the seed-matched Phase I repetition folders, writes only its own files,
     resumes, and repeats the comparison without flagged seeds; the cold-start
     results are left untouched;
  3. detector x architecture control (phases/arch2x2_check.py): the analysis
     recovers known simple, main and interaction effects and the production-gap
     identity; the cross configurations swap only the architecture; the two new
     cells train for real (one epoch); the run (training and evaluation mocked)
     pairs seeds with production repetitions, writes its outputs and resumes;
  4. notebook: the cells exist and parse; the evaluation cells keep the Phase I
     models and come before the three training cells, which release them; the
     multi-SNR cells run the cold and the warm start.

Usage:  python3 tests/test_controls.py
"""
import os, sys, copy, json, tempfile, ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.config import CONFIG, set_seed
from src.models import SwinJSCCSystem, MaxLogitDetector, PrototypeDetector
from src.train import save_ckpt
from src.evaluate import _eval_one_system
import phases.location_prior_check as LP
import phases.multisnr_check as MS
import phases.arch2x2_check as AX

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

cfg = copy.deepcopy(CONFIG)
cfg["USE_TORCH_COMPILE"] = False
# The production architectures selected in Phase I.
cfg.update({"BACKBONE_STAGE_ML": 2, "PATCH_GRID_ML": 14, "PATCH_DIM_ML": 8,
            "BACKBONE_STAGE_PR": 3, "PATCH_GRID_PR": 7, "PATCH_DIM_PR": 32})
K = 13
snrs = list(cfg["SNR_RANGE_DB"])

class Obj:
    def __init__(self, **k): self.__dict__.update(k)

# 1. Location audit.
groups = ["t5"] * 7 + ["t6"] * 5
partner, rule = LP.partner_images(groups, 12)
check(f"partner of every image is in the other town ({rule})",
      all(groups[i] != groups[partner[i]] for i in range(12)) and rule == "image from the other town")
p1g, rule1 = LP.partner_images(["t5"] * 6, 6)
check("one town only: partner = next image", list(p1g) == [1, 2, 3, 4, 5, 0] and "next image" in rule1)
check("no group information: partner = next image", list(LP.partner_images(None, 4)[0]) == [1, 2, 3, 0])

rng = np.random.default_rng(0)
hw, n_img = 30, 12
labels = (rng.random(n_img * hw) < 0.2).astype(np.int64)
idx = rng.choice(labels.size, 100, replace=False)
ys = LP.shuffled_labels(labels, idx, hw, partner)
brute = np.array([labels[partner[i // hw] * hw + i % hw] for i in idx])
check("shuffled labels = partner image's label at the same pixel (brute force)", np.array_equal(ys, brute))
loc = LP.location_baseline_scores(labels, idx, hw, groups)
L2 = labels.reshape(n_img, hw)
brute_loc = np.array([np.mean([L2[j, i % hw] for j in range(n_img) if groups[j] != groups[i // hw]]) for i in idx])
check("location baseline = anomaly frequency over the other town (brute force)", np.allclose(loc, brute_loc))
loo = LP.location_baseline_scores(labels, idx, hw, None)
brute_loo = np.array([np.mean([L2[j, i % hw] for j in range(n_img) if j != i // hw]) for i in idx])
check("no towns: leave-one-out frequency (brute force)", np.allclose(loo, brute_loo))
check("pixel subsample identical to src/evaluate.py's and cached",
      np.array_equal(LP.pixel_subsample(250_000), np.random.RandomState(42).choice(250_000, 200_000, replace=False))
      and LP.pixel_subsample(250_000) is LP.pixel_subsample(250_000))

# Image-independent and image-dependent score maps.
hw2, n2 = 400, 600
grp2 = ["t5"] * 300 + ["t6"] * 300
# Location prior: anomalies more likely towards the end of the image.
prob = np.linspace(0.0, 0.2, hw2)
lab2 = (rng.random((n2, hw2)) < prob).astype(np.int64).ravel()
part2, _ = LP.partner_images(grp2, n2)
const = np.tile(prob, n2)
a_const = LP.location_aucs(lab2, const.astype(np.float64), hw2, part2)
oracle = lab2 + 0.3 * rng.random(lab2.size)
a_orc = LP.location_aucs(lab2, oracle, hw2, part2)
check(f"input-independent map: matched {a_const['matched']:.3f} ~= shuffled {a_const['shuffled']:.3f}",
      abs(a_const["matched"] - a_const["shuffled"]) < 0.02 and a_const["matched"] > 0.6)
check(f"image-dependent map: matched {a_orc['matched']:.3f} >> shuffled {a_orc['shuffled']:.3f}",
      a_orc["matched"] - a_orc["shuffled"] > 0.3)

# Matched AUROC = src/evaluate.py's pixel AUROC on a real model with seeded noise.
set_seed(3)
model = SwinJSCCSystem(256, pretrained=False, patch_grid=14, patch_dim=8, stage_idx=2).eval()
det = MaxLogitDetector(8, 14, K).eval()
g = torch.Generator().manual_seed(1)
batches = []
for _ in range(3):
    m = torch.zeros(2, 224, 224, dtype=torch.long); m[:, 150:, 60:160] = (torch.rand(2, 74, 100, generator=g) < 0.5).long()
    batches.append({"image": torch.randn(2, 3, 224, 224, generator=g), "mask": m,
                    "has_anomaly": torch.ones(2, dtype=torch.long)})
with torch.no_grad():
    ref = _eval_one_system(model, det, -1.0, 0.01, batches, 0, cfg, torch.device("cpu"), "awgn", noise_seed=777)
lab, sc, hw3 = LP.score_maps(model, det, batches, 0, torch.device("cpu"), "awgn", noise_seed=777)
a = LP.location_aucs(lab, sc, hw3, LP.partner_images(None, lab.size // hw3)[0])
check(f"matched AUROC == evaluate.py pixel AUROC on a real model ({a['matched']:.6f} vs {ref['auc_roc_pixel']:.6f})",
      a["matched"] == ref["auc_roc_pixel"] and hw3 == 224 * 224)
lab_b, sc_b, _ = LP.score_maps(model, det, batches, 0, torch.device("cpu"), "awgn", noise_seed=778)
check("a different noise seed changes the scores (noise is seeded, not ignored)", not np.array_equal(sc, sc_b))
dep = LP.input_dependence(model, det, batches, 0, cfg, torch.device("cpu"))
check("input_dependence runs on a real model (no town info -> next-image partner)",
      "matched_minus_shuffled" in dep and "next image" in dep["partner_rule"])

# Run with scoring mocked.
class FakeSubset:
    def __init__(self, n, paths):
        self.indices = list(range(n)); self.dataset = Obj(img_paths=paths)
    def __len__(self): return len(self.indices)
paths = [Path(f"/x/test/{t}/{i}.png") for i, t in enumerate(grp2)]
fake_loader = Obj(dataset=FakeSubset(n2, paths))
with tempfile.TemporaryDirectory() as d:
    ps = {"awgn": [[{"snr_db": s, "max_logit_auc_roc_pixel": None, "prototype_auc_roc_pixel": None} for s in snrs]
                   for _ in range(3)], "snr_range_db": snrs}
    def fake_scores(model, detector, loader, snr_db, device, channel_type="awgn", noise_seed=None):
        k = model.k
        s = oracle if k == "a" else loc_only
        return lab2, s.astype(np.float64), hw2
    # Location only, no image content.
    loc_only = const + 1e-6 * np.random.default_rng(5).random(const.size)
    idx2 = LP.pixel_subsample(lab2.size)
    for r in range(3):
        for j, s in enumerate(snrs):
            ps["awgn"][r][j]["max_logit_auc_roc_pixel"] = float(roc_auc_score(lab2[idx2], oracle[idx2]))
            ps["awgn"][r][j]["prototype_auc_roc_pixel"] = float(roc_auc_score(lab2[idx2], loc_only[idx2]))
    json.dump(ps, open(Path(d) / "phase2_per_seed.json", "w"))
    runs = [{"jscc_a": Obj(k="a"), "jscc_b": Obj(k="b"), "max_logit_det": None, "prototype_det": None,
             "tau_max_logit": -9.0, "tau_prototype": 0.02, "loaders": {"test": fake_loader}} for _ in range(3)]
    calls = []
    orig = LP.score_maps
    LP.score_maps = lambda *a_, **k_: (calls.append(1), fake_scores(*a_, **k_))[1]
    try:
        an = LP.run_location_prior_check({"final_runs": runs, "out": Path(d)}, cfg, torch.device("cpu"))
        n1 = len(calls)
        an2 = LP.run_location_prior_check({"final_runs": runs, "out": Path(d)}, cfg, torch.device("cpu"))
    finally:
        LP.score_maps = orig
    check(f"location run: {n1} scorings (3 reps x 2 systems x 4 SNRs), none on resume",
          n1 == 3 * 2 * 4 and len(calls) == n1)
    check(f"location run: matched reproduces the stored Phase II values (max |diff| {an['max_abs_diff_from_phase2']:.1e})",
          an["max_abs_diff_from_phase2"] < 1e-12)
    check(f"location run: town rule used and counted ({an['partner_rule']}, {an['town_counts']})",
          an["partner_rule"] == "image from the other town" and an["town_counts"] == {"t5": 300, "t6": 300})
    sh_pr = an["systems"]["prototype"]["40.0"]["location_share"]["mean"]
    sh_ml = an["systems"]["max_logit"]["40.0"]["location_share"]["mean"]
    check(f"location run: location share high for the location-driven system ({sh_pr:.2f}), low for the "
          f"image-driven one ({sh_ml:.2f})", sh_pr > 0.7 and sh_ml < 0.3)
    for f in ("location_prior/location_state.json", "location_prior/location_summary.json",
              "tables/location_prior_check.md"):
        check(f"location run: wrote {f}", (Path(d) / f).exists())
    runs[0]["jscc_a"] = None
    try:
        LP.run_location_prior_check({"final_runs": runs, "out": Path(d)}, cfg, torch.device("cpu")); raised = False
    except RuntimeError:
        raised = True
    check("location run: refuses to run when the Phase I models were released", raised)

# 2. Training over a range of SNRs, warm start.
def point(s, x0_ml, x0_pr, ch="awgn", flat=False):
    x = min(s, 60)
    out = {"snr_db": s, "channel": ch}
    for det_, x0, top in (("max_logit", x0_ml, 0.81), ("prototype", x0_pr, 0.77)):
        auc = (0.72 if flat else 0.5 + (top - 0.5) / (1 + np.exp(-0.3 * (x - x0))))
        out.update({f"{det_}_auc_roc_pixel": float(auc),
                    f"{det_}_fpr95": float(0.95 - (0.95 - 0.55) * (auc - 0.5) / (top - 0.5)),
                    f"{det_}_dsem": float(0.3 * np.exp(-(x + 20) / 8))})
    return out

good = [point(s, -14.0, -17.0) for s in snrs]
flat = [point(s, -14.0, -17.0, flat=True) for s in snrs]
dep_ok = {"max_logit": {"matched_minus_shuffled": 0.15}, "prototype": {"matched_minus_shuffled": 0.12}}
dep_bad = {"max_logit": {"matched_minus_shuffled": 0.15}, "prototype": {"matched_minus_shuffled": 0.004}}
g_ok = MS.collapse_flags(good, dep_ok, cfg)
g_flat = MS.collapse_flags(flat, dep_ok, cfg)
g_dep = MS.collapse_flags(good, dep_bad, cfg)
check("guard: a normal model is not flagged", not g_ok["max_logit"]["flagged"] and not g_ok["prototype"]["flagged"])
check("guard: a flat AUROC curve is flagged (both detectors)", g_flat["max_logit"]["flagged"] and g_flat["prototype"]["flagged"])
check("guard: matched ~= shuffled is flagged (that detector only)",
      g_dep["prototype"]["flagged"] and not g_dep["max_logit"]["flagged"])
check("guard: without a dependence measurement only the range rule applies",
      not MS.collapse_flags(good, None, cfg)["prototype"]["flagged"] and MS.collapse_flags(flat, None, cfg)["prototype"]["flagged"])

# _train_pair loads the production weights before training.
batches_tr = [{"image": torch.randn(2, 3, 224, 224, generator=g),
               "class_map": torch.randint(0, K, (2, 1, 224, 224), generator=g)} for _ in range(2)]
with tempfile.TemporaryDirectory() as d:
    rep_dir = Path(d) / "run_0_rep0"; rep_dir.mkdir()
    set_seed(11)
    prod = {}
    for det_name, stage, grid, dim, mk in (("max_logit", 2, 14, 8, lambda: MaxLogitDetector(8, 14, K)),
                                           ("prototype", 3, 7, 32, lambda: PrototypeDetector(32, 7, K, cfg["PROTOTYPE_SCALE"]))):
        j = SwinJSCCSystem(cfg["CHANNEL_SYMBOLS"][cfg["DEFAULT_CR"]], pretrained=False, patch_grid=grid,
                           patch_dim=dim, stage_idx=stage)
        dd = mk()
        with torch.no_grad():
            for p_ in list(j.parameters()) + list(dd.parameters()):
                # Distinct from any fresh initialisation.
                p_.add_(0.123)
        jf, df = MS.PRODUCTION_CKPTS[det_name]
        save_ckpt(j, None, 1, 0.5, rep_dir / jf)
        torch.save(dd.state_dict(), rep_dir / df)
        prod[det_name] = (j.state_dict(), dd.state_dict())
    seen, pre = {}, []
    orig_tw, orig_sys = MS._train_with_channel, MS.SwinJSCCSystem
    def fake_tw(jscc, detector, channel, **kw):
        name = "max_logit" if isinstance(detector, MaxLogitDetector) else "prototype"
        seen[name] = all(torch.equal(v, prod[name][0][k]) for k, v in jscc.state_dict().items()) and \
                     all(torch.equal(v, prod[name][1][k]) for k, v in detector.state_dict().items())
        return {"epochs_run": 3, "loss": 0.4}
    def rec_sys(nsym, pretrained=True, **k):
        pre.append(pretrained); return orig_sys(nsym, pretrained=False, **k)
    MS._train_with_channel, MS.SwinJSCCSystem = fake_tw, rec_sys
    try:
        out = MS._train_pair(cfg, torch.device("cpu"), {"loaders": {"cls_train": batches_tr, "cls_val": batches_tr},
                                                       "num_classes": K}, 42, 2.0, 1.5, 15,
                             cfg["CHANNEL_SYMBOLS"][cfg["DEFAULT_CR"]], Path(d), init_dir=rep_dir)
        seen_warm = dict(seen)
        MS._train_pair(cfg, torch.device("cpu"), {"loaders": {"cls_train": batches_tr, "cls_val": batches_tr},
                                                 "num_classes": K}, 42, 2.0, 1.5, 15,
                       cfg["CHANNEL_SYMBOLS"][cfg["DEFAULT_CR"]], Path(d))
    finally:
        MS._train_with_channel, MS.SwinJSCCSystem = orig_tw, orig_sys
    check("warm start: both systems hold their production weights when training starts",
          seen_warm.get("max_logit") is True and seen_warm.get("prototype") is True)
    check("cold start: fresh weights, not the production ones", seen.get("max_logit") is False)
    check(f"warm start builds without downloading pretrained weights; cold start still asks for them ({pre})",
          pre == [False, False, True, True])
    try:
        (rep_dir / MS.PRODUCTION_CKPTS["prototype"][1]).unlink()
        MS._train_pair(cfg, torch.device("cpu"), {"loaders": {}, "num_classes": K}, 42, 2.0, 1.5, 15, 2048,
                       Path(d), init_dir=rep_dir); raised = False
    except FileNotFoundError:
        raised = True
    check("warm start: a missing production checkpoint raises before any training", raised)

with tempfile.TemporaryDirectory() as d:
    per_seed_all = {"awgn": [[point(s, -11.5 + 0.1 * i, -16.5 + 0.1 * i) for s in snrs] for i in range(15)],
                    "rayleigh": [[point(s, -1.5, -7.0, "rayleigh") for s in snrs] for i in range(15)],
                    "snr_range_db": snrs, "n_seeds": 15}
    json.dump(per_seed_all, open(Path(d) / "phase2_per_seed.json", "w"))
    (Path(d) / "multisnr").mkdir()
    (Path(d) / "multisnr" / "multisnr_state.json").write_text('{"old": "cold-start results"}')
    runs = []
    for i in range(15):
        rd = Path(d) / f"run_0_rep{i}"; rd.mkdir()
        for det_name in MS.PRODUCTION_CKPTS:
            for f in MS.PRODUCTION_CKPTS[det_name]:
                (rd / f).write_bytes(b"x")
        runs.append({"loaders": {"val_mixed": None, "test": None, "cls_train": None, "cls_val": None},
                     "num_classes": K, "out": rd, "tau_max_logit": -9.0 - i, "tau_prototype": 0.02})
    p1_grid = {"final_runs": runs, "out": Path(d),
               "grid_analysis": {"winning_lambda_sem": {"max_logit": 2.0, "prototype": 2.0}, "derived_patience": 15}}
    # run_multisnr_check, warm start, with training and evaluation mocked.
    seeds = list(cfg["SEEDS"][:cfg["MULTISNR_N_SEEDS"]])
    trained = []
    def fake_train_pair(cfg_, device, p1_, seed, lam_ml, lam_pr, patience, nsym, out_dir, init_dir=None):
        trained.append((seed, init_dir))
        return (Obj(seed=seed), Obj(), Obj(seed=seed), Obj(), {"epochs_run": 9, "loss": 0.5}, {"epochs_run": 11, "loss": 0.5})
    def fake_eval(ja, ml, tml, sml, jb, pr, tpr, spr, loader, snr, cfg_, device, ch, noise_seed=None):
        # The last seed collapses to a flat curve.
        return point(snr, -14.0, -17.0, ch, flat=(ja.seed == seeds[-1]))
    orig = (MS._train_pair, MS.calibrate_tau, MS.calibrate_sdop_threshold, MS.eval_both_systems, MS._input_dependence)
    MS._train_pair = fake_train_pair; MS.calibrate_tau = lambda *a_, **k_: -9.0
    MS.calibrate_sdop_threshold = lambda *a_, **k_: 0.01; MS.eval_both_systems = fake_eval
    MS._input_dependence = lambda *a_, **k_: {"matched": 0.8, "shuffled": 0.7, "matched_minus_shuffled": 0.1}
    try:
        an1 = MS.run_multisnr_check(p1_grid, cfg, torch.device("cpu"))
        n_first = len(trained)
        MS.run_multisnr_check(p1_grid, cfg, torch.device("cpu"))
    finally:
        (MS._train_pair, MS.calibrate_tau, MS.calibrate_sdop_threshold, MS.eval_both_systems,
         MS._input_dependence) = orig
    check(f"warm run: default is the warm start (MULTISNR_WARM_START={cfg['MULTISNR_WARM_START']})",
          an1["warm_start"] is True)
    check("warm run: each seed fine-tuned once from ITS OWN Phase I repetition folder, none on resume",
          n_first == len(seeds) and len(trained) == len(seeds)
          and all(init == Path(d) / f"run_0_rep{cfg['SEEDS'].index(s)}" for s, init in trained))
    for f in ("multisnr_ft/multisnr_ft_state.json", "multisnr_ft/multisnr_ft_summary.json",
              "tables/multisnr_finetune.md", "plots/11c_multisnr_ft_dr_auc.png", "plots/11d_multisnr_ft_auroc.png"):
        check(f"warm run: wrote {f}", (Path(d) / f).exists())
    check("warm run: the cold-start results and plots are untouched",
          (Path(d) / "multisnr" / "multisnr_state.json").read_text() == '{"old": "cold-start results"}'
          and not (Path(d) / "plots" / "11_multisnr_dr_auc.png").exists()
          and not (Path(d) / "tables" / "multisnr_check.md").exists())
    check(f"warm run: collapsed seed flagged ({an1['flagged_seeds']}) and comparison repeated without it",
          an1["flagged_seeds"] == [seeds[-1]] and an1["excluding_flagged"]["seeds"] == seeds[:-1])
    txt = (Path(d) / "tables" / "multisnr_finetune.md").read_text()
    check("warm run: table reports censoring counts, the guard and the exclusion analysis",
          "crossed," in txt and "FLAGGED" in txt and "WITHOUT THE FLAGGED SEEDS" in txt and "fine-tuned over SNRs" in txt)
    st = json.load(open(Path(d) / "multisnr_ft" / "multisnr_ft_state.json"))
    fp_w = json.loads(st[str(seeds[0])]["_fp"])
    check("warm run: cached results are tied to the production models (version 2, their taus)",
          fp_w["version"] == 2 and fp_w["warm_start"] is True and fp_w["init_taus"][1] == [-10.0, 0.02])

# 3. Detector x architecture control.
A = AX.architectures(cfg)
c_ml, c_pr = AX.cross_cfgs(cfg)
check("cross configurations: Max Logit gets A_PR, Prototype gets A_ML, nothing else changes",
      (c_ml["BACKBONE_STAGE_ML"], c_ml["PATCH_GRID_ML"], c_ml["PATCH_DIM_ML"]) == (3, 7, 32)
      and (c_pr["BACKBONE_STAGE_PR"], c_pr["PATCH_GRID_PR"], c_pr["PATCH_DIM_PR"]) == (2, 14, 8)
      and all(c_ml[k] == cfg[k] for k in cfg if not k.endswith("_ML"))
      and all(c_pr[k] == cfg[k] for k in cfg if not k.endswith("_PR"))
      and cfg["PATCH_GRID_ML"] == 14 and cfg["PATCH_GRID_PR"] == 7)
check("both latents hold the same number of values (1568)",
      A["A_ML"]["grid"] ** 2 * A["A_ML"]["dim"] == A["A_PR"]["grid"] ** 2 * A["A_PR"]["dim"] == 1568)

x0 = {"ml_aml": -11.5, "pr_apr": -16.5, "ml_apr": -13.5, "pr_aml": -14.0}
def pt2(s, x_ml, x_pr, ch="awgn"):
    return point(s, x_ml, x_pr, ch)
n5 = 5
prod5 = {"awgn": [[pt2(s, x0["ml_aml"] + 0.2 * i, x0["pr_apr"] + 0.2 * i) for s in snrs] for i in range(n5)],
         "rayleigh": [[pt2(s, x0["ml_aml"] + 10, x0["pr_apr"] + 10, "rayleigh") for s in snrs] for i in range(n5)],
         "snr_range_db": snrs}
cross5 = {"awgn": [[pt2(s, x0["ml_apr"] + 0.2 * i, x0["pr_aml"] + 0.2 * i) for s in snrs] for i in range(n5)],
          "rayleigh": [[pt2(s, x0["ml_apr"] + 10, x0["pr_aml"] + 10, "rayleigh") for s in snrs] for i in range(n5)],
          "snr_range_db": snrs}
an = AX.arch2x2_analysis(prod5, cross5, snrs, cfg, cfg["SEEDS"][:5])
e = an["results"]["awgn"]["ref"]["0.5"]["effects"]
expect = {"detector_on_A_ML": -2.5, "detector_on_A_PR": -3.0, "architecture_ML": -2.0, "architecture_PR": -2.5,
          "main_detector": -2.75, "main_architecture": -2.25, "interaction": -0.5, "production_gap": -5.0}
check("2x2: every effect recovered within 0.15 dB: " + ", ".join(f"{k} {e[k]['mean_db']:+.2f}" for k in expect),
      all(abs(e[k]["mean_db"] - v) < 0.15 for k, v in expect.items()))
check("2x2: production gap = main detector + main architecture effect, per seed",
      np.allclose(e["production_gap"]["values"],
                  np.add(e["main_detector"]["values"], e["main_architecture"]["values"])))
check("2x2: sign counts 5/5 negative for the detector effects",
      e["detector_on_A_ML"]["n_negative"] == 5 and e["detector_on_A_PR"]["n_negative"] == 5)
check("2x2: Rayleigh results present (40 dB AWGN reference), no Rayleigh 'peak' block",
      "0.5" in an["results"]["rayleigh"]["ref"] and "peak" not in an["results"]["rayleigh"])
check(f"2x2: clean AUROC per cell ({an['clean_auroc']['ml_aml']['auc_40db_mean']:.3f})",
      abs(an["clean_auroc"]["ml_aml"]["auc_40db_mean"] - 0.81) < 0.01)
json.dumps(an); check("2x2 analysis is JSON-serialisable", True)
txt = AX._report(dict(an, guard={}, flagged_seeds=[]))
check("2x2 report renders all effects", all(name in txt for _, name in AX.EFFECT_NAMES))

# Real training of the two new cells (one epoch; random weights instead of
# pretrained ones, so the test runs offline).
cfg_tp = copy.deepcopy(cfg); cfg_tp["JOINT_MAX_EPOCHS"] = 1
orig_sys = AX.SwinJSCCSystem
built = []
def rec_sys2(nsym, pretrained=True, **k):
    built.append(k); return orig_sys(nsym, pretrained=False, **k)
AX.SwinJSCCSystem = rec_sys2
try:
    with tempfile.TemporaryDirectory() as d:
        p1_tp = {"loaders": {"cls_train": batches_tr, "cls_val": batches_tr}, "num_classes": K}
        jm, dm, jp, dp, rm, rp = AX._train_cross_pair(cfg_tp, torch.device("cpu"), p1_tp, 42, 2.0, 1.5, 15, 2048, Path(d))
finally:
    AX.SwinJSCCSystem = orig_sys
check(f"2x2: Max Logit trained on stage 3, 7x7x32; Prototype on stage 2, 14x14x8 ({built})",
      built == [{"patch_grid": 7, "patch_dim": 32, "stage_idx": 3}, {"patch_grid": 14, "patch_dim": 8, "stage_idx": 2}]
      and rm["epochs_run"] == 1 and rp["epochs_run"] == 1
      and (dm.patch_grid, dm.patch_dim) == (7, 32) and (dp.patch_grid, dp.patch_dim) == (14, 8))
with torch.no_grad():
    jm.eval(); jp.eval()
    xb = torch.randn(1, 3, 224, 224)
    sm, _ = dm.anomaly_score(jm(xb)[2], img_size=224); sp, _ = dp.anomaly_score(jp(xb)[2], img_size=224)
check("2x2: both new cells give full-resolution anomaly maps", tuple(sm.shape[-2:]) == (224, 224) == tuple(sp.shape[-2:]))

with tempfile.TemporaryDirectory() as d:
    # Run with training and evaluation mocked.
    per_seed_all = {"awgn": [[pt2(s, x0["ml_aml"] + 0.1 * i, x0["pr_apr"] + 0.1 * i) for s in snrs] for i in range(15)],
                    "rayleigh": [[pt2(s, x0["ml_aml"] + 10, x0["pr_apr"] + 10, "rayleigh") for s in snrs] for i in range(15)],
                    "snr_range_db": snrs, "n_seeds": 15}
    json.dump(per_seed_all, open(Path(d) / "phase2_per_seed.json", "w"))
    p1_grid = {"final_runs": [{"loaders": {"val_mixed": None, "test": None, "cls_train": None, "cls_val": None},
                               "num_classes": K}], "out": Path(d),
               "grid_analysis": {"winning_lambda_sem": {"max_logit": 2.0, "prototype": 1.5}, "derived_patience": 15}}
    trained = []
    def fake_cross(cfg_, device, p1_, seed, lam_ml, lam_pr, patience, nsym, out_dir):
        trained.append((seed, lam_ml, lam_pr))
        return (Obj(seed=seed), Obj(), Obj(seed=seed), Obj(), {"epochs_run": 30, "loss": 0.5}, {"epochs_run": 40, "loss": 0.5})
    def fake_eval2(ja, ml, tml, sml, jb, pr, tpr, spr, loader, snr, cfg_, device, ch, noise_seed=None):
        i = cfg["SEEDS"].index(ja.seed)
        return pt2(snr, x0["ml_apr"] + 0.1 * i + (10 if ch == "rayleigh" else 0),
                   x0["pr_aml"] + 0.1 * i + (10 if ch == "rayleigh" else 0), ch)
    orig = (AX._train_cross_pair, AX.calibrate_tau, AX.calibrate_sdop_threshold, AX.eval_both_systems, AX._input_dependence)
    AX._train_cross_pair = fake_cross; AX.calibrate_tau = lambda *a_, **k_: -9.0
    AX.calibrate_sdop_threshold = lambda *a_, **k_: 0.01; AX.eval_both_systems = fake_eval2
    AX._input_dependence = lambda *a_, **k_: {"matched": 0.8, "shuffled": 0.7, "matched_minus_shuffled": 0.1}
    try:
        anr = AX.run_arch2x2_check(p1_grid, cfg, torch.device("cpu"))
        n_first = len(trained)
        AX.run_arch2x2_check(p1_grid, cfg, torch.device("cpu"))
    finally:
        (AX._train_cross_pair, AX.calibrate_tau, AX.calibrate_sdop_threshold, AX.eval_both_systems,
         AX._input_dependence) = orig
    check(f"2x2 run: trains each of the {cfg['ARCH2X2_N_SEEDS']} seeds once with the Phase I lambdas, none on resume",
          n_first == cfg["ARCH2X2_N_SEEDS"] and len(trained) == n_first and all(t[1:] == (2.0, 1.5) for t in trained))
    er = anr["results"]["awgn"]["ref"]["0.5"]["effects"]
    check(f"2x2 run: production cells are the seed-matched repetitions 0-4 (detector effect on A_ML "
          f"{er['detector_on_A_ML']['mean_db']:+.2f}, expected -2.50)", abs(er["detector_on_A_ML"]["mean_db"] + 2.5) < 0.15)
    for f in ("arch2x2/arch2x2_state.json", "arch2x2/arch2x2_summary.json", "tables/arch2x2_check.md",
              "plots/13_arch2x2_snrx.png", "plots/13b_arch2x2_auroc.png"):
        check(f"2x2 run: wrote {f}", (Path(d) / f).exists())
    check("2x2 run: no existing Phase II/III/IV file was written",
          sorted(p.name for p in Path(d).iterdir()) == ["arch2x2", "phase2_per_seed.json", "plots", "tables"])
    check("2x2 run: no seed flagged by the guard", anr["flagged_seeds"] == [])

# 4. Notebook.
nb = json.load(open(os.path.join(os.path.dirname(__file__), "..", "run_experiment.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
evaluation = ["mechanism_retention", "run_digital_baseline", "run_location_prior_check",
              "run_stratified_sweep", "run_stratified_supplement"]
training = ["run_multisnr_check", "run_arch2x2_check"]
idx = {k: [i for i, s in enumerate(cells) if k in s] for k in evaluation + training}
parsed = True
for k in evaluation + training:
    for i in idx[k]:
        try:
            ast.parse(cells[i])
        except SyntaxError:
            parsed = False
check("notebook: one cell per evaluation control, two multi-SNR cells and one 2x2 cell, all parse",
      all(len(idx[k]) == 1 for k in evaluation) and len(idx["run_multisnr_check"]) == 2
      and len(idx["run_arch2x2_check"]) == 1 and parsed)
rel = lambda s: "_r[_k] = None" in s
ev_cells = [i for k in evaluation for i in idx[k]]
tr_cells = [i for k in training for i in idx[k]]
check("notebook: the evaluation cells keep the Phase I models; the three training cells release them",
      bool(tr_cells) and not any(rel(cells[i]) for i in ev_cells) and all(rel(cells[i]) for i in tr_cells))
check("notebook: every evaluation cell comes before the first training cell",
      bool(ev_cells) and bool(tr_cells) and max(ev_cells) < min(tr_cells))
ms = [cells[i] for i in idx["run_multisnr_check"]]
check("notebook: one multi-SNR cell runs the cold start, the other the warm start",
      sorted(("warm_start=False" in s, "warm_start=True" in s) for s in ms) == [(False, True), (True, False)])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All control checks passed.")
