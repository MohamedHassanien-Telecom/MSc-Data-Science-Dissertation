"""Training over a range of SNRs (cold start), the scoring-rule check and the ideal
digital baseline (dissertation Appendices F.4-F.6):
  1. RandomSNRAWGNChannel: per-image SNR inside the range, random in training,
     identical draws on every validation pass;
  2. _train_with_channel runs the real train_joint_system with the random-SNR
     channel (training and validation passes go through it) and removes the
     override afterwards, also when training fails; the class method is untouched;
  3. multisnr_analysis recovers known shifts; run_multisnr_check (cold start,
     training and evaluation mocked) writes its outputs, resumes, and pairs each
     seed with the Phase I repetition of the same seed;
  4. mechanism_retention recovers known SNR50 orderings and separates a
     reference effect from a scoring effect;
  5. digital baseline: threshold formula, quantiser (levels, clipping, channel
     layout), DigitalLink without a quantiser equals SwinJSCCSystem with an
     identity channel (real model), the analysis, and run_digital_baseline
     (evaluation mocked);
  6. the notebook's mechanism, digital and two multi-SNR cells exist and parse.

Usage:  python3 tests/test_multisnr_mechanism_digital.py
"""
import os, sys, copy, json, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from src.config import CONFIG, set_seed
from src.models import SwinJSCCSystem, MaxLogitDetector
import phases.multisnr_check as MS
import phases.digital_baseline as DB
from src.mechanism import mechanism_retention, mechanism_report

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

cfg = copy.deepcopy(CONFIG)
cfg["USE_TORCH_COMPILE"] = False

torch.manual_seed(0)
# 1. Random-SNR channel.
ch = MS.RandomSNRAWGNChannel(-20, 20, eval_seed=5)
X = torch.randn(64, 1024, 2)
ch.train()
Y1 = ch(X); Y2 = ch(X)
sig = X.pow(2).mean()
noise_pow = (Y1 - X).pow(2).mean(dim=(1, 2))
snr_hat = 10 * torch.log10(sig / noise_pow)
check(f"training: per-image SNR within [-20, 20] dB (estimated {snr_hat.min():.1f} .. {snr_hat.max():.1f})",
      bool((snr_hat > -21).all() and (snr_hat < 21).all()))
check("training: per-image SNRs differ (spread > 20 dB)", float(snr_hat.max() - snr_hat.min()) > 20)
check("training: two forward passes draw different noise", not torch.allclose(Y1, Y2))
ch.eval(); E1 = ch(X); E1b = ch(X)
ch.train(); ch(X); ch.eval(); E2 = ch(X)
check("validation: every pass (each switch to eval mode) uses identical draws", torch.allclose(E1, E2))
check("validation: successive batches within a pass get different draws", not torch.allclose(E1, E1b))
try:
    MS.RandomSNRAWGNChannel(5, 5, 1); bad = False
except ValueError:
    bad = True
check("an empty SNR range is refused", bad)

# 2. Real training through the random-SNR channel.
K = 13
g = torch.Generator().manual_seed(0)
batches = [{"image": torch.randn(2, 3, 224, 224, generator=g),
            "class_map": torch.randint(0, K, (2, 1, 224, 224), generator=g)} for _ in range(2)]

class CountingChannel(MS.RandomSNRAWGNChannel):
    def __init__(self, *a):
        super().__init__(*a); self.n_train = 0; self.n_eval = 0
    def forward(self, X):
        if self.training: self.n_train += 1
        else: self.n_eval += 1
        return super().forward(X)

with tempfile.TemporaryDirectory() as d:
    set_seed(1)
    jscc = SwinJSCCSystem(256, pretrained=False, patch_grid=14, patch_dim=8, stage_idx=2)
    det = MaxLogitDetector(8, 14, K)
    chan = CountingChannel(-20, 20, 7)
    r = MS._train_with_channel(jscc, det, chan, loader=batches, val_loader=batches, cfg=cfg,
                               device=torch.device("cpu"), num_classes=K,
                               jscc_save_path=os.path.join(d, "j.pt"), detector_save_path=os.path.join(d, "d.pt"),
                               lambda_rec=0.5, lambda_sem=2.0, patience=15, joint_lr=0.003, joint_momentum=0.9,
                               weight_decay=1e-4, joint_lr_power=1.0, min_rel_improve=0.01, max_epochs=1)
    check(f"real train_joint_system trained through the random-SNR channel "
          f"({chan.n_train} training, {chan.n_eval} validation batches)", chan.n_train == 2 and chan.n_eval == 2)
    check("the instance override is removed after training", "set_channel" not in jscc.__dict__)
    jscc.set_channel("awgn", 40.0)
    check("set_channel works normally afterwards (class method, AWGN 40 dB)",
          type(jscc.channel).__name__ == "AWGNChannel" and jscc.snr_db == 40.0)
    jscc2 = SwinJSCCSystem(256, pretrained=False, patch_grid=14, patch_dim=8, stage_idx=2)
    try:
        MS._train_with_channel(jscc2, MaxLogitDetector(8, 14, K), CountingChannel(-20, 20, 7),
                               loader=batches, val_loader=batches, cfg=cfg, device=torch.device("cpu"),
                               num_classes=K, jscc_save_path=os.path.join(d, "j2.pt"),
                               detector_save_path=os.path.join(d, "d2.pt"), lambda_rec=0.5, lambda_sem=2.0,
                               patience=15, joint_lr=None, joint_momentum=0.9, weight_decay=1e-4,
                               joint_lr_power=1.0, min_rel_improve=0.01, max_epochs=1)
        raised = False
    except ValueError:
        raised = True
    check("a failing training call still removes the override", raised and "set_channel" not in jscc2.__dict__)
    check("SwinJSCCSystem.set_channel itself is unchanged",
          SwinJSCCSystem.set_channel.__qualname__ == "SwinJSCCSystem.set_channel")

snrs = list(cfg["SNR_RANGE_DB"])
# 3. Multi-SNR analysis and run.
def point(s, x0_ml, x0_pr, ch="awgn"):
    x = min(s, 60)
    out = {"snr_db": s, "channel": ch}
    for det, x0, top in (("max_logit", x0_ml, 0.81), ("prototype", x0_pr, 0.77)):
        auc = 0.5 + (top - 0.5) / (1 + np.exp(-0.3 * (x - x0)))
        out.update({f"{det}_auc_roc_pixel": float(auc), f"{det}_fpr95": float(0.95 - (0.95 - 0.55) * (auc - 0.5) / (top - 0.5)),
                    f"{det}_dsem": float(0.3 * np.exp(-(x + 20) / 8))})
    return out

seeds = list(cfg["SEEDS"][:5])
fixed = {"awgn": [[point(s, -11.5, -16.5) for s in snrs] for _ in seeds],
         "rayleigh": [[point(s, -1.5, -7.0, "rayleigh") for s in snrs] for _ in seeds], "snr_range_db": snrs}
multi = {"awgn": [[point(s, -14.0, -17.0) for s in snrs] for _ in seeds],
         "rayleigh": [[point(s, -4.0, -8.0, "rayleigh") for s in snrs] for _ in seeds], "snr_range_db": snrs}
an = MS.multisnr_analysis(multi, fixed, snrs, cfg, seeds)
c = an["comparisons"]["awgn"]["0.5"]
check(f"gap with 40 dB training ~ -5 dB ({c['gap_fixed_prototype_minus_max_logit']['mean_diff_db']:.2f})",
      abs(c["gap_fixed_prototype_minus_max_logit"]["mean_diff_db"] + 5.0) < 0.3)
check(f"gap with multi-SNR training ~ -3 dB ({c['gap_multi_prototype_minus_max_logit']['mean_diff_db']:.2f})",
      abs(c["gap_multi_prototype_minus_max_logit"]["mean_diff_db"] + 3.0) < 0.3)
check(f"Max Logit multi minus fixed ~ -2.5 dB, lower in 5/5 ({c['max_logit_multi_minus_fixed']['mean_diff_db']:.2f})",
      abs(c["max_logit_multi_minus_fixed"]["mean_diff_db"] + 2.5) < 0.3
      and c["max_logit_multi_minus_fixed"]["n_first_lower"] == 5)
json.dumps(an); check("analysis is JSON-serialisable", True)

class Obj:
    def __init__(self, **k): self.__dict__.update(k)

with tempfile.TemporaryDirectory() as d:
    per_seed_all = {"awgn": [[point(s, -11.5 + 0.1 * i, -16.5 + 0.1 * i) for s in snrs] for i in range(15)],
                    "rayleigh": [[point(s, -1.5, -7.0, "rayleigh") for s in snrs] for i in range(15)],
                    "snr_range_db": snrs, "n_seeds": 15}
    json.dump(per_seed_all, open(Path(d) / "phase2_per_seed.json", "w"))
    p1 = {"loaders": {"val_mixed": None, "test": None, "cls_train": None, "cls_val": None}, "num_classes": K}
    p1_grid = {"final_runs": [p1], "out": Path(d),
               "grid_analysis": {"winning_lambda_sem": {"max_logit": 2.0, "prototype": 2.0}, "derived_patience": 15}}
    trained = []
    def fake_train_pair(cfg_, device, p1_, seed, lam_ml, lam_pr, patience, nsym, out_dir):
        trained.append(seed)
        return (Obj(seed=seed), Obj(), Obj(seed=seed), Obj(), {"epochs_run": 20, "loss": 0.5}, {"epochs_run": 40, "loss": 0.5})
    def fake_eval(ja, ml, tml, sml, jb, pr, tpr, spr, loader, snr, cfg_, device, ch, noise_seed=None):
        return point(snr, -14.0, -17.0, ch)
    orig = (MS._train_pair, MS.calibrate_tau, MS.calibrate_sdop_threshold, MS.eval_both_systems,
            MS._input_dependence)
    MS._train_pair = fake_train_pair; MS.calibrate_tau = lambda *a, **k: -9.0
    MS.calibrate_sdop_threshold = lambda *a, **k: 0.01; MS.eval_both_systems = fake_eval
    MS._input_dependence = lambda *a, **k: {"matched": 0.8, "shuffled": 0.7, "matched_minus_shuffled": 0.1}
    try:
        # Cold start; the warm start is tested in test_controls.py.
        an1 = MS.run_multisnr_check(p1_grid, cfg, torch.device("cpu"), warm_start=False)
        n_first = len(trained)
        an2 = MS.run_multisnr_check(p1_grid, cfg, torch.device("cpu"), warm_start=False)
    finally:
        (MS._train_pair, MS.calibrate_tau, MS.calibrate_sdop_threshold, MS.eval_both_systems,
         MS._input_dependence) = orig
    check(f"run: trains each of the {len(seeds)} seeds once, resumes without retraining",
          n_first == len(seeds) and len(trained) == len(seeds))
    for f in ("multisnr/multisnr_state.json", "multisnr/multisnr_summary.json", "tables/multisnr_check.md",
              "plots/11_multisnr_dr_auc.png", "plots/11b_multisnr_auroc.png"):
        check(f"run: wrote {f}", (Path(d) / f).exists())
    s50 = an1["dr_fixed"]["channels"]["awgn"]["auc"]["ref"]["detectors"]["max_logit"]["levels"]["0.5"]["per_seed_snr_db"]
    check(f"run: the 40 dB comparison uses repetitions 0-4 (seed-matched): SNR50 {[round(x, 2) for x in s50]}",
          np.allclose(s50, [-11.5 + 0.1 * i for i in range(5)], atol=0.15))
    check("run: no existing Phase II/III/IV file was written",
          sorted(p.name for p in Path(d).iterdir()) == ["multisnr", "phase2_per_seed.json", "plots", "tables"])

# 4. Scoring-rule check (mechanism_retention).
ex_snrs = [20, 10, 5, 0, -5, -10, -15, -20]
rng = np.random.default_rng(3)
mag = {s: {"ml": [], "pr": [], "pr_mag": []} for s in ex_snrs}
for r in range(15):
    for s in ex_snrs:
        for v, x0, top in (("ml", -10.0, 0.81), ("pr", -15.0, 0.77), ("pr_mag", -12.5, 0.79)):
            mag[s][v].append(float(0.5 + (top - 0.5) / (1 + np.exp(-0.3 * (s - x0 - rng.normal(0, 0.3))))))
res = mechanism_retention(mag, ex_snrs, [0.5, 0.9])
pm = res["pairs"]["pr_mag_minus_pr"]["0.5"]; pml = res["pairs"]["pr_minus_ml"]["0.5"]
check(f"mechanism: Prototype-mag keeps decisions only at a higher SNR than Prototype in "
      f"{pm['n_first_higher']}/{pm['n']} reps ({pm['mean_diff_db']:+.2f} dB)", pm["n_first_higher"] == 15 and pm["mean_diff_db"] > 1)
check(f"mechanism: Prototype lower than Max Logit ({pml['mean_diff_db']:+.2f} dB)", pml["mean_diff_db"] < -3)
txt = mechanism_report(res)
check("mechanism report renders", "Prototype-mag - Prototype" in txt)
check("mechanism report shows both references and the share of the gap",
      "Reference 'peak'" in txt and "Share of the Prototype - Max Logit gap" in txt)
json.dumps(res); check("mechanism result is JSON-serialisable", True)
sg = res["share_of_gap"]["ref"]["0.5"]
check(f"mechanism: share of gap = (pr_mag - pr) / (ml - pr) = {sg:.3f}",
      abs(sg - pm["mean_diff_db"] / -pml["mean_diff_db"]) < 1e-12)

# Reference effect: Prototype dips at the top SNRs; Prototype-mag is the same curve
# without the dip. Against the highest-SNR reference the dip lowers Prototype's
# SNR50; against each repetition's peak the two coincide.
def _f(s): return 0.5 + 0.28 / (1 + np.exp(-0.3 * (s + 15.0)))
mag2 = {s: {"ml": [float(0.5 + 0.31 / (1 + np.exp(-0.3 * (s + 10.0))))] * 15,
            "pr": [float(_f(s) - (0.015 if s >= 10 else 0.0))] * 15,
            "pr_mag": [float(_f(s))] * 15} for s in ex_snrs}
res2 = mechanism_retention(mag2, ex_snrs, [0.5, 0.9])
d_ref = res2["pairs"]["pr_mag_minus_pr"]["0.5"]["mean_diff_db"]
d_peak = res2["peak"]["pairs"]["pr_mag_minus_pr"]["0.5"]["mean_diff_db"]
check(f"mechanism: dip-only difference shows under 'ref' ({d_ref:+.3f} dB) and vanishes under 'peak' "
      f"({d_peak:+.3f} dB)", d_ref > 0.2 and abs(d_peak) < 0.05)
check("mechanism: 'peak' reference of Prototype = its best AUROC (mid-range), not the top-SNR value",
      abs(res2["peak"]["variants"]["pr"]["auc_ref_mean"] - max(_f(s) - (0.015 if s >= 10 else 0.0) for s in ex_snrs)) < 1e-12)

# 5. Ideal digital baseline.
thr2 = DB.digital_threshold_db(2, 1568, 2048)
check(f"threshold, 2 bits: 10log10(2^(1568*2/2048)-1) = {thr2:.3f} dB",
      abs(thr2 - 10 * math.log10(2 ** (1568 * 2 / 2048) - 1)) < 1e-12)
check("threshold rises with bit depth",
      all(DB.digital_threshold_db(b, 1568, 2048) < DB.digital_threshold_db(b + 1, 1568, 2048) for b in range(1, 8)))
q = DB.LatentQuantiser([0.0, -1.0], [1.0, 1.0], 2, patch_dim=2, patch_grid=2)
z = torch.tensor([[0.1, 0.3, 0.6, 2.0, -5.0, -0.6, 0.1, 0.9]])
zq = q(z)
check(f"quantiser: channel 0 (values in [0,1], step 0.25) -> {zq[0, :4].tolist()}",
      torch.allclose(zq[0, :4], torch.tensor([0.125, 0.375, 0.625, 0.875])))
check(f"quantiser: channel 1 (values in [-1,1], step 0.5, clipped) -> {zq[0, 4:].tolist()}",
      torch.allclose(zq[0, 4:], torch.tensor([-0.75, -0.75, 0.25, 0.75])))
check("quantiser: 2 bits -> at most 4 distinct values per channel",
      len(set(zq[0, :4].tolist())) <= 4 and len(set(zq[0, 4:].tolist())) <= 4)
set_seed(2)
j = SwinJSCCSystem(2048, pretrained=False, patch_grid=14, patch_dim=8, stage_idx=2).eval()
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    link_out = DB.DigitalLink(j, None)(x)
    j.channel = nn.Identity()
    ref_out = j(x)
check("DigitalLink without quantiser == SwinJSCCSystem with an identity channel (real model)",
      all(torch.allclose(a, b) for a, b in zip(link_out, ref_out)))

gq = torch.Generator().manual_seed(4)
cal_batches = [{"image": torch.randn(2, 3, 224, 224, generator=gq)} for _ in range(2)]
lo, hi = DB.calibrate_quantiser_range(j, cal_batches, torch.device("cpu"), [0.1, 99.9])
check(f"quantiser range from a real model: {len(lo)} channels, hi > lo everywhere",
      len(lo) == 8 and all(b > a for a, b in zip(lo, hi)))
masks = torch.zeros(2, 1, 224, 224, dtype=torch.long); masks[:, :, :64, :64] = 1
test_batches = [{"image": torch.randn(2, 3, 224, 224, generator=gq), "mask": masks,
                 "has_anomaly": torch.ones(2, dtype=torch.long)} for _ in range(2)]
from src.evaluate import _eval_one_system
r_real = _eval_one_system(DB.DigitalLink(j, DB.LatentQuantiser(lo, hi, 4, 8, 14)), MaxLogitDetector(8, 14, K).eval(),
                          -1.0, 0.01, test_batches, None, cfg, torch.device("cpu"), "awgn", noise_seed=None)
check(f"real _eval_one_system on the digital link: AUROC {r_real['auc_roc_pixel']:.3f}, FPR95 {r_real['fpr95']:.3f}",
      0.0 <= r_real["auc_roc_pixel"] <= 1.0 and 0.0 <= r_real["fpr95"] <= 1.0)

# _train_pair end to end (one epoch; random weights instead of pretrained ones,
# so the test runs offline).
cfg_tp = copy.deepcopy(cfg); cfg_tp["JOINT_MAX_EPOCHS"] = 1
_orig_sys = MS.SwinJSCCSystem
MS.SwinJSCCSystem = lambda nsym, pretrained=True, **k: _orig_sys(nsym, pretrained=False, **k)
try:
    with tempfile.TemporaryDirectory() as d:
        p1_tp = {"loaders": {"cls_train": batches, "cls_val": batches}, "num_classes": K}
        ja, mla, jb, prb, rml, rpr = MS._train_pair(cfg_tp, torch.device("cpu"), p1_tp, 42, 2.0, 2.0, 15, 256, Path(d))
finally:
    MS.SwinJSCCSystem = _orig_sys
check("_train_pair trains both systems and leaves both on the plain 40 dB AWGN channel",
      rml["epochs_run"] == 1 and rpr["epochs_run"] == 1
      and all(type(m.channel).__name__ == "AWGNChannel" and m.snr_db == cfg["CLS_FIT_SNR"] for m in (ja, jb))
      and all("set_channel" not in m.__dict__ for m in (ja, jb)))

state = {str(r): {det: {"none": {"auc_roc_pixel": 0.80, "fpr95": 0.55, "dsem": 0.0, "psnr": 20.0},
                        **{str(b): {"auc_roc_pixel": 0.5 + 0.3 * (1 - 2 ** -b), "fpr95": 0.6, "dsem": 0.01, "psnr": 20.0}
                           for b in cfg["DIGITAL_BITS"]}}
                  for det in ("max_logit", "prototype")} for r in range(3)}
analog = {"awgn": [[point(s, -11.5, -16.5) for s in snrs] for _ in range(3)], "snr_range_db": snrs}
dan = DB.digital_analysis(state, analog, cfg, {"max_logit": 1568, "prototype": 1568}, 2048)
v1 = dan["detectors"]["max_logit"]["digital"]["1"]; v8 = dan["detectors"]["max_logit"]["digital"]["8"]
check(f"digital: 1 bit -> AUROC 0.65 above {v1['threshold_snr_db']:+.2f} dB, 0.5 below",
      abs(v1["auc_mean"] - 0.65) < 1e-9 and all((a == 0.5) == (s < v1["threshold_snr_db"])
                                                 for a, s in zip(v1["curve_auc_mean"], dan["snr_db"])))
sx, sy = DB.digital_step_curve(-20.0, 20.0, v1["threshold_snr_db"], 0.65)
check(f"digital plot: cliff drawn exactly at the threshold ({sx}, {sy})",
      sx == [-20.0, v1["threshold_snr_db"], v1["threshold_snr_db"], 20.0] and sy == [0.5, 0.5, 0.65, 0.65])
check("digital plot: threshold above the range -> chance throughout",
      DB.digital_step_curve(-20.0, 20.0, 25.0, 0.8) == ([-20.0, 20.0], [0.5, 0.5]))
check(f"digital: analogue SNR50 lies below the 8-bit threshold "
      f"({v8['analogue_snr50_minus_threshold_db_mean']:+.2f} dB)", v8["analogue_snr50_minus_threshold_db_mean"] < 0)

with tempfile.TemporaryDirectory() as d:
    json.dump({"awgn": [[point(s, -11.5, -16.5) for s in snrs] for _ in range(2)], "snr_range_db": snrs,
               "n_seeds": 2}, open(Path(d) / "phase2_per_seed.json", "w"))
    mk = lambda pd_, pg: Obj(patch_dim=pd_, patch_grid=pg)
    runs = [{"jscc_a": mk(8, 14), "jscc_b": mk(32, 7), "max_logit_det": None, "prototype_det": None,
             "tau_max_logit": -9.0, "tau_prototype": 0.02, "sdop_threshold_a": 0.01, "sdop_threshold_b": 0.01,
             "loaders": {"val_mixed": None, "test": None}} for _ in range(2)]
    calls = []
    def fake_eval_one(model, det, tau, sdop, loader, snr, cfg_, device, ch, noise_seed=None):
        calls.append(model.quantiser.bits if model.quantiser is not None else None)
        b = model.quantiser.bits if model.quantiser is not None else 32
        return {"auc_roc_pixel": 0.5 + 0.3 * (1 - 2 ** -b), "fpr95": 0.6, "dsem": 0.0, "psnr": 20.0}
    orig = (DB._eval_one_system, DB.calibrate_quantiser_range)
    DB._eval_one_system = fake_eval_one
    DB.calibrate_quantiser_range = lambda jscc, loader, device, pct: ([-1.0] * jscc.patch_dim, [1.0] * jscc.patch_dim)
    try:
        g1 = DB.run_digital_baseline({"final_runs": runs, "out": Path(d)}, cfg, torch.device("cpu"))
        n1 = len(calls)
        DB.run_digital_baseline({"final_runs": runs, "out": Path(d)}, cfg, torch.device("cpu"))
    finally:
        DB._eval_one_system, DB.calibrate_quantiser_range = orig
    check(f"digital run: {n1} evaluations (2 reps x 2 systems x (1 + {len(cfg['DIGITAL_BITS'])})), none on resume",
          n1 == 2 * 2 * (1 + len(cfg["DIGITAL_BITS"])) and len(calls) == n1)
    for f in ("digital_baseline/digital_state.json", "digital_baseline/digital_summary.json",
              "tables/digital_baseline.md", "plots/12_digital_vs_analogue_auroc.png"):
        check(f"digital run: wrote {f}", (Path(d) / f).exists())

# 6. Notebook.
nb = json.load(open(os.path.join(os.path.dirname(__file__), "..", "run_experiment.ipynb")))
import ast
supp = [c for c in nb["cells"] if c["cell_type"] == "code"
        and any(k in "".join(c["source"]) for k in ("mechanism_retention", "run_digital_baseline", "run_multisnr_check"))]
parsed = True
for c in supp:
    try:
        ast.parse("".join(c["source"]))
    except SyntaxError:
        parsed = False
check(f"notebook: the mechanism, digital and two multi-SNR cells exist and parse ({len(supp)} found)",
      len(supp) == 4 and parsed)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All multi-SNR, scoring-rule and digital-baseline checks passed.")
