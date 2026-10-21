"""Location-stratified pixel AUROC (phases/stratified_eval.py):
  1. stratified_auc equals a brute-force within-region Mann-Whitney computation
     (ties 1/2), equals plain AUROC with one region, is exactly 0.5 for any map
     that depends only on the region and 1.0 for a perfectly separating map;
  2. region_index maps pixels to the right 16 x 16 region on 224 x 224;
  3. on synthetic data where anomalies follow a location prior, a map that is the
     prior scores a high plain AUROC but LS-AUROC of about 0.5;
  4. on a real model, the plain AUROC of map_metrics equals src/evaluate.py's;
  5. run_stratified_sweep (scoring mocked) reproduces the stored Phase II values,
     resumes per (repetition, channel), writes its outputs, refuses to run without
     the Phase I models, and recovers a known reversal: a system that relies on a
     location prior looks more robust on plain AUROC and less robust on LS-AUROC;
  6. run_digital_stratified (scoring mocked) uses the digital thresholds, scores
     the outage at max(0.5, the location map's LS-AUROC), writes its outputs and
     resumes;
  7. the notebook has one cell that runs the sweep and then the digital run, and
     it does not release the Phase I models.

Usage:  python3 tests/test_stratified_eval.py
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
from src.models import SwinJSCCSystem, MaxLogitDetector
from src.evaluate import _eval_one_system
import phases.location_prior_check as LP
import phases.stratified_eval as SE
import phases.digital_baseline as DB

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

class Obj:
    def __init__(self, **k): self.__dict__.update(k)

rng = np.random.default_rng(0)

# 1. stratified_auc against a brute-force computation.
n = 5000
reg = rng.integers(0, 7, n)
y = (rng.random(n) < 0.1 + 0.05 * reg).astype(int)
# Rounded scores: many ties.
s = np.round(rng.normal(0, 1, n) + 0.8 * y + 0.3 * reg, 1)
auc, pairs = SE.stratified_auc(y, s, reg)
U, P = 0.0, 0.0
for r in range(7):
    m = reg == r
    yr, sr = y[m], s[m]
    pos, neg = sr[yr == 1], sr[yr == 0]
    if pos.size and neg.size:
        U += float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()))
        P += pos.size * neg.size
check(f"stratified AUROC = brute-force within-region Mann-Whitney with ties ({auc:.6f} vs {U / P:.6f})",
      abs(auc - U / P) < 1e-12 and pairs == P)
check("one region: equals plain AUROC", abs(SE.stratified_auc(y, s, np.zeros(n, int))[0] - roc_auc_score(y, s)) < 1e-12)
check("a map that depends only on the region scores exactly 0.5",
      SE.stratified_auc(y, reg * 1.7 + 3.0, reg)[0] == 0.5)
check("a perfectly separating map scores 1.0", SE.stratified_auc(y, y + 0.01 * reg, reg)[0] == 1.0)
check("no region with both classes -> NaN", np.isnan(SE.stratified_auc(np.ones(5), np.arange(5.0), np.arange(5))[0]))

# 2. region_index.
pix = np.array([0, 15, 16, 223, 224 * 16, 224 * 16 + 17, 224 * 224 - 1])
check(f"region_index 14x14 on 224x224: {SE.region_index(pix, 224, 224, 14).tolist()}",
      SE.region_index(pix, 224, 224, 14).tolist() == [0, 0, 1, 13, 14, 15, 195])

# 3. A map that is the location prior and nothing else.
H = W = 28; hw = H * W; n_img = 600
# Anomalies are more likely lower in the frame.
prior = np.repeat(np.linspace(0.0, 0.25, H), W)
lab = (rng.random((n_img, hw)) < prior).astype(np.int64).ravel()
pixpos = np.tile(np.arange(hw), n_img)
prior_scores = np.tile(prior, n_img)
raw = roc_auc_score(lab, prior_scores)
ls28 = SE.stratified_auc(lab, prior_scores, SE.region_index(pixpos, H, W, 28))[0]
ls14 = SE.stratified_auc(lab, prior_scores, SE.region_index(pixpos, H, W, 14))[0]
check(f"location-only map: plain AUROC {raw:.3f} high, LS-AUROC {ls28:.3f} (1-pixel regions) = 0.5, "
      f"{ls14:.3f} (2-pixel regions, small within-region residual) ~0.5",
      raw > 0.65 and ls28 == 0.5 and abs(ls14 - 0.5) < 0.03)

# 4. Real model: map_metrics against src/evaluate.py.
cfg = copy.deepcopy(CONFIG); cfg["USE_TORCH_COMPILE"] = False
set_seed(3)
model = SwinJSCCSystem(256, pretrained=False, patch_grid=14, patch_dim=8, stage_idx=2).eval()
det = MaxLogitDetector(8, 14, 13).eval()
g = torch.Generator().manual_seed(1)
batches = []
for _ in range(3):
    m = torch.zeros(2, 224, 224, dtype=torch.long); m[:, 150:, 60:160] = (torch.rand(2, 74, 100, generator=g) < 0.5).long()
    batches.append({"image": torch.randn(2, 3, 224, 224, generator=g), "mask": m, "has_anomaly": torch.ones(2, dtype=torch.long)})
with torch.no_grad():
    ref = _eval_one_system(model, det, -1.0, 0.01, batches, 0, cfg, torch.device("cpu"), "awgn", noise_seed=777)
labels, scores, hw_r = LP.score_maps(model, det, batches, 0, torch.device("cpu"), "awgn", noise_seed=777)
mm = SE.map_metrics(labels, scores, hw_r, LP.partner_images(None, labels.size // hw_r)[0], cfg)
check(f"real model: plain AUROC = evaluate.py ({mm['matched']:.6f} vs {ref['auc_roc_pixel']:.6f}); LS grids "
      f"{sorted(mm['ls'])}", mm["matched"] == ref["auc_roc_pixel"] and sorted(mm["ls"]) == ["14", "28", "7"]
      and all(0.0 <= v <= 1.0 for v in mm["ls"].values()))

# 5. run_stratified_sweep with mocked scoring.
cfg5 = copy.deepcopy(CONFIG)
cfg5.update({"IMG_H": H, "IMG_W": W, "USE_RAYLEIGH": True})
snrs = list(cfg5["SNR_RANGE_DB"])
n_img5 = 200
groups5 = ["t5"] * 120 + ["t6"] * 80
lab5 = (rng.random((n_img5, hw)) < prior).astype(np.int64).ravel()
prior5 = np.tile(prior, n_img5)

def sig(x, x0, top):
    return top / (1 + np.exp(-0.3 * (x - x0)))

def fake_scores(model, snr, ch):
    snr = float(snr); shift = 0.0 if ch == "awgn" else 8.0
    r = np.random.default_rng(int(1000 * model.rep + 10 * (snr + 30) + (ch == "rayleigh")))
    noise = r.normal(0, 1, lab5.size)
    # Max Logit: image content only, kept down to about -12 dB.
    # Prototype: dominated by the location prior, weak content down to about -5 dB.
    if model.det == "max_logit":
        return sig(snr - shift, -12.0, 3.0) * lab5 + noise
    return 8.0 * prior5 + sig(snr - shift, -5.0, 0.5) * lab5 + 0.5 * noise

class FakeSubset:
    def __init__(self, n, paths):
        self.indices = list(range(n)); self.dataset = Obj(img_paths=paths)
    def __len__(self): return len(self.indices)
loader5 = Obj(dataset=FakeSubset(n_img5, [Path(f"/x/test/{t}/{i}.png") for i, t in enumerate(groups5)]))
partner5, _ = LP.partner_images(groups5, n_img5)

with tempfile.TemporaryDirectory() as d:
    runs = [{"jscc_a": Obj(det="max_logit", rep=i, patch_dim=8, patch_grid=14),
             "jscc_b": Obj(det="prototype", rep=i, patch_dim=32, patch_grid=7),
             "max_logit_det": None, "prototype_det": None, "tau_max_logit": -9.0, "tau_prototype": 0.02,
             "loaders": {"test": loader5, "val_mixed": None}} for i in range(3)]
    p2s = {"snr_range_db": snrs}
    for ch in ("awgn", "rayleigh"):
        p2s[ch] = []
        for i in range(3):
            run = []
            for snr in snrs:
                pt = {"snr_db": snr}
                for detn, jk in (("max_logit", "jscc_a"), ("prototype", "jscc_b")):
                    sc = fake_scores(runs[i][jk], snr, ch)
                    pt[f"{detn}_auc_roc_pixel"] = LP.location_aucs(lab5, sc, hw, partner5)["matched"]
                    pt[f"{detn}_fpr95"] = 0.6; pt[f"{detn}_dsem"] = 0.3 * np.exp(-(snr + 20) / 8)
                run.append(pt)
            p2s[ch].append(run)
    json.dump(p2s, open(Path(d) / "phase2_per_seed.json", "w"))
    calls = []
    def fake_score_maps(model, detector, loader, snr_db, device, channel_type="awgn", noise_seed=None):
        calls.append((getattr(model, 'rep', None), getattr(model, 'det', None), channel_type, snr_db))
        # DigitalLink (part 6): AUROC grows with the bit depth.
        if hasattr(model, "quantiser"):
            b = None if model.quantiser is None else model.quantiser.bits
            k = {None: 3.0, 1: 0.4, 2: 1.5, 4: 2.6, 8: 3.0}[b]
            r = np.random.default_rng(7)
            base = 8.0 * prior5 if model.jscc.det == "prototype" else 0.0
            return lab5, base + k * lab5 + r.normal(0, 1, lab5.size), hw
        return lab5, fake_scores(model, snr_db, channel_type), hw
    orig = LP.score_maps
    LP.score_maps = fake_score_maps
    p1g = {"final_runs": runs, "out": Path(d)}
    try:
        an = SE.run_stratified_sweep(p1g, cfg5, "cpu")
        n1 = len(calls)
        SE.run_stratified_sweep(p1g, cfg5, "cpu")
        n2 = len(calls) - n1
        st = json.load(open(Path(d) / "stratified" / "stratified_state.json"))
        del st["reps"]["1"]["rayleigh"]
        json.dump(st, open(Path(d) / "stratified" / "stratified_state.json", "w"))
        SE.run_stratified_sweep(p1g, cfg5, "cpu")
        n3 = len(calls) - n1 - n2
        check(f"sweep: {n1} scorings (3 reps x {len(snrs)} SNRs x 2 channels x 2 systems), none on resume, "
              f"{n3} after deleting one (repetition, channel)",
              n1 == 3 * len(snrs) * 2 * 2 and n2 == 0 and n3 == len(snrs) * 2)
        check(f"sweep: plain AUROC reproduces Phase II (max |diff| {an['max_abs_diff_from_phase2']:.1e})",
              an["max_abs_diff_from_phase2"] == 0.0)
        b = an["baseline"]
        check(f"sweep: location map plain AUROC {b['raw']:.3f} high; LS-AUROC 28x28 {b['ls']['28']:.3f} = 0.5 "
              f"(1-pixel regions, same map within a town), 14x14 {b['ls']['14']:.3f} ~0.5",
              b["raw"] > 0.6 and b["ls"]["28"] == 0.5 and abs(b["ls"]["14"] - 0.5) < 0.03)
        dr_raw = an["dr"]["raw"]["channels"]["awgn"]["ref"]["detectors"]
        dr_ls = an["dr"]["ls14"]["channels"]["awgn"]["ref"]["detectors"]
        def s50(dr, det_):
            lv = dr[det_]["levels"]["0.5"]
            return [(x if st_ == "crossed" else st_) for x, st_ in zip(lv["per_seed_snr_db"], lv["per_seed_status"])]
        ml_ls, pr_ls = s50(dr_ls, "max_logit"), s50(dr_ls, "prototype")
        ml_raw, pr_raw = s50(dr_raw, "max_logit"), s50(dr_raw, "prototype")
        check(f"sweep: on LS-AUROC Max Logit keeps decisions to a lower SNR than Prototype in 3/3 seeds "
              f"(SNR50 ML {[round(x, 1) for x in ml_ls]}, PR {[round(x, 1) for x in pr_ls]})",
              all(isinstance(a_, float) and isinstance(b_, float) and a_ < b_ - 3 for a_, b_ in zip(ml_ls, pr_ls)))
        check(f"sweep: on plain AUROC the location prior makes Prototype look more robust "
              f"(SNR50 ML {[round(x, 1) if isinstance(x, float) else x for x in ml_raw]}, PR {pr_raw})",
              all((p_ == "left_censored") or (isinstance(p_, float) and p_ < m_) for m_, p_ in zip(ml_raw, pr_raw)))
        lr = an["dr"]["ls14"]["channels"]["rayleigh"]["ref"]["detectors"]["max_logit"]["levels"]["0.5"]["snr_db"]["mean"]
        check(f"sweep: Rayleigh analysed (Max Logit SNR50 on LS {lr:+.1f} dB, shifted up as simulated)", lr > -10)
        for f in ("stratified/stratified_state.json", "stratified/stratified_summary.json", "tables/stratified_eval.md",
                  "plots/14_stratified_auroc.png", "plots/14b_stratified_dr.png"):
            check(f"sweep: wrote {f}", (Path(d) / f).exists())
        txt = (Path(d) / "tables" / "stratified_eval.md").read_text()
        check("sweep: table has the per-SNR comparison, DR on every variant, SRI",
              "LS: PR - ML" in txt and "(PRIMARY)" in txt and "sensitivity" in txt and "matched - shuffled" in txt
              and "excess SRI_LS" in txt)
        json.dumps(an); check("sweep analysis is JSON-serialisable", True)

        # 6. run_digital_stratified with mocked scoring.
        orig_cal = DB.calibrate_quantiser_range
        DB.calibrate_quantiser_range = lambda jscc, loader, device, pct: ([-1.0] * jscc.patch_dim, [1.0] * jscc.patch_dim)
        try:
            c0 = len(calls)
            dan = SE.run_digital_stratified(p1g, cfg5, "cpu")
            nd1 = len(calls) - c0
            SE.run_digital_stratified(p1g, cfg5, "cpu")
            nd2 = len(calls) - c0 - nd1
        finally:
            DB.calibrate_quantiser_range = orig_cal
        check(f"digital: {nd1} scorings (3 reps x 2 systems x {1 + len(cfg5['DIGITAL_BITS'])}), none on resume",
              nd1 == 3 * 2 * (1 + len(cfg5["DIGITAL_BITS"])) and nd2 == 0)
        v1 = dan["detectors"]["max_logit"]["digital"]["1"]
        check(f"digital: thresholds as in digital_baseline ({v1['threshold_snr_db']:+.2f} dB for 1 bit, 1568 values)",
              abs(v1["threshold_snr_db"] - DB.digital_threshold_db(1, 8 * 14 * 14, cfg5["CHANNEL_SYMBOLS"][cfg5["DEFAULT_CR"]])) < 1e-12)
        check(f"digital: outage scored at max(0.5, location map's LS-AUROC) ({dan['fallback_ls_auroc']:.4f})",
              dan["fallback_ls_auroc"] == max(0.5, an["baseline"]["ls"]["14"]))
        prv = dan["detectors"]["prototype"]["digital"]
        check(f"digital: LS-AUROC grows with bits for Prototype ({[round(prv[k]['ls_auroc']['mean'], 3) for k in prv]})",
              prv["1"]["ls_auroc"]["mean"] < prv["2"]["ls_auroc"]["mean"] < prv["8"]["ls_auroc"]["mean"])
        for f in ("stratified/digital_stratified_state.json", "stratified/digital_stratified_summary.json",
                  "tables/digital_stratified.md", "plots/14c_digital_stratified.png"):
            check(f"digital: wrote {f}", (Path(d) / f).exists())
        runs[0]["jscc_b"] = None
        try:
            SE.run_stratified_sweep(p1g, cfg5, "cpu"); raised = False
        except RuntimeError:
            raised = True
        check("sweep: refuses to run when the Phase I models were released", raised)
    finally:
        LP.score_maps = orig
    check("no existing Phase II/III/IV file written",
          sorted(p.name for p in Path(d).iterdir()) == ["phase2_per_seed.json", "plots", "stratified", "tables"])

check("digital step curve: fallback below, value above the threshold",
      SE.digital_ls_step(-20.0, 20.0, 2.5, 0.7, 0.51) == ([-20.0, 2.5, 2.5, 20.0], [0.51, 0.51, 0.7, 0.7]))

# 7. Notebook cell.
nb = json.load(open(os.path.join(os.path.dirname(__file__), "..", "run_experiment.ipynb")))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
strat = [c for c in cells if "run_stratified_sweep" in c]
parsed = True
for c in strat:
    try:
        ast.parse(c)
    except SyntaxError:
        parsed = False
check("notebook: one stratified cell, parses, calls the sweep then the digital run, does not release the models",
      len(strat) == 1 and parsed and strat[0].index("run_stratified_sweep(") < strat[0].index("run_digital_stratified(")
      and "_r[_k] = None" not in strat[0])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All stratified-evaluation checks passed.")
