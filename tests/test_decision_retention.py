"""Decision Retention (src/decision_retention.py) and its Phase IV outputs
(src/visualise.py):
  1. DR values: reference -> 1, chance -> 0, FPR95 direction, undefined reference;
  2. SNR_X: linear interpolation, left-censoring, not reached, missing values, and
     "stays >= X at every higher SNR" (a dip above the crossing moves it up);
  3. the per-seed analysis recovers the known 50% points of synthetic S-curves,
     and the paired comparison counts and tests the right direction;
  4. Phase III retention per compression ratio from a per-seed state dict; an
     incomplete ratio gives None;
  5. phase4() writes plots 10/10b/10c (10d with the Phase III state), the DR
     tables and the report keys, and draws no published reference lines on
     Plots 1/1b.

Usage:  python3 tests/test_decision_retention.py
"""
import os, sys, copy, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from src.config import CONFIG
from src.decision_retention import (dr_values, retention_snr, decision_retention_analysis,
                                    phase3_retention_by_cr)

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# 1. DR values.
d = dr_values([0.8, 0.5, 0.65, 0.45], 0.8, "auc")
check("DR_AUC: reference -> 1, chance -> 0, midpoint -> 0.5, below chance -> negative",
      np.allclose(d, [1.0, 0.0, 0.5, -1/6]))
d = dr_values([0.55, 0.95, 0.75, None], 0.55, "op")
check("DR_op: reference FPR95 -> 1, FPR95 0.95 -> 0, midpoint -> 0.5, missing -> NaN",
      np.allclose(d[:3], [1.0, 0.0, 0.5]) and np.isnan(d[3]))
check("DR undefined when the reference is not better than chance",
      np.all(np.isnan(dr_values([0.6], 0.5, "auc"))) and np.all(np.isnan(dr_values([0.6], 0.97, "op"))))

# 2. SNR_X.
snr = np.array([-20., -18., -16., -14.])
x, st = retention_snr(snr, np.array([0.1, 0.3, 0.7, 0.9]), 0.5)
check(f"SNR50 interpolated between -18 and -16 dB ({x})", st == "crossed" and abs(x - (-17.0)) < 1e-12)
x, st = retention_snr(snr, np.array([0.6, 0.7, 0.8, 0.9]), 0.5)
check("DR >= level at the lowest SNR -> left-censored", st == "left_censored" and x is None)
x, st = retention_snr(snr, np.array([0.1, 0.2, 0.3, 0.4]), 0.5)
check("DR < level at the highest SNR -> not reached", st == "not_reached" and x is None)
x, st = retention_snr(snr, np.array([0.1, np.nan, 0.7, 0.9]), 0.5)
check("missing value -> undefined", st == "undefined")
x, st = retention_snr(snr, np.array([0.6, 0.4, 0.8, 0.9]), 0.5)
check(f"a dip below the level moves SNR50 above the dip ({x})", st == "crossed" and abs(x - (-17.5)) < 1e-12)

# 3. Synthetic per-seed S-curves with known 50% points.
snrs = list(CONFIG["SNR_RANGE_DB"])
cfg = copy.deepcopy(CONFIG)
rng = np.random.default_rng(7)
def logistic(s, x0, k=0.3):
    return 1 / (1 + np.exp(-k * (s - x0)))
def make_point(s, x0_ml, x0_pr, ch):
    out = {"snr_db": s, "channel": ch}
    for det, x0, top in (("max_logit", x0_ml, 0.81), ("prototype", x0_pr, 0.77)):
        base = 0.5 + (top - 0.5) * logistic(min(s, 60), x0)
        auc = base + rng.normal(0, 0.002)
        fpr = 0.95 - (0.95 - 0.55) * logistic(min(s, 60), x0 + 3) + rng.normal(0, 0.002)
        out.update({f"{det}_auc_roc_pixel": float(auc), f"{det}_fpr95": float(fpr),
                    f"{det}_dsem": float(0.3 * np.exp(-(s + 20) / 8)), f"{det}_sri_auc": 1.0,
                    f"{det}_sri_op": 1.0, f"{det}_sdop": 0.1, f"{det}_psnr": 20.0,
                    f"{det}_tpr_at_tau": 0.9, f"{det}_fpr_at_tau": 0.5})
    return out
n_seeds = 8
x0_ml = [-11.5 + rng.normal(0, 0.3) for _ in range(n_seeds)]
x0_pr = [-16.5 + rng.normal(0, 0.3) for _ in range(n_seeds)]
awgn = [[make_point(s, x0_ml[i], x0_pr[i], "awgn") for s in snrs] for i in range(n_seeds)]
ray = [[make_point(s, x0_ml[i] + 8, x0_pr[i] + 8, "rayleigh") for s in snrs] for i in range(n_seeds)]
per_seed = {"awgn": awgn, "rayleigh": ray, "snr_range_db": snrs, "n_seeds": n_seeds}
dr = decision_retention_analysis(per_seed, snrs, cfg)
b = dr["channels"]["awgn"]["auc"]["ref"]
m_ml = b["detectors"]["max_logit"]["levels"]["0.5"]["snr_db"]["mean"]
m_pr = b["detectors"]["prototype"]["levels"]["0.5"]["snr_db"]["mean"]
check(f"AWGN SNR50 recovers the true midpoints (ML {m_ml:.2f} vs {np.mean(x0_ml):.2f}; "
      f"PR {m_pr:.2f} vs {np.mean(x0_pr):.2f})",
      abs(m_ml - np.mean(x0_ml)) < 0.5 and abs(m_pr - np.mean(x0_pr)) < 0.5)
P = b["paired_prototype_minus_max_logit"]["0.5"]
check(f"paired: Prototype lower in {P['prototype_lower_count']}/{P['n_order_determinable']}, "
      f"diff {P['mean_diff_db']:.2f} dB, p={P['wilcoxon_p']}",
      P["prototype_lower_count"] == n_seeds and P["mean_diff_db"] < -4 and P["wilcoxon_p"] < 0.01)
check(f"unpaired Mann-Whitney check reported (p={P['mannwhitney_p']})",
      P["mannwhitney_p"] is not None and P["mannwhitney_p"] < 0.01)
check("40 dB reference point is not part of the DR curve", 40.0 not in dr["snr_db"] and len(dr["snr_db"]) == len(snrs) - 1)
check("steepest increase of the mean DR curve within one grid step of SNR50",
      abs(b["detectors"]["max_logit"]["steepest_increase_snr_db"] - m_ml) <= 2.0)
rb = dr["channels"]["rayleigh"]["auc"]["ref"]["detectors"]["max_logit"]["levels"]["0.5"]["snr_db"]["mean"]
check(f"Rayleigh curves use the AWGN 40 dB reference and are shifted (+8 dB): {rb:.2f}",
      abs(rb - (np.mean(x0_ml) + 8)) < 0.6)
check("no sensitivity ('peak') block for Rayleigh", "peak" not in dr["channels"]["rayleigh"]["auc"])
json.dumps(dr)
check("analysis is JSON-serialisable", True)

# 4. Phase III retention per compression ratio.
state = {}
for cr, shift in (("low", 2.0), ("medium", 0.0), ("high", -2.0)):
    state[cr] = {str(s): {"sweep": {str(sn): make_point(sn, x0_ml[i] + shift, x0_pr[i] + shift, "awgn")
                                    for sn in snrs}}
                 for i, s in enumerate(cfg["SEEDS"][:n_seeds])}
state["high"].pop(str(cfg["SEEDS"][0]))
res = phase3_retention_by_cr(state, ["low", "medium", "high"], cfg["SEEDS"][:n_seeds], snrs, cfg)
lo_m = res["by_cr"]["low"]["max_logit"]["0.5"]["snr_db"]["mean"]
md_m = res["by_cr"]["medium"]["max_logit"]["0.5"]["snr_db"]["mean"]
check(f"Phase III: low CR's SNR50 is ~2 dB above medium's ({lo_m:.2f} vs {md_m:.2f})", abs((lo_m - md_m) - 2.0) < 0.6)
check("Phase III: a CR with missing seeds -> None", res["by_cr"]["high"] is None)

# 5. Phase IV outputs.
import src.visualise as V
from phases.phase2_sweep import _aggregate_metric_dicts
p2 = {"awgn": [_aggregate_metric_dicts([run[i] for run in awgn]) for i in range(len(snrs))],
      "rayleigh": [_aggregate_metric_dicts([run[i] for run in ray]) for i in range(len(snrs))]}
cfg5 = copy.deepcopy(cfg); cfg5["BREAK_TEST_N_BOOTSTRAP"] = 50
with tempfile.TemporaryDirectory() as dtmp:
    json.dump(per_seed, open(Path(dtmp) / "phase2_per_seed.json", "w"))
    full_state = {cr: {str(s): {"sweep": {str(sn): make_point(sn, x0_ml[i % n_seeds], x0_pr[i % n_seeds], "awgn") for sn in snrs}}
                       for i, s in enumerate(cfg5["SEEDS"])} for cr in ("low", "medium", "high")}
    json.dump(full_state, open(Path(dtmp) / "phase3_multiseed_state.json", "w"))
    p3 = {"by_cr": {}, "snr_by_cr": {cr: p2["awgn"] for cr in ("low", "medium", "high")}, "n_runs": 15}
    rep = V.phase4(p2, p3, dtmp, cfg5)
    plots = {p.name for p in (Path(dtmp) / "plots").glob("*.png")}
    for prefix in ("10_dr_auc_vs_snr__", "10b_dr_op_vs_snr__", "10c_dr_snr50_snr90_paired__", "10d_dr_snr_vs_cr__"):
        check(f"plot {prefix}* written", any(p.startswith(prefix) for p in plots))
    for t in ("decision_retention.csv", "decision_retention.md"):
        check(f"table {t} written", (Path(dtmp) / "tables" / t).exists())
    import csv as _csv
    rows = list(_csv.reader(open(Path(dtmp) / "tables" / "decision_retention.csv")))
    check(f"every CSV row has the header's {len(rows[0])} columns",
          all(len(r) == len(rows[0]) for r in rows) and len(rows) > 1)
    saved = json.load(open(Path(dtmp) / "collapse_report.json"))
    check("collapse_report.json has decision_retention and decision_retention_by_cr",
          saved.get("decision_retention") is not None and saved.get("decision_retention_by_cr") is not None)
    svb = saved["stage4_structural_break"]["max_logit"]["smooth_vs_break"]
    check(f"Stage 4 reports the smooth-vs-break comparison with a seed bootstrap (verdict {svb['verdict']})",
          svb["bootstrap"] is not None and svb["verdict"] in ("smooth", "break", "inconclusive"))
    import inspect
    src_v = inspect.getsource(V.phase4)
    check("published no-channel reference lines are not drawn on Plots 1/1b",
          "0.893" not in src_v and "0.911" not in src_v and "0.265" not in src_v and "0.232" not in src_v)
    # Without the per-seed file, DR is skipped and Phase IV still completes.
    os.remove(Path(dtmp) / "phase2_per_seed.json")
    rep2 = V.phase4(p2, {"by_cr": {}, "snr_by_cr": {}}, dtmp, cfg5)
    check("without phase2_per_seed.json, DR is skipped and Phase IV completes",
          rep2["decision_retention"] is None)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All Decision Retention checks passed.")
