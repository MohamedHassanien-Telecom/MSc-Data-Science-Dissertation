"""Inputs of the Phase IV collapse analysis (src/visualise.py phase4,
src/collapse.py):
  1. the CLS_FIT_SNR reference point (40 dB) is excluded from Stages 1-4 and
     recorded as excluded; the derivative arrays line up with the SNRs used;
  2. Stage 2 verifies only Stage-1 candidate points (no plateau points);
  3. elbow_confirmed holds exactly when the headline elbow is corroborated at a
     strict majority of the tested percentiles;
  4. a tied headline elbow does not break any plot;
  5. a per-seed file that does not match the aggregate is refused.
Synthetic Phase II data, aggregated with the pipeline's own
_aggregate_metric_dicts; plots are written to a temporary folder.

Usage:  python3 tests/test_phase4_collapse_inputs.py
"""
import os, sys, copy, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from src.config import CONFIG
import src.visualise as V
from src.collapse import detect_collapse
from phases.phase2_sweep import _aggregate_metric_dicts

cfg = copy.deepcopy(CONFIG)
snrs = list(cfg["SNR_RANGE_DB"])
assert float(snrs[-1]) == float(cfg["CLS_FIT_SNR"]), "test assumes the 40 dB reference point is in the sweep"
rng = np.random.default_rng(3)

def point(snr, s):
    x = min(snr, 20)
    out = {"snr_db": snr, "channel": "awgn"}
    for det, off in (("max_logit", 0.0), ("prototype", -0.05)):
        auc = 0.55 + off + 0.27 / (1 + np.exp(-(x + 4) / 3.0)) + rng.normal(0, 0.004)
        dsem = 0.02 + 0.30 * np.exp(-(x + 20) / 8.0) + rng.normal(0, 0.0005)
        out.update({f"{det}_auc_roc_pixel": float(auc), f"{det}_dsem": float(dsem),
                    f"{det}_fpr95": float(1 - auc), f"{det}_sri_auc": float(auc / dsem),
                    f"{det}_sri_op": float(auc / dsem), f"{det}_sdop": 0.1, f"{det}_psnr": 20.0,
                    f"{det}_tpr_at_tau": 0.9, f"{det}_fpr_at_tau": 0.5})
    return out

per_seed = [[point(s, k) for s in snrs] for k in range(15)]
p2 = {"awgn": [_aggregate_metric_dicts([run[i] for run in per_seed]) for i in range(len(snrs))], "rayleigh": []}

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

with tempfile.TemporaryDirectory() as d:
    json.dump({"awgn": per_seed, "rayleigh": [], "snr_range_db": snrs, "n_seeds": 15},
              open(Path(d) / "phase2_per_seed.json", "w"))
    rep = V.phase4(p2, {"by_cr": {}, "snr_by_cr": {}}, d, cfg)
    used = [s for s in snrs if float(s) != float(cfg["CLS_FIT_SNR"])]
    for det in ("max_logit", "prototype"):
        c = rep[det]
        check(f"{det}: collapse SNRs = sweep without 40 dB ({len(used)} points)", c["snr_db"] == [float(s) for s in used])
        check(f"{det}: 40 dB recorded as excluded", c["excluded_snr_db"] == [float(cfg["CLS_FIT_SNR"])])
        check(f"{det}: derivative arrays aligned with the SNRs used",
              len(c["first_derivative"]) == len(c["second_derivative"]) == len(used))
        b = rep["stage3_bootstrap"][det]; s4 = rep["stage4_structural_break"][det]
        check(f"{det}: Stage 4 breakpoint inside the swept range", -20 <= s4["observed_breakpoint_snr"] <= 20)
        check(f"{det}: Stage 3 elbow is one of the swept SNRs",
              b["original_elbow_snr"] is None or b["original_elbow_snr"] in used)
        n_maj = len(c["elbow_corroborated_percentiles"])
        check(f"{det}: elbow_confirmed == strict majority ({n_maj}/{len(cfg['COLLAPSE_SENSITIVITY_PERCENTILES'])})",
              c["elbow_confirmed"] == (c["elbow_snr"] is not None
                                       and n_maj > len(cfg["COLLAPSE_SENSITIVITY_PERCENTILES"]) / 2))
    check("plots written", (Path(d) / "plots" / "3b_collapse_derivatives__semantic_collapse_detection_derivative_curves_underlying_the_elbow_point.png").exists())

    # 2. Stage 2 verifies candidate points only.
    auc_m = [r["max_logit_auc_roc_pixel_mean"] for r in p2["awgn"]][:-1]
    dsem_m = [r["max_logit_dsem_mean"] for r in p2["awgn"]][:-1]
    res = detect_collapse(used, auc_m, dsem_m, cfg)
    g = np.abs(np.gradient(np.array(dsem_m), np.array(used, dtype=float)))
    bad = []
    for key, r in res["per_percentile_results"].items():
        cand = {used[i] for i in range(len(used)) if g[i] >= np.percentile(g, int(key[1:]))}
        for s in r["verified_collapse_snrs"]:
            i = used.index(s)
            # A verified SNR s is the upper point of a pair (s - step, s); both must be candidates.
            if used[i] not in cand or used[i - 1] not in cand:
                bad.append((key, s))
    check(f"Stage 2 verifies candidate points only (violations: {bad})", not bad)

    # 4. A tied headline elbow.
    real = V.detect_collapse
    def tied(*a, **k):
        r = real(*a, **k); s = r["snr_db"]
        r.update({"elbow_snr": None, "elbow_tie_detected": True, "elbow_tied_values": [s[3], s[6]],
                  "elbow_confirmed": False, "elbow_corroborated_percentiles": []})
        return r
    V.detect_collapse = tied
    try:
        V.phase4(p2, {"by_cr": {}, "snr_by_cr": {}}, d, cfg); crashed = None
    except Exception as e:
        crashed = repr(e)
    finally:
        V.detect_collapse = real
    check(f"tied headline elbow: every plot runs ({crashed})", crashed is None)

    # 5. A per-seed file that does not match the aggregate.
    bad_seed = copy.deepcopy(per_seed); bad_seed[0][0]["max_logit_auc_roc_pixel"] += 0.1
    json.dump({"awgn": bad_seed, "rayleigh": [], "snr_range_db": snrs, "n_seeds": 15},
              open(Path(d) / "phase2_per_seed.json", "w"))
    try:
        V.phase4(p2, {"by_cr": {}, "snr_by_cr": {}}, d, cfg); refused = False
    except ValueError as e:
        refused = "differs from the Phase II aggregate" in str(e)
    check("per-seed file inconsistent with the aggregate is refused", refused)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All Phase IV collapse-input checks passed.")
