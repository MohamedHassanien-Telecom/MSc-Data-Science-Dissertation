"""Annotations of the Phase IV figures (src/visualise.py):
  1. _elbow_marker: "supported" only for a Stage-2-confirmed headline elbow that
     Stage 4 does not reject (verdict other than "smooth"); otherwise
     "candidate", "tie" or None;
  2. with smooth S-curves (Stage 4 verdict "smooth"), Plot 1 (AUROC vs SNR) and
     Plot 3 (D_sem vs AUROC) draw no elbow line, Plot 3 has no empty legend, and
     every elbow line on Plot 3b is labelled "not confirmed";
  3. with a confirmed elbow and Stage 4 verdict "break", Plots 1 and 3 draw it;
  4. Plot 7 (SDOP) marks the clean-channel level set by the calibration
     percentile (10% for the 90th percentile).
Synthetic Phase II data; the figures are inspected through matplotlib when they
are saved (no image comparison).

Usage:  python3 tests/test_phase4_figure_markers.py
"""
import os, sys, copy, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from pathlib import Path
from src.config import CONFIG
import src.visualise as V
from phases.phase2_sweep import _aggregate_metric_dicts

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

# 1. _elbow_marker.
def s34(verdict):
    return {"structural_break": {"smooth_vs_break": {"verdict": verdict}}, "bootstrap": {}}
base = {"elbow_snr": -8, "elbow_confirmed": True, "elbow_tie_detected": False, "elbow_tied_values": None}
check("confirmed + verdict break -> supported", V._elbow_marker(base, s34("break")) == "supported")
check("confirmed + verdict inconclusive -> supported", V._elbow_marker(base, s34("inconclusive")) == "supported")
check("confirmed + Stage 4 not run -> supported", V._elbow_marker(base, None) == "supported")
check("confirmed + verdict smooth -> candidate", V._elbow_marker(base, s34("smooth")) == "candidate")
check("not confirmed + verdict break -> candidate",
      V._elbow_marker(dict(base, elbow_confirmed=False), s34("break")) == "candidate")
tie = {"elbow_snr": None, "elbow_confirmed": False, "elbow_tie_detected": True, "elbow_tied_values": [-14, -8]}
check("tie -> tie", V._elbow_marker(tie, s34("smooth")) == "tie")
check("no elbow -> None", V._elbow_marker({"elbow_snr": None, "elbow_tie_detected": False}, None) is None)

# Figure recorder.
records = {}
_orig_savefig = Figure.savefig
def _recording_savefig(self, fname, *a, **k):
    """Figure.savefig that records each figure's vertical and horizontal lines, legends and texts."""
    name = Path(str(fname)).name
    if name.endswith(".png"):
        info = {"vlines": [], "hlines": [], "legends": [], "texts": []}
        for ax in self.axes:
            for ln in ax.lines:
                x, y = list(ln.get_xdata()), list(ln.get_ydata())
                if len(x) == 2 and x[0] == x[1] and list(y) == [0, 1]:
                    info["vlines"].append((float(x[0]), ln.get_label()))
                elif len(y) == 2 and y[0] == y[1] and list(x) == [0, 1]:
                    info["hlines"].append((float(y[0]), ln.get_label()))
            lg = ax.get_legend()
            if lg is not None:
                info["legends"].append(len(lg.get_texts()))
            info["texts"] += [t.get_text() for t in ax.texts]
        records[name.split("__")[0].rsplit(".", 1)[0]] = info
    return _orig_savefig(self, fname, *a, **k)

cfg = copy.deepcopy(CONFIG)
snrs = list(cfg["SNR_RANGE_DB"])
rng = np.random.default_rng(5)

def point(snr):
    x = min(snr, 20)
    out = {"snr_db": snr, "channel": "awgn"}
    for det, x0 in (("max_logit", -11.0), ("prototype", -15.0)):
        auc = 0.52 + 0.29 / (1 + np.exp(-(x - x0) / 3.5)) + rng.normal(0, 0.002)
        dsem = 0.01 + 0.30 / (1 + np.exp((x - x0) / 5.0)) + rng.normal(0, 0.0005)
        out.update({f"{det}_auc_roc_pixel": float(auc), f"{det}_dsem": float(max(dsem, 1e-4)),
                    f"{det}_fpr95": float(1.3 - auc), f"{det}_sri_auc": float(auc / dsem),
                    f"{det}_sri_op": float(auc / dsem), f"{det}_sdop": 0.1, f"{det}_psnr": 20.0,
                    f"{det}_tpr_at_tau": 0.9, f"{det}_fpr_at_tau": 0.5})
    return out

per_seed = [[point(s) for s in snrs] for _ in range(15)]
p2 = {"awgn": [_aggregate_metric_dicts([run[i] for run in per_seed]) for i in range(len(snrs))], "rayleigh": []}

def run(tmp):
    json.dump({"awgn": per_seed, "rayleigh": [], "snr_range_db": snrs, "n_seeds": 15},
              open(Path(tmp) / "phase2_per_seed.json", "w"))
    records.clear()
    return V.phase4(p2, {"by_cr": {}, "snr_by_cr": {}}, tmp, cfg)

Figure.savefig = _recording_savefig
try:
    with tempfile.TemporaryDirectory() as d:
        # 2. Smooth S-curves.
        rep = run(d)
        verdicts = [rep["stage4_structural_break"][k]["smooth_vs_break"]["verdict"] for k in ("max_logit", "prototype")]
        check(f"synthetic S-curves: Stage 4 verdicts {verdicts}", verdicts == ["smooth", "smooth"])
        r1, r3, r3b, r7 = (records.get(k) for k in ("1_snr_vs_auc", "3_dsem_vs_auc", "3b_collapse_derivatives", "7_sdop_vs_snr"))
        check("all four figures were written", None not in (r1, r3, r3b, r7))
        check(f"Plot 1: no elbow line ({r1['vlines']})", r1["vlines"] == [])
        check(f"Plot 3: no elbow line ({r3['vlines']})", r3["vlines"] == [])
        check(f"Plot 3: no empty legend box (legend entry counts {r3['legends']})", all(n > 0 for n in r3["legends"]))
        check("Plot 3: no 'Tied candidates' text", not any("Tied" in t for t in r3["texts"]))
        lab3b = [l for _, l in r3b["vlines"] if not l.startswith("_")]
        cand = [c for c in (rep["max_logit"], rep["prototype"]) if c["elbow_snr"] is not None or c["elbow_tie_detected"]]
        check(f"Plot 3b: candidates labelled 'not confirmed' ({lab3b})",
              (len(lab3b) == len(cand)) and all("not confirmed" in l for l in lab3b))
        check(f"Plot 7: clean-channel line at 0.10 with '10%' label ({r7['hlines']})",
              any(abs(y - 0.10) < 1e-12 and "10%" in l for y, l in r7["hlines"]))
        check("Plot 7: no 5% line", not any(abs(y - 0.05) < 1e-12 for y, _ in r7["hlines"]))

    # 3. Confirmed elbow and Stage 4 verdict "break".
    orig_dc, orig_s34 = V.detect_collapse, V._run_collapse_stages_3_4
    def fake_dc(*a, **k):
        r = orig_dc(*a, **k)
        r.update({"elbow_snr": -8, "elbow_confirmed": True, "elbow_tie_detected": False, "elbow_tied_values": None})
        return r
    def fake_s34(*a, **k):
        return {"max_logit": s34("break"), "prototype": s34("break")}
    V.detect_collapse, V._run_collapse_stages_3_4 = fake_dc, fake_s34
    try:
        with tempfile.TemporaryDirectory() as d:
            run(d)
            r1, r3, r3b = records["1_snr_vs_auc"], records["3_dsem_vs_auc"], records["3b_collapse_derivatives"]
            check(f"supported elbow: Plot 1 draws both ({r1['vlines']})",
                  [x for x, _ in r1["vlines"]] == [-8.0, -8.0] and all("elbow" in l for _, l in r1["vlines"]))
            check(f"supported elbow: Plot 3 draws one per panel ({len(r3['vlines'])})", len(r3["vlines"]) == 2)
            check("supported elbow: Plot 3b labels without 'not confirmed'",
                  len(r3b["vlines"]) == 2 and not any("not confirmed" in l for _, l in r3b["vlines"]))
    finally:
        V.detect_collapse, V._run_collapse_stages_3_4 = orig_dc, orig_s34
finally:
    Figure.savefig = _orig_savefig

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All Phase IV figure-marker checks passed.")
