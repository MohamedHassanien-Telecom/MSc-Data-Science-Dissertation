"""The notebook's paired test of the scoring-rule check (run_experiment.ipynb,
"SCORING-RULE CHECK 2/3") states the direction of a significant difference. The
cell is run on synthetic per-repetition AUROCs with known directions, and its
printed verdicts are checked.

Usage:  python3 tests/test_notebook_paired_cell.py
"""
import os, sys, io, json, contextlib
import numpy as np

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

nb = json.load(open(os.path.join(os.path.dirname(__file__), "..", "run_experiment.ipynb")))
cells = [c for c in nb["cells"] if c["cell_type"] == "code" and "paired by repetition" in "".join(c["source"])[:400]]
check(f"the paired-by-repetition cell exists ({len(cells)} found)", len(cells) == 1)
code = "".join(cells[0]["source"])

SNRS = [20, 10, 5, 0, -5, -10, -15, -20]
def run(ml_off, mag_off, n=15, seed=0):
    """Run the cell with Max Logit and Prototype-mag offset from Prototype by ml_off and mag_off."""
    rng = np.random.default_rng(seed)
    base = {s: 0.5 + 0.28 / (1 + np.exp(-0.3 * (s + 12))) for s in SNRS}
    mag = {s: {"pr": [], "pr_mag": [], "ml": []} for s in SNRS}
    for r in range(n):
        e = rng.normal(0, 0.002)
        for s in SNRS:
            mag[s]["pr"].append(base[s] + e)
            mag[s]["pr_mag"].append(base[s] + e + mag_off + rng.normal(0, 0.001))
            mag[s]["ml"].append(base[s] + e + ml_off + rng.normal(0, 0.001))
    ns = {"mag_ablation": mag, "n_reps": n, "_EXPLAIN_SNRS": SNRS}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "paired_cell", "exec"), ns)
    return buf.getvalue()

out = run(ml_off=-0.016, mag_off=+0.011)
check("Prototype-mag above Max Logit -> 'Prototype-mag significantly HIGHER than Max Logit'",
      "Prototype-mag significantly HIGHER than Max Logit" in out)
check("magnitude-aware scoring raises Prototype -> 'consistently RAISES'", "consistently RAISES" in out)
out = run(ml_off=+0.030, mag_off=+0.011)
check("Max Logit above Prototype-mag -> 'Max Logit significantly HIGHER than Prototype-mag'",
      "Max Logit significantly HIGHER than Prototype-mag" in out)
out = run(ml_off=+0.011, mag_off=-0.010)
check("magnitude-aware scoring lowers Prototype -> 'consistently LOWERS'", "consistently LOWERS" in out)
out = run(ml_off=0.0, mag_off=0.0)
check("no differences -> neither verdict claims significance",
      "not statistically consistent" in out and "no statistically distinguishable difference" in out)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All paired-cell checks passed.")
