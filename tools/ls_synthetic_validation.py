"""Synthetic validation of LS-AUROC (dissertation Appendix F.7; CPU only, no data, no
model): does LS-AUROC behave as intended when the strength of a location prior is
controlled exactly?

Synthetic test set: N images of 224 x 224 pixels, each with one or two
rectangular anomalies at location-biased positions (centre row ~ N(150, 22),
centre column ~ N(112, 45), clipped; sides 12-56 px). The location prior pi is
the anomaly frequency of an independent set of masks from the same distribution.

Score maps are constant inside 16 x 16-pixel cells (14 x 14 cells), like the
Max Logit system's maps:

    score(i, c) = a * z(pi_c) + b * f(i, c) + e(i, c),   e ~ N(0, 1)

  a      prior strength: weight of the image-independent cell prior pi_c
         (standardised to mean 0, SD 1 over cells);
  b      image-signal strength: weight of f(i, c), the fraction of anomalous
         pixels of image i in cell c;
  e      independent noise per image and cell (the same draws for every a and b
         within a repetition).

A third variant gives the prior at pixel resolution (it varies inside a cell), as
the model-free location map does in the study.

Metrics as in phases/stratified_eval.py: plain pixel AUROC (seeded
200,000-pixel subsample), LS-AUROC at 14x14, 28x28 and 7x7 regions (seeded
2,000,000-pixel subsample; phases.stratified_eval.stratified_auc), and the
region-balanced variant (mean of the per-region AUROCs) at 14x14.

Expected behaviour:
  1. plain AUROC changes with a; LS-AUROC at 14x14 and 28x28 does not (a
     cell-constant, image-independent map cannot change the order of pixels
     inside a region of either grid);
  2. LS-AUROC rises with b, from chance at b = 0;
  3. at 7x7 (32 x 32-pixel regions, four cells each) the prior varies inside a
     region, so LS-AUROC there rises with a;
  4. a pixel-resolution prior raises LS-AUROC at every grid: the within-region
     residual that LS-AUROC does not remove.

Usage:  python3 tools/ls_synthetic_validation.py [out_dir]
        python3 tools/ls_synthetic_validation.py --plot out_dir   (redraw from the JSON)
Writes ls_synthetic_validation.md, .json and .png.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phases.stratified_eval import stratified_auc, region_index

H = W = 224
CELL = 16
# 14 cells per side.
NC = H // CELL
N_IMAGES = 400
N_PRIOR_IMAGES = 2000
N_REPS = 5
PLAIN_SAMPLE = 200_000
LS_SAMPLE = 2_000_000
A_GRID = (0.0, 1.0, 2.0, 4.0, 8.0)
B_GRID = (0.0, 1.0, 2.0, 4.0, 8.0)
A_FIXED = 4.0
# Weak and strong image signal for the prior-strength sweep.
B_FOR_A_SWEEP = (1.0, 4.0)


def make_masks(n, rng):
    """n binary masks (n, H, W) with 1-2 rectangles at location-biased positions."""
    m = np.zeros((n, H, W), dtype=bool)
    for i in range(n):
        for _ in range(rng.integers(1, 3)):
            hgt, wid = rng.integers(12, 57, size=2)
            r = int(np.clip(rng.normal(150, 22), hgt // 2, H - 1 - hgt // 2))
            c = int(np.clip(rng.normal(112, 45), wid // 2, W - 1 - wid // 2))
            m[i, r - hgt // 2:r + hgt // 2, c - wid // 2:c + wid // 2] = True
    return m


def cell_fraction(masks):
    """(n, NC, NC) fraction of anomalous pixels per 16 x 16 cell."""
    n = masks.shape[0]
    return masks.reshape(n, NC, CELL, NC, CELL).mean(axis=(2, 4))


def upsample(cells):
    """(n, NC, NC) -> (n, H, W), nearest (block-constant)."""
    return np.repeat(np.repeat(cells, CELL, axis=1), CELL, axis=2)


def zscore(x):
    """Standardise to mean 0, SD 1."""
    return (x - x.mean()) / (x.std() + 1e-12)


def region_balanced_auc(y, s, reg):
    """Mean of per-region AUROCs over regions that contain both classes."""
    vals = []
    order = np.argsort(reg, kind="stable")
    r, yy, ss = reg[order], y[order], s[order]
    cuts = np.flatnonzero(np.diff(r)) + 1
    for yg, sg in zip(np.split(yy, cuts), np.split(ss, cuts)):
        if 0 < yg.sum() < yg.size:
            vals.append(roc_auc_score(yg, sg))
    return float(np.mean(vals)) if vals else float("nan")


def metrics(labels, scores, idx_plain, idx_ls, pix_ls):
    """Plain AUROC, LS-AUROC at 14x14, 28x28 and 7x7, and region-balanced LS-AUROC at 14x14."""
    out = {"plain": float(roc_auc_score(labels[idx_plain], scores[idx_plain]))}
    y, s = labels[idx_ls], scores[idx_ls]
    for g in (14, 28, 7):
        out[f"ls{g}"] = stratified_auc(y, s, region_index(pix_ls, H, W, g))[0]
    out["rb14"] = region_balanced_auc(y, s, region_index(pix_ls, H, W, 14))
    return out


def run(out_dir):
    """Run the four experiments over N_REPS repetitions and write the table, JSON and figure."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = {"A_prior_strength_b1": {}, "A_prior_strength_b4": {}, "B_signal_strength": {}, "C_pixel_prior": {}}
    for rep in range(N_REPS):
        rng = np.random.default_rng(20261020 + rep)
        # Prior from an independent set of masks.
        prior_pix = make_masks(N_PRIOR_IMAGES, rng).mean(axis=0)
        prior_cell = zscore(prior_pix.reshape(NC, CELL, NC, CELL).mean(axis=(1, 3)))
        prior_pix_z = zscore(prior_pix)
        masks = make_masks(N_IMAGES, rng)
        labels = masks.reshape(-1).astype(np.int8)
        f = cell_fraction(masks)
        noise = rng.standard_normal(f.shape)
        n = labels.size
        idx_plain = np.random.RandomState(42).choice(n, size=PLAIN_SAMPLE, replace=False)
        idx_ls = np.random.RandomState(42).choice(n, size=LS_SAMPLE, replace=False)
        pix_ls = idx_ls % (H * W)

        def score(a, b, pixel_prior=False):
            cells = b * f + noise
            s = upsample(cells)
            s = s + a * (prior_pix_z[None] if pixel_prior else upsample(np.broadcast_to(prior_cell, f.shape)))
            return s.reshape(-1)

        for bb in B_FOR_A_SWEEP:
            for a in A_GRID:
                results[f"A_prior_strength_b{bb:g}"].setdefault(str(a), []).append(
                    metrics(labels, score(a, bb), idx_plain, idx_ls, pix_ls))
        for b in B_GRID:
            results["B_signal_strength"].setdefault(str(b), []).append(metrics(labels, score(A_FIXED, b), idx_plain, idx_ls, pix_ls))
        for a in A_GRID:
            results["C_pixel_prior"].setdefault(str(a), []).append(metrics(labels, score(a, 0.0, pixel_prior=True), idx_plain, idx_ls, pix_ls))
        print(f"  repetition {rep + 1}/{N_REPS} done")

    keys = ("plain", "ls14", "ls28", "ls7", "rb14")
    summ = {exp: {v: {k: (float(np.mean([r[k] for r in rs])), (float(np.std([r[k] for r in rs], ddof=1)) if len(rs) > 1 else 0.0))
                      for k in keys} for v, rs in d.items()} for exp, d in results.items()}
    # Invariance check: largest change of LS-AUROC at 14x14 and 28x28 with a, per repetition.
    inv = {g: max(abs(results[e][str(a)][r][g] - results[e]["0.0"][r][g])
                  for e in ("A_prior_strength_b1", "A_prior_strength_b4") for a in A_GRID for r in range(N_REPS))
           for g in ("ls14", "ls28", "rb14")}
    summ["max_change_with_a"] = inv
    json.dump(summ, open(out_dir / "ls_synthetic_validation.json", "w"), indent=2)

    lines = ["# Synthetic validation of LS-AUROC", "",
             f"{N_IMAGES} synthetic images, {N_REPS} repetitions (mean +/- SD); scores constant in 16x16 cells.",
             f"Largest change with prior strength a (cell-constant prior): LS 14x14 {inv['ls14']:.2e}, LS 28x28 {inv['ls28']:.2e}, region-balanced 14x14 {inv['rb14']:.2e}", ""]
    titles = {"A_prior_strength_b1": "A1. Prior strength a, weak image signal (b = 1)",
              "A_prior_strength_b4": "A2. Prior strength a, strong image signal (b = 4)",
              "B_signal_strength": f"B. Image-signal strength b (prior a = {A_FIXED:g})",
              "C_pixel_prior": "C. Pixel-resolution prior a, no image signal (b = 0)"}
    for exp in ("A_prior_strength_b1", "A_prior_strength_b4", "B_signal_strength", "C_pixel_prior"):
        lines += [f"## {titles[exp]}", "", "| value | plain AUROC | LS 14x14 | LS 28x28 | LS 7x7 | region-balanced 14x14 |",
                  "|---:|---|---|---|---|---|"]
        for v, m in summ[exp].items():
            lines.append(f"| {float(v):g} | " + " | ".join(f"{m[k][0]:.4f} +/- {m[k][1]:.4f}" for k in keys) + " |")
        lines.append("")
    (out_dir / "ls_synthetic_validation.md").write_text("\n".join(lines))

    plot(summ, out_dir)
    print("\n".join(lines))
    return summ


def plot(summ, out_dir):
    """2 x 2 figure of the four experiments (also accepts a saved JSON summary)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.6), sharey=True)
    styles = {"plain": ("Plain pixel AUROC", "black", "-", "o"), "ls14": ("LS-AUROC 14x14", "#1f77b4", "-", "s"),
              "ls28": ("LS-AUROC 28x28", "#17becf", "--", "^"), "ls7": ("LS-AUROC 7x7", "#d62728", ":", "v"),
              "rb14": ("Region-balanced 14x14", "#9467bd", "-.", "D")}
    panels = (("A_prior_strength_b1", "(a) Cell-constant prior, weak signal (b = 1)", "Prior strength a"),
              ("A_prior_strength_b4", "(b) Cell-constant prior, strong signal (b = 4)", "Prior strength a"),
              ("B_signal_strength", "(c) Image signal, prior a = 4", "Image-signal strength b"),
              ("C_pixel_prior", "(d) Pixel-resolution prior, no signal (b = 0)", "Prior strength a"))
    for ax, (exp, title, xl) in zip(axs.ravel(), panels):
        xs = [float(v) for v in summ[exp]]
        for k, (lab, c, ls, mk) in styles.items():
            m = np.array([summ[exp][v][k][0] for v in summ[exp]]); sd = np.array([summ[exp][v][k][1] for v in summ[exp]])
            ax.errorbar(xs, m, yerr=sd, label=lab, color=c, ls=ls, marker=mk, ms=4, capsize=2)
        ax.axhline(0.5, color="grey", ls=":", lw=.8)
        ax.set_title(title, fontsize=10); ax.set_xlabel(xl)
    for ax in axs[:, 0]:
        ax.set_ylabel("AUROC")
    axs[0, 1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "ls_synthetic_validation.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--plot":
        plot(json.load(open(Path(sys.argv[2]) / "ls_synthetic_validation.json")), sys.argv[2])
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else ".")
