"""Compression-ratio statistics (dissertation Section 3.2.6; CPU only), computed from
run_0/phase3_multiseed_state.json:

  * mean pixel AUROC at 0 dB per ratio and system (AWGN);
  * each seed's mean AUROC over the 21 swept SNRs (-20..+20 dB, AWGN);
  * Friedman test across the three ratios (src.effect_sizes.
    friedman_permutation_test, 20,000 permutations, fixed seed) with Kendall's W.
    The permutation p has a Monte Carlo error of about +/-0.0004 near 0.003 and
    depends slightly on row order, so it is reported to one significant figure;
  * pairwise exact Wilcoxon signed-rank tests paired by seed, Holm-corrected
    (Holm, 1979), with the number of seeds in which the larger ratio scored
    higher and the matched-pairs rank-biserial r.

Usage:  python3 tools/compression_stats.py PATH/TO/phase3_multiseed_state.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.effect_sizes import friedman_permutation_test   # noqa: E402

RATIOS = ("low", "medium", "high")
SWEEP = [str(x) for x in range(-20, 21, 2)]


def holm(pvals):
    """Holm-adjusted p-values, in the input order."""
    order = np.argsort(pvals); k = len(pvals); adj = [0.0] * k; running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (k - i) * pvals[idx])); adj[idx] = running
    return adj


def main(path):
    d = json.load(open(path))
    # Seeds present at every ratio, in the state file's order.
    seeds = [s for s in d["low"] if all(s in d[r] for r in RATIOS)]
    out = {}
    for det in ("max_logit", "prototype"):
        key = f"{det}_auc_roc_pixel"
        m = {r: np.array([np.mean([d[r][s]["sweep"][x][key] for x in SWEEP]) for s in seeds]) for r in RATIOS}
        a0 = {r: float(np.mean([d[r][s]["sweep"]["0"][key] for s in seeds])) for r in RATIOS}
        stat, p = friedman_permutation_test(np.column_stack([m[r] for r in RATIOS]))
        w = stat / (len(seeds) * (len(RATIOS) - 1))
        pairs = [("low", "medium"), ("low", "high"), ("medium", "high")]
        raw, rows = [], []
        for a, b in pairs:
            diff = m[b] - m[a]
            ranks = rankdata(np.abs(diff)); rp, rn = ranks[diff > 0].sum(), ranks[diff < 0].sum()
            raw.append(float(wilcoxon(m[b], m[a], method="exact").pvalue))
            rows.append({"pair": f"{a} vs {b}", "n_larger_higher": int((diff > 0).sum()), "r": float((rp - rn) / (rp + rn))})
        for row, pr, pa in zip(rows, raw, holm(raw)):
            row.update({"p_raw": pr, "p_holm": pa, "significant_holm_0.05": pa < 0.05})
        out[det] = {"n_seeds": len(seeds), "auroc_0db": a0, "mean_auroc_21snr": {r: float(m[r].mean()) for r in RATIOS},
                    "friedman_stat": float(stat), "friedman_p": float(p), "kendalls_w": float(w), "pairwise": rows}
        print(f"{det}: AUROC at 0 dB " + ", ".join(f"{r} {a0[r]:.4f}" for r in RATIOS)
              + "; mean over 21 SNRs " + " / ".join(f"{m[r].mean():.4f}" for r in RATIOS)
              + f"; Friedman p={p:.2g}, W={w:.2f}")
        for row in rows:
            print(f"    {row['pair']}: p={row['p_raw']:.2g} (Holm {row['p_holm']:.2g}){' *' if row['significant_holm_0.05'] else ''}, "
                  f"{row['n_larger_higher']}/{len(seeds)}, r={row['r']:.2f}")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 tools/compression_stats.py PATH/TO/phase3_multiseed_state.json")
    main(sys.argv[1])
