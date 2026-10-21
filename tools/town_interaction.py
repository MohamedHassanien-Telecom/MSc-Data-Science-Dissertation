"""Town x system test on Decision Retention (dissertation Appendix F.8; CPU only):
does the Prototype - Max Logit gap in SNR_X differ between the two test towns?

Reads run_0/stratified_supp/stratified_supp_summary.json (phases/stratified_supp.py)
and, for LS-AUROC and plain AUROC, both references (40 dB and each seed's best
value) and SNR50 / SNR90, computes per seed

    gap_t   = SNR_X(Prototype, town t) - SNR_X(Max Logit, town t)
    inter   = gap_t5 - gap_t6

It reports the mean +/- SD of inter, the number of seeds with inter < 0, the
exact two-sided Wilcoxon signed-rank p, the paired t-test p, and each system's
own town difference (Town 5 - Town 6). Seeds are paired because the same models
are scored on both towns; comparisons with censored seeds are skipped. The test
is over training repetitions on fixed images: it shows whether the town
difference exceeds seed-to-seed variation, not why the towns differ.

Usage:  python3 tools/town_interaction.py PATH/TO/stratified_supp_summary.json
"""
import json
import sys

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def per_seed(d, key, ref, det, lev):
    """Per-seed SNR_X and censoring status of one variant, reference, detector and level."""
    lv = d["dr"][key]["channels"]["awgn"][ref]["detectors"][det]["levels"][lev]
    return np.array(lv["per_seed_snr_db"], dtype=float), lv["per_seed_status"]


def main(path):
    d = json.load(open(path))
    towns = d["towns"]
    if len(towns) != 2:
        sys.exit(f"expected two towns, found {towns}")
    a, b = towns
    out = {}
    print(f"Towns {a} vs {b}; images {d['n_images_per_town']}")
    for metric in ("ls", "plain"):
        for ref in ("ref", "peak"):
            for lev in ("0.5", "0.9"):
                vals, censored = {}, False
                for t in towns:
                    m, sm = per_seed(d, f"{metric}_{t}", ref, "max_logit", lev)
                    p, sp = per_seed(d, f"{metric}_{t}", ref, "prototype", lev)
                    censored |= any(s != "crossed" for s in list(sm) + list(sp))
                    vals[t] = (m, p)
                name = f"{metric} {'40 dB ref' if ref == 'ref' else 'best value'} SNR{int(float(lev) * 100)}"
                if censored:
                    print(f"  {name}: censored seeds present - skipped")
                    continue
                gap = {t: vals[t][1] - vals[t][0] for t in towns}
                inter = gap[a] - gap[b]
                dm = vals[a][0] - vals[b][0]
                dp = vals[a][1] - vals[b][1]
                r = {"gap_" + a: float(gap[a].mean()), "gap_" + b: float(gap[b].mean()),
                     "interaction_mean": float(inter.mean()), "interaction_sd": float(inter.std(ddof=1)),
                     "n_interaction_negative": int((inter < 0).sum()), "n": int(inter.size),
                     "wilcoxon_p": float(wilcoxon(inter, method="exact").pvalue),
                     "paired_t_p": float(ttest_rel(gap[a], gap[b]).pvalue),
                     "max_logit_town_diff": float(dm.mean()), "max_logit_town_diff_sd": float(dm.std(ddof=1)),
                     "max_logit_town_p": float(wilcoxon(dm, method="exact").pvalue),
                     "prototype_town_diff": float(dp.mean()), "prototype_town_diff_sd": float(dp.std(ddof=1)),
                     "prototype_town_p": float(wilcoxon(dp, method="exact").pvalue)}
                out[name] = r
                print(f"  {name}: gap {a} {r['gap_' + a]:+.2f}, {b} {r['gap_' + b]:+.2f}; "
                      f"{a} - {b} {r['interaction_mean']:+.2f} +/- {r['interaction_sd']:.2f} dB "
                      f"(negative in {r['n_interaction_negative']}/{r['n']}; Wilcoxon p={r['wilcoxon_p']:.2g}; "
                      f"paired t p={r['paired_t_p']:.2g}); Max Logit {a}-{b} {r['max_logit_town_diff']:+.2f} "
                      f"(p={r['max_logit_town_p']:.2g}), Prototype {a}-{b} {r['prototype_town_diff']:+.2f} "
                      f"(p={r['prototype_town_p']:.2g})")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 tools/town_interaction.py PATH/TO/stratified_supp_summary.json")
    main(sys.argv[1])
