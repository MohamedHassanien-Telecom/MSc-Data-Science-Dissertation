"""Recompute Decision Retention summaries from the saved per-seed AUROC curves
(dissertation Appendix F.10).

The script imports no project code: it reads only the JSON state files and
re-implements every step, so it checks the reported SNR_X values and system
comparisons independently. It needs numpy and scipy and runs on a CPU in a few
seconds.

Definitions (dissertation Section 3.1.2 and Appendix A.6):
    DR(SNR) = (AUROC(SNR) - 0.5) / (AUROC_ref - 0.5)
        reference "40dB": the same seed's AWGN value at 40 dB (also for Rayleigh);
        reference "best": the seed's highest AWGN value over the swept SNRs.
    SNR_X   = the lowest swept SNR (-20 ... +20 dB) from which DR stays >= X at every
              higher swept SNR, linearly interpolated between the last point below X
              and the next one. "L": DR >= X already at -20 dB; "NR": DR < X at +20 dB.
    Paired comparison: Prototype - Max Logit SNR_X on the seeds where both crossed,
              with a t-based and a bootstrap (seeds resampled) 95% confidence interval
              and an exact two-sided Wilcoxon signed-rank p-value.
    Town interaction: (Prototype - Max Logit in Town 5) - (the same in Town 6), per seed.

Inputs (all written to OUTPUT_DIR/run_0/ by the notebook):
    stratified/stratified_state.json            plain, LS (7/14/28) and matched-shuffled AUROC
    stratified_supp/stratified_supp_state.json  region-balanced and per-town AUROC (optional)
    stratified_supp/stratified_supp_summary.json  stored per-seed SNR_X, compared if given

Usage:
    python3 tools/verify_retention.py --state PATH/stratified_state.json \
        [--supp-state PATH/stratified_supp_state.json] \
        [--supp-summary PATH/stratified_supp_summary.json] [--out DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

LEVELS = (0.5, 0.9)
REF_SNR = 40.0
N_BOOT = 10000
BOOT_SEED = 0
DETS = ("max_logit", "prototype")

# Quantities read from each state entry. Gain is shifted by 0.5 so that the DR
# formula gives gain / gain at the reference.
MAIN_VARIANTS = {
    "plain": ("Plain pixel AUROC", lambda e: e["matched"]),
    "ls14": ("LS-AUROC, 14x14 regions (primary)", lambda e: e["ls"]["14"]),
    "ls7": ("LS-AUROC, 7x7 regions", lambda e: e["ls"]["7"]),
    "ls28": ("LS-AUROC, 28x28 regions", lambda e: e["ls"]["28"]),
    "gain": ("Matched - shuffled AUROC", lambda e: None if e.get("shuffled") is None
             else 0.5 + e["matched"] - e["shuffled"]),
}
SUPP_VARIANTS = {
    "ls": ("LS-AUROC, pooled", lambda e: e["ls"]),
    "rb": ("Region-balanced LS-AUROC", lambda e: e["rb"]),
    "ls_t5": ("LS-AUROC, Town 5", lambda e: e["town"]["t5"]["ls"]),
    "plain_t5": ("Plain AUROC, Town 5", lambda e: e["town"]["t5"]["plain"]),
    "ls_t6": ("LS-AUROC, Town 6", lambda e: e["town"]["t6"]["ls"]),
    "plain_t6": ("Plain AUROC, Town 6", lambda e: e["town"]["t6"]["plain"]),
}


# ------------------------------------------------------------------ core steps
def curves(state, fn, channel, det):
    """Per-seed AUROC arrays (seeds x SNRs) and the SNR grid for one variant."""
    reps = sorted(state["reps"], key=int)
    snr = np.array([p["snr_db"] for p in state["reps"][reps[0]]["awgn"][det]], dtype=float)
    vals = np.array([[np.nan if fn(p) is None else fn(p) for p in state["reps"][r][channel][det]]
                     for r in reps], dtype=float)
    ref = np.array([[np.nan if fn(p) is None else fn(p) for p in state["reps"][r]["awgn"][det]]
                    for r in reps], dtype=float)
    return snr, vals, ref


def retention(snr, vals, ref_vals, reference):
    """DR curves over the swept SNRs (40 dB excluded) for every seed."""
    swept = snr != REF_SNR
    if reference == "40dB":
        ref = ref_vals[:, np.flatnonzero(snr == REF_SNR)[0]]
    else:
        ref = np.nanmax(ref_vals[:, swept], axis=1)
    return snr[swept], (vals[:, swept] - 0.5) / (ref[:, None] - 0.5)


def snr_at_level(snr, dr, level):
    """SNR_X for one seed: (value, "crossed") or (None, "L" / "NR" / "undefined")."""
    if not np.all(np.isfinite(dr)):
        return None, "undefined"
    k = len(dr) - 1
    if dr[k] < level:
        return None, "NR"
    while k > 0 and dr[k - 1] >= level:
        k -= 1
    if k == 0:
        return None, "L"
    s0, s1, d0, d1 = snr[k - 1], snr[k], dr[k - 1], dr[k]
    return float(s0 + (level - d0) * (s1 - s0) / (d1 - d0)), "crossed"


def wilcoxon_exact(d):
    """Exact two-sided Wilcoxon signed-rank p-value (zeros dropped; tied ranks
    are averaged and the null distribution enumerated over the actual ranks)."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return None
    r2 = np.rint(2 * stats.rankdata(np.abs(d))).astype(int)      # doubled ranks are integers
    dist = np.zeros(r2.sum() + 1)
    dist[0] = 1.0
    for r in r2:                                                 # each sign + or - with p = 1/2
        dist[r:] = dist[r:] + dist[:-r] if r else dist
    dist /= dist.sum()
    w = int(r2[d > 0].sum())
    p = 2 * min(dist[:w + 1].sum(), dist[w:].sum())
    return float(min(p, 1.0))


def interval(d):
    """Mean, SD, t-based and bootstrap 95% confidence intervals of the mean."""
    d = np.asarray(d, dtype=float)
    n = d.size
    m, sd = float(d.mean()), float(d.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    rng = np.random.default_rng(BOOT_SEED)
    boot = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n": n, "mean": m, "sd": sd, "t_ci": [m - h, m + h], "boot_ci": [float(lo), float(hi)],
            "wilcoxon_p": wilcoxon_exact(d)}


def order_key(x, status):
    """Position of a seed for the ordering count: L below every value, NR above."""
    return {"crossed": x, "L": -np.inf, "NR": np.inf}.get(status)


def compare(cross_pr, cross_ml):
    """Prototype - Max Logit over seeds; ordering count includes censored pairs
    whose order is known."""
    diffs = [a[0] - b[0] for a, b in zip(cross_pr, cross_ml) if a[1] == b[1] == "crossed"]
    lower = known = 0
    for (xa, sa), (xb, sb) in zip(cross_pr, cross_ml):
        ka, kb = order_key(xa, sa), order_key(xb, sb)
        if ka is None or kb is None or (sa == sb != "crossed"):
            continue
        known += 1
        lower += int(ka < kb)
    out = interval(diffs) if len(diffs) >= 2 else {"n": len(diffs)}
    out.update({"prototype_lower": lower, "n_order_known": known})
    return out


def describe(cross):
    v = np.array([x for x, s in cross if s == "crossed"], dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()) if v.size else None,
            "sd": float(v.std(ddof=1)) if v.size > 1 else None,
            "L": sum(s == "L" for _, s in cross), "NR": sum(s == "NR" for _, s in cross),
            "per_seed": [x for x, _ in cross], "status": [s for _, s in cross]}


def analyse(state, variants):
    """All variants x channels x references x levels for one state file."""
    rep0 = state["reps"][sorted(state["reps"], key=int)[0]]
    out = {}
    for key, (label, fn) in variants.items():
        for ch in rep0:
            for reference in ("40dB", "best"):
                if ch != "awgn" and reference == "best":
                    continue
                for lv in LEVELS:
                    cross = {}
                    for det in DETS:
                        snr, vals, ref = curves(state, fn, ch, det)
                        sw, dr = retention(snr, vals, ref, reference)
                        cross[det] = [snr_at_level(sw, row, lv) for row in dr]
                    out[(key, ch, reference, lv)] = {
                        "label": label,
                        "max_logit": describe(cross["max_logit"]),
                        "prototype": describe(cross["prototype"]),
                        "prototype_minus_max_logit": compare(cross["prototype"], cross["max_logit"]),
                    }
    return out


def town_interaction(supp):
    """(PR - ML in Town 5) - (PR - ML in Town 6) per seed, for LS and plain AUROC."""
    out = {}
    for metric in ("ls", "plain"):
        for reference in ("40dB", "best"):
            for lv in LEVELS:
                a, b = supp[(f"{metric}_t5", "awgn", reference, lv)], supp[(f"{metric}_t6", "awgn", reference, lv)]
                if any(s != "crossed" for r in (a, b) for det in DETS for s in r[det]["status"]):
                    out[(metric, reference, lv)] = None                 # censored seeds: not tested
                    continue
                gap = {t: np.array(r["prototype"]["per_seed"]) - np.array(r["max_logit"]["per_seed"])
                       for t, r in (("t5", a), ("t6", b))}
                out[(metric, reference, lv)] = interval(gap["t5"] - gap["t6"])
    return out


# ------------------------------------------------------------------ output
def fmt_ci(c):
    return f"[{c[0]:+.2f}, {c[1]:+.2f}]"


def fmt_p(p):
    return "-" if p is None else f"{p:.2g}"


def table(res, title):
    lines = [f"## {title}", "",
             "| Variant | Channel | Reference | Level | Max Logit SNR_X | Prototype SNR_X | PR - ML (dB) | t 95% CI | bootstrap 95% CI | Wilcoxon p | PR lower |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (key, ch, ref, lv), r in res.items():
        def s(d):
            c = f" (L {d['L']}, NR {d['NR']})" if d["L"] or d["NR"] else ""
            return "-" if d["mean"] is None else f"{d['mean']:+.2f} ± {d['sd']:.2f}{c}"
        c = r["prototype_minus_max_logit"]
        diff = (f"{c['mean']:+.2f} (n={c['n']})", fmt_ci(c["t_ci"]), fmt_ci(c["boot_ci"]), fmt_p(c["wilcoxon_p"])) \
            if "mean" in c else ("-", "-", "-", "-")
        lines.append(f"| {key} | {ch} | {ref} | SNR{int(lv * 100)} | {s(r['max_logit'])} | {s(r['prototype'])} | "
                     f"{' | '.join(diff)} | {c['prototype_lower']}/{c['n_order_known']} |")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--supp-state")
    ap.add_argument("--supp-summary")
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)

    main_res = analyse(json.load(open(a.state)), MAIN_VARIANTS)
    lines = ["# Decision Retention recomputed from per-seed AUROC curves", "",
             "SNR_X: mean ± SD over seeds that crossed (dB). PR - ML: Prototype minus Max Logit, paired by seed; "
             "negative = Prototype keeps the level down to a lower SNR. "
             f"Bootstrap: {N_BOOT} resamples of the seeds (seed {BOOT_SEED}), percentile interval.", ""]
    lines += table(main_res, "Main stratified evaluation (stratified_state.json)")
    result = {"main": main_res}

    if a.supp_state:
        supp = analyse(json.load(open(a.supp_state)), SUPP_VARIANTS)
        lines += [""] + table(supp, "Region-balanced and per-town evaluation (stratified_supp_state.json)")
        inter = town_interaction(supp)
        lines += ["", "## Town interaction: (PR - ML, Town 5) - (PR - ML, Town 6)", "",
                  "| Metric | Reference | Level | Mean ± SD (dB) | t 95% CI | bootstrap 95% CI | Wilcoxon p |",
                  "|---|---|---|---|---|---|---|"]
        for (metric, ref, lv), r in inter.items():
            lines.append(f"| {metric} | {ref} | SNR{int(lv * 100)} | " + (
                "censored seeds; not tested | - | - | -" if r is None else
                f"{r['mean']:+.2f} ± {r['sd']:.2f} (n={r['n']}) | {fmt_ci(r['t_ci'])} | "
                f"{fmt_ci(r['boot_ci'])} | {fmt_p(r['wilcoxon_p'])}") + " |")
        result.update({"supplement": supp, "town_interaction": inter})

        if a.supp_summary:
            stored = json.load(open(a.supp_summary))["dr"]
            names = {"40dB": "ref", "best": "peak"}
            worst, n_cmp = 0.0, 0
            for (key, ch, ref, lv), r in supp.items():
                blk = stored[key]["channels"][ch][names[ref]]["detectors"]
                for det in DETS:
                    old = blk[det]["levels"][str(lv)]["per_seed_snr_db"]
                    for x, y in zip(r[det]["per_seed"], old):
                        if (x is None) != (y is None):
                            worst = np.inf
                        elif x is not None:
                            worst = max(worst, abs(x - y))
                        n_cmp += 1
            lines += ["", f"Stored per-seed SNR_X in {Path(a.supp_summary).name}: {n_cmp} values compared, "
                          f"largest absolute difference {worst:.2e} dB."]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify_retention.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump({sec: {" | ".join(map(str, k)): v for k, v in res.items()} for sec, res in result.items()},
              open(out / "verify_retention.json", "w"), indent=1)
    print("\n".join(lines))
    return result


if __name__ == "__main__":
    main()
