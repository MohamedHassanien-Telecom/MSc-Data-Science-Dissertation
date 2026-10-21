"""Decision Retention (DR): the share of a system's own clean-channel, above-chance
detection ability that remains at a given SNR (dissertation Section 3.1.2).

    DR_AUC(SNR) = (AUC(SNR) - 0.5)   / (AUC_ref - 0.5)
    DR_op(SNR)  = (0.95 - FPR95(SNR)) / (0.95 - FPR95_ref)

0.5 and 0.95 are the values of a detector that guesses. DR = 1 is the reference
ability, 0 is guessing and values above 1 are better than the reference.

References: "ref" is the same seed's value at CLS_FIT_SNR (40 dB, AWGN), also
used for the Rayleigh curves; "peak" (sensitivity check, AWGN only) is the seed's
best value over the swept SNRs.

SNR_X is the lowest swept SNR from which DR stays >= X at every higher swept SNR,
linearly interpolated between the last point below X and the next point. It is
"left_censored" if DR >= X already at the lowest swept SNR and "not_reached" if
DR < X at the highest. D_sem at SNR_X is interpolated from the seed's own D_sem
curve and is comparable within a system only.
"""
import numpy as np
from scipy import stats

CHANCE = {"auc": 0.5, "op": 0.95}      # chance AUROC; chance FPR95 (= the TPR level)


def dr_values(values, ref, metric):
    """DR of AUROC ("auc") or FPR95 ("op") values against one reference value.
    NaN where a value is missing or the reference is not better than chance."""
    v = np.array([np.nan if x is None else x for x in np.atleast_1d(values)], dtype=float)
    if ref is None or not np.isfinite(ref):
        return np.full_like(v, np.nan)
    c = CHANCE[metric]
    denom = (ref - c) if metric == "auc" else (c - ref)
    if denom <= 0:
        return np.full_like(v, np.nan)
    return (v - c) / denom if metric == "auc" else (c - v) / denom


def retention_snr(snr, dr, level):
    """SNR_level of one DR curve. Returns (value, status) with status
    "crossed", "left_censored", "not_reached" or "undefined" (NaN in the curve)."""
    snr = np.asarray(snr, dtype=float)
    dr = np.asarray(dr, dtype=float)
    if not np.all(np.isfinite(dr)):
        return None, "undefined"
    if dr[-1] < level:
        return None, "not_reached"
    below = np.where(dr < level)[0]
    if below.size == 0:
        return None, "left_censored"
    j = int(below[-1])                       # last point below the level; j+1 is >= level
    x = snr[j] + (level - dr[j]) * (snr[j + 1] - snr[j]) / (dr[j + 1] - dr[j])
    return float(x), "crossed"


def _summary(vals):
    """Mean, SD and t-based 95% CI half-width of the non-missing values."""
    v = np.array([x for x in vals if x is not None], dtype=float)
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95_half_width": None}
    sd = float(np.std(v, ddof=1)) if n > 1 else None
    hw = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
    return {"n": n, "mean": float(np.mean(v)), "sd": sd, "ci95_half_width": hw}


def _mean_band(curves):
    """Per-SNR mean and t-based 95% CI half-width across seeds, ignoring NaN."""
    import warnings
    a = np.asarray(curves, dtype=float)
    n = np.sum(np.isfinite(a), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN columns give NaN
        mean = np.nanmean(a, axis=0)
        sd = np.nanstd(a, axis=0, ddof=1)
    hw = np.where(n > 1, stats.t.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return [None if not np.isfinite(v) else float(v) for v in mean], \
           [None if not np.isfinite(v) else float(v) for v in hw]


def per_seed_retention(per_seed_points, ref_points, sweep_idx, ref_idx, det, metric, reference,
                       levels, snr_sweep):
    """DR curves, SNR_X values and D_sem at SNR_X for one detector and metric,
    one entry per seed.

    per_seed_points: list over seeds of lists over SNR points of metric dicts.
    ref_points:      the matching AWGN lists the reference is taken from.
    sweep_idx:       indices of the swept SNRs (CLS_FIT_SNR excluded).
    ref_idx:         index of CLS_FIT_SNR in ref_points.
    """
    key = f"{det}_auc_roc_pixel" if metric == "auc" else f"{det}_fpr95"
    curves, crossings, dsem_at = [], {lv: [] for lv in levels}, {lv: [] for lv in levels}
    for run, ref_run in zip(per_seed_points, ref_points):
        vals = [run[i].get(key) for i in sweep_idx]
        if reference == "ref":
            ref = ref_run[ref_idx].get(key)
        else:
            ref_sweep = np.array([np.nan if ref_run[i].get(key) is None else ref_run[i][key]
                                  for i in sweep_idx], dtype=float)
            ref = float(np.nanmax(ref_sweep) if metric == "auc" else np.nanmin(ref_sweep))
        d = dr_values(vals, ref, metric)
        curves.append(d.tolist())
        dsem = np.array([np.nan if run[i].get(f"{det}_dsem") is None else run[i][f"{det}_dsem"]
                         for i in sweep_idx], dtype=float)
        for lv in levels:
            x, status = retention_snr(snr_sweep, d, lv)
            crossings[lv].append((x, status))
            dsem_at[lv].append(float(np.interp(x, snr_sweep, dsem)) if status == "crossed" else None)
    return curves, crossings, dsem_at


def _paired(a, b, lo, hi):
    """Prototype (a) - Max Logit (b) comparison of SNR_X over the same seeds.

    a, b: lists of (value, status). Wilcoxon signed-rank test on the seeds where
    both crossed (paired by seed) and Mann-Whitney U on the crossed values as an
    unpaired check (Appendix A.2). The ordering count also uses censored pairs
    whose order is known (left-censored lies below lo, not-reached above hi)."""
    def _pos(v, s):
        return v if s == "crossed" else (lo - 1e9 if s == "left_censored" else (hi + 1e9 if s == "not_reached" else None))
    diffs, a_lower, determinable = [], 0, 0
    for (va, sa), (vb, sb) in zip(a, b):
        pa, pb = _pos(va, sa), _pos(vb, sb)
        if sa == "crossed" and sb == "crossed":
            diffs.append(va - vb)
        if pa is not None and pb is not None and not (sa == sb and sa != "crossed"):
            determinable += 1
            a_lower += int(pa < pb)
    d = np.asarray(diffs, dtype=float)
    p = None
    if d.size >= 2 and np.any(d != 0):
        p = float(stats.wilcoxon(d).pvalue)
    xa = [v for v, s in a if s == "crossed"]
    xb = [v for v, s in b if s == "crossed"]
    p_mw = (float(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
            if len(xa) >= 2 and len(xb) >= 2 else None)
    return {"n_both_crossed": int(d.size), "mean_diff_db": float(d.mean()) if d.size else None,
            "sd_diff_db": float(d.std(ddof=1)) if d.size > 1 else None, "wilcoxon_p": p,
            "mannwhitney_p": p_mw,
            "prototype_lower_count": a_lower, "n_order_determinable": determinable}


def decision_retention_analysis(per_seed, snrs, cfg, channels=("awgn", "rayleigh")):
    """DR analysis of Phase II per-seed results (the content of phase2_per_seed.json).

    Returns, per channel, metric and reference: the mean DR curve with its 95% CI
    band, every seed's SNR_X and status, summaries over the seeds that crossed
    (mean, SD, 95% CI half-width; censoring counts), D_sem at SNR_X, the paired
    Prototype - Max Logit comparison and the SNR where the mean DR rises fastest."""
    levels = list(cfg["DR_LEVELS"])
    snrs = [float(s) for s in snrs]
    ref_snr = float(cfg["CLS_FIT_SNR"])
    ref_idx = snrs.index(ref_snr)
    sweep_idx = [i for i, s in enumerate(snrs) if s != ref_snr]
    snr_sweep = np.array([snrs[i] for i in sweep_idx])
    lo, hi = float(snr_sweep.min()), float(snr_sweep.max())
    out = {"levels": levels, "snr_db": snr_sweep.tolist(), "reference_snr_db": ref_snr,
           "reference_channel": "awgn", "n_seeds": len(per_seed["awgn"]), "channels": {}}
    for ch in channels:
        pts = per_seed.get(ch) or []
        if not pts:
            continue
        out["channels"][ch] = {}
        for metric in ("auc", "op"):
            out["channels"][ch][metric] = {}
            for reference in ("ref", "peak"):
                if ch != "awgn" and reference == "peak":
                    continue                      # the "peak" reference is AWGN only
                block = {"detectors": {}, "paired_prototype_minus_max_logit": {}}
                cross = {}
                for det in ("max_logit", "prototype"):
                    curves, crossings, dsem_at = per_seed_retention(
                        pts, per_seed["awgn"], sweep_idx, ref_idx, det, metric, reference,
                        levels, snr_sweep)
                    cross[det] = crossings
                    mean, hw = _mean_band(curves)
                    m = np.array([np.nan if v is None else v for v in mean], dtype=float)
                    steep = (float(snr_sweep[int(np.nanargmax(np.gradient(m, snr_sweep)))])
                             if np.any(np.isfinite(m)) else None)
                    block["detectors"][det] = {
                        "mean_curve": mean, "ci95_half_width_curve": hw,
                        "max_mean_dr": float(np.nanmax(m)) if np.any(np.isfinite(m)) else None,
                        "steepest_increase_snr_db": steep,
                        "levels": {str(lv): {
                            "per_seed_snr_db": [x for x, _ in crossings[lv]],
                            "per_seed_status": [s for _, s in crossings[lv]],
                            "n_left_censored": sum(s == "left_censored" for _, s in crossings[lv]),
                            "n_not_reached": sum(s == "not_reached" for _, s in crossings[lv]),
                            "n_undefined": sum(s == "undefined" for _, s in crossings[lv]),
                            "snr_db": _summary([x for x, _ in crossings[lv]]),
                            "dsem_at_snr": _summary(dsem_at[lv]),
                        } for lv in levels},
                    }
                for lv in levels:
                    block["paired_prototype_minus_max_logit"][str(lv)] = _paired(
                        cross["prototype"][lv], cross["max_logit"][lv], lo, hi)
                out["channels"][ch][metric][reference] = block
    return out


def phase3_retention_by_cr(state, cr_order, seeds, snrs, cfg):
    """SNR_X of DR_AUC per compression ratio from phase3_multiseed_state.json.
    Each CR is referenced to its own 40 dB value for the same seed. A CR whose
    per-seed sweeps are incomplete gives None."""
    levels = list(cfg["DR_LEVELS"])
    snrs = [float(s) for s in snrs]
    ref_snr = float(cfg["CLS_FIT_SNR"])
    sweep = [s for s in snrs if s != ref_snr]
    snr_sweep = np.array(sweep)
    out = {"levels": levels, "by_cr": {}}
    for cr in cr_order:
        runs = []
        for s in seeds:
            entry = (state.get(cr) or {}).get(str(s)) or {}
            sw = entry.get("sweep") or {}
            keys = {float(k): v for k, v in sw.items() if k.lstrip("-").replace(".", "", 1).isdigit()}
            if not all(x in keys for x in snrs):
                runs = None
                break
            runs.append([keys[x] for x in snrs])
        if runs is None or not runs:
            out["by_cr"][cr] = None
            continue
        ref_idx = snrs.index(ref_snr)
        sweep_idx = [i for i, x in enumerate(snrs) if x != ref_snr]
        res = {}
        cross = {}
        for det in ("max_logit", "prototype"):
            _, crossings, _ = per_seed_retention(runs, runs, sweep_idx, ref_idx, det, "auc", "ref",
                                                 levels, snr_sweep)
            cross[det] = crossings
            res[det] = {str(lv): {"snr_db": _summary([x for x, _ in crossings[lv]]),
                                  "per_seed_status": [st for _, st in crossings[lv]]} for lv in levels}
        res["paired_prototype_minus_max_logit"] = {
            str(lv): _paired(cross["prototype"][lv], cross["max_logit"][lv],
                             float(snr_sweep.min()), float(snr_sweep.max())) for lv in levels}
        res["n_seeds"] = len(runs)
        out["by_cr"][cr] = res
    return out
