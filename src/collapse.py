"""Semantic-collapse analysis of the mean AUROC-vs-SNR trajectory (dissertation
Section 3.1.8 and Appendix D).

  Stage 1  elbow: point of maximum |d2 AUC / d D_sem2| (second derivative of AUC
           with respect to D_sem), excluding points where D_sem barely changes
  Stage 2  verification: |dAUC/dD_sem| above a percentile threshold at two
           consecutive SNR points near the elbow
  Stage 3  seed-bootstrap stability of the Stage 1 elbow
  Stage 4  smooth S-curve versus discrete break, compared by delta AIC

Stages 1-2 are run at every percentile in cfg["COLLAPSE_SENSITIVITY_PERCENTILES"]
(the same percentile for the plateau exclusion and the gradient threshold).
"""
import numpy as np


def _detect_collapse_at_percentile(snr_arr, auc_arr, dsem_arr, plateau_pct, grad_pct,
                                     grad_threshold_override=None):
    """Stages 1 and 2 at one percentile setting."""
    auc_a  = np.array(auc_arr,  dtype=float)
    dsem_a = np.array(dsem_arr, dtype=float)
    snr_a  = np.array(snr_arr,  dtype=float)

    # Derivatives with respect to D_sem via the chain rule over the SNR grid.
    d_auc_d_snr  = np.gradient(auc_a, snr_a)
    d_dsem_d_snr = np.gradient(dsem_a, snr_a)

    safe_denom = np.where(np.abs(d_dsem_d_snr) < 1e-6,
                          1e-6 * np.sign(d_dsem_d_snr + 1e-12),
                          d_dsem_d_snr)
    first_deriv = d_auc_d_snr / safe_denom
    d_first_d_snr = np.gradient(first_deriv, snr_a)
    second_deriv  = d_first_d_snr / safe_denom

    # Candidate points: |dD_sem/dSNR| at or above the plateau percentile. Where D_sem
    # is flat the derivatives above divide by a near-zero gradient.
    grad_floor  = np.percentile(np.abs(d_dsem_d_snr), plateau_pct)
    candidate_mask = np.abs(d_dsem_d_snr) >= grad_floor
    if not candidate_mask.any():
        candidate_mask = np.ones_like(candidate_mask, dtype=bool)

    # Derivative curves returned for plotting, NaN outside the candidate points.
    first_deriv_plot = np.where(candidate_mask, first_deriv, np.nan)
    second_deriv_plot = np.where(candidate_mask, second_deriv, np.nan)

    candidate_idx = np.where(candidate_mask)[0]
    elbow_idx     = int(candidate_idx[np.argmax(np.abs(second_deriv[candidate_idx]))])
    elbow_snr     = snr_arr[elbow_idx]

    # Stage 2 on the same candidate points.
    abs_grad = np.abs(first_deriv)
    if grad_threshold_override is not None:
        threshold = float(grad_threshold_override); src = "config override"
    else:
        threshold = float(np.percentile(abs_grad[candidate_mask], grad_pct))
        src = f"empirical (p{grad_pct} over candidate points)"

    above = candidate_mask & (abs_grad > threshold)
    ver_idx  = [i for i in range(len(above)-1) if above[i] and above[i+1]]
    # snr_arr is ascending; the higher-SNR member of a qualifying pair is where the
    # elevated-gradient span begins when coming from good channel conditions.
    ver_snrs = [snr_arr[i+1] for i in ver_idx]
    # The elbow is confirmed if a verified SNR lies within one grid step of it.
    snr_step = float(np.median(np.diff(np.sort(np.unique(snr_arr)))))
    confirmed = elbow_snr in ver_snrs or any(abs(elbow_snr - s) <= snr_step for s in ver_snrs)

    return {"elbow_snr": elbow_snr, "elbow_idx": elbow_idx,
            "confirmed": bool(confirmed), "threshold": threshold, "threshold_source": src,
            "verified_collapse_snrs": ver_snrs,
            "first_derivative": first_deriv_plot.tolist(), "second_derivative": second_deriv_plot.tolist()}


def _modal_elbow(elbow_snrs):
    """Headline elbow: the most frequent elbow SNR across the percentile settings.

    Returns (headline or None, agreement fraction, tie_detected, tied values). A
    tie gives no headline; a setting without a candidate counts as the value None.
    """
    counts = {}
    for v in elbow_snrs:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    tied = sorted((v for v, c in counts.items() if c == max_count),
                  key=lambda x: (x is None, x))
    if len(tied) == 1:
        return tied[0], max_count / len(elbow_snrs), False, None
    return None, max_count / len(elbow_snrs), True, tied


def detect_collapse(snr_arr, auc_arr, dsem_arr, cfg):
    """Stages 1 and 2 at every sensitivity percentile, and the headline elbow.

    The headline elbow is the modal elbow across percentiles (no headline on a
    tie). It is confirmed when a Stage-2 verified SNR lies within one grid step of
    it at a strict majority of the percentiles. The derivative curves returned are
    those of the median percentile.
    """
    pcts = cfg["COLLAPSE_SENSITIVITY_PERCENTILES"]
    override = cfg["COLLAPSE_GRAD_THRESHOLD"]  # None: threshold from the data

    per_percentile = {}
    for p in pcts:
        r = _detect_collapse_at_percentile(snr_arr, auc_arr, dsem_arr,
                                            plateau_pct=p, grad_pct=p,
                                            grad_threshold_override=override)
        per_percentile[f"p{p}"] = r
        print(f"  [Collapse p{p}] elbow_snr={r['elbow_snr']} confirmed={r['confirmed']} "
              f"threshold={r['threshold']:.4f} ({r['threshold_source']})")

    elbow_snrs = [r["elbow_snr"] for r in per_percentile.values()]
    headline_elbow, agreement, tie_detected, tied_values = _modal_elbow(elbow_snrs)
    if not tie_detected:
        pass
    else:
        print(f"  [Collapse] TIE: {len(tied_values)} SNR values ({tied_values}) share the highest "
              f"frequency (agreement {agreement:.0%} of percentile settings) -- no single modal "
              f"elbow reported.")

    print(f"  [Collapse] headline elbow_snr={headline_elbow} "
          f"(agreement across {len(pcts)} tested percentiles: {agreement:.0%})"
          + (f" -- TIED: {tied_values}" if tie_detected else ""))
    if agreement < 1.0:
        print(f"  [Collapse] NOTE: the elbow differs between percentile settings ({elbow_snrs}).")

    median_pct_key = f"p{sorted(pcts)[len(pcts)//2]}"
    representative = per_percentile[median_pct_key]

    snr_step = float(np.median(np.diff(np.sort(np.unique(np.asarray(snr_arr, dtype=float))))))
    corroborated = []
    if headline_elbow is not None:
        corroborated = [k for k, r in per_percentile.items()
                        if any(abs(float(headline_elbow) - float(s)) <= snr_step
                               for s in r["verified_collapse_snrs"])]
    elbow_confirmed = headline_elbow is not None and len(corroborated) > len(pcts) / 2
    # SNRs verified by Stage 2 at a strict majority of the percentiles.
    ver_counts = {}
    for r in per_percentile.values():
        for s in set(r["verified_collapse_snrs"]):
            ver_counts[s] = ver_counts.get(s, 0) + 1
    verified_majority = sorted(s for s, c in ver_counts.items() if c > len(pcts) / 2)
    print(f"  [Collapse] headline corroborated by Stage 2 at {len(corroborated)}/{len(pcts)} "
          f"percentiles {corroborated} -> confirmed={elbow_confirmed}")

    return {
        "snr_db":            [float(s) for s in snr_arr],
        "first_derivative":  representative["first_derivative"],
        "second_derivative": representative["second_derivative"],
        "derivative_percentile_key": median_pct_key,
        "elbow_snr":         headline_elbow,
        "elbow_agreement_fraction": agreement,
        "elbow_tie_detected": tie_detected,
        "elbow_tied_values": tied_values if tie_detected else None,
        "elbow_confirmed":   bool(elbow_confirmed),
        "elbow_corroborated_percentiles": corroborated,
        "per_percentile_results": {k: {kk: vv for kk, vv in v.items()
                                        if kk not in ("first_derivative", "second_derivative")}
                                    for k, v in per_percentile.items()},
        "verified_collapse_snrs": verified_majority,
        "sensitivity_table": {k: v["threshold"] for k, v in per_percentile.items()},
    }


def bootstrap_elbow_stability(per_seed_auc, per_seed_dsem, snr_arr, cfg,
                               n_bootstrap=2000, seed=20260825):
    """Stage 3: is the headline elbow a stable feature of the data?

    Any smooth declining curve has a point of maximum curvature, so the elbow of
    the mean trajectory alone does not show a transition. The production seeds
    are resampled with replacement n_bootstrap times; on each resample the mean
    AUC and D_sem trajectories are recomputed and the full headline procedure
    (every sensitivity percentile, then the modal rule) is re-run.

    per_seed_auc, per_seed_dsem: arrays of shape (N_seeds, N_SNR).
    Returns the original headline elbow, the 95% percentile interval of the
    resampled elbows, the fraction of all resamples within one grid step of the
    original (a stability score, not a p-value), the fractions of tied and
    no-candidate resamples, and the resampled elbows.
    """
    per_seed_auc = np.asarray(per_seed_auc, dtype=float)
    per_seed_dsem = np.asarray(per_seed_dsem, dtype=float)
    snr_arr = np.asarray(snr_arr, dtype=float)
    n_seeds = per_seed_auc.shape[0]
    if n_seeds < 2:
        raise ValueError(f"Bootstrap stability requires at least 2 seeds to resample from; got {n_seeds}.")
    if per_seed_auc.shape != per_seed_dsem.shape:
        raise ValueError(f"per_seed_auc shape {per_seed_auc.shape} != per_seed_dsem shape {per_seed_dsem.shape}")
    if per_seed_auc.shape[1] != len(snr_arr):
        raise ValueError(f"Trajectory length {per_seed_auc.shape[1]} != len(snr_arr) {len(snr_arr)}")

    pcts = cfg["COLLAPSE_SENSITIVITY_PERCENTILES"]

    def _headline_for(auc_mean, dsem_mean):
        elbows = [_detect_collapse_at_percentile(snr_arr, auc_mean, dsem_mean,
                                                  plateau_pct=pp, grad_pct=pp)["elbow_snr"]
                  for pp in pcts]
        headline, _, tie, _ = _modal_elbow(elbows)
        return headline, tie

    rng = np.random.default_rng(seed)
    resample_headlines, n_tied, n_none = [], 0, 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_seeds, size=n_seeds)  # seeds resampled with replacement
        h, tie = _headline_for(per_seed_auc[idx].mean(axis=0),
                               per_seed_dsem[idx].mean(axis=0))
        if tie:
            n_tied += 1
        elif h is None:
            n_none += 1
        else:
            resample_headlines.append(h)

    original_elbow, original_tie = _headline_for(per_seed_auc.mean(axis=0),
                                                  per_seed_dsem.mean(axis=0))

    located = np.array(resample_headlines, dtype=float)
    snr_step = float(np.median(np.diff(np.sort(np.unique(snr_arr)))))
    if original_elbow is not None and located.size:
        within_one_step = float(np.sum(np.abs(located - original_elbow) <= snr_step)) / n_bootstrap
    else:
        within_one_step = float("nan")  # no single headline to be stable around
    if located.size:
        ci_low, ci_high = (float(v) for v in np.percentile(located, [2.5, 97.5]))
    else:
        ci_low = ci_high = float("nan")

    print(f"  [Bootstrap stability, full headline procedure x{len(pcts)} percentiles, "
          f"N={n_bootstrap}] original headline={original_elbow}dB"
          + (" (TIED -- no single headline)" if original_tie else "")
          + f", located-resample 95% interval=[{ci_low:.1f}, {ci_high:.1f}]dB, "
          f"stability (within 1 grid step, over ALL resamples)={within_one_step:.0%}, "
          f"tied resamples={n_tied/n_bootstrap:.1%}, no-candidate resamples={n_none/n_bootstrap:.1%}")

    return {
        "original_elbow_snr": original_elbow,
        "original_headline_tied": original_tie,
        "percentiles_used": list(pcts),
        "n_bootstrap": n_bootstrap,
        "n_seeds": n_seeds,
        "bootstrap_ci_95": (ci_low, ci_high),
        "stability_fraction_within_one_grid_step": within_one_step,
        "tie_fraction": n_tied / n_bootstrap,
        "no_candidate_fraction": n_none / n_bootstrap,
        "bootstrap_elbow_std_db": float(np.std(located)) if located.size else float("nan"),
        "bootstrap_elbow_snrs": located.tolist(),
    }


def _fit_single_line_rss(x, y):
    """Least-squares straight line. Returns (RSS, slope, intercept)."""
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ coef
    rss = float(np.sum((y - fitted) ** 2))
    return rss, float(coef[0]), float(coef[1])


def _fit_two_line_rss(x, y, breakpoint_idx):
    """Total RSS of two independent least-squares lines split after breakpoint_idx."""
    x1, y1 = x[:breakpoint_idx + 1], y[:breakpoint_idx + 1]
    x2, y2 = x[breakpoint_idx + 1:], y[breakpoint_idx + 1:]
    rss1, _, _ = _fit_single_line_rss(x1, y1)
    rss2, _, _ = _fit_single_line_rss(x2, y2)
    return rss1 + rss2


def _best_breakpoint_rss(x, y, min_points_per_side=3):
    """Best two-segment (broken-stick) fit over all interior split points with at
    least min_points_per_side points per segment. Returns (RSS, split index)."""
    n = len(x)
    best_rss, best_idx = np.inf, None
    for idx in range(min_points_per_side - 1, n - min_points_per_side):
        rss = _fit_two_line_rss(x, y, idx)
        if rss < best_rss:
            best_rss, best_idx = rss, idx
    return best_rss, best_idx


def _s_curve_design(x, x0, k):
    """Design matrix [1, logistic(k (x - x0)), x] of the S-curve for fixed (x0, k)."""
    z = np.clip(k * (x - x0), -60.0, 60.0)
    return np.column_stack([np.ones_like(x), 1.0 / (1.0 + np.exp(-z)), x])


def _s_curve_rss_at(x, y, x0, k):
    A = _s_curve_design(x, x0, k)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.sum((y - A @ coef) ** 2)), coef


def _fit_s_curve_rss(x, y, k_max, k_min=1e-3, n_x0=81, n_k=40):
    """Least-squares fit of the smooth S-curve model

        y = b + a / (1 + exp(-k (x - x0))) + c * x          (5 parameters)

    a logistic transition (centre x0, steepness k) plus a linear trend. For fixed
    (x0, k) the model is linear in (b, a, c), which are solved by least squares;
    (x0, k) are found by a grid search (x0 over the sweep extended by half its
    width on each side, k log-spaced in [k_min, k_max]) and a bounded local
    refinement. The fit is deterministic. k is capped at k_max so that the smooth
    model cannot become a step.

    Returns (rss, params) with params = {b, a, c, x0, k, k_at_bound}.
    """
    from scipy.optimize import minimize
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(x.max() - x.min())
    x0_lo, x0_hi = float(x.min() - span / 2), float(x.max() + span / 2)
    # Grid search: the 3-parameter least-squares problem of every (x0, k) is solved
    # through its normal equations in one batched call.
    X0, K = np.meshgrid(np.linspace(x0_lo, x0_hi, n_x0), np.geomspace(k_min, k_max, n_k), indexing="ij")
    X0, K = X0.ravel(), K.ravel()
    S = 1.0 / (1.0 + np.exp(-np.clip(K[:, None] * (x[None, :] - X0[:, None]), -60.0, 60.0)))
    A = np.stack([np.ones_like(S), S, np.broadcast_to(x, S.shape)], axis=2)      # (G, n, 3)
    M = np.einsum("gni,gnj->gij", A, A) + 1e-12 * np.eye(3)
    coef = np.linalg.solve(M, np.einsum("gni,n->gi", A, y)[..., None])[..., 0]
    rss_grid = np.sum((y[None, :] - np.einsum("gni,gi->gn", A, coef)) ** 2, axis=1)
    g = int(np.argmin(rss_grid))
    best = (float(rss_grid[g]), float(X0[g]), float(K[g]))
    _, x0_b, k_b = best

    def _obj(p):
        return _s_curve_rss_at(x, y, p[0], np.exp(p[1]))[0]

    res = minimize(_obj, x0=[x0_b, np.log(k_b)], method="L-BFGS-B",
                   bounds=[(x0_lo, x0_hi), (np.log(k_min), np.log(k_max))])
    if res.fun <= best[0]:
        x0_f, k_f = float(res.x[0]), float(np.exp(res.x[1]))
    else:
        x0_f, k_f = float(x0_b), float(k_b)
    rss, coef = _s_curve_rss_at(x, y, x0_f, k_f)
    return rss, {"b": float(coef[0]), "a": float(coef[1]), "c": float(coef[2]),
                 "x0": x0_f, "k": k_f, "k_at_bound": bool(k_f >= k_max * (1 - 1e-6))}


def _delta_aic_break_minus_smooth(x, y, k_max, min_points_per_side):
    """AIC(broken stick) - AIC(S-curve) for one trajectory.

    Both models have five parameters (S-curve: b, a, c, x0, k; broken stick: four
    line coefficients and the breakpoint), so the difference is
    n ln(RSS_break / RSS_smooth), also under AICc. Positive favours the S-curve.
    """
    n = len(x)
    rss_s, params = _fit_s_curve_rss(x, y, k_max)
    rss_b, idx = _best_breakpoint_rss(x, y, min_points_per_side)
    if max(rss_b, rss_s) < 1e-20:          # both fit exactly (e.g. a constant trajectory)
        return 0.0, rss_s, params, rss_b, idx
    floor = 1e-300
    return (n * np.log(max(rss_b, floor) / max(rss_s, floor)), rss_s, params, rss_b, idx)


def structural_break_test(snr_arr, auc_mean, cfg, n_permutations=5000, seed=20260825,
                          min_points_per_side=3, per_seed_auc=None):
    """Stage 4: what kind of curve is the AUC-vs-SNR trajectory?

    Part 1, non-linearity (p_value): the best broken stick against a single
    straight line, with a permutation p-value; the straight-line residuals are
    permuted and the full breakpoint search is repeated on each permutation,
    which accounts for searching over breakpoints. A small p_value says the
    trajectory is not a straight line, not that a discrete break exists. The
    residuals are assumed exchangeable across SNR.

    Part 2, smooth transition or discrete break (smooth_vs_break): the S-curve
    of _fit_s_curve_rss, with k capped so that its 10-90% width 2 ln(9) / k is at
    least two SNR grid steps, against the broken stick (both 5 parameters).
    delta_aic = AIC_break - AIC_smooth (Akaike, 1974); positive favours the
    S-curve. With per_seed_auc (N_seeds x N_SNR), the seeds are resampled with
    replacement cfg["BREAK_TEST_N_BOOTSTRAP"] times and both models refitted to
    each resampled mean trajectory, giving a 95% interval for delta_aic.
    Verdict, with t = cfg["BREAK_TEST_DELTA_AIC"] (|delta AIC| > 2; Burnham and
    Anderson, 2002):
      "smooth"        delta_aic > t  and (no bootstrap or 2.5th percentile > 0)
      "break"         delta_aic < -t and (no bootstrap or 97.5th percentile < 0)
      "inconclusive"  otherwise.
    """
    snr_arr = np.asarray(snr_arr, dtype=float)
    auc_mean = np.asarray(auc_mean, dtype=float)
    n = len(snr_arr)
    if n < 2 * min_points_per_side + 1:
        raise ValueError(f"Need at least {2*min_points_per_side+1} points for a two-segment fit "
                          f"with min_points_per_side={min_points_per_side}; got {n}.")

    # Part 1: broken stick vs single line
    null_rss, slope, intercept = _fit_single_line_rss(snr_arr, auc_mean)
    best_two_rss, best_idx = _best_breakpoint_rss(snr_arr, auc_mean, min_points_per_side)
    observed_reduction = null_rss - best_two_rss
    observed_breakpoint_snr = float(snr_arr[best_idx])

    fitted_null = slope * snr_arr + intercept
    residuals = auc_mean - fitted_null
    rng = np.random.default_rng(seed)
    permuted_reductions = np.empty(n_permutations)
    for b in range(n_permutations):
        perm_y = fitted_null + rng.permutation(residuals)
        perm_null_rss, _, _ = _fit_single_line_rss(snr_arr, perm_y)
        perm_best_rss, _ = _best_breakpoint_rss(snr_arr, perm_y, min_points_per_side)
        permuted_reductions[b] = perm_null_rss - perm_best_rss
    p_value = (1 + np.sum(permuted_reductions >= observed_reduction)) / (1 + n_permutations)
    print(f"  [Structural break test, N_perm={n_permutations}] best breakpoint={observed_breakpoint_snr}dB, "
          f"RSS reduction={observed_reduction:.5f}, p={p_value:.4f} "
          f"(vs a straight line: tests non-linearity only, not a discrete break)")

    # Part 2: smooth S-curve vs discrete break
    snr_step = float(np.median(np.diff(np.sort(np.unique(snr_arr)))))
    k_max = 2.0 * np.log(9.0) / (2.0 * snr_step)   # 10-90% width >= two grid steps
    thr = float(cfg.get("BREAK_TEST_DELTA_AIC", 2.0))
    d_aic, rss_s, s_params, rss_b, b_idx = _delta_aic_break_minus_smooth(
        snr_arr, auc_mean, k_max, min_points_per_side)

    boot = None
    if per_seed_auc is not None:
        per_seed_auc = np.asarray(per_seed_auc, dtype=float)
        if per_seed_auc.ndim != 2 or per_seed_auc.shape[1] != n:
            raise ValueError(f"per_seed_auc shape {per_seed_auc.shape} does not match {n} SNR points")
        if not np.allclose(np.nanmean(per_seed_auc, axis=0), auc_mean, atol=1e-9):
            raise ValueError("per_seed_auc's mean trajectory differs from auc_mean")
        n_boot = int(cfg.get("BREAK_TEST_N_BOOTSTRAP", 1000))
        n_seeds = per_seed_auc.shape[0]
        rng_b = np.random.default_rng(seed + 1)
        d_boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng_b.integers(0, n_seeds, size=n_seeds)
            d_boot[i] = _delta_aic_break_minus_smooth(
                snr_arr, np.nanmean(per_seed_auc[idx], axis=0), k_max, min_points_per_side)[0]
        lo, hi = (float(v) for v in np.percentile(d_boot, [2.5, 97.5]))
        boot = {"n_bootstrap": n_boot, "n_seeds": int(n_seeds), "delta_aic_ci95": (lo, hi),
                "fraction_favouring_smooth": float(np.mean(d_boot > 0)), "seed": seed + 1}

    if d_aic > thr and (boot is None or boot["delta_aic_ci95"][0] > 0):
        verdict = "smooth"
    elif d_aic < -thr and (boot is None or boot["delta_aic_ci95"][1] < 0):
        verdict = "break"
    else:
        verdict = "inconclusive"
    width = 2.0 * np.log(9.0) / s_params["k"]
    print(f"  [Smooth S-curve vs discrete break] RSS S-curve={rss_s:.6f} "
          f"(centre {s_params['x0']:+.2f} dB, 10-90% width {width:.1f} dB"
          + (", k AT ITS CAP" if s_params["k_at_bound"] else "") + f"), "
          f"RSS broken stick={rss_b:.6f} (break at {float(snr_arr[b_idx])} dB), "
          f"delta AIC (break - smooth)={d_aic:+.1f}"
          + (f", seed-bootstrap 95% interval [{boot['delta_aic_ci95'][0]:+.1f}, "
             f"{boot['delta_aic_ci95'][1]:+.1f}], smooth favoured in "
             f"{boot['fraction_favouring_smooth']:.0%} of {boot['n_bootstrap']} resamples" if boot else "")
          + f" -> verdict: {verdict}")

    return {
        "observed_breakpoint_snr": observed_breakpoint_snr,
        "observed_rss_reduction": float(observed_reduction),
        "null_single_line_rss": float(null_rss),
        "best_two_segment_rss": float(best_two_rss),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "smooth_vs_break": {
            "smooth_model": "b + a/(1+exp(-k(x-x0))) + c*x (5 parameters)",
            "break_model": "two independent OLS lines + breakpoint (5 parameters)",
            "rss_smooth": float(rss_s), "rss_break": float(rss_b),
            "break_snr": float(snr_arr[b_idx]),
            "smooth_params": s_params,
            "smooth_width_10_90_db": float(width),
            "k_cap_per_db": float(k_max),
            "delta_aic_break_minus_smooth": float(d_aic),
            "threshold": thr,
            "bootstrap": boot,
            "verdict": verdict,
        },
    }
