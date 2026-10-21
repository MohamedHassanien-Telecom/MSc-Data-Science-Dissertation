"""structural_break_test (src/collapse.py), Stage 4 of the semantic-collapse
analysis.

Part 1, broken stick against a straight line (residual-permutation p-value):
  1. a genuine break (two slopes meeting at a known SNR) gives p < 0.05;
  2. a straight line plus noise gives p >= 0.05;
  3. across many noise draws the false-positive rate stays near the nominal 5%,
     despite the search over breakpoints;
  4. the same seed gives the same p-value.

Part 2, smooth S-curve against a discrete break (delta AIC with a seed
bootstrap): smooth S-curves of three shapes are classified "smooth"; a sharp
hockey stick and a gentle two-slope kink are classified "break"; a straight line
is not classified "smooth". Without per-seed data the mean-curve rule applies,
and a per-seed array that does not match the mean curve is refused.

Usage:  python3 tests/test_structural_break_test.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.collapse import structural_break_test


def _default_cfg():
    return {}


def test_genuine_break_is_significant():
    rng = np.random.default_rng(1)
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    n = len(snr_arr)
    breakpoint_snr = -8.0
    auc = np.where(
        snr_arr <= breakpoint_snr,
        0.55 + 0.005 * (snr_arr + 20),
        0.55 + 0.005 * (breakpoint_snr + 20) + 0.025 * (snr_arr - breakpoint_snr)
    )
    auc += rng.normal(0, 0.004, size=n)

    result = structural_break_test(snr_arr, auc, _default_cfg(), n_permutations=2000, seed=10)
    print(f"test_genuine_break: breakpoint={result['observed_breakpoint_snr']}dB, p={result['p_value']:.4f}")
    assert result["p_value"] < 0.05, f"Expected a significant p-value for a genuine break, got p={result['p_value']:.4f}"
    print("test_genuine_break_is_significant: PASS")


def test_pure_noise_is_not_significant():
    rng = np.random.default_rng(2)
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    n = len(snr_arr)
    auc = 0.5 + 0.01 * (snr_arr + 20) + rng.normal(0, 0.015, size=n)

    result = structural_break_test(snr_arr, auc, _default_cfg(), n_permutations=2000, seed=11)
    print(f"test_pure_noise: breakpoint={result['observed_breakpoint_snr']}dB, p={result['p_value']:.4f}")
    assert result["p_value"] >= 0.05, (
        f"Expected a non-significant p-value for a single true line with no break, got p={result['p_value']:.4f} "
        "-- if this fails, the permutation correction is not properly controlling false positives from "
        "searching over many candidate breakpoints."
    )
    print("test_pure_noise_is_not_significant: PASS")


def test_null_calibration_false_positive_rate():
    """Repeats the straight-line case over 40 noise draws: the false-positive rate stays near 5%."""
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    n = len(snr_arr)
    n_trials = 40
    false_positives = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        auc = 0.5 + 0.01 * (snr_arr + 20) + rng.normal(0, 0.015, size=n)
        result = structural_break_test(snr_arr, auc, _default_cfg(), n_permutations=500, seed=2000 + trial)
        if result["p_value"] < 0.05:
            false_positives += 1
    fp_rate = false_positives / n_trials
    print(f"test_null_calibration: {false_positives}/{n_trials} false positives ({fp_rate:.1%})")
    # Generous bound for 40 trials; an uncorrected best-of-many-breakpoints search
    # would flag nearly every trial.
    assert fp_rate <= 0.20, (
        f"False-positive rate {fp_rate:.1%} is too high for a nominal 5% test -- "
        "the permutation correction is not adequately controlling for the "
        "multiple-breakpoint search."
    )
    print("test_null_calibration_false_positive_rate: PASS")


def test_reproducible_with_fixed_seed():
    rng = np.random.default_rng(3)
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    auc = 0.5 + 0.01*(snr_arr+20) + rng.normal(0, 0.01, size=len(snr_arr))
    r1 = structural_break_test(snr_arr, auc, _default_cfg(), n_permutations=300, seed=777)
    r2 = structural_break_test(snr_arr, auc, _default_cfg(), n_permutations=300, seed=777)
    assert r1["p_value"] == r2["p_value"]
    print("test_reproducible_with_fixed_seed: PASS")


def _verdicts(shape, n_trials, seed_noise_sd=0.008, n_seeds=15, n_boot=200):
    """Count the smooth_vs_break verdicts over n_trials data sets of n_seeds noisy copies of shape."""
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    cfg = {"BREAK_TEST_DELTA_AIC": 2.0, "BREAK_TEST_N_BOOTSTRAP": n_boot}
    counts = {"smooth": 0, "break": 0, "inconclusive": 0}
    for t in range(n_trials):
        rng = np.random.default_rng(10000 + t)
        per_seed = shape(snr_arr)[None, :] + rng.normal(0, seed_noise_sd, (n_seeds, len(snr_arr)))
        r = structural_break_test(snr_arr, per_seed.mean(axis=0), cfg, n_permutations=100,
                                  seed=t, per_seed_auc=per_seed)
        counts[r["smooth_vs_break"]["verdict"]] += 1
    return counts


def test_smooth_s_curves_are_classified_smooth():
    """Smooth S-curves are classified smooth: off-centre (Max Logit-like), centred
    and wide, and with a mild high-SNR decline (Prototype-like).
    """
    shapes = {
        "off-centre logistic (Max Logit-like)": lambda x: 0.5 + 0.31 / (1 + np.exp(-0.26 * (x + 11.5))),
        "centred wide logistic": lambda x: 0.55 + 0.35 / (1 + np.exp(-(x + 2) / 4.5)),
        "logistic + high-SNR decline (Prototype-like)":
            lambda x: 0.46 + 0.33 / (1 + np.exp(-0.2 * (x + 15.4))) - 0.0014 * x,
    }
    n_trials = 10
    for name, shape in shapes.items():
        c = _verdicts(shape, n_trials)
        print(f"  {name}: {c}")
        assert c["break"] == 0, f"{name}: a smooth S-curve was classified as a discrete break: {c}"
        assert c["smooth"] >= 0.9 * n_trials, f"{name}: smooth S-curve not recognised as smooth: {c}"
    print("test_smooth_s_curves_are_classified_smooth: PASS")


def test_genuine_breaks_are_classified_break():
    """Genuine breaks are classified break: a sharp hockey stick (flat above -6 dB,
    slope 0.045/dB below) and a gentle two-slope kink (0.005 and 0.025/dB at -8 dB).
    """
    shapes = {
        "sharp hockey stick": lambda x: np.where(x >= -6, 0.85, 0.85 + 0.045 * (x + 6)),
        "gentle two-slope kink": lambda x: np.where(x <= -8, 0.55 + 0.005 * (x + 20),
                                                    0.55 + 0.005 * 12 + 0.025 * (x + 8)),
    }
    n_trials = 10
    for name, shape in shapes.items():
        c = _verdicts(shape, n_trials)
        print(f"  {name}: {c}")
        assert c["smooth"] == 0, f"{name}: a genuine break was classified as smooth: {c}"
        assert c["break"] >= 0.9 * n_trials, f"{name}: genuine break not detected: {c}"
    print("test_genuine_breaks_are_classified_break: PASS")


def test_straight_line_is_not_called_smooth_s_curve():
    """A straight line has neither an S-shape nor a break: the verdict must not be smooth."""
    c = _verdicts(lambda x: 0.5 + 0.01 * (x + 20), 10)
    print(f"  straight line: {c}")
    assert c["smooth"] == 0, f"straight line classified as a smooth S-curve: {c}"
    assert c["inconclusive"] >= 8, f"straight line should mostly be inconclusive: {c}"
    print("test_straight_line_is_not_called_smooth_s_curve: PASS")


def test_without_per_seed_data_mean_only_rule_applies():
    """Without per-seed trajectories there is no bootstrap; the verdict uses the mean-curve delta AIC."""
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    auc = 0.5 + 0.31 / (1 + np.exp(-0.26 * (snr_arr + 11.5)))
    r = structural_break_test(snr_arr, auc, {"BREAK_TEST_DELTA_AIC": 2.0}, n_permutations=100, seed=1)
    svb = r["smooth_vs_break"]
    assert svb["bootstrap"] is None and svb["verdict"] == "smooth", svb
    print("test_without_per_seed_data_mean_only_rule_applies: PASS")


def test_per_seed_mismatch_is_refused():
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    per_seed = np.tile(0.5 + 0.01 * (snr_arr + 20), (5, 1))
    try:
        structural_break_test(snr_arr, per_seed.mean(axis=0) + 0.01, {}, n_permutations=10,
                              per_seed_auc=per_seed)
        refused = False
    except ValueError:
        refused = True
    assert refused, "a per-seed array whose mean differs from auc_mean must be refused"
    print("test_per_seed_mismatch_is_refused: PASS")


if __name__ == "__main__":
    test_genuine_break_is_significant()
    test_pure_noise_is_not_significant()
    test_null_calibration_false_positive_rate()
    test_reproducible_with_fixed_seed()
    test_smooth_s_curves_are_classified_smooth()
    test_genuine_breaks_are_classified_break()
    test_straight_line_is_not_called_smooth_s_curve()
    test_without_per_seed_data_mean_only_rule_applies()
    test_per_seed_mismatch_is_refused()
    print("\nAll structural_break_test checks passed.")
