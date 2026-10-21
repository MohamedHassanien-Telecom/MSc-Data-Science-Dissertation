"""bootstrap_elbow_stability (src/collapse.py), Stage 3 of the semantic-collapse
analysis: is the candidate elbow a reproducible feature of the data or an
artefact of averaging noisy per-seed trajectories?
  1. 15 noisy seeds sharing one sharp AUROC corner, with the D_sem transition at
     the same SNR, give high stability and a narrow bootstrap interval;
  2. 15 seeds of pure noise give low stability;
  3. fewer than 2 seeds, or mismatched shapes, are refused;
  4. the same seed gives the same bootstrap result.

Usage:  python3 tests/test_bootstrap_elbow_stability.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.collapse import bootstrap_elbow_stability


def _default_cfg():
    return {"COLLAPSE_SENSITIVITY_PERCENTILES": [50, 60, 70, 75, 80, 90]}


def test_genuine_consistent_elbow_shows_high_stability():
    rng = np.random.default_rng(42)
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    n_seeds = 15
    n_snr = len(snr_arr)

    per_seed_auc = []
    per_seed_dsem = []
    for _ in range(n_seeds):
        # One corner: AUROC flat above -8 dB and declining linearly below, plus small noise.
        auc = np.where(snr_arr >= -8, 0.80, 0.80 + 0.030 * (snr_arr + 8))
        auc = auc + rng.normal(0, 0.004, size=n_snr)
        # D_sem rises around the same -8 dB, so the distortion accelerates where
        # detection degrades.
        dsem = 0.02 + 0.20 / (1 + np.exp((snr_arr - (-8)) / 3.0)) + rng.normal(0, 0.0008, size=n_snr)
        dsem = np.clip(dsem, 1e-6, None)
        per_seed_auc.append(auc)
        per_seed_dsem.append(dsem)

    result = bootstrap_elbow_stability(per_seed_auc, per_seed_dsem, snr_arr, _default_cfg(),
                                        n_bootstrap=500, seed=1)
    print(f"test_genuine_consistent_elbow: original={result['original_elbow_snr']}dB, "
          f"CI={result['bootstrap_ci_95']}, stability={result['stability_fraction_within_one_grid_step']:.0%}")

    assert result["stability_fraction_within_one_grid_step"] >= 0.7, (
        f"Expected high stability (>=70%) for a genuinely shared, consistent elbow, "
        f"got {result['stability_fraction_within_one_grid_step']:.0%}"
    )
    ci_width = result["bootstrap_ci_95"][1] - result["bootstrap_ci_95"][0]
    assert ci_width <= 8.0, f"Expected a narrow bootstrap CI for a genuine shared feature, got width={ci_width}dB"
    print("test_genuine_consistent_elbow_shows_high_stability: PASS")


def test_pure_noise_shows_low_stability():
    rng = np.random.default_rng(7)
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    n_seeds = 15
    n_snr = len(snr_arr)

    # No shared structure: AUROC is independent noise, unrelated to D_sem.
    per_seed_auc = [rng.normal(0.7, 0.05, size=n_snr) for _ in range(n_seeds)]
    per_seed_dsem = [np.clip(0.02 + 0.15*np.exp(-(snr_arr+20)/10.0) + rng.normal(0, 0.02, size=n_snr), 1e-6, None)
                      for _ in range(n_seeds)]

    result = bootstrap_elbow_stability(per_seed_auc, per_seed_dsem, snr_arr, _default_cfg(),
                                        n_bootstrap=500, seed=2)
    print(f"test_pure_noise: original={result['original_elbow_snr']}dB, "
          f"CI={result['bootstrap_ci_95']}, stability={result['stability_fraction_within_one_grid_step']:.0%}")

    # A tied original headline gives NaN stability, which also counts as not high.
    assert not (result["stability_fraction_within_one_grid_step"] > 0.5), (
        f"Expected low stability (<=50%) for pure noise with no shared structure, "
        f"got {result['stability_fraction_within_one_grid_step']:.0%}"
    )
    print("test_pure_noise_shows_low_stability: PASS")


def test_rejects_single_seed():
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    try:
        bootstrap_elbow_stability([[0.5]*len(snr_arr)], [[0.01]*len(snr_arr)], snr_arr, _default_cfg())
        assert False, "Expected ValueError for n_seeds < 2"
    except ValueError as e:
        assert "at least 2 seeds" in str(e)
        print(f"test_rejects_single_seed: PASS ({e})")


def test_rejects_mismatched_shapes():
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    try:
        bootstrap_elbow_stability([[0.5]*len(snr_arr)]*3, [[0.01]*3]*3, snr_arr, _default_cfg())
        assert False, "Expected ValueError for shape mismatch"
    except ValueError as e:
        print(f"test_rejects_mismatched_shapes: PASS ({e})")


def test_reproducible_with_fixed_seed():
    rng = np.random.default_rng(99)
    snr_arr = np.arange(-20, 22, 2, dtype=float)
    n_snr = len(snr_arr)
    per_seed_auc = [0.7 - 0.1/(1+np.exp(-(snr_arr+5)/2)) + rng.normal(0,0.01,n_snr) for _ in range(10)]
    per_seed_dsem = [np.clip(0.01 + 0.001*(20-snr_arr) + rng.normal(0,0.0005,n_snr), 1e-6, None) for _ in range(10)]

    r1 = bootstrap_elbow_stability(per_seed_auc, per_seed_dsem, snr_arr, _default_cfg(), n_bootstrap=200, seed=555)
    r2 = bootstrap_elbow_stability(per_seed_auc, per_seed_dsem, snr_arr, _default_cfg(), n_bootstrap=200, seed=555)
    assert r1["bootstrap_elbow_snrs"] == r2["bootstrap_elbow_snrs"], "Same seed must give identical bootstrap results"
    print("test_reproducible_with_fixed_seed: PASS")


if __name__ == "__main__":
    test_genuine_consistent_elbow_shows_high_stability()
    test_pure_noise_shows_low_stability()
    test_rejects_single_seed()
    test_rejects_mismatched_shapes()
    test_reproducible_with_fixed_seed()
    print("\nAll bootstrap_elbow_stability checks passed.")
