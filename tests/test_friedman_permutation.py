"""friedman_permutation_test (src/effect_sizes.py), the Friedman test used by the
hyperparameter search and the compression-ratio study:
  1. its statistic equals scipy.stats.friedmanchisquare at the search's shapes;
  2. the same seed gives the same p-value;
  3. under the null, p-values are not concentrated near 0;
  4. a strong candidate effect gives a small p-value;
  5. the p-value lies in (0, 1];
  6. degenerate input (fewer than 2 candidates or blocks) returns (0.0, 1.0).

Usage:  python3 tests/test_friedman_permutation.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from scipy.stats import friedmanchisquare
from src.effect_sizes import friedman_permutation_test


def test_h_statistic_matches_scipy_exactly():
    rng = np.random.default_rng(7)
    # scipy's friedmanchisquare needs at least 3 candidates; the search uses
    # Wilcoxon for two-candidate steps.
    shapes = [(5, 3), (5, 7), (6, 3), (5, 4)]
    for n, k in shapes:
        for _ in range(5):
            matrix = rng.normal(size=(n, k))
            scipy_h = friedmanchisquare(*[matrix[:, j] for j in range(k)]).statistic
            my_h, _ = friedman_permutation_test(matrix.tolist(), n_permutations=500)
            assert np.isclose(scipy_h, my_h, atol=1e-9), \
                f"H-statistic mismatch at shape ({n},{k}): scipy={scipy_h}, mine={my_h}"
    print("test_h_statistic_matches_scipy_exactly: PASS")


def test_deterministic_same_seed():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(5, 7)).tolist()
    h1, p1 = friedman_permutation_test(matrix, n_permutations=2000, seed=20260824)
    h2, p2 = friedman_permutation_test(matrix, n_permutations=2000, seed=20260824)
    assert h1 == h2, "h_stat should be bit-identical (deterministic, no RNG involved)"
    assert p1 == p2, "p-value should be bit-identical across repeated calls with same seed"
    print(f"test_deterministic_same_seed: PASS (h={h1:.4f}, p={p1:.5f})")


def test_null_data_not_spuriously_significant():
    """Under a true null (candidates drawn i.i.d.), about 5% of p-values fall below 0.05."""
    rng = np.random.default_rng(11)
    n, k = 5, 7
    p_values = []
    for _ in range(200):
        matrix = rng.normal(size=(n, k))
        _, p = friedman_permutation_test(matrix.tolist(), n_permutations=2000, seed=int(rng.integers(1e9)))
        p_values.append(p)
    frac_below_05 = np.mean(np.array(p_values) < 0.05)
    # Generous bound: 200 draws with 2,000 permutations each.
    assert frac_below_05 < 0.15, \
        f"Too many spurious 'significant' results under null data: {frac_below_05:.3f} below p=0.05"
    print(f"test_null_data_not_spuriously_significant: PASS ({frac_below_05:.3f} below p=0.05, expected ~0.05)")


def test_strong_effect_detected():
    """One candidate far above the others in every block gives p < 0.01."""
    rng = np.random.default_rng(5)
    n, k = 6, 5
    matrix = rng.normal(loc=0, scale=0.1, size=(n, k))
    matrix[:, 2] += 5.0
    _, p = friedman_permutation_test(matrix.tolist(), n_permutations=20000)
    assert p < 0.01, f"Expected a strong, obvious effect to produce p<0.01, got p={p:.5f}"
    print(f"test_strong_effect_detected: PASS (p={p:.5f})")


def test_pvalue_bounds():
    rng = np.random.default_rng(13)
    for _ in range(10):
        n, k = int(rng.integers(5, 7)), int(rng.integers(2, 8))
        matrix = rng.normal(size=(n, k)).tolist()
        _, p = friedman_permutation_test(matrix, n_permutations=1000)
        assert 0.0 < p <= 1.0, f"p-value {p} outside valid (0, 1] range"
    print("test_pvalue_bounds: PASS")


def test_degenerate_input():
    h, p = friedman_permutation_test([[1.0]], n_permutations=100)
    assert (h, p) == (0.0, 1.0), f"k=1 should return (0.0, 1.0), got ({h}, {p})"
    h, p = friedman_permutation_test([[1.0, 2.0]], n_permutations=100)
    assert (h, p) == (0.0, 1.0), f"n=1 should return (0.0, 1.0), got ({h}, {p})"
    print("test_degenerate_input: PASS")


if __name__ == "__main__":
    test_h_statistic_matches_scipy_exactly()
    test_deterministic_same_seed()
    test_null_data_not_spuriously_significant()
    test_strong_effect_detected()
    test_pvalue_bounds()
    test_degenerate_input()
    print("\nAll friedman_permutation_test checks passed.")
