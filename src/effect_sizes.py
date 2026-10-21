"""Paired significance tests and effect sizes for the hyperparameter search and
the compression study (dissertation Appendix B.3).

Every candidate of a search step is trained with the same seeds, so comparisons
are paired by seed: a Wilcoxon signed-rank test with the matched-pairs
rank-biserial r for two candidates, and a Friedman permutation test with
Kendall's W for three or more. At the small numbers of seeds used, Kendall's W
is biased upwards; interpret_kendalls_w therefore compares an observed W with a
simulated null value for the same (n, k).
"""
import numpy as np


def rank_biserial_wilcoxon(paired_diffs):
    """Matched-pairs rank-biserial r of paired differences, in [-1, 1]:
    (sum of ranks of positive differences - sum of ranks of negative ones) /
    sum of all ranks. Zero differences are dropped and tied |differences| get
    average ranks, as in scipy.stats.wilcoxon."""
    from scipy.stats import rankdata
    diffs = np.asarray(paired_diffs, dtype=float)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = rankdata(np.abs(diffs))
    pos_rank_sum = ranks[diffs > 0].sum()
    neg_rank_sum = ranks[diffs < 0].sum()
    total = pos_rank_sum + neg_rank_sum
    return (pos_rank_sum - neg_rank_sum) / total if total > 0 else 0.0


def kendalls_w(matrix):
    """Kendall's coefficient of concordance W in [0, 1] for a seeds x candidates
    matrix: 0 = no consistent ranking of the candidates across seeds, 1 = every
    seed ranks them in the same order."""
    n, k = len(matrix), len(matrix[0])
    if k < 2 or n < 2:
        return 0.0
    from scipy.stats import rankdata
    rank_matrix = [rankdata(row) for row in matrix]
    col_sums = [sum(rank_matrix[i][j] for i in range(n)) for j in range(k)]
    mean_rank_sum = sum(col_sums) / k
    ss = sum((s - mean_rank_sum) ** 2 for s in col_sums)
    return (12 * ss) / (n ** 2 * (k ** 3 - k)) if k > 1 else 0.0


def friedman_permutation_test(matrix, n_permutations=20000, seed=20260824):
    """Friedman test with a permutation p-value, for a seeds x candidates matrix.

    Under the null hypothesis the candidates' ranks are exchangeable within each
    seed, so each row's ranks are shuffled independently n_permutations times and
    the Friedman statistic recomputed. p = (number of permuted statistics >= the
    observed one + 1) / (n_permutations + 1). Returns (statistic, p); the
    statistic equals scipy.stats.friedmanchisquare's. The seed is fixed, so the
    p-value is reproducible.
    """
    from scipy.stats import rankdata
    n, k = len(matrix), len(matrix[0])
    if k < 2 or n < 2:
        return 0.0, 1.0
    arr = np.array(matrix, dtype=float)

    def _h_from_ranks(ranks):
        col_sums = ranks.sum(axis=0)
        return (12.0 / (n * k * (k + 1))) * np.sum(col_sums ** 2) - 3 * n * (k + 1)

    rank_matrix = np.array([rankdata(row) for row in arr])
    h_observed = _h_from_ranks(rank_matrix)

    rng = np.random.default_rng(seed)
    h_null = np.empty(n_permutations)
    for i in range(n_permutations):
        permuted = np.array([rng.permutation(row) for row in rank_matrix])
        h_null[i] = _h_from_ranks(permuted)

    p_value = (np.sum(h_null >= h_observed) + 1) / (n_permutations + 1)
    return float(h_observed), float(p_value)


def kendalls_w_null_floor(n, k, percentile=50, n_sims=3000, seed=20260822):
    """Simulated percentile of Kendall's W when the k candidates do not differ
    (n seeds, Gaussian seed effect plus Gaussian noise). Fixed seed."""
    from scipy.stats import friedmanchisquare
    rng = np.random.default_rng(seed)
    ws = []
    for _ in range(n_sims):
        baseline = rng.normal(0, 1, size=n).reshape(-1, 1)
        noise = rng.normal(0, 1, size=(n, k))
        data = baseline + noise
        cols = [data[:, j] for j in range(k)]
        stat, _ = friedmanchisquare(*cols)
        ws.append(stat / (n * (k - 1)))
    return float(np.percentile(ws, percentile))


_kendalls_w_null_floor_cache = {}


def _cached_null_floor(n, k):
    key = (n, k)
    if key not in _kendalls_w_null_floor_cache:
        _kendalls_w_null_floor_cache[key] = kendalls_w_null_floor(n, k)
    return _kendalls_w_null_floor_cache[key]


def rank_biserial_null_floor(n, percentile=75, n_sims=3000, seed=20260822):
    """Simulated percentile of |rank-biserial r| for n paired differences that are
    pure Gaussian noise. Used by the tie test of the stopping rule
    (phases.phase1_grid._tie_test): a non-significant difference whose |r|
    reaches this value is not treated as a tie. Fixed seed."""
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_sims):
        diffs = rng.normal(0, 1, size=n)
        rs.append(abs(rank_biserial_wilcoxon(list(diffs))))
    return float(np.percentile(rs, percentile))


def interpret_rank_biserial(r):
    """Conventional label for |r|: below 0.1 negligible, 0.1 small, 0.3 medium, 0.5 large."""
    mag = abs(r)
    if mag < 0.1:
        label = "negligible"
    elif mag < 0.3:
        label = "small"
    elif mag < 0.5:
        label = "medium"
    else:
        label = "large"
    return label


def interpret_epsilon_squared(eps2):
    """Conventional label for a proportion-of-variance effect size
    (0.01 / 0.06 / 0.14 = small / medium / large)."""
    if eps2 < 0.01:
        return "negligible"
    elif eps2 < 0.06:
        return "small"
    elif eps2 < 0.14:
        return "medium"
    else:
        return "large"


def interpret_kendalls_w(w, n=None, k=None):
    """Label for Kendall's W on the same bands as interpret_epsilon_squared. With n
    and k, a caution is added when W does not exceed the simulated null median
    for that (n, k), since such a W is not evidence of a real difference."""
    label = interpret_epsilon_squared(w)
    if n is not None and k is not None:
        null_floor = _cached_null_floor(n, k)
        if w <= null_floor:
            label += (f" [CAUTION: at/below the N={n},k={k} null-hypothesis "
                      f"median W~={null_floor:.3f} -- not reliable evidence of "
                      f"a real difference at this sample size; see Appendix B.3]")
    return label
