"""Installed statistics libraries against requirements.txt:
  1. reports whether scipy, numpy and scikit-learn match the pinned versions (a
     mismatch is printed as a warning; results may then not reproduce exactly);
  2. scipy.stats.mannwhitneyu with method="auto" gives the expected exact result
     for a 5 vs 5 toy case;
  3. scipy.stats.wilcoxon runs on a 21-point paired toy case.

Run once per new environment.
Usage:  python3 tests/test_dependency_versions.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scipy, numpy, sklearn
from scipy import stats


def _read_pinned_version(package_name):
    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    with open(req_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{package_name}=="):
                return line.split("==")[1]
    return None


def test_installed_versions_match_pinned():
    checks = [("scipy", scipy.__version__), ("numpy", numpy.__version__),
              ("scikit-learn", sklearn.__version__)]
    mismatches = []
    for pkg, installed in checks:
        pinned = _read_pinned_version(pkg)
        print(f"  {pkg}: installed={installed}, pinned in requirements.txt={pinned}")
        if pinned is not None and installed != pinned:
            mismatches.append((pkg, installed, pinned))
    if mismatches:
        print(f"\n  WARNING -- {len(mismatches)} version mismatch(es) against requirements.txt:")
        for pkg, installed, pinned in mismatches:
            print(f"    {pkg}: installed {installed}, requirements.txt pins {pinned}")
        print("  Statistical outputs may not reproduce the reported results exactly.")
    else:
        print("  All three pinned packages match requirements.txt exactly.")


def test_mannwhitneyu_auto_method_behavior():
    """method="auto" chooses between the exact and the asymptotic p-value; for this
    5 vs 5 case it gives U = 12, p = 1.0 with scipy 1.17.1.
    """
    a = [1.1, 2.2, 3.3, 4.4, 5.5]
    b = [1.5, 2.6, 3.1, 4.9, 5.2]
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    print(f"  N=5 vs N=5, method='auto' -> U={u_stat}, p={p_val:.6f}")
    assert abs(u_stat - 12.0) < 1e-9 and abs(p_val - 1.000000) < 1e-5, \
        f"mannwhitneyu method='auto' gave U={u_stat}, p={p_val:.6f}; expected U=12.0, p=1.0 " \
        f"(scipy 1.17.1)."


def test_wilcoxon_auto_method_behavior():
    """Output ranges of wilcoxon on a 21-point paired toy case."""
    diffs = [0.01, -0.02, 0.03, 0.015, -0.005, 0.025, -0.01, 0.02, 0.005, -0.015,
             0.03, 0.01, -0.02, 0.025, 0.005, -0.01, 0.015, 0.02, -0.005, 0.03, 0.01]
    a = [0.5] * len(diffs)
    b = [0.5 + d for d in diffs]
    w_stat, p_val = stats.wilcoxon(a, b)
    print(f"  N=21 paired, method='auto' -> W={w_stat}, p={p_val:.6f}")
    assert w_stat > 0 and 0 <= p_val <= 1, "sanity check on output ranges"


if __name__ == "__main__":
    print("=== Version match check ===")
    test_installed_versions_match_pinned()
    print("\n=== mannwhitneyu method='auto' behavior lock ===")
    test_mannwhitneyu_auto_method_behavior()
    print("PASS")
    print("\n=== wilcoxon method='auto' behavior (reference) ===")
    test_wilcoxon_auto_method_behavior()
    print("PASS")
    print("\nAll dependency reproducibility checks completed.")
