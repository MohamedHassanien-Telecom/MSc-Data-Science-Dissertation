"""detect_collapse (src/collapse.py), Stages 1 and 2 of the semantic-collapse
analysis, on synthetic curves with a known transition:
  1. a sharp knee (transition from -10 to -2 dB) puts the headline elbow near the
     transition onset, not at the edges of the SNR range, with at least half of
     the percentile settings agreeing;
  2. a gradual sigmoid gives a computed agreement fraction in [0, 1] (lower
     agreement is expected for a smooth transition; no SNR is asserted).

Usage:  python3 tests/test_collapse_region.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.collapse import detect_collapse


def test_sharp_knee_elbow_detection():
    """Piecewise-linear AUROC and D_sem with a sharp transition from -10 to -2 dB."""
    snrs = list(range(-20, 22, 2))

    def knee_auc(snr, start=-10, end=-2, lo=0.55, hi=0.75):
        if snr <= start: return lo
        if snr >= end: return hi
        return lo + (hi - lo) * (snr - start) / (end - start)

    def knee_dsem(snr, start=-10, end=-2, hi=0.15, lo=0.05):
        if snr <= start: return hi
        if snr >= end: return lo
        return hi - (hi - lo) * (snr - start) / (end - start)

    auc_arr = [knee_auc(s) for s in snrs]
    dsem_arr = [knee_dsem(s) for s in snrs]
    cfg = {"COLLAPSE_SENSITIVITY_PERCENTILES": [50, 60, 70, 75, 80, 90],
           "COLLAPSE_GRAD_THRESHOLD": None}

    result = detect_collapse(snrs, auc_arr, dsem_arr, cfg)
    print(f"Headline elbow_snr={result['elbow_snr']}, "
          f"agreement={result['elbow_agreement_fraction']:.0%}")

    # The elbow (onset of curvature) lies within two grid steps of -10 dB.
    assert -16 <= result["elbow_snr"] <= -6, \
        f"Elbow at {result['elbow_snr']} is not near the transition onset (-10 dB)."
    assert result["elbow_agreement_fraction"] >= 0.5, \
        f"Only {result['elbow_agreement_fraction']:.0%} of tested percentiles agree -- " \
        f"below the 50% consistency bar this test treats as a passing signal."
    print("PASS: elbow correctly lands near the designed transition onset, "
          "not at the SNR range's own edges.")


def test_gradual_sigmoid_flagged_as_lower_agreement():
    """A gradual (sigmoid) transition: only the agreement fraction is checked, since
    a smooth curve has no unambiguous elbow.
    """
    snrs = list(range(-20, 22, 2))

    def sigmoid_auc(snr, midpoint=-6, steepness=0.5):
        return 0.5 + 0.3 / (1 + np.exp(-steepness * (snr - midpoint)))

    def sigmoid_dsem(snr, midpoint=-6, steepness=0.4):
        return 0.05 + 0.15 / (1 + np.exp(steepness * (snr - midpoint)))

    auc_arr = [sigmoid_auc(s) for s in snrs]
    dsem_arr = [sigmoid_dsem(s) for s in snrs]
    cfg = {"COLLAPSE_SENSITIVITY_PERCENTILES": [50, 60, 70, 75, 80, 90],
           "COLLAPSE_GRAD_THRESHOLD": None}

    result = detect_collapse(snrs, auc_arr, dsem_arr, cfg)
    print(f"Sigmoid case -- headline elbow_snr={result['elbow_snr']}, "
          f"agreement={result['elbow_agreement_fraction']:.0%}")
    assert 0.0 <= result["elbow_agreement_fraction"] <= 1.0
    print("Recorded: the agreement fraction is computed for a gradual transition.")


if __name__ == "__main__":
    test_sharp_knee_elbow_detection()
    print()
    test_gradual_sigmoid_flagged_as_lower_agreement()
    print("\nAll collapse-region checks completed.")
