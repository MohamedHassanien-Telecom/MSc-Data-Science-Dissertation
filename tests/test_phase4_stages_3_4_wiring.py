"""src/visualise.py _run_collapse_stages_3_4 connects Stage 3
(bootstrap_elbow_stability) and Stage 4 (structural_break_test) to Phase IV. It
reads phase2_per_seed.json, passes each detector's own per-seed trajectories to
both stages, and skips them when the file is missing. The statistics of the two
stages are tested in test_bootstrap_elbow_stability.py and
test_structural_break_test.py.
  1. without the per-seed file, both stages are skipped (None) without error;
  2. with it, both stages return their report keys for both detectors, and the
     per-seed trajectories reach Stage 4's seed bootstrap;
  3. each detector's result uses that detector's own data.

Usage:  python3 tests/test_phase4_stages_3_4_wiring.py
"""
import sys
import json
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")
from src.visualise import _run_collapse_stages_3_4


def _default_cfg():
    return {"COLLAPSE_SENSITIVITY_PERCENTILES": [50, 60, 70, 75, 80, 90],
            "COLLAPSE_GRAD_THRESHOLD": None}


def _write_fake_per_seed_file(tmpdir, n_seeds, snrs):
    """Write a phase2_per_seed.json with the layout phase2_aggregated writes; returns the AWGN runs."""
    rng = np.random.default_rng(42)
    awgn = []
    for s in range(n_seeds):
        seed_run = []
        for snr in snrs:
            ml_auc = float(np.clip(0.55 + 0.30 / (1 + np.exp(-(snr + 2) / 4.0))
                                    + rng.normal(0, 0.01), 0, 1))
            pr_auc = float(np.clip(0.50 + 0.25 / (1 + np.exp(-(snr + 4) / 4.0))
                                    + rng.normal(0, 0.01), 0, 1))
            ml_dsem = float(max(1e-6, 0.02 + 0.15 * np.exp(-(snr + 20) / 15.0) + rng.normal(0, 0.001)))
            pr_dsem = float(max(1e-6, 0.03 + 0.20 * np.exp(-(snr + 20) / 15.0) + rng.normal(0, 0.001)))
            seed_run.append({
                "max_logit_auc_roc_pixel": ml_auc, "max_logit_dsem": ml_dsem,
                "prototype_auc_roc_pixel": pr_auc, "prototype_dsem": pr_dsem,
            })
        awgn.append(seed_run)
    per_seed = {"awgn": awgn, "rayleigh": [], "snr_range_db": snrs, "n_seeds": n_seeds}
    with open(Path(tmpdir) / "phase2_per_seed.json", "w") as f:
        json.dump(per_seed, f)
    return awgn


def test_missing_file_degrades_gracefully():
    """Without the per-seed file, Stages 3 and 4 are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snrs = list(range(-20, 22, 2))
        result = _run_collapse_stages_3_4(tmpdir, snrs, [0.7] * len(snrs), [0.6] * len(snrs), _default_cfg())
        assert result == {"max_logit": None, "prototype": None}
    print("test_missing_file_degrades_gracefully: PASS")


def test_wiring_produces_both_stages_for_both_detectors():
    """With the per-seed file, both stages report their keys for both detectors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        snrs = list(range(-20, 22, 2))
        n_seeds = 8
        awgn = _write_fake_per_seed_file(tmpdir, n_seeds, snrs)

        # Mean trajectories, as phase4 computes them from the aggregate.
        ml_auc = [float(np.mean([run[i]["max_logit_auc_roc_pixel"] for run in awgn]))
                   for i in range(len(snrs))]
        pr_auc = [float(np.mean([run[i]["prototype_auc_roc_pixel"] for run in awgn]))
                   for i in range(len(snrs))]

        result = _run_collapse_stages_3_4(tmpdir, snrs, ml_auc, pr_auc, _default_cfg())

        for det in ("max_logit", "prototype"):
            assert result[det] is not None, f"{det}: Stage 3/4 result missing entirely"
            boot = result[det]["bootstrap"]
            brk = result[det]["structural_break"]
            # Keys written to collapse_report.json.
            for key in ("original_elbow_snr", "stability_fraction_within_one_grid_step",
                        "tie_fraction", "no_candidate_fraction", "bootstrap_ci_95"):
                assert key in boot, f"{det} bootstrap result missing '{key}'"
            for key in ("p_value", "smooth_vs_break", "observed_breakpoint_snr"):
                assert key in brk, f"{det} structural-break result missing '{key}'"
            svb = brk["smooth_vs_break"]
            # The per-seed trajectories reach Stage 4's seed bootstrap.
            assert svb["bootstrap"] is not None and svb["bootstrap"]["n_seeds"] == n_seeds, svb["bootstrap"]
            assert svb["verdict"] in ("smooth", "break", "inconclusive")
        for det in ("max_logit", "prototype"):
            svb = result[det]["structural_break"]["smooth_vs_break"]
            print(f"  {det}:  bootstrap stability={result[det]['bootstrap']['stability_fraction_within_one_grid_step']:.0%}, "
                  f"p={result[det]['structural_break']['p_value']:.4f}, "
                  f"delta AIC={svb['delta_aic_break_minus_smooth']:+.1f}, verdict={svb['verdict']}")
    print("test_wiring_produces_both_stages_for_both_detectors: PASS")


def test_wiring_reads_correct_detector_specific_keys():
    """Max Logit's AUROC is flat and Prototype's rises with SNR: Max Logit's Stage 4
    result must show no significant break, so it was computed from its own data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        snrs = list(range(-20, 22, 2))
        n_seeds = 6
        awgn = []
        for s in range(n_seeds):
            seed_run = []
            for snr in snrs:
                seed_run.append({
                    "max_logit_auc_roc_pixel": 0.9, "max_logit_dsem": 1e-5,
                    "prototype_auc_roc_pixel": 0.5 + 0.3 * (snr + 20) / 40,
                    "prototype_dsem": 0.5,
                })
            awgn.append(seed_run)
        with open(Path(tmpdir) / "phase2_per_seed.json", "w") as f:
            json.dump({"awgn": awgn, "rayleigh": [], "snr_range_db": snrs, "n_seeds": n_seeds}, f)

        ml_auc = [0.9] * len(snrs)
        pr_auc = [0.5 + 0.3 * (snr + 20) / 40 for snr in snrs]
        result = _run_collapse_stages_3_4(tmpdir, snrs, ml_auc, pr_auc, _default_cfg())

        assert result["max_logit"]["structural_break"]["p_value"] > 0.05, (
            "Max Logit's flat trajectory shows a significant break: the other detector's data was used."
        )
        print("test_wiring_reads_correct_detector_specific_keys: PASS")


if __name__ == "__main__":
    test_missing_file_degrades_gracefully()
    test_wiring_produces_both_stages_for_both_detectors()
    test_wiring_reads_correct_detector_specific_keys()
    print("\nAll Phase IV Stage 3/4 wiring checks passed.")
