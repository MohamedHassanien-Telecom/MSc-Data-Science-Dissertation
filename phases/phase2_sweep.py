"""Phase II: SNR sweep of every production repetition on the AWGN and Rayleigh
channels, and the mean +/- SD over repetitions.

Writes, per repetition, run_0_rep{i}/phase2.json, and for all repetitions
run_0/phase2_aggregated.json (mean and SD per SNR point) and
run_0/phase2_per_seed.json (every repetition's results, used by Phase IV and the
later controls).
"""
from src.evaluate import eval_both_systems, eval_noise_seed
from src.config import (training_fingerprint, evaluation_fingerprint, fingerprint_key)
import json
import math
import statistics
from pathlib import Path


def _phase2_stamp(p1, cfg):
    """Identity of one repetition's sweep: the evaluation settings, both systems'
    training settings and the repetition's calibrated thresholds."""
    return {"eval": evaluation_fingerprint(cfg),
            "max_logit": training_fingerprint(cfg, "max_logit"),
            "prototype": training_fingerprint(cfg, "prototype"),
            "tau_max_logit": p1["tau_max_logit"], "tau_prototype": p1["tau_prototype"],
            "sdop_threshold_a": p1["sdop_threshold_a"], "sdop_threshold_b": p1["sdop_threshold_b"]}


def phase2(p1, cfg, device, resume=True):
    """SNR sweep of one repetition (both systems, every SNR in cfg["SNR_RANGE_DB"]).

    Channel noise is seeded per (channel, SNR), so the sweep is reproducible from
    the checkpoints. A repetition whose phase2.json was written with the same
    stamp (phase2_stamp.json) is loaded instead of re-evaluated.
    """
    out = Path(p1["out"])
    stamp = _phase2_stamp(p1, cfg)
    res_path, stamp_path = out / "phase2.json", out / "phase2_stamp.json"
    if resume and res_path.exists() and stamp_path.exists():
        try:
            if fingerprint_key(json.load(open(stamp_path))) == fingerprint_key(stamp):
                print(f"  [Phase II] {out.name}: already evaluated with the current settings -- loaded")
                return json.load(open(res_path))
        except (ValueError, OSError):
            pass
    print(f"\n{'='*62}\n  PHASE II -- AWGN Sweep  SNR{cfg['SNR_RANGE_DB']}\n{'='*62}")
    jscc_a=p1["jscc_a"]; jscc_b=p1["jscc_b"]
    max_logit_det=p1["max_logit_det"]; tau_max_logit=p1["tau_max_logit"]
    prototype_det=p1["prototype_det"]; tau_prototype=p1["tau_prototype"]
    sdop_threshold_a=p1["sdop_threshold_a"]; sdop_threshold_b=p1["sdop_threshold_b"]
    ldr=p1["loaders"]["test"]
    results={"awgn":[],"rayleigh":[]}
    for snr in cfg["SNR_RANGE_DB"]:
        r = eval_both_systems(jscc_a, max_logit_det, tau_max_logit, sdop_threshold_a,
                               jscc_b, prototype_det, tau_prototype, sdop_threshold_b,
                               ldr, snr, cfg, device, "awgn",
                               noise_seed=eval_noise_seed(cfg, snr, "awgn"))
        results["awgn"].append(r)
        print(f"  SNR={snr:+4d}dB  MaxLogit AUC={r['max_logit_auc_roc_pixel']:.3f} "
              f"FPR95={r['max_logit_fpr95']:.3f} SRI_AUC={r['max_logit_sri_auc']:.3f} "
              f"Dsem={r['max_logit_dsem']:.4f}  |  "
              f"Prototype AUC={r['prototype_auc_roc_pixel']:.3f} FPR95={r['prototype_fpr95']:.3f} "
              f"SRI_AUC={r['prototype_sri_auc']:.3f} Dsem={r['prototype_dsem']:.4f}")
        if cfg["USE_RAYLEIGH"]:
            rr = eval_both_systems(jscc_a, max_logit_det, tau_max_logit, sdop_threshold_a,
                                    jscc_b, prototype_det, tau_prototype, sdop_threshold_b,
                                    ldr, snr, cfg, device, "rayleigh",
                                    noise_seed=eval_noise_seed(cfg, snr, "rayleigh"))
            results["rayleigh"].append(rr)
            print(f"       Rayleigh  MaxLogit AUC={rr['max_logit_auc_roc_pixel']:.3f}  |  "
                  f"Prototype AUC={rr['prototype_auc_roc_pixel']:.3f}")
    with open(res_path,"w") as f: json.dump(results,f,indent=2)
    with open(stamp_path,"w") as f: json.dump(stamp,f,indent=2)   # written last: marks completion
    print(f"\n[Phase II] Done -> {res_path}")
    return results


_NON_METRIC_KEYS = {"snr_db", "channel"}


def _mean_std(values):
    """Mean and sample SD of the finite values; returns (mean, sd, n_nonfinite)."""
    vals = [float(v) for v in values]
    finite = [v for v in vals if math.isfinite(v)]
    n_bad = len(vals) - len(finite)
    if not finite:
        return float("nan"), None, n_bad
    return statistics.mean(finite), (statistics.stdev(finite) if len(finite) > 1 else None), n_bad


def _aggregate_metric_dicts(dicts):
    """Mean and sample SD of every numeric field over the per-repetition results of
    one (SNR, channel) point. Fields that are None for some repetitions (sri_op
    when FPR95 came from the fallback) are averaged over the others, with the
    number of None values stored as {field}_n_none."""
    agg = {}
    for k in dicts[0]:
        if k in _NON_METRIC_KEYS:
            agg[k] = dicts[0][k]
            continue
        values = [d[k] for d in dicts]
        if any(v is None for v in values):
            non_none = [v for v in values if v is not None]
            agg[f"{k}_n_none"] = len(values) - len(non_none)
            if non_none:
                agg[f"{k}_mean"], agg[f"{k}_std"], _nb = _mean_std(non_none)
            else:
                agg[f"{k}_mean"] = None
                agg[f"{k}_std"] = None
            continue
        try:
            m, sd, n_bad = _mean_std(values)
        except (TypeError, ValueError):
            agg[k] = dicts[0][k]
            continue
        agg[f"{k}_mean"] = m
        agg[f"{k}_std"] = sd
        if n_bad:
            agg[f"{k}_n_nonfinite"] = n_bad
    return agg


def phase2_aggregated(p1_grid, cfg, device):
    """Sweep every production repetition (phase2) and aggregate mean +/- SD per SNR
    point; both detectors use the same repetitions. Writes phase2_aggregated.json
    and phase2_per_seed.json to p1_grid["out"] and returns the aggregate."""
    print(f"\n{'='*62}\n  PHASE II (AGGREGATED) -- {len(p1_grid['final_runs'])} repeated runs"
          + f", mean +/- std\n{'='*62}")

    per_run_results = []
    for i, p1 in enumerate(p1_grid["final_runs"]):
        print(f"\n--- Phase II, repetition {i+1}/{len(p1_grid['final_runs'])} ---")
        per_run_results.append(phase2(p1, cfg, device))

    aggregated = {"awgn": [], "rayleigh": []}
    n_snr_points = len(per_run_results[0]["awgn"])
    for channel in ["awgn", "rayleigh"]:
        if not per_run_results[0][channel]:
            continue
        for snr_idx in range(n_snr_points):
            shared_n_dicts = [run[channel][snr_idx] for run in per_run_results]
            aggregated[channel].append(_aggregate_metric_dicts(shared_n_dicts))

    out = p1_grid["out"]
    with open(out / "phase2_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)
    per_seed_export = {
        "awgn": [[run["awgn"][i] for i in range(n_snr_points)] for run in per_run_results],
        "rayleigh": ([[run["rayleigh"][i] for i in range(len(run["rayleigh"]))] for run in per_run_results]
                     if per_run_results[0]["rayleigh"] else []),
        "snr_range_db": cfg["SNR_RANGE_DB"],
        "n_seeds": len(per_run_results),
    }
    with open(out / "phase2_per_seed.json", "w") as f:
        json.dump(per_seed_export, f, indent=2)
    print(f"\n[Phase II Aggregated] Done -> {out}/phase2_aggregated.json")
    print(f"[Phase II Aggregated] Per-seed trajectories also saved -> {out}/phase2_per_seed.json")
    return aggregated
