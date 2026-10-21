"""Phase I: hyperparameter search to convergence, then the production repetitions
(dissertation Section 3.1.5 and Appendix B).

Each round of the search runs these steps in order, every candidate being trained
for both detectors on the same selection seeds (cfg["SELECTION_SEEDS"], disjoint
from the production seeds) and scored by pixel AUROC on the calibration images at
DEFAULT_SNR_DB:

  Step 0  backbone stage (Swin-Tiny stage 0-3, each at its native grid, budget-matched
          to 1,568 transmitted values)
  Step A  patch grid / dimension (stage 2 only)
  Step B  JOINT_LR
  Step C  JOINT_MOMENTUM, JOINT_LR_POWER, WEIGHT_DECAY, MIN_REL_IMPROVE (one at a time)
  Step D  PROTOTYPE_SCALE (Prototype only)
  Step E  lambda_sem grid
  Step F  LAMBDA_REC

Per detector, the candidate with the highest mean AUROC is selected; a Friedman
permutation test with Kendall's W (or a Wilcoxon test for two candidates) is
reported alongside. Steps before Step E use a provisional lambda_sem in round 0
and the previous round's winner afterwards. From round 1 on, the search stops when
neither detector's complete round-end configuration improves significantly on the
previous round's (phases/objective_convergence.py). Every (candidate, seed) result
is saved as it finishes, stamped with the settings it was trained under, so an
interrupted search resumes without retraining.
"""
from pathlib import Path
import json
from src.effect_sizes import (interpret_rank_biserial, rank_biserial_null_floor,
                                rank_biserial_wilcoxon, kendalls_w, interpret_kendalls_w,
                                friedman_permutation_test)
import statistics
from phases import objective_convergence as _oc
from src.dataset import build_dataloaders, detect_anomaly_value, discover_known_classes
from src.gpu_utils import detect_max_batch_size
from src.config import set_seed
from src.models import SwinJSCCSystem, MaxLogitDetector, PrototypeDetector
from src.train import train_joint_system
from src.metrics import calibrate_tau, calibrate_sdop_threshold
from phases.phase1_train import train_one_system, phase1, load_phase1

_DETECTOR_CLASSES = {"max_logit": MaxLogitDetector, "prototype": PrototypeDetector}


def _lambda_rec_upstream_ok(cached, key, current_value, round_idx):
    """Cache check of LAMBDA_REC for the steps that run before Step F.

    In round 0 LAMBDA_REC still has its initial value during Steps 0-E (Step F,
    which sets it, runs last), so it is not checked. From round 1 on it must match;
    an entry without the field is treated as not matching.
    """
    if round_idx == 0:
        return True
    if cached is None or key not in cached:
        return False
    return cached[key] == current_value


# Optimiser settings stamped on every cached result of Steps 0, A and B.
_UPSTREAM_TRAIN_KEYS = {"_joint_lr": "JOINT_LR", "_joint_momentum": "JOINT_MOMENTUM",
                          "_joint_lr_power": "JOINT_LR_POWER", "_weight_decay": "WEIGHT_DECAY",
                          "_min_rel_improve": "MIN_REL_IMPROVE"}


def _record_upstream_train_keys(entry, cfg, sfx, exclude=()):
    """Stamp the current optimiser settings of detector suffix sfx ("ml"/"pr") onto
    a cache entry (and PROTOTYPE_SCALE for "pr"), except the keys in `exclude`."""
    for key, cfg_prefix in _UPSTREAM_TRAIN_KEYS.items():
        if cfg_prefix in exclude:
            continue
        entry[f"{key}_{sfx}"] = cfg[f"{cfg_prefix}_{sfx.upper()}"]
    if sfx == "pr" and "PROTOTYPE_SCALE" not in exclude:
        entry["_prototype_scale"] = cfg["PROTOTYPE_SCALE"]


def _upstream_train_ok(cached, sfx, cfg, round_idx, exclude=()):
    """True if every optimiser setting stamped on `cached` equals the current one.

    Fields that are absent are accepted. `exclude` names the setting the calling
    step is itself sweeping. PROTOTYPE_SCALE is checked from round 1 on only (it is
    first settled in round 0's Step D).
    """
    for key, cfg_prefix in _UPSTREAM_TRAIN_KEYS.items():
        if cfg_prefix in exclude:
            continue
        cache_key = f"{key}_{sfx}"
        if cache_key not in cached:
            continue
        if cached[cache_key] != cfg[f"{cfg_prefix}_{sfx.upper()}"]:
            return False
    if sfx == "pr" and round_idx > 0 and "_prototype_scale" in cached:
        if cached["_prototype_scale"] != cfg["PROTOTYPE_SCALE"]:
            return False
    return True


def _run_grid_search(cfg, device, loaders, num_classes, nsym, grid_dir, round_idx=0):
    """Step E: train every (detector, lambda_sem, seed) combination of
    cfg["LAMBDA_SEM_GRID_MAXLOGIT"] / cfg["LAMBDA_SEM_GRID_PROTOTYPE"] on the first
    six selection seeds, with patience cfg["GRID_SEARCH_PATIENCE"].

    Each result (validation task loss, pixel AUROC and FPR95 on the calibration
    images, and the settings it was trained under) is appended to
    grid_partial_results_round{N}.json as it finishes, and its checkpoints are
    deleted. Returns the records that match the current settings, one per
    combination, in a fixed order.
    """
    partial_path = grid_dir.parent / f"grid_partial_results_round{round_idx}.json"
    def _tag_for(r):
        # Identity of a record: detector, lambda_sem, seed and every setting it was
        # trained under. In round 0, LAMBDA_REC is not part of it (Step F runs later)
        # and a missing MIN_REL_IMPROVE field takes the current value.
        _mri_fallback = ((cfg["MIN_REL_IMPROVE_ML"] if r["detector_type"] == "max_logit"
                          else cfg["MIN_REL_IMPROVE_PR"]) if round_idx == 0 else None)
        _lambda_rec_tag_val = "_ROUND0_LAMBDA_REC_IGNORED_" if round_idx == 0 else r.get("lambda_rec_at_train")
        base = (r["detector_type"], r["lambda_sem"], r.get("seed"),
                r.get("patch_grid_at_train"), r.get("patch_dim_at_train"), r.get("joint_lr_at_train"),
                r.get("joint_momentum_at_train"), r.get("joint_lr_power_at_train"), r.get("weight_decay_at_train"),
                r.get("min_rel_improve_at_train", _mri_fallback), _lambda_rec_tag_val)
        if r["detector_type"] == "prototype":
            return base + (r.get("prototype_scale_at_train"),)
        return base

    results = []
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        done_tags = {_tag_for(r) for r in results}
        print(f"[grid:round{round_idx}] RESUMING -- found {len(results)} already-completed "
              f"combination(s) at {partial_path}")
    else:
        done_tags = set()

    _grid_by_detector = {"max_logit": cfg["LAMBDA_SEM_GRID_MAXLOGIT"],
                          "prototype": cfg["LAMBDA_SEM_GRID_PROTOTYPE"]}
    from src.evaluate import _eval_one_system
    grid_seeds = cfg["SELECTION_SEEDS"][:6]   # six seeds: the smallest N at which a two-sided exact
                                              # Wilcoxon test can reach p < 0.05
    _current_tags_in_order = []
    for detector_type in ["max_logit", "prototype"]:
        for lam_sem in _grid_by_detector[detector_type]:
            for seed in grid_seeds:
                _lambda_rec_check_val = (
                    "_ROUND0_LAMBDA_REC_IGNORED_" if round_idx == 0
                    else (cfg["LAMBDA_REC_ML"] if detector_type == "max_logit" else cfg["LAMBDA_REC_PR"]))
                check_tag = (detector_type, lam_sem, seed,
                             (cfg["PATCH_GRID_ML"] if detector_type == "max_logit" else cfg["PATCH_GRID_PR"]),
                             (cfg["PATCH_DIM_ML"] if detector_type == "max_logit" else cfg["PATCH_DIM_PR"]),
                             (cfg["JOINT_LR_ML"] if detector_type == "max_logit" else cfg["JOINT_LR_PR"]),
                             (cfg["JOINT_MOMENTUM_ML"] if detector_type == "max_logit" else cfg["JOINT_MOMENTUM_PR"]),
                             (cfg["JOINT_LR_POWER_ML"] if detector_type == "max_logit" else cfg["JOINT_LR_POWER_PR"]),
                             (cfg["WEIGHT_DECAY_ML"] if detector_type == "max_logit" else cfg["WEIGHT_DECAY_PR"]),
                             (cfg["MIN_REL_IMPROVE_ML"] if detector_type == "max_logit" else cfg["MIN_REL_IMPROVE_PR"]),
                             _lambda_rec_check_val)
                if detector_type == "prototype":
                    check_tag = check_tag + (cfg["PROTOTYPE_SCALE"],)
                _current_tags_in_order.append(check_tag)
                if check_tag in done_tags:
                    print(f"[grid:round{round_idx}] SKIP {detector_type}, lambda_sem={lam_sem}, "
                          f"seed={seed} -- already completed")
                    continue
                set_seed(seed)
                tag = f"grid_round{round_idx}_{detector_type}_sem{lam_sem}_seed{seed}"
                print(f"\n[grid:round{round_idx}] {detector_type}, lambda_sem={lam_sem}, seed={seed} "
                      f"(patience={cfg['GRID_SEARCH_PATIENCE']})...")
                r = train_one_system(cfg, device, loaders, num_classes, nsym, detector_type,
                                      lam_sem, cfg["GRID_SEARCH_PATIENCE"], grid_dir, tag)
                # Training sees no anomalies, so AUROC is measured afterwards on the
                # calibration images, as in every other step.
                jscc, detector = r["jscc"], r["detector"]
                jscc.eval(); detector.eval()
                tau = calibrate_tau(jscc, detector, loaders["val_mixed"], device,
                                     target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
                sdop = calibrate_sdop_threshold(jscc, loaders["val_mixed"], device,
                                                 snr_db=cfg["CLS_FIT_SNR"],
                                                 percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
                r_eval = _eval_one_system(jscc, detector, tau, sdop, loaders["val_mixed"],
                                           cfg["DEFAULT_SNR_DB"], cfg, device, "awgn")
                r_light = {k: v for k, v in r.items() if k not in ("jscc", "detector")}
                r_light["auc_roc_pixel"] = float(r_eval["auc_roc_pixel"])
                r_light["fpr95"] = float(r_eval["fpr95"])
                r_light["detector_type"] = detector_type
                r_light["lambda_sem"] = lam_sem
                r_light["seed"] = seed
                r_light["patch_grid_at_train"] = (cfg["PATCH_GRID_ML"] if detector_type == "max_logit"
                                                   else cfg["PATCH_GRID_PR"])
                r_light["patch_dim_at_train"] = (cfg["PATCH_DIM_ML"] if detector_type == "max_logit"
                                                  else cfg["PATCH_DIM_PR"])
                r_light["joint_lr_at_train"] = (cfg["JOINT_LR_ML"] if detector_type == "max_logit"
                                                 else cfg["JOINT_LR_PR"])
                r_light["joint_momentum_at_train"] = (cfg["JOINT_MOMENTUM_ML"] if detector_type == "max_logit"
                                                       else cfg["JOINT_MOMENTUM_PR"])
                r_light["joint_lr_power_at_train"] = (cfg["JOINT_LR_POWER_ML"] if detector_type == "max_logit"
                                                       else cfg["JOINT_LR_POWER_PR"])
                r_light["weight_decay_at_train"] = (cfg["WEIGHT_DECAY_ML"] if detector_type == "max_logit"
                                                     else cfg["WEIGHT_DECAY_PR"])
                r_light["min_rel_improve_at_train"] = (cfg["MIN_REL_IMPROVE_ML"] if detector_type == "max_logit"
                                                        else cfg["MIN_REL_IMPROVE_PR"])
                r_light["lambda_rec_at_train"] = (cfg["LAMBDA_REC_ML"] if detector_type == "max_logit"
                                                   else cfg["LAMBDA_REC_PR"])
                if detector_type == "prototype":
                    r_light["prototype_scale_at_train"] = cfg["PROTOTYPE_SCALE"]
                results.append(r_light)
                print(f"[grid]   -> best_val_task_loss={r['loss']:.4f}, AUC={r_light['auc_roc_pixel']:.4f}, "
                      f"epochs_run={r['epochs_run']}, "
                      f"epochs_since_best_at_stop={r['epochs_since_best_at_stop']}")

                for suffix in ("_jscc.pt", "_detector.pt"):
                    ckpt_path = grid_dir / f"{tag}{suffix}"
                    if ckpt_path.exists():
                        ckpt_path.unlink()
                print("[grid]   -> checkpoint files removed")

                with open(partial_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
    # Return only the records matching the current settings (the latest one per
    # combination); records of other settings stay in the file but are not analysed.
    _by_tag = {}
    for r in results:
        _by_tag[_tag_for(r)] = r
    current = [_by_tag[t] for t in _current_tags_in_order if t in _by_tag]
    if len(current) != len(results):
        print(f"[grid:round{round_idx}] {len(results) - len(current)} cached record(s) belong to a "
              f"different upstream configuration -- excluded from this round's lambda_sem analysis "
              f"(kept in {partial_path.name}).")
    return current


def _analyze_grid_results(grid_results, grid_search_patience):
    """Winning lambda_sem per detector: the highest mean AUROC over the six seeds.

    The winner is compared with the runner-up by a paired Wilcoxon test and the
    rank-biserial r. Also reports the median fraction of epochs run before the
    best validation point. The production patience equals the grid patience.
    """
    from scipy import stats as scipy_stats

    by_detector_lambda = {}
    for r in grid_results:
        key = (r["detector_type"], r["lambda_sem"])
        by_detector_lambda.setdefault(key, []).append(r["auc_roc_pixel"])

    by_detector = {"max_logit": {}, "prototype": {}}
    for (det, lam), aucs in by_detector_lambda.items():
        by_detector[det][lam] = aucs

    winning_lambda_sem = {}
    lambda_sem_margins = {}
    for det, lam_to_aucs in by_detector.items():
        sorted_lams = sorted(lam_to_aucs.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True)
        (best_lam, best_aucs), (second_lam, second_aucs) = sorted_lams[0], sorted_lams[1]
        winning_lambda_sem[det] = best_lam
        best_mean, second_mean = statistics.mean(best_aucs), statistics.mean(second_aucs)
        rel_gap = (best_mean - second_mean) / max(abs(best_mean), 1e-8)
        if len(best_aucs) >= 2 and len(second_aucs) >= 2:
            diffs = [a - b for a, b in zip(best_aucs, second_aucs)]
            try:
                _, p_val = scipy_stats.wilcoxon(best_aucs, second_aucs)
            except ValueError:
                _, p_val = 0.0, 1.0   # all paired differences are zero
            effect_r = rank_biserial_wilcoxon(diffs)
        else:
            _, p_val, effect_r = float("nan"), float("nan"), float("nan")
        reliable = (p_val < 0.05) if not (p_val != p_val) else (rel_gap >= 0.05)
        lambda_sem_margins[det] = {
            "best_lambda_sem": best_lam, "best_mean_auc": best_mean, "best_aucs": best_aucs,
            "second_lambda_sem": second_lam, "second_mean_auc": second_mean, "second_aucs": second_aucs,
            "relative_gap": rel_gap, "wilcoxon_p": p_val, "reliable": reliable,
            "effect_size_rank_biserial": effect_r,
            "effect_size_interpretation": interpret_rank_biserial(effect_r) if effect_r == effect_r else "n/a",
        }
        if effect_r == effect_r:
            print(f"[grid] {det}: effect size (rank-biserial r) = {effect_r:.3f} "
                  f"({interpret_rank_biserial(effect_r)})")
        if not reliable:
            print(f"[grid] NOTE -- {det}: lambda_sem={best_lam} has the highest mean AUC but does not "
                  f"differ significantly from lambda_sem={second_lam} ({best_mean:.4f} vs "
                  f"{second_mean:.4f}, N={len(best_aucs)}, Wilcoxon p={p_val:.4f}).")
        else:
            print(f"[grid] {det}: winning lambda_sem={best_lam} beats lambda_sem={second_lam} "
                  f"significantly on AUC (Wilcoxon p={p_val:.4f}, N={len(best_aucs)} each).")

    early_best_fractions = []
    for r in grid_results:
        frac = (r["best_epoch_idx"] + 1) / max(1, r["epochs_run"])
        early_best_fractions.append(frac)
    median_early_fraction = statistics.median(early_best_fractions)
    lr_mismatch_flag = median_early_fraction < 0.35

    analysis = {
        "winning_lambda_sem": winning_lambda_sem,
        "lambda_sem_margins": lambda_sem_margins,
        "derived_patience": grid_search_patience,
        "all_epochs_since_best_at_stop": [r["epochs_since_best_at_stop"] for r in grid_results],
        "median_early_best_fraction": median_early_fraction,
        "lr_mismatch_suspected": lr_mismatch_flag,
        "lr_mismatch_note": (
            "Best validation point reached, on median, within the first "
            f"{median_early_fraction:.0%} of each grid run's total epochs, "
            "consistently across the lambda_sem values tested; the learning rate "
            "rather than lambda_sem may be the cause."
        ) if lr_mismatch_flag else (
            "No consistent early-overfitting pattern found across the grid."
        ),
    }
    return analysis


def _verify_backbone_stage(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_override=None, round_idx=0):
    """Step 0: Swin-Tiny stage of each detector, from cfg["STAGE_CANDIDATES"].

    Each stage is used at its native grid (56, 28, 14, 7) with the cell dimension
    chosen to keep about 1,568 transmitted values (stage 0 cannot go below 3,136).
    Both detectors are trained per stage and seed on cfg["STAGE_VERIFY_N"] seeds.
    When a detector's stage changes, its PATCH_GRID / PATCH_DIM are set to that
    stage's grid and dimension, and Step A is skipped for it. Updates cfg in place.
    """
    from src.metrics import calibrate_tau, calibrate_sdop_threshold
    from src.evaluate import _eval_one_system
    from src.effect_sizes import friedman_permutation_test, kendalls_w, interpret_kendalls_w
    from scipy import stats

    _native_grid = {0: 56, 1: 28, 2: 14, 3: 7}
    _target_total = 14 * 14 * 8   # 1,568 values
    _matched_dim = {s: max(1, round(_target_total / (g * g))) for s, g in _native_grid.items()}

    round_tag = f"round{round_idx}"
    state_path = verify_dir.parent / f"stage_verify_state_{round_tag}.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    if state:
        n_done = sum(len(v) for v in state.values())
        print(f"[stage-verify:{round_tag}] RESUMING -- found {n_done} already-completed "
              f"(stage, seed) result(s) at {state_path}")

    verify_seeds = cfg["SELECTION_SEEDS"][:cfg["STAGE_VERIFY_N"]]
    if lambda_sem_override is not None:
        lambda_sem_for_run = lambda_sem_override
        print(f"[stage-verify:{round_tag}] using the previous round's lambda_sem winners: "
              f"ml={lambda_sem_for_run['max_logit']}, pr={lambda_sem_for_run['prototype']}")
    else:
        lambda_sem_for_run = {"max_logit": cfg["PROVISIONAL_LAMBDA_SEM_ML"],
                               "prototype": cfg["PROVISIONAL_LAMBDA_SEM_PR"]}
        print(f"[stage-verify:{round_tag}] using the provisional lambda_sem: "
              f"ml={lambda_sem_for_run['max_logit']}, pr={lambda_sem_for_run['prototype']}")

    for stage_idx in cfg["STAGE_CANDIDATES"]:
        stage_key = f"stage{stage_idx}"
        native_grid = _native_grid[stage_idx]
        matched_dim = _matched_dim[stage_idx]
        state.setdefault(stage_key, {})
        print(f"\n[stage-verify:{round_tag}] STAGE={stage_idx} (native grid {native_grid}x{native_grid}, "
              f"dim={matched_dim}, total={native_grid*native_grid*matched_dim})")

        for seed in verify_seeds:
            seed_key = str(seed)
            _cached = state[stage_key].get(seed_key)
            _cache_valid = (_cached is not None
                             and _cached.get("_lambda_sem_ml") == lambda_sem_for_run["max_logit"]
                             and _cached.get("_lambda_sem_pr") == lambda_sem_for_run["prototype"]
                             and _lambda_rec_upstream_ok(_cached, "_lambda_rec_ml", cfg["LAMBDA_REC_ML"], round_idx)
                             and _lambda_rec_upstream_ok(_cached, "_lambda_rec_pr", cfg["LAMBDA_REC_PR"], round_idx)
                             and _upstream_train_ok(_cached, "ml", cfg, round_idx)
                             and _upstream_train_ok(_cached, "pr", cfg, round_idx))
            if _cache_valid:
                print(f"  seed={seed}: SKIP, already done (resumed)")
                continue
            elif _cached is not None:
                print(f"  seed={seed}: cached result was trained under different settings -- re-running.")
            print(f"  seed={seed}: training both systems...")

            _orig = {k: cfg[k] for k in ("PATCH_GRID_ML", "PATCH_DIM_ML", "PATCH_GRID_PR",
                                          "PATCH_DIM_PR", "BACKBONE_STAGE_ML", "BACKBONE_STAGE_PR")}
            cfg["PATCH_GRID_ML"] = cfg["PATCH_GRID_PR"] = native_grid
            cfg["PATCH_DIM_ML"]  = cfg["PATCH_DIM_PR"]  = matched_dim
            cfg["BACKBONE_STAGE_ML"] = cfg["BACKBONE_STAGE_PR"] = stage_idx
            try:
                results_this_seed = {}
                for det_name in ["max_logit", "prototype"]:
                    set_seed(seed)
                    tag = f"stageverify_{round_tag}_{stage_key}_{det_name}_seed{seed}"
                    r = train_one_system(cfg, device, loaders, num_classes, nsym, det_name,
                                          lambda_sem_for_run[det_name], cfg["GRID_SEARCH_PATIENCE"],
                                          verify_dir, tag)
                    jscc, detector = r["jscc"], r["detector"]
                    jscc.eval(); detector.eval()
                    tau = calibrate_tau(jscc, detector, loaders["val_mixed"], device,
                                         target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
                    sdop = calibrate_sdop_threshold(jscc, loaders["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
                    r_eval = _eval_one_system(jscc, detector, tau, sdop, loaders["val_mixed"],
                                               cfg["DEFAULT_SNR_DB"], cfg, device, "awgn")
                    results_this_seed[det_name] = float(r_eval["auc_roc_pixel"])
                    for suffix in ("_jscc.pt", "_detector.pt"):
                        (verify_dir / f"{tag}{suffix}").unlink(missing_ok=True)
            finally:
                for k, v in _orig.items():
                    cfg[k] = v

            state[stage_key][seed_key] = results_this_seed
            state[stage_key][seed_key]["_lambda_sem_ml"] = lambda_sem_for_run["max_logit"]
            state[stage_key][seed_key]["_lambda_sem_pr"] = lambda_sem_for_run["prototype"]
            state[stage_key][seed_key]["_lambda_rec_ml"] = cfg["LAMBDA_REC_ML"]
            state[stage_key][seed_key]["_lambda_rec_pr"] = cfg["LAMBDA_REC_PR"]
            _record_upstream_train_keys(state[stage_key][seed_key], cfg, "ml")
            _record_upstream_train_keys(state[stage_key][seed_key], cfg, "pr")
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            print(f"  seed={seed}: done, ML_AUC={results_this_seed['max_logit']:.4f}, "
                  f"PR_AUC={results_this_seed['prototype']:.4f}, checkpoints deleted")

    print(f"\n[stage-verify:{round_tag}] Results at N={cfg['STAGE_VERIFY_N']}:")
    results_by_stage = {}
    for stage_idx in cfg["STAGE_CANDIDATES"]:
        stage_key = f"stage{stage_idx}"
        ml_aucs = [state[stage_key][str(s)]["max_logit"] for s in verify_seeds]
        pr_aucs = [state[stage_key][str(s)]["prototype"] for s in verify_seeds]
        results_by_stage[stage_key] = {"max_logit": ml_aucs, "prototype": pr_aucs}
        print(f"  {stage_key} (grid={_native_grid[stage_idx]}, dim={_matched_dim[stage_idx]}): ML mean={statistics.mean(ml_aucs):.4f} "
              f"std={statistics.stdev(ml_aucs):.4f}, PR mean={statistics.mean(pr_aucs):.4f} "
              f"std={statistics.stdev(pr_aucs):.4f}")

    stage_keys = list(results_by_stage.keys())
    per_detector_winners = {}
    for det_name in ("max_logit", "prototype"):
        det_scores = {k: v[det_name] for k, v in results_by_stage.items()}
        label = "Max Logit" if det_name == "max_logit" else "Prototype"

        # Reported test: Wilcoxon for two candidates, Friedman permutation test otherwise.
        if len(stage_keys) < 2:
            print(f"[stage-verify:{round_tag}] {label}: only one candidate in "
                  f"STAGE_CANDIDATES -- no comparison to make.")
        elif len(stage_keys) == 2:
            a_vals, b_vals = det_scores[stage_keys[0]], det_scores[stage_keys[1]]
            diffs = [a - b for a, b in zip(a_vals, b_vals)]
            try:
                _, p_val = stats.wilcoxon(a_vals, b_vals)
            except ValueError:
                _, p_val = 0.0, 1.0   # all paired differences are zero
            effect_r = rank_biserial_wilcoxon(diffs)
            print(f"[stage-verify:{round_tag}] {label} -- Wilcoxon signed-rank (paired by seed, "
                  f"{stage_keys[0]} vs {stage_keys[1]}): p={p_val:.4f}, effect size "
                  f"(matched-pairs rank-biserial r)={effect_r:.3f} ({interpret_rank_biserial(effect_r)})")
        else:
            matrix = [[det_scores[k][i] for k in stage_keys] for i in range(len(verify_seeds))]
            h_stat, p_val = friedman_permutation_test(matrix)
            effect_size = kendalls_w(matrix)
            effect_label, effect_interp = "Kendall's W", interpret_kendalls_w(
                effect_size, n=len(verify_seeds), k=len(cfg["STAGE_CANDIDATES"]))
            print(f"[stage-verify:{round_tag}] {label} -- Friedman permutation test (paired by seed, "
                  f"20000 permutations) across {len(cfg['STAGE_CANDIDATES'])} candidates: p={p_val:.4f}, "
                  f"effect size ({effect_label})={effect_size:.4f} ({effect_interp})")

        best_key = max(det_scores, key=lambda k: statistics.mean(det_scores[k]))
        best_stage = int(best_key.replace("stage", ""))
        cfg_key = "BACKBONE_STAGE_ML" if det_name == "max_logit" else "BACKBONE_STAGE_PR"
        grid_key, dim_key = (("PATCH_GRID_ML", "PATCH_DIM_ML") if det_name == "max_logit"
                              else ("PATCH_GRID_PR", "PATCH_DIM_PR"))
        incumbent_stage = cfg[cfg_key]
        final_stage = best_stage
        print(f"[stage-verify:{round_tag}] {label}: highest mean at N={len(det_scores[best_key])} "
              f"is stage {best_stage} (previous: stage {incumbent_stage}).")

        per_detector_winners[det_name] = {"best_stage": final_stage, "previous_stage": incumbent_stage}
        if final_stage != cfg[cfg_key]:
            print(f"[stage-verify:{round_tag}] *** {label}'s BACKBONE_STAGE changing from "
                  f"{cfg[cfg_key]} to {final_stage}; PATCH_GRID/PATCH_DIM set to "
                  f"({_native_grid[final_stage]},{_matched_dim[final_stage]}) and Step A skipped "
                  f"for this detector. ***")
            cfg[cfg_key]  = final_stage
            cfg[grid_key] = _native_grid[final_stage]
            cfg[dim_key]  = _matched_dim[final_stage]
        else:
            print(f"[stage-verify:{round_tag}] {label}: existing stage {final_stage} confirmed at "
                  f"N={cfg['STAGE_VERIFY_N']}.")
    return per_detector_winners


def _verify_patch_config(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_override=None, round_idx=0):
    """Step A: (patch_grid, patch_dim) of each detector from
    cfg["PATCH_CONFIG_CANDIDATES"] (stage-2 grids), on cfg["PATCH_CONFIG_VERIFY_N"]
    seeds. Skipped for a detector whose stage is not 2. Updates cfg in place."""
    from src.metrics import calibrate_tau, calibrate_sdop_threshold
    from scipy import stats
    from src.evaluate import _eval_one_system

    round_tag = f"round{round_idx}"
    state_path = verify_dir.parent / f"patch_config_verify_state_{round_tag}.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    if state:
        n_done = sum(len(v) for v in state.values())
        print(f"[patch-verify:{round_tag}] RESUMING -- found {n_done} already-completed "
              f"(config, seed) result(s) at {state_path}")

    verify_seeds = cfg["SELECTION_SEEDS"][:cfg["PATCH_CONFIG_VERIFY_N"]]
    if lambda_sem_override is not None:
        lambda_sem_for_run = lambda_sem_override
        print(f"[patch-verify:{round_tag}] using the previous round's lambda_sem winners: "
              f"ml={lambda_sem_for_run['max_logit']}, pr={lambda_sem_for_run['prototype']}")
    else:
        lambda_sem_for_run = {"max_logit": cfg["PROVISIONAL_LAMBDA_SEM_ML"],
                               "prototype": cfg["PROVISIONAL_LAMBDA_SEM_PR"]}
        print(f"[patch-verify:{round_tag}] using the provisional lambda_sem: "
              f"ml={lambda_sem_for_run['max_logit']}, pr={lambda_sem_for_run['prototype']}")

    _skip_ml = cfg.get("BACKBONE_STAGE_ML", 2) != 2
    _skip_pr = cfg.get("BACKBONE_STAGE_PR", 2) != 2
    if _skip_ml or _skip_pr:
        _skipped = ", ".join(d for d, s in [("Max Logit", _skip_ml), ("Prototype", _skip_pr)] if s)
        print(f"[patch-verify:{round_tag}] SKIPPING for: {_skipped} -- BACKBONE_STAGE moved away "
              f"from 2 for this detector; (14,8)/(8,16) do not apply to a different stage.")

    for patch_grid, patch_dim in cfg["PATCH_CONFIG_CANDIDATES"]:
        config_key = f"{patch_grid}x{patch_grid}_{patch_dim}"
        state.setdefault(config_key, {})
        print(f"\n[patch-verify:{round_tag}] PATCH_GRID={patch_grid}, PATCH_DIM={patch_dim}")

        for seed in verify_seeds:
            seed_key = str(seed)
            _cached = state[config_key].get(seed_key)
            _cache_valid = (_cached is not None
                             and _cached.get("_lambda_sem_ml") == lambda_sem_for_run["max_logit"]
                             and _cached.get("_lambda_sem_pr") == lambda_sem_for_run["prototype"]
                             and _lambda_rec_upstream_ok(_cached, "_lambda_rec_ml", cfg["LAMBDA_REC_ML"], round_idx)
                             and _lambda_rec_upstream_ok(_cached, "_lambda_rec_pr", cfg["LAMBDA_REC_PR"], round_idx)
                             and _upstream_train_ok(_cached, "ml", cfg, round_idx)
                             and _upstream_train_ok(_cached, "pr", cfg, round_idx))
            if _cache_valid:
                print(f"  seed={seed}: SKIP, already done (resumed)")
                continue
            elif _cached is not None:
                print(f"  seed={seed}: cached result was trained under different settings -- re-running.")
            print(f"  seed={seed}: training both systems...")

            _orig_grid_ml, _orig_dim_ml = cfg["PATCH_GRID_ML"], cfg["PATCH_DIM_ML"]
            _orig_grid_pr, _orig_dim_pr = cfg["PATCH_GRID_PR"], cfg["PATCH_DIM_PR"]
            cfg["PATCH_GRID_ML"], cfg["PATCH_DIM_ML"] = patch_grid, patch_dim
            cfg["PATCH_GRID_PR"], cfg["PATCH_DIM_PR"] = patch_grid, patch_dim
            try:
                results_this_seed = {}
                for det_name in ["max_logit", "prototype"]:
                    if (det_name == "max_logit" and _skip_ml) or (det_name == "prototype" and _skip_pr):
                        continue
                    set_seed(seed)
                    tag = f"patchverify_{round_tag}_{config_key}_{det_name}_seed{seed}"
                    r = train_one_system(cfg, device, loaders, num_classes, nsym, det_name,
                                          lambda_sem_for_run[det_name], cfg["GRID_SEARCH_PATIENCE"],
                                          verify_dir, tag)
                    jscc, detector = r["jscc"], r["detector"]
                    jscc.eval(); detector.eval()
                    tau = calibrate_tau(jscc, detector, loaders["val_mixed"], device,
                                         target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
                    sdop = calibrate_sdop_threshold(jscc, loaders["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
                    r_eval = _eval_one_system(jscc, detector, tau, sdop, loaders["val_mixed"],
                                               cfg["DEFAULT_SNR_DB"], cfg, device, "awgn")
                    results_this_seed[det_name] = float(r_eval["auc_roc_pixel"])
                    for suffix in ("_jscc.pt", "_detector.pt"):
                        (verify_dir / f"{tag}{suffix}").unlink(missing_ok=True)
            finally:
                cfg["PATCH_GRID_ML"], cfg["PATCH_DIM_ML"] = _orig_grid_ml, _orig_dim_ml
                cfg["PATCH_GRID_PR"], cfg["PATCH_DIM_PR"] = _orig_grid_pr, _orig_dim_pr

            state[config_key][seed_key] = results_this_seed
            state[config_key][seed_key]["_lambda_sem_ml"] = lambda_sem_for_run["max_logit"]
            state[config_key][seed_key]["_lambda_sem_pr"] = lambda_sem_for_run["prototype"]
            state[config_key][seed_key]["_lambda_rec_ml"] = cfg["LAMBDA_REC_ML"]
            state[config_key][seed_key]["_lambda_rec_pr"] = cfg["LAMBDA_REC_PR"]
            _record_upstream_train_keys(state[config_key][seed_key], cfg, "ml")
            _record_upstream_train_keys(state[config_key][seed_key], cfg, "pr")
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            _ml_str = (f"ML_AUC={results_this_seed['max_logit']:.4f}" if "max_logit" in results_this_seed
                       else "ML SKIPPED (stage != 2)")
            _pr_str = (f"PR_AUC={results_this_seed['prototype']:.4f}" if "prototype" in results_this_seed
                       else "PR SKIPPED (stage != 2)")
            print(f"  seed={seed}: done, {_ml_str}, {_pr_str}, checkpoints deleted")

    print(f"\n[patch-verify:{round_tag}] Results at N={cfg['PATCH_CONFIG_VERIFY_N']}:")
    results_by_config = {}
    for patch_grid, patch_dim in cfg["PATCH_CONFIG_CANDIDATES"]:
        config_key = f"{patch_grid}x{patch_grid}_{patch_dim}"
        # A detector skipped this round has no entries.
        ml_aucs = [state[config_key][str(s)]["max_logit"] for s in verify_seeds
                   if "max_logit" in state[config_key][str(s)]]
        pr_aucs = [state[config_key][str(s)]["prototype"] for s in verify_seeds
                   if "prototype" in state[config_key][str(s)]]
        results_by_config[config_key] = {"max_logit": ml_aucs, "prototype": pr_aucs}
        if ml_aucs:
            print(f"  {config_key}: ML mean={statistics.mean(ml_aucs):.4f} std={statistics.stdev(ml_aucs):.4f}", end="")
        else:
            print(f"  {config_key}: ML SKIPPED (stage != 2)", end="")
        if pr_aucs:
            print(f", PR mean={statistics.mean(pr_aucs):.4f} std={statistics.stdev(pr_aucs):.4f}")
        else:
            print(", PR SKIPPED (stage != 2)")

    config_keys = list(results_by_config.keys())
    per_detector_winners = {}
    for det_name in ("max_logit", "prototype"):
        if (det_name == "max_logit" and _skip_ml) or (det_name == "prototype" and _skip_pr):
            print(f"[patch-verify:{round_tag}] {'Max Logit' if det_name=='max_logit' else 'Prototype'}: "
                  f"skipped this round -- Step 0 set its PATCH_GRID/PATCH_DIM.")
            continue
        det_scores = {k: v[det_name] for k, v in results_by_config.items()}
        if len(config_keys) == 2:
            a_vals, b_vals = det_scores[config_keys[0]], det_scores[config_keys[1]]
            diffs = [a - b for a, b in zip(a_vals, b_vals)]
            try:
                _, p = stats.wilcoxon(a_vals, b_vals)
            except ValueError:
                _, p = 0.0, 1.0   # all paired differences are zero
            effect_r = rank_biserial_wilcoxon(diffs)
            label = "Max Logit" if det_name == "max_logit" else "Prototype"
            print(f"[patch-verify:{round_tag}] {label} -- Wilcoxon signed-rank (paired by seed): "
                  f"p={p:.4f}, effect size (matched-pairs rank-biserial r)={effect_r:.3f} "
                  f"({interpret_rank_biserial(effect_r)})")
        best_key = max(config_keys, key=lambda k: statistics.mean(det_scores[k]))
        best_grid, best_dim = [(g, d) for g, d in cfg["PATCH_CONFIG_CANDIDATES"] if f"{g}x{g}_{d}" == best_key][0]
        grid_cfg_key, dim_cfg_key = (f"PATCH_GRID_{'ML' if det_name == 'max_logit' else 'PR'}",
                                      f"PATCH_DIM_{'ML' if det_name == 'max_logit' else 'PR'}")
        label = "Max Logit" if det_name == "max_logit" else "Prototype"
        prev = (cfg[grid_cfg_key], cfg[dim_cfg_key])
        per_detector_winners[det_name] = {"best_config": best_key, "previous_value": prev}
        if (best_grid, best_dim) != prev:
            print(f"[patch-verify:{round_tag}] *** {label}'s PATCH_GRID/PATCH_DIM changing from "
                  f"{prev} to ({best_grid},{best_dim}) based on N={cfg['PATCH_CONFIG_VERIFY_N']} "
                  f"verification ***")
            cfg[grid_cfg_key], cfg[dim_cfg_key] = best_grid, best_dim
        else:
            print(f"[patch-verify:{round_tag}] {label}: existing value {prev} confirmed at "
                  f"N={cfg['PATCH_CONFIG_VERIFY_N']}.")

    verify_summary = {"results_by_config": {k: {"max_logit": v["max_logit"], "prototype": v["prototype"]}
                                              for k, v in results_by_config.items()},
                       "per_detector": per_detector_winners}
    with open(verify_dir.parent / f"patch_config_verify_summary_{round_tag}.json", "w") as f:
        json.dump(verify_summary, f, indent=2)

    return {"max_logit": (cfg["PATCH_GRID_ML"], cfg["PATCH_DIM_ML"]),
            "prototype": (cfg["PATCH_GRID_PR"], cfg["PATCH_DIM_PR"])}, verify_summary


def _verify_joint_lr(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_override=None, round_idx=0):
    """Step B: JOINT_LR of each detector from cfg["JOINT_LR_CANDIDATES"], on
    cfg["JOINT_LR_VERIFY_N"] seeds. A cached result is reused per detector when that
    detector's settings are unchanged. Updates cfg in place."""
    from src.metrics import calibrate_tau, calibrate_sdop_threshold
    from src.evaluate import _eval_one_system

    round_tag = f"round{round_idx}"
    state_path = verify_dir.parent / f"joint_lr_verify_state_{round_tag}.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    if state:
        n_done = sum(len(v) for v in state.values())
        print(f"[lr-verify:{round_tag}] RESUMING -- found {n_done} already-completed (lr, seed) result(s) at {state_path}")

    verify_seeds = cfg["SELECTION_SEEDS"][:cfg["JOINT_LR_VERIFY_N"]]
    if lambda_sem_override is not None:
        lambda_sem_for_run = lambda_sem_override
        print(f"[lr-verify:{round_tag}] using the previous round's lambda_sem winners: "
              f"ml={lambda_sem_for_run['max_logit']}, pr={lambda_sem_for_run['prototype']}")
    else:
        lambda_sem_for_run = {"max_logit": cfg["PROVISIONAL_LAMBDA_SEM_ML"],
                               "prototype": cfg["PROVISIONAL_LAMBDA_SEM_PR"]}
        print(f"[lr-verify:{round_tag}] using the provisional lambda_sem: "
              f"ml={lambda_sem_for_run['max_logit']}, pr={lambda_sem_for_run['prototype']}")

    for lr in cfg["JOINT_LR_CANDIDATES"]:
        lr_key = str(lr)
        state.setdefault(lr_key, {})
        print(f"\n[lr-verify:{round_tag}] JOINT_LR={lr}")
        for seed in verify_seeds:
            seed_key = str(seed)
            _cached = state[lr_key].get(seed_key, {})
            _valid = {}
            _suffix = {"max_logit": "ml", "prototype": "pr"}
            for det_name, grid_key, dim_key in [
                    ("max_logit", "PATCH_GRID_ML", "PATCH_DIM_ML"),
                    ("prototype", "PATCH_GRID_PR", "PATCH_DIM_PR")]:
                sfx = _suffix[det_name]
                # JOINT_LR is excluded: cfg still holds the previous candidate here.
                _valid[det_name] = (
                    det_name in _cached
                    and _cached.get(f"_patch_grid_{sfx}") == cfg[grid_key]
                    and _cached.get(f"_patch_dim_{sfx}") == cfg[dim_key]
                    and _cached.get(f"_lambda_sem_{sfx}") == lambda_sem_for_run[det_name]
                    and _lambda_rec_upstream_ok(_cached, f"_lambda_rec_{sfx}", cfg[f"LAMBDA_REC_{sfx.upper()}"], round_idx)
                    and _upstream_train_ok(_cached, sfx, cfg, round_idx, exclude={"JOINT_LR"}))

            if _valid["max_logit"] and _valid["prototype"]:
                print(f"  seed={seed}: SKIP, already done (resumed)")
                results_this_seed = {"max_logit": _cached["max_logit"], "prototype": _cached["prototype"]}
            else:
                for det_name in ["max_logit", "prototype"]:
                    if _valid[det_name]:
                        print(f"  seed={seed}, {det_name}: SKIP, already done (resumed)")
                    elif det_name in _cached:
                        print(f"  seed={seed}, {det_name}: cached result was trained under different "
                              f"settings -- re-running.")
                    else:
                        print(f"  seed={seed}, {det_name}: training...")

                _orig_lr_ml = cfg["JOINT_LR_ML"]
                _orig_lr_pr = cfg["JOINT_LR_PR"]
                cfg["JOINT_LR_ML"] = lr
                cfg["JOINT_LR_PR"] = lr
                try:
                    results_this_seed = {}
                    for det_name in ["max_logit", "prototype"]:
                        if _valid[det_name]:
                            results_this_seed[det_name] = _cached[det_name]
                            continue
                        set_seed(seed)
                        tag = f"lrverify_{round_tag}_{lr_key}_{det_name}_seed{seed}"
                        r = train_one_system(cfg, device, loaders, num_classes, nsym, det_name,
                                              lambda_sem_for_run[det_name], cfg["GRID_SEARCH_PATIENCE"],
                                              verify_dir, tag)
                        jscc, detector = r["jscc"], r["detector"]
                        jscc.eval(); detector.eval()
                        tau = calibrate_tau(jscc, detector, loaders["val_mixed"], device,
                                             target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
                        sdop = calibrate_sdop_threshold(jscc, loaders["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
                        r_eval = _eval_one_system(jscc, detector, tau, sdop, loaders["val_mixed"],
                                                   cfg["DEFAULT_SNR_DB"], cfg, device, "awgn")
                        results_this_seed[det_name] = float(r_eval["auc_roc_pixel"])
                        for suffix in ("_jscc.pt", "_detector.pt"):
                            (verify_dir / f"{tag}{suffix}").unlink(missing_ok=True)
                        print(f"  seed={seed}, {det_name}: done, AUC={results_this_seed[det_name]:.4f}, checkpoints deleted")
                finally:
                    cfg["JOINT_LR_ML"] = _orig_lr_ml
                    cfg["JOINT_LR_PR"] = _orig_lr_pr

            # The learning rate is recorded as `lr`: cfg has been restored to the previous value.
            state[lr_key][seed_key] = {
                "max_logit": results_this_seed["max_logit"],
                "prototype": results_this_seed["prototype"],
                "_patch_grid_ml": cfg["PATCH_GRID_ML"], "_patch_dim_ml": cfg["PATCH_DIM_ML"],
                "_patch_grid_pr": cfg["PATCH_GRID_PR"], "_patch_dim_pr": cfg["PATCH_DIM_PR"],
                "_lambda_sem_ml": lambda_sem_for_run["max_logit"], "_lambda_sem_pr": lambda_sem_for_run["prototype"],
                "_lambda_rec_ml": cfg["LAMBDA_REC_ML"], "_lambda_rec_pr": cfg["LAMBDA_REC_PR"],
                "_joint_lr_ml": lr, "_joint_lr_pr": lr}
            _record_upstream_train_keys(state[lr_key][seed_key], cfg, "ml", exclude={"JOINT_LR"})
            _record_upstream_train_keys(state[lr_key][seed_key], cfg, "pr", exclude={"JOINT_LR"})
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            if not (_valid["max_logit"] and _valid["prototype"]):
                print(f"  seed={seed}: done, ML_AUC={results_this_seed['max_logit']:.4f}, "
                      f"PR_AUC={results_this_seed['prototype']:.4f}, checkpoints deleted")

    print(f"\n[lr-verify:{round_tag}] Results at N={cfg['JOINT_LR_VERIFY_N']}:")
    results_by_lr = {}
    for lr in cfg["JOINT_LR_CANDIDATES"]:
        lr_key = str(lr)
        ml_aucs = [state[lr_key][str(s)]["max_logit"] for s in verify_seeds]
        pr_aucs = [state[lr_key][str(s)]["prototype"] for s in verify_seeds]
        results_by_lr[lr] = {"max_logit": ml_aucs, "prototype": pr_aucs}
        print(f"  lr={lr}: ML mean={statistics.mean(ml_aucs):.4f} std={statistics.stdev(ml_aucs):.4f}, "
              f"PR mean={statistics.mean(pr_aucs):.4f} std={statistics.stdev(pr_aucs):.4f}")

    per_detector_winners = {}
    for det_name in ("max_logit", "prototype"):
        det_scores = {lr: v[det_name] for lr, v in results_by_lr.items()}
        lr_order = list(det_scores.keys())
        matrix = [[det_scores[lr][i] for lr in lr_order] for i in range(len(verify_seeds))]   # seeds x candidates
        h_stat, p_kw = friedman_permutation_test(matrix)
        eps2 = kendalls_w(matrix)
        best_lr = max(det_scores, key=lambda lr: statistics.mean(det_scores[lr]))
        cfg_key = "JOINT_LR_ML" if det_name == "max_logit" else "JOINT_LR_PR"
        label = "Max Logit" if det_name == "max_logit" else "Prototype"
        print(f"[lr-verify:{round_tag}] {label} -- Friedman permutation test (paired by seed, 20000 permutations) across "
              f"{len(cfg['JOINT_LR_CANDIDATES'])} candidates: stat={h_stat:.3f}, p={p_kw:.4f}, "
              f"effect size (Kendall's W)={eps2:.4f} "
              f"({interpret_kendalls_w(eps2, n=len(verify_seeds), k=len(cfg['JOINT_LR_CANDIDATES']))})")
        print(f"[lr-verify:{round_tag}] {label} best at N={cfg['JOINT_LR_VERIFY_N']}: {best_lr} "
              f"(previous config value: {cfg[cfg_key]})")
        incumbent_lr = cfg[cfg_key]
        final_lr = best_lr
        # The summary keys "kruskal_p" and "epsilon_squared" hold the Friedman
        # permutation p-value and Kendall's W.
        per_detector_winners[det_name] = {"best_lr": final_lr, "kruskal_p": p_kw,
                                           "epsilon_squared": eps2, "previous_value": incumbent_lr}
        if final_lr != cfg[cfg_key]:
            print(f"[lr-verify:{round_tag}] *** {cfg_key} changing from {cfg[cfg_key]} to "
                  f"{final_lr} based on N={cfg['JOINT_LR_VERIFY_N']} verification ***")
            cfg[cfg_key] = final_lr
        else:
            print(f"[lr-verify:{round_tag}] {label}: existing value {cfg[cfg_key]} confirmed "
                  f"at N={cfg['JOINT_LR_VERIFY_N']}.")

    verify_summary = {"results_by_lr": {str(k): v for k, v in results_by_lr.items()},
                       "per_detector": per_detector_winners}
    with open(verify_dir.parent / f"joint_lr_verify_summary_{round_tag}.json", "w") as f:
        json.dump(verify_summary, f, indent=2)

    return (cfg["JOINT_LR_ML"], cfg["JOINT_LR_PR"]), verify_summary


def _verify_one_optimizer_param(cfg, device, loaders, num_classes, nsym, verify_dir,
                                  param_name, candidates, lambda_sem_for_run, round_tag, verify_n, round_idx=0):
    """One Step C setting (JOINT_MOMENTUM, JOINT_LR_POWER, WEIGHT_DECAY or
    MIN_REL_IMPROVE): the same candidate value is trained for both detectors on
    verify_n seeds, and each detector selects its own winner. The other settings
    keep each detector's current values. Updates cfg in place."""
    from src.metrics import calibrate_tau, calibrate_sdop_threshold
    from scipy import stats
    from src.evaluate import _eval_one_system

    state_path = verify_dir.parent / f"{param_name.lower()}_verify_state_{round_tag}.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    if state:
        n_done = sum(len(v) for v in state.values())
        print(f"[{param_name}-verify:{round_tag}] RESUMING -- found {n_done} already-completed "
              f"(value, seed) result(s) at {state_path}")

    verify_seeds = cfg["SELECTION_SEEDS"][:verify_n]
    for candidate in candidates:
        cand_key = str(candidate)
        state.setdefault(cand_key, {})
        print(f"\n[{param_name}-verify:{round_tag}] {param_name}={candidate}")
        for seed in verify_seeds:
            seed_key = str(seed)
            _cached = state[cand_key].get(seed_key)
            # Cache check: the swept setting itself is excluded (cfg still holds the
            # previous candidate here); PROTOTYPE_SCALE is checked from round 1 on.
            _other_optimizer_keys = {"JOINT_MOMENTUM": "_joint_momentum",
                                       "JOINT_LR_POWER": "_joint_lr_power",
                                       "WEIGHT_DECAY": "_weight_decay",
                                       "MIN_REL_IMPROVE": "_min_rel_improve"}
            _other_optimizer_keys.pop(param_name)

            def _valid_for(sfx, det_name):
                if _cached is None:
                    return False
                ok = (_cached.get(f"_patch_grid_{sfx}") == cfg[f"PATCH_GRID_{sfx.upper()}"]
                      and _cached.get(f"_patch_dim_{sfx}") == cfg[f"PATCH_DIM_{sfx.upper()}"]
                      and _cached.get(f"_joint_lr_{sfx}") == cfg[f"JOINT_LR_{sfx.upper()}"]
                      and _cached.get(f"_lambda_sem_{sfx}") == lambda_sem_for_run[det_name]
                      and _lambda_rec_upstream_ok(_cached, f"_lambda_rec_{sfx}", cfg[f"LAMBDA_REC_{sfx.upper()}"], round_idx)
                      and all(_cached.get(f"{k}_{sfx}") == cfg[f"{p}_{sfx.upper()}"]
                              for p, k in _other_optimizer_keys.items())
                      and det_name in _cached)
                if det_name == "prototype" and round_idx > 0:
                    ok = ok and _cached.get("_prototype_scale") == cfg["PROTOTYPE_SCALE"]
                return ok

            _valid = {"max_logit": _valid_for("ml", "max_logit"),
                      "prototype": _valid_for("pr", "prototype")}

            _used_optimizer_vals = {}
            for p, k in {"JOINT_MOMENTUM": "_joint_momentum", "JOINT_LR_POWER": "_joint_lr_power",
                         "WEIGHT_DECAY": "_weight_decay", "MIN_REL_IMPROVE": "_min_rel_improve"}.items():
                if p == param_name:
                    continue
                _used_optimizer_vals[f"{k}_ml"] = cfg[f"{p}_ML"]
                _used_optimizer_vals[f"{k}_pr"] = cfg[f"{p}_PR"]

            _all_param_keys = {"JOINT_MOMENTUM": "_joint_momentum", "JOINT_LR_POWER": "_joint_lr_power",
                                "WEIGHT_DECAY": "_weight_decay", "MIN_REL_IMPROVE": "_min_rel_improve"}
            if _valid["max_logit"] and _valid["prototype"]:
                print(f"  seed={seed}: SKIP, already done (resumed)")
                results_this_seed = {"max_logit": _cached["max_logit"], "prototype": _cached["prototype"]}
                _used_optimizer_vals[f"{_all_param_keys[param_name]}_ml"] = candidate
                _used_optimizer_vals[f"{_all_param_keys[param_name]}_pr"] = candidate
            else:
                for det_name in ["max_logit", "prototype"]:
                    if _valid[det_name]:
                        print(f"  seed={seed}, {det_name}: SKIP, already done (resumed)")
                    elif _cached is not None and det_name in _cached:
                        print(f"  seed={seed}, {det_name}: cached result was trained under different settings -- re-running.")
                    else:
                        print(f"  seed={seed}, {det_name}: training...")

                _orig_val_ml, _orig_val_pr = cfg[f"{param_name}_ML"], cfg[f"{param_name}_PR"]
                cfg[f"{param_name}_ML"] = candidate
                cfg[f"{param_name}_PR"] = candidate
                try:
                    # Record the settings as trained, before `finally` restores the swept one.
                    _used_optimizer_vals = {}
                    for p, k in {"JOINT_MOMENTUM": "_joint_momentum", "JOINT_LR_POWER": "_joint_lr_power",
                                 "WEIGHT_DECAY": "_weight_decay", "MIN_REL_IMPROVE": "_min_rel_improve"}.items():
                        _used_optimizer_vals[f"{k}_ml"] = cfg[f"{p}_ML"]
                        _used_optimizer_vals[f"{k}_pr"] = cfg[f"{p}_PR"]
                    results_this_seed = {}
                    for det_name in ["max_logit", "prototype"]:
                        if _valid[det_name]:
                            results_this_seed[det_name] = _cached[det_name]
                            continue
                        set_seed(seed)
                        tag = f"{param_name.lower()}verify_{round_tag}_{cand_key}_{det_name}_seed{seed}"
                        r = train_one_system(cfg, device, loaders, num_classes, nsym, det_name,
                                              lambda_sem_for_run[det_name], cfg["GRID_SEARCH_PATIENCE"],
                                              verify_dir, tag)
                        jscc, detector = r["jscc"], r["detector"]
                        jscc.eval(); detector.eval()
                        tau = calibrate_tau(jscc, detector, loaders["val_mixed"], device,
                                             target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
                        sdop = calibrate_sdop_threshold(jscc, loaders["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
                        r_eval = _eval_one_system(jscc, detector, tau, sdop, loaders["val_mixed"],
                                                   cfg["DEFAULT_SNR_DB"], cfg, device, "awgn")
                        results_this_seed[det_name] = float(r_eval["auc_roc_pixel"])
                        for suffix in ("_jscc.pt", "_detector.pt"):
                            (verify_dir / f"{tag}{suffix}").unlink(missing_ok=True)
                        print(f"  seed={seed}, {det_name}: done, AUC={results_this_seed[det_name]:.4f}, checkpoints deleted")
                finally:
                    cfg[f"{param_name}_ML"] = _orig_val_ml
                    cfg[f"{param_name}_PR"] = _orig_val_pr

            results_this_seed["_patch_grid_ml"] = cfg["PATCH_GRID_ML"]
            results_this_seed["_patch_dim_ml"] = cfg["PATCH_DIM_ML"]
            results_this_seed["_patch_grid_pr"] = cfg["PATCH_GRID_PR"]
            results_this_seed["_patch_dim_pr"] = cfg["PATCH_DIM_PR"]
            results_this_seed["_joint_lr_ml"] = cfg["JOINT_LR_ML"]
            results_this_seed["_joint_lr_pr"] = cfg["JOINT_LR_PR"]
            results_this_seed["_prototype_scale"] = cfg["PROTOTYPE_SCALE"]
            results_this_seed["_lambda_sem_ml"] = lambda_sem_for_run["max_logit"]
            results_this_seed["_lambda_sem_pr"] = lambda_sem_for_run["prototype"]
            results_this_seed["_lambda_rec_ml"] = cfg["LAMBDA_REC_ML"]
            results_this_seed["_lambda_rec_pr"] = cfg["LAMBDA_REC_PR"]
            results_this_seed.update(_used_optimizer_vals)
            state[cand_key][seed_key] = results_this_seed
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            if not (_valid["max_logit"] and _valid["prototype"]):
                print(f"  seed={seed}: done, ML_AUC={results_this_seed['max_logit']:.4f}, "
                      f"PR_AUC={results_this_seed['prototype']:.4f}, checkpoints deleted")

    results_by_val = {}
    for candidate in candidates:
        cand_key = str(candidate)
        ml_aucs = [state[cand_key][str(s)]["max_logit"] for s in verify_seeds]
        pr_aucs = [state[cand_key][str(s)]["prototype"] for s in verify_seeds]
        results_by_val[candidate] = {"max_logit": ml_aucs, "prototype": pr_aucs}
        print(f"[{param_name}-verify:{round_tag}] {param_name}={candidate}: "
              f"ML mean={statistics.mean(ml_aucs):.4f}, PR mean={statistics.mean(pr_aucs):.4f}")

    per_detector_winners = {}
    for det_name in ("max_logit", "prototype"):
        det_scores = {c: v[det_name] for c, v in results_by_val.items()}
        cand_order = list(det_scores.keys())
        if len(candidates) > 2:
            matrix = [[det_scores[c][i] for c in cand_order] for i in range(len(verify_seeds))]
            h_stat, p_val = friedman_permutation_test(matrix)
            test_name = "Friedman permutation test (paired by seed, 20000 permutations)"
            effect_size = kendalls_w(matrix)
            effect_label, effect_interp = "Kendall's W", interpret_kendalls_w(
                effect_size, n=len(verify_seeds), k=len(candidates))
        else:
            a_vals, b_vals = det_scores[cand_order[0]], det_scores[cand_order[1]]
            diffs = [a - b for a, b in zip(a_vals, b_vals)]
            try:
                _, p_val = stats.wilcoxon(a_vals, b_vals)
            except ValueError:
                _, p_val = 0.0, 1.0   # all paired differences are zero
            test_name = "Wilcoxon signed-rank (paired by seed)"
            effect_size = rank_biserial_wilcoxon(diffs)
            effect_label, effect_interp = "matched-pairs rank-biserial r", interpret_rank_biserial(effect_size)
        best_val = max(det_scores, key=lambda v: statistics.mean(det_scores[v]))
        cfg_key = f"{param_name}_{'ML' if det_name == 'max_logit' else 'PR'}"
        label = "Max Logit" if det_name == "max_logit" else "Prototype"
        print(f"[{param_name}-verify:{round_tag}] {label} -- {test_name} across "
              f"{len(candidates)} candidates: p={p_val:.4f}, effect size ({effect_label})="
              f"{effect_size:.4f} ({effect_interp})")
        print(f"[{param_name}-verify:{round_tag}] {label} best at N={verify_n}: {best_val} "
              f"(previous config value: {cfg[cfg_key]})")
        incumbent_val = cfg[cfg_key]
        final_val = best_val
        per_detector_winners[det_name] = {"test_name": test_name, "p_value": p_val,
                                           "effect_size": effect_size, "effect_size_type": effect_label,
                                           "effect_size_interpretation": effect_interp,
                                           "best_value": final_val, "previous_value": incumbent_val}
        if final_val != cfg[cfg_key]:
            print(f"[{param_name}-verify:{round_tag}] *** {label}'s {param_name} changing from "
                  f"{cfg[cfg_key]} to {final_val} based on N={verify_n} verification ***")
            cfg[cfg_key] = final_val
        else:
            print(f"[{param_name}-verify:{round_tag}] {label}: existing value {cfg[cfg_key]} "
                  f"confirmed at N={verify_n}.")

    verify_summary = {"results_by_value": {str(k): v for k, v in results_by_val.items()},
                       "per_detector": per_detector_winners}
    with open(verify_dir.parent / f"{param_name.lower()}_verify_summary_{round_tag}.json", "w") as f:
        json.dump(verify_summary, f, indent=2)

    return {"max_logit": cfg[f"{param_name}_ML"], "prototype": cfg[f"{param_name}_PR"]}, verify_summary


def _verify_optimizer_hparams(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_override=None, round_idx=0):
    """Step C: coordinate search over JOINT_MOMENTUM, JOINT_LR_POWER and WEIGHT_DECAY
    (cfg["LR_POWER_WD_VERIFY_N"] seeds each), then MIN_REL_IMPROVE
    (cfg["OPTIMIZER_HPARAM_VERIFY_N"] seeds), each holding the values already
    settled."""
    round_tag = f"round{round_idx}"
    if lambda_sem_override is not None:
        lambda_sem_for_run = lambda_sem_override
    else:
        lambda_sem_for_run = {"max_logit": cfg["PROVISIONAL_LAMBDA_SEM_ML"],
                               "prototype": cfg["PROVISIONAL_LAMBDA_SEM_PR"]}

    verify_n = cfg["OPTIMIZER_HPARAM_VERIFY_N"]
    summaries = {}
    _optimizer_params = [("JOINT_MOMENTUM", cfg["JOINT_MOMENTUM_CANDIDATES"], cfg["LR_POWER_WD_VERIFY_N"]),
                          ("JOINT_LR_POWER", cfg["JOINT_LR_POWER_CANDIDATES"], cfg["LR_POWER_WD_VERIFY_N"]),
                          ("WEIGHT_DECAY", cfg["WEIGHT_DECAY_CANDIDATES"], cfg["LR_POWER_WD_VERIFY_N"]),
                          ("MIN_REL_IMPROVE", cfg["MIN_REL_IMPROVE_CANDIDATES"], verify_n)]
    for param_name, candidates, _n in _optimizer_params:
        _, summary = _verify_one_optimizer_param(cfg, device, loaders, num_classes, nsym, verify_dir,
                                                    param_name, candidates, lambda_sem_for_run, round_tag,
                                                    _n, round_idx=round_idx)
        summaries[param_name] = summary
    return summaries


def _verify_prototype_scale(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_override=None, round_idx=0):
    """Step D: PROTOTYPE_SCALE from cfg["PROTOTYPE_SCALE_GRID"] (Prototype only), on
    cfg["PROTOTYPE_SCALE_VERIFY_N"] seeds, at Prototype's current stage, grid and
    optimiser settings. Runs before the lambda_sem grid, which trains at the
    selected scale. Updates cfg["PROTOTYPE_SCALE"] in place."""
    from src.metrics import calibrate_tau, calibrate_sdop_threshold
    from src.evaluate import _eval_one_system

    round_tag = f"round{round_idx}"
    state_path = verify_dir.parent / f"prototype_scale_verify_state_{round_tag}.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    if state:
        n_done = sum(len(v) for v in state.values())
        print(f"[scale-verify:{round_tag}] RESUMING -- found {n_done} already-completed "
              f"(scale, seed) result(s) at {state_path}")

    verify_seeds = cfg["SELECTION_SEEDS"][:cfg["PROTOTYPE_SCALE_VERIFY_N"]]
    if lambda_sem_override is not None:
        lambda_sem_for_run = lambda_sem_override["prototype"]
        print(f"[scale-verify:{round_tag}] using the previous round's lambda_sem winner: pr={lambda_sem_for_run}")
    else:
        lambda_sem_for_run = cfg["PROVISIONAL_LAMBDA_SEM_PR"]
        print(f"[scale-verify:{round_tag}] using the provisional lambda_sem: pr={lambda_sem_for_run}")

    for scale in cfg["PROTOTYPE_SCALE_GRID"]:
        scale_key = str(scale)
        state.setdefault(scale_key, {})
        print(f"\n[scale-verify:{round_tag}] PROTOTYPE_SCALE={scale}")
        for seed in verify_seeds:
            seed_key = str(seed)
            _cached = state[scale_key].get(seed_key)
            _cache_valid = (_cached is not None
                             and _cached.get("_patch_grid_pr") == cfg["PATCH_GRID_PR"]
                             and _cached.get("_patch_dim_pr") == cfg["PATCH_DIM_PR"]
                             and _cached.get("_joint_lr_pr") == cfg["JOINT_LR_PR"]
                             and _cached.get("_joint_momentum_pr") == cfg["JOINT_MOMENTUM_PR"]
                             and _cached.get("_joint_lr_power_pr") == cfg["JOINT_LR_POWER_PR"]
                             and _cached.get("_weight_decay_pr") == cfg["WEIGHT_DECAY_PR"]
                             and _cached.get("_min_rel_improve_pr") == cfg["MIN_REL_IMPROVE_PR"]
                             and _lambda_rec_upstream_ok(_cached, "_lambda_rec_pr", cfg["LAMBDA_REC_PR"], round_idx)
                             and _cached.get("_lambda_sem_pr") == lambda_sem_for_run
                             and _cached.get("_backbone_stage_pr") == cfg.get("BACKBONE_STAGE_PR", 2))
            if _cache_valid:
                print(f"  seed={seed}: SKIP, already done (resumed)")
                continue
            elif _cached is not None:
                print(f"  seed={seed}: cached result was trained under different settings -- re-running.")
            set_seed(seed)
            print(f"  seed={seed}: training...")
            jscc = SwinJSCCSystem(nsym, pretrained=True,
                                   patch_grid=cfg["PATCH_GRID_PR"], patch_dim=cfg["PATCH_DIM_PR"],
                                   stage_idx=cfg.get("BACKBONE_STAGE_PR", 2)).to(device)
            detector = PrototypeDetector(cfg["PATCH_DIM_PR"], cfg["PATCH_GRID_PR"], num_classes, scale).to(device)
            tag = f"scaleverify_{round_tag}_{scale_key}_seed{seed}"
            train_joint_system(jscc, detector, loaders["cls_train"], loaders["cls_val"],
                                cfg, device, num_classes,
                                jscc_save_path=verify_dir / f"{tag}_jscc.pt",
                                detector_save_path=verify_dir / f"{tag}_detector.pt",
                                lambda_rec=cfg["LAMBDA_REC_PR"], lambda_sem=lambda_sem_for_run,
                                joint_lr=cfg["JOINT_LR_PR"],
                                joint_momentum=cfg["JOINT_MOMENTUM_PR"], weight_decay=cfg["WEIGHT_DECAY_PR"],
                                joint_lr_power=cfg["JOINT_LR_POWER_PR"],
                                patience=cfg["GRID_SEARCH_PATIENCE"], min_rel_improve=cfg["MIN_REL_IMPROVE_PR"],
                                max_epochs=cfg["JOINT_MAX_EPOCHS"])
            jscc.eval(); detector.eval()
            tau = calibrate_tau(jscc, detector, loaders["val_mixed"], device,
                                 target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
            sdop = calibrate_sdop_threshold(jscc, loaders["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
            r = _eval_one_system(jscc, detector, tau, sdop, loaders["val_mixed"],
                                  cfg["DEFAULT_SNR_DB"], cfg, device, "awgn")
            state[scale_key][seed_key] = {"auc": float(r["auc_roc_pixel"]),
                                            "_patch_grid_pr": cfg["PATCH_GRID_PR"], "_patch_dim_pr": cfg["PATCH_DIM_PR"],
                                            "_joint_lr_pr": cfg["JOINT_LR_PR"], "_joint_momentum_pr": cfg["JOINT_MOMENTUM_PR"],
                                            "_joint_lr_power_pr": cfg["JOINT_LR_POWER_PR"],
                                            "_weight_decay_pr": cfg["WEIGHT_DECAY_PR"],
                                            "_min_rel_improve_pr": cfg["MIN_REL_IMPROVE_PR"],
                                            "_lambda_rec_pr": cfg["LAMBDA_REC_PR"],
                                            "_lambda_sem_pr": lambda_sem_for_run,
                                            "_backbone_stage_pr": cfg.get("BACKBONE_STAGE_PR", 2)}
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            for suffix in ("_jscc.pt", "_detector.pt"):
                (verify_dir / f"{tag}{suffix}").unlink(missing_ok=True)
            print(f"  seed={seed}: done, AUC={r['auc_roc_pixel']:.4f}, checkpoints deleted")

    results_by_scale = {s: [state[str(s)][str(seed)]["auc"] for seed in verify_seeds]
                         for s in cfg["PROTOTYPE_SCALE_GRID"]}
    for s, aucs in results_by_scale.items():
        print(f"[scale-verify:{round_tag}] scale={s}: mean={statistics.mean(aucs):.4f}, "
              f"std={statistics.stdev(aucs):.4f}")

    scale_order = list(results_by_scale.keys())
    scale_matrix = [[results_by_scale[s][i] for s in scale_order] for i in range(len(verify_seeds))]
    h_stat, p_kw = friedman_permutation_test(scale_matrix)
    eps2_scale = kendalls_w(scale_matrix)
    best_scale = max(results_by_scale, key=lambda s: statistics.mean(results_by_scale[s]))
    print(f"[scale-verify:{round_tag}] Friedman permutation test (paired by seed, 20000 permutations) across {len(cfg['PROTOTYPE_SCALE_GRID'])} "
          f"candidates: stat={h_stat:.3f}, p={p_kw:.4f}, effect size (Kendall's W)={eps2_scale:.4f} "
          f"({interpret_kendalls_w(eps2_scale, n=len(verify_seeds), k=len(cfg['PROTOTYPE_SCALE_GRID']))})")
    print(f"[scale-verify:{round_tag}] Best at N={cfg['PROTOTYPE_SCALE_VERIFY_N']}: {best_scale} "
          f"(previous config value: {cfg['PROTOTYPE_SCALE']})")
    incumbent_scale = cfg["PROTOTYPE_SCALE"]
    final_scale = best_scale

    # "kruskal_p" and "effect_size_epsilon_sq" hold the Friedman permutation p-value and Kendall's W.
    verify_summary = {"results_by_scale": results_by_scale, "kruskal_p": p_kw,
                       "effect_size_epsilon_sq": eps2_scale,
                       "best_scale": final_scale, "previous_value": incumbent_scale}
    with open(verify_dir.parent / f"prototype_scale_verify_summary_{round_tag}.json", "w") as f:
        json.dump(verify_summary, f, indent=2)

    if final_scale != cfg["PROTOTYPE_SCALE"]:
        print(f"[scale-verify:{round_tag}] *** PROTOTYPE_SCALE changing from {cfg['PROTOTYPE_SCALE']} to "
              f"{final_scale} based on N={cfg['PROTOTYPE_SCALE_VERIFY_N']} verification ***")
        cfg["PROTOTYPE_SCALE"] = final_scale
    else:
        print(f"[scale-verify:{round_tag}] Existing value {cfg['PROTOTYPE_SCALE']} confirmed at "
              f"N={cfg['PROTOTYPE_SCALE_VERIFY_N']}.")
    return cfg["PROTOTYPE_SCALE"], verify_summary


def _verify_lambda_rec(cfg, device, loaders, num_classes, nsym, verify_dir, lambda_sem_for_run, round_idx=0):
    """Step F: LAMBDA_REC of each detector from cfg["LAMBDA_REC_SANITY_CANDIDATES"], on
    cfg["OPTIMIZER_HPARAM_VERIFY_N"] seeds, with this round's lambda_sem winners.

    Only the ratio lambda_sem / lambda_rec would matter if the loss scale did not
    interact with the optimiser; this step tests that. It runs last in a round, so
    its candidates are trained under every other final setting of the round, and the
    winner's per-seed AUROCs measure the round's complete configuration (used by the
    stopping rule). Updates cfg in place.
    """
    from src.metrics import calibrate_tau, calibrate_sdop_threshold
    from src.evaluate import _eval_one_system

    round_tag = f"round{round_idx}"
    state_path = verify_dir.parent / f"lambda_rec_verify_state_{round_tag}.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    if state:
        n_done = sum(len(v) for v in state.values())
        print(f"[LAMBDA_REC-verify:{round_tag}] RESUMING -- found {n_done} already-completed "
              f"(value, seed) result(s) at {state_path}")

    _initial_lambda_rec_ml = cfg["LAMBDA_REC_ML"]
    _initial_lambda_rec_pr = cfg["LAMBDA_REC_PR"]

    candidates = cfg["LAMBDA_REC_SANITY_CANDIDATES"]
    verify_n = cfg["OPTIMIZER_HPARAM_VERIFY_N"]
    verify_seeds = cfg["SELECTION_SEEDS"][:verify_n]

    for candidate in candidates:
        cand_key = str(candidate)
        state.setdefault(cand_key, {})
        print(f"\n[LAMBDA_REC-verify:{round_tag}] LAMBDA_REC={candidate}")
        for seed in verify_seeds:
            seed_key = str(seed)
            _cached = state[cand_key].get(seed_key)

            def _valid_for(sfx, det_name):
                if _cached is None:
                    return False
                ok = (_cached.get(f"_patch_grid_{sfx}") == cfg[f"PATCH_GRID_{sfx.upper()}"]
                      and _cached.get(f"_patch_dim_{sfx}") == cfg[f"PATCH_DIM_{sfx.upper()}"]
                      and _cached.get(f"_joint_lr_{sfx}") == cfg[f"JOINT_LR_{sfx.upper()}"]
                      and _cached.get(f"_joint_momentum_{sfx}") == cfg[f"JOINT_MOMENTUM_{sfx.upper()}"]
                      and _cached.get(f"_joint_lr_power_{sfx}") == cfg[f"JOINT_LR_POWER_{sfx.upper()}"]
                      and _cached.get(f"_weight_decay_{sfx}") == cfg[f"WEIGHT_DECAY_{sfx.upper()}"]
                      and _cached.get(f"_min_rel_improve_{sfx}") == cfg[f"MIN_REL_IMPROVE_{sfx.upper()}"]
                      and _cached.get(f"_lambda_sem_{det_name}") == lambda_sem_for_run[det_name]
                      and det_name in _cached)
                if det_name == "prototype" and round_idx > 0:
                    ok = ok and _cached.get("_prototype_scale") == cfg["PROTOTYPE_SCALE"]
                return ok

            _valid = {"max_logit": _valid_for("ml", "max_logit"),
                      "prototype": _valid_for("pr", "prototype")}

            if _valid["max_logit"] and _valid["prototype"]:
                print(f"  seed={seed}: SKIP, already done (resumed)")
                results_this_seed = {"max_logit": _cached["max_logit"], "prototype": _cached["prototype"]}
            else:
                for det_name in ["max_logit", "prototype"]:
                    if _valid[det_name]:
                        print(f"  seed={seed}, {det_name}: SKIP, already done (resumed)")
                    elif _cached is not None and det_name in _cached:
                        print(f"  seed={seed}, {det_name}: cached result was trained under different settings -- re-running.")
                    else:
                        print(f"  seed={seed}, {det_name}: training...")

                cfg["LAMBDA_REC_ML"] = candidate
                cfg["LAMBDA_REC_PR"] = candidate
                try:
                    results_this_seed = {}
                    for det_name in ["max_logit", "prototype"]:
                        if _valid[det_name]:
                            results_this_seed[det_name] = _cached[det_name]
                            continue
                        set_seed(seed)
                        tag = f"lambdarecverify_{round_tag}_{cand_key}_{det_name}_seed{seed}"
                        r = train_one_system(cfg, device, loaders, num_classes, nsym, det_name,
                                              lambda_sem_for_run[det_name], cfg["GRID_SEARCH_PATIENCE"],
                                              verify_dir, tag)
                        jscc, detector = r["jscc"], r["detector"]
                        jscc.eval(); detector.eval()
                        tau = calibrate_tau(jscc, detector, loaders["val_mixed"], device,
                                             target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
                        sdop = calibrate_sdop_threshold(jscc, loaders["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
                        r_eval = _eval_one_system(jscc, detector, tau, sdop, loaders["val_mixed"],
                                                   cfg["DEFAULT_SNR_DB"], cfg, device, "awgn")
                        results_this_seed[det_name] = float(r_eval["auc_roc_pixel"])
                        for suffix in ("_jscc.pt", "_detector.pt"):
                            (verify_dir / f"{tag}{suffix}").unlink(missing_ok=True)
                        print(f"  seed={seed}, {det_name}: done, AUC={results_this_seed[det_name]:.4f}, checkpoints deleted")
                finally:
                    cfg["LAMBDA_REC_ML"] = _initial_lambda_rec_ml
                    cfg["LAMBDA_REC_PR"] = _initial_lambda_rec_pr

            results_this_seed["_patch_grid_ml"] = cfg["PATCH_GRID_ML"]
            results_this_seed["_patch_dim_ml"] = cfg["PATCH_DIM_ML"]
            results_this_seed["_patch_grid_pr"] = cfg["PATCH_GRID_PR"]
            results_this_seed["_patch_dim_pr"] = cfg["PATCH_DIM_PR"]
            results_this_seed["_joint_lr_ml"] = cfg["JOINT_LR_ML"]
            results_this_seed["_joint_lr_pr"] = cfg["JOINT_LR_PR"]
            results_this_seed["_joint_momentum_ml"] = cfg["JOINT_MOMENTUM_ML"]
            results_this_seed["_joint_momentum_pr"] = cfg["JOINT_MOMENTUM_PR"]
            results_this_seed["_joint_lr_power_ml"] = cfg["JOINT_LR_POWER_ML"]
            results_this_seed["_joint_lr_power_pr"] = cfg["JOINT_LR_POWER_PR"]
            results_this_seed["_weight_decay_ml"] = cfg["WEIGHT_DECAY_ML"]
            results_this_seed["_weight_decay_pr"] = cfg["WEIGHT_DECAY_PR"]
            results_this_seed["_min_rel_improve_ml"] = cfg["MIN_REL_IMPROVE_ML"]
            results_this_seed["_min_rel_improve_pr"] = cfg["MIN_REL_IMPROVE_PR"]
            results_this_seed["_prototype_scale"] = cfg["PROTOTYPE_SCALE"]
            results_this_seed["_lambda_sem_max_logit"] = lambda_sem_for_run["max_logit"]
            results_this_seed["_lambda_sem_prototype"] = lambda_sem_for_run["prototype"]
            state[cand_key][seed_key] = results_this_seed
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)

    results_by_val = {}
    for candidate in candidates:
        cand_key = str(candidate)
        ml_aucs = [state[cand_key][str(s)]["max_logit"] for s in verify_seeds]
        pr_aucs = [state[cand_key][str(s)]["prototype"] for s in verify_seeds]
        results_by_val[candidate] = {"max_logit": ml_aucs, "prototype": pr_aucs}
        print(f"[LAMBDA_REC-verify:{round_tag}] LAMBDA_REC={candidate}: "
              f"ML mean={statistics.mean(ml_aucs):.4f}, PR mean={statistics.mean(pr_aucs):.4f}")

    per_detector_winners = {}
    for det_name in ("max_logit", "prototype"):
        det_scores = {c: v[det_name] for c, v in results_by_val.items()}
        cand_order = list(det_scores.keys())
        matrix = [[det_scores[c][i] for c in cand_order] for i in range(len(verify_seeds))]
        h_stat, p_kw = friedman_permutation_test(matrix)
        eps2 = kendalls_w(matrix)
        best_val = max(det_scores, key=lambda v: statistics.mean(det_scores[v]))
        cfg_key = f"LAMBDA_REC_{'ML' if det_name == 'max_logit' else 'PR'}"
        label = "Max Logit" if det_name == "max_logit" else "Prototype"
        initial_val = _initial_lambda_rec_ml if det_name == "max_logit" else _initial_lambda_rec_pr
        print(f"[LAMBDA_REC-verify:{round_tag}] {label} -- Friedman permutation test (paired by seed, 20000 permutations) across "
              f"{len(candidates)} candidates: stat={h_stat:.3f}, p={p_kw:.4f}, effect size "
              f"(Kendall's W)={eps2:.4f} "
              f"({interpret_kendalls_w(eps2, n=len(verify_seeds), k=len(candidates))})")
        final_val = best_val

        per_detector_winners[det_name] = {"friedman_p": p_kw, "effect_size_kendalls_w": eps2,
                                           "best_value": final_val, "previous_value": initial_val}
        if final_val != initial_val:
            print(f"[LAMBDA_REC-verify:{round_tag}] *** {label}'s LAMBDA_REC changing from "
                  f"{initial_val} to {final_val} based on N={verify_n} verification ***")
        else:
            print(f"[LAMBDA_REC-verify:{round_tag}] {label}: existing value {initial_val} "
                  f"confirmed at N={verify_n}.")
        cfg[cfg_key] = final_val

    verify_summary = {"results_by_value": {str(k): v for k, v in results_by_val.items()},
                       "per_detector": per_detector_winners}
    with open(verify_dir / f"lambda_rec_verify_summary_{round_tag}.json", "w") as f:
        json.dump(verify_summary, f, indent=2)

    return {"max_logit": cfg["LAMBDA_REC_ML"], "prototype": cfg["LAMBDA_REC_PR"]}, verify_summary


def _tie_test(values_a, values_b, alpha):
    """Paired tie test of two sets of per-seed AUROCs (same seeds, same order).

    Tied when the two-sided Wilcoxon p >= alpha and |rank-biserial r| stays below
    the simulated null value for this N (effect_sizes.rank_biserial_null_floor).
    At five pairs the exact Wilcoxon p cannot fall below 0.0625, so alpha must be
    above that (cfg["CONVERGENCE_TIE_ALPHA"] = 0.1). Returns (tied, p, r); r is
    positive when values_a is higher.
    """
    from scipy import stats as _stats
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return False, float("nan"), float("nan")
    diffs = [a - b for a, b in zip(values_a, values_b)]
    if all(d == 0 for d in diffs):
        return True, 1.0, 0.0
    try:
        _, p_val = _stats.wilcoxon(values_a, values_b)
    except ValueError:
        p_val = 1.0
    effect_r = rank_biserial_wilcoxon(diffs)
    tied_by_p = (p_val >= alpha)
    if tied_by_p and abs(effect_r) >= rank_biserial_null_floor(len(values_a)):
        tied_by_p = False
    return tied_by_p, p_val, effect_r


def _run_hyperparameter_search_to_convergence(cfg, device, loaders, num_classes, nsym, out, device_for_batch):
    """Repeat Steps 0, A-F until the stopping rule is met or
    cfg["MAX_CONVERGENCE_ROUNDS"] rounds have run.

    From round 1 on, each detector's round-end configuration (the per-seed AUROCs of
    its Step F winner) is compared with the previous round's on the same seeds
    (objective_convergence.objective_decision, using _tie_test). The search stops
    when neither detector improved significantly; each detector then keeps the
    configuration with the higher mean AUROC (the previous one is restored if its
    patch configuration is the same). Each round's lambda_sem winners are used by
    the next round's Steps 0-D. If Step 0 or A changes a patch configuration, the
    batch size is re-set and the loaders are rebuilt.

    Writes convergence_summary.json and returns
    (analysis, loaders, num_classes, snapshots, outcome), outcome being
    "converged_objective" or "max_rounds_exceeded".
    """
    verify_dirs = {
        "patch": out / "patch_config_verify", "lr": out / "joint_lr_verify",
        "optim": out / "optimizer_hparam_verify", "scale": out / "prototype_scale_verify",
        "lambda_rec": out / "lambda_rec_verify",
    }
    for d in verify_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    grid_dir = out / "grid_search"
    grid_dir.mkdir(parents=True, exist_ok=True)

    max_rounds = cfg.get("MAX_CONVERGENCE_ROUNDS", 5)
    snapshots = []
    round_scores, decisions = [], []
    lambda_sem_override = None
    analysis = None
    outcome = None

    def _snapshot():
        return {
            "patch_grid_ml": cfg["PATCH_GRID_ML"], "patch_dim_ml": cfg["PATCH_DIM_ML"],
            "patch_grid_pr": cfg["PATCH_GRID_PR"], "patch_dim_pr": cfg["PATCH_DIM_PR"],
            "joint_lr_ml": cfg["JOINT_LR_ML"], "joint_lr_pr": cfg["JOINT_LR_PR"],
            "joint_momentum_ml": cfg["JOINT_MOMENTUM_ML"], "joint_momentum_pr": cfg["JOINT_MOMENTUM_PR"],
            "joint_lr_power_ml": cfg["JOINT_LR_POWER_ML"], "joint_lr_power_pr": cfg["JOINT_LR_POWER_PR"],
            "weight_decay_ml": cfg["WEIGHT_DECAY_ML"], "weight_decay_pr": cfg["WEIGHT_DECAY_PR"],
            "min_rel_improve_ml": cfg["MIN_REL_IMPROVE_ML"], "min_rel_improve_pr": cfg["MIN_REL_IMPROVE_PR"],
            "prototype_scale": cfg["PROTOTYPE_SCALE"],
            "lambda_rec_ml": cfg["LAMBDA_REC_ML"], "lambda_rec_pr": cfg["LAMBDA_REC_PR"],
            "lambda_sem_max_logit": analysis["winning_lambda_sem"]["max_logit"],
            "lambda_sem_prototype": analysis["winning_lambda_sem"]["prototype"],
        }

    for round_idx in range(max_rounds):
        print(f"\n{'#'*62}\n  HYPERPARAMETER SEARCH -- ROUND {round_idx} "
              f"(max {max_rounds})\n{'#'*62}")

        _before_patch = (cfg["PATCH_GRID_ML"], cfg["PATCH_DIM_ML"], cfg["PATCH_GRID_PR"], cfg["PATCH_DIM_PR"])
        print(f"\n{'-'*62}\n  Round {round_idx}, Step 0 -- BACKBONE_STAGE\n{'-'*62}")
        _verify_backbone_stage(cfg, device, loaders, num_classes, nsym, verify_dirs["patch"],
                                lambda_sem_override=lambda_sem_override, round_idx=round_idx)

        print(f"\n{'-'*62}\n  Round {round_idx}, Step A -- PATCH_GRID/PATCH_DIM\n{'-'*62}")
        _verify_patch_config(cfg, device, loaders, num_classes, nsym, verify_dirs["patch"],
                              lambda_sem_override=lambda_sem_override, round_idx=round_idx)
        if (cfg["PATCH_GRID_ML"], cfg["PATCH_DIM_ML"], cfg["PATCH_GRID_PR"], cfg["PATCH_DIM_PR"]) != _before_patch:
            print(f"\n[round{round_idx}] PATCH_GRID/PATCH_DIM changed for at least one detector -- "
                  f"re-setting the batch size and rebuilding the loaders.")
            def _model_builder():
                # The batch-size probe uses the larger of the two latent sizes.
                _ml_size = cfg["PATCH_GRID_ML"] ** 2 * cfg["PATCH_DIM_ML"]
                _pr_size = cfg["PATCH_GRID_PR"] ** 2 * cfg["PATCH_DIM_PR"]
                _grid, _dim = ((cfg["PATCH_GRID_ML"], cfg["PATCH_DIM_ML"]) if _ml_size >= _pr_size
                               else (cfg["PATCH_GRID_PR"], cfg["PATCH_DIM_PR"]))
                jscc = SwinJSCCSystem(nsym, pretrained=False, patch_grid=_grid, patch_dim=_dim)
                detector = MaxLogitDetector(_dim, _grid, num_classes)
                return jscc, detector
            if cfg.get("BATCH_SIZE_OVERRIDE") is not None:
                cfg["BATCH_SIZE"] = cfg["BATCH_SIZE_OVERRIDE"]
                print(f"[round{round_idx}] BATCH_SIZE_OVERRIDE={cfg['BATCH_SIZE']} kept for the new "
                      f"patch configuration (set it to None to probe the batch size instead).")
            else:
                cfg["BATCH_SIZE"] = detect_max_batch_size(_model_builder, device_for_batch, img_size=cfg["IMG_H"], safety_factor=cfg["BATCH_SIZE_SAFETY_FACTOR"], warn_above=cfg["BATCH_SIZE_WARN_ABOVE"])
            loaders = build_dataloaders(cfg)
            num_classes = len(loaders["known_class_ids"])

        print(f"\n{'-'*62}\n  Round {round_idx}, Step B -- JOINT_LR\n{'-'*62}")
        _verify_joint_lr(cfg, device, loaders, num_classes, nsym, verify_dirs["lr"],
                          lambda_sem_override=lambda_sem_override, round_idx=round_idx)

        print(f"\n{'-'*62}\n  Round {round_idx}, Step C -- Optimizer hyperparameters\n{'-'*62}")
        _verify_optimizer_hparams(cfg, device, loaders, num_classes, nsym, verify_dirs["optim"],
                                    lambda_sem_override=lambda_sem_override, round_idx=round_idx)

        print(f"\n{'-'*62}\n  Round {round_idx}, Step D -- PROTOTYPE_SCALE\n{'-'*62}")
        _verify_prototype_scale(cfg, device, loaders, num_classes, nsym, verify_dirs["scale"],
                                  lambda_sem_override=lambda_sem_override, round_idx=round_idx)

        print(f"\n{'-'*62}\n  Round {round_idx}, Step E -- lambda_sem grid search + analysis\n{'-'*62}")
        grid_results = _run_grid_search(cfg, device, loaders, num_classes, nsym, grid_dir, round_idx=round_idx)
        analysis = _analyze_grid_results(grid_results, cfg["GRID_SEARCH_PATIENCE"])
        print(f"[round{round_idx}] Winning lambda_sem: {analysis['winning_lambda_sem']}")

        print(f"\n{'-'*62}\n  Round {round_idx}, Step F -- LAMBDA_REC\n{'-'*62}")
        _verify_lambda_rec(cfg, device, loaders, num_classes, nsym, verify_dirs["lambda_rec"],
                            lambda_sem_for_run=analysis["winning_lambda_sem"], round_idx=round_idx)

        snap = _snapshot()
        print(f"\n[round{round_idx}] Settled values this round: {snap}")

        # Stopping rule: the Step F winner's per-seed AUROCs measure this round's
        # complete configuration (no extra training).
        round_scores.append(_oc.round_end_scores(
            out / f"lambda_rec_verify_state_round{round_idx}.json",
            {"max_logit": cfg["LAMBDA_REC_ML"], "prototype": cfg["LAMBDA_REC_PR"]}))
        snapshots.append(snap)
        if round_idx >= 1:
            _alpha = cfg.get("CONVERGENCE_TIE_ALPHA", 0.1)
            stop, per_det = _oc.objective_decision(round_scores[-2], round_scores[-1], _tie_test,
                                                     _alpha, cfg["SELECTION_SEEDS"])
            decisions.append({"round": round_idx, "stop": stop, "per_detector": per_det})
            for det, d in per_det.items():
                if not d["testable"]:
                    print(f"[round{round_idx}] objective check {det}: only {d['n']} common seed(s) -- cannot test.")
                    continue
                print(f"[round{round_idx}] objective check {det}: config_{round_idx} mean AUC={d['mean_this']:.4f} vs "
                      f"config_{round_idx-1} {d['mean_prev']:.4f} (diff {d['diff']:+.4f}, n={d['n']} paired seeds) -- "
                      f"Wilcoxon p={d['p_value']:.4f}, r={d['rank_biserial_r']:+.3f} -> "
                      + ("SIGNIFICANT IMPROVEMENT" if d["improved"] else "no significant improvement"))
            if stop:
                for det, d in per_det.items():
                    if d["keep"] == "prev":
                        if _oc.revert_detector_config(cfg, analysis, snapshots[-1], snapshots[-2], det):
                            print(f"[round{round_idx}] {det}: previous round's configuration has the higher mean AUC "
                                  f"-- restored it as the final configuration.")
                        else:
                            print(f"[round{round_idx}] {det}: previous round's configuration has the higher mean AUC "
                                  f"but its patch configuration differs, so it was not restored -- keeping round "
                                  f"{round_idx}'s.")
                snapshots[-1] = _snapshot()
                print(f"\n{'='*62}\n  CONVERGED (objective) after round {round_idx} -- neither detector's complete "
                      f"configuration improved significantly over round {round_idx-1}'s. Final per detector = the "
                      f"configuration with the higher mean AUC.\n{'='*62}")
                outcome = "converged_objective"
                break
        lambda_sem_override = analysis["winning_lambda_sem"]
    else:
        print(f"\n{'!'*62}\n  DID NOT CONVERGE after {max_rounds} rounds -- using the final "
              f"round's values.\n{'!'*62}")
        outcome = "max_rounds_exceeded"

    with open(out / "convergence_summary.json", "w") as f:
        json.dump({"outcome": outcome, "n_rounds_run": len(snapshots), "snapshots": snapshots,
                   "convergence_mode": "objective",
                   "round_end_scores": [{d: {str(k): v for k, v in sc.items()} for d, sc in r.items()} for r in round_scores],
                   "objective_decisions": decisions}, f, indent=2, default=str)

    return analysis, loaders, num_classes, snapshots, outcome


# Set in cfg by the first phase1_grid_search_and_train call.
_PHASE1_CALLED_KEY = "_PHASE1_CALLED_IN_THIS_SESSION"


def phase1_grid_search_and_train(cfg, device, run_idx=0):
    """Complete Phase I: batch size, hyperparameter search to convergence, then
    cfg["N_RUNS"] production repetitions (seeds cfg["SEEDS"]) of both systems.

    The search writes every selected value into cfg, so this function may be
    called only once per Python session (a second call is refused before anything
    runs). A production repetition whose phase1_meta.json records the current
    settings is loaded instead of retrained. Returns {"final_runs": [...],
    "grid_analysis", "loaders", "num_classes", "nsym", "out"}.
    """
    if cfg.get(_PHASE1_CALLED_KEY):
        raise RuntimeError(
            "Phase I already ran in this Python session, and it changed CONFIG in place (selected "
            "stage, patch and optimizer settings). Running it again now would make its cached search "
            "results look stale and retrain them. Nothing was trained or written. Restart the session "
            "(Runtime -> Restart session), then run the setup cells and the Phase I cell once.")
    cfg[_PHASE1_CALLED_KEY] = True
    out = Path(cfg["OUTPUT_DIR"]) / f"run_{run_idx}"
    out.mkdir(parents=True, exist_ok=True)
    grid_dir = out / "grid_search"
    grid_dir.mkdir(parents=True, exist_ok=True)
    nsym = cfg["CHANNEL_SYMBOLS"][cfg["DEFAULT_CR"]]

    print(f"\n{'='*62}\n  STEP 1 -- Batch size\n{'='*62}")
    # The class count is needed to build a detector for the batch-size probe.
    _anomaly_value_probe = detect_anomaly_value(cfg["STREETHAZARDS_ROOT"], sample_size=cfg["ANOMALY_VALUE_SAMPLE_SIZE"])
    _known_ids_probe, _ = discover_known_classes(cfg["STREETHAZARDS_ROOT"], _anomaly_value_probe)
    _num_classes_probe = len(_known_ids_probe)

    def _model_builder():
        _ml_size = cfg["PATCH_GRID_ML"] ** 2 * cfg["PATCH_DIM_ML"]
        _pr_size = cfg["PATCH_GRID_PR"] ** 2 * cfg["PATCH_DIM_PR"]
        _grid, _dim = ((cfg["PATCH_GRID_ML"], cfg["PATCH_DIM_ML"]) if _ml_size >= _pr_size
                       else (cfg["PATCH_GRID_PR"], cfg["PATCH_DIM_PR"]))
        jscc = SwinJSCCSystem(nsym, pretrained=False, patch_grid=_grid, patch_dim=_dim)
        detector = MaxLogitDetector(_dim, _grid, _num_classes_probe)
        return jscc, detector
    if cfg.get("BATCH_SIZE_OVERRIDE") is not None:
        detected_bs = cfg["BATCH_SIZE_OVERRIDE"]
        print(f"[Phase I] BATCH_SIZE_OVERRIDE={detected_bs} -- batch-size probe skipped.")
    else:
        detected_bs = detect_max_batch_size(_model_builder, device, img_size=cfg["IMG_H"], safety_factor=cfg["BATCH_SIZE_SAFETY_FACTOR"], warn_above=cfg["BATCH_SIZE_WARN_ABOVE"])
    cfg["BATCH_SIZE"] = detected_bs
    print(f"[Phase I] Using batch_size={detected_bs} for all "
          f"subsequent training in this run.")

    loaders = build_dataloaders(cfg)
    known_class_ids = loaders["known_class_ids"]
    num_classes = len(known_class_ids)

    print(f"\n{'='*62}\n  STEP 2 -- Hyperparameter search to convergence "
          f"(patch config, JOINT_LR, optimizer hparams, PROTOTYPE_SCALE, "
          f"lambda_sem, LAMBDA_REC)\n{'='*62}")
    analysis, loaders, num_classes, snapshots, outcome = _run_hyperparameter_search_to_convergence(
        cfg, device, loaders, num_classes, nsym, out, device)

    with open(out / "grid_search_results.json", "w") as f:
        json.dump({"analysis": analysis}, f, indent=2, default=str)

    print(f"\n{'='*62}\n  STEP 3 -- Final production training, "
          f"{cfg['N_RUNS']} repeated runs per detector\n{'='*62}")
    final_runs = []
    for rep in range(cfg["N_RUNS"]):
        rep_seed = cfg["SEEDS"][rep]
        rep_run_idx = f"{run_idx}_rep{rep}"
        rep_meta_path = Path(cfg["OUTPUT_DIR"]) / f"run_{rep_run_idx}" / "phase1_meta.json"

        # phase1_meta.json is written at the end of a completed repetition; it is
        # reused only if it records the current value of every selected setting.
        resume_ok = False
        if rep_meta_path.exists():
            with open(rep_meta_path) as f:
                _saved_meta = json.load(f)
            _saved_ml = _saved_meta.get("lambda_sem_max_logit")
            _saved_pr = _saved_meta.get("lambda_sem_prototype")
            _saved_scale = _saved_meta.get("prototype_scale")
            _saved_lr_ml = _saved_meta.get("joint_lr_ml")
            _saved_lr_pr = _saved_meta.get("joint_lr_pr")
            _saved_patch_grid_ml = _saved_meta.get("patch_grid_ml")
            _saved_patch_dim_ml = _saved_meta.get("patch_dim_ml")
            _saved_patch_grid_pr = _saved_meta.get("patch_grid_pr")
            _saved_patch_dim_pr = _saved_meta.get("patch_dim_pr")
            _saved_mom_ml = _saved_meta.get("joint_momentum_ml")
            _saved_mom_pr = _saved_meta.get("joint_momentum_pr")
            _saved_wd_ml = _saved_meta.get("weight_decay_ml")
            _saved_wd_pr = _saved_meta.get("weight_decay_pr")
            _saved_lrp_ml = _saved_meta.get("joint_lr_power_ml")
            _saved_lrp_pr = _saved_meta.get("joint_lr_power_pr")
            _saved_mri_ml = _saved_meta.get("min_rel_improve_ml")
            _saved_mri_pr = _saved_meta.get("min_rel_improve_pr")
            _saved_lrec_ml = _saved_meta.get("lambda_rec_ml")
            _saved_lrec_pr = _saved_meta.get("lambda_rec_pr")
            _current_ml = analysis["winning_lambda_sem"]["max_logit"]
            _current_pr = analysis["winning_lambda_sem"]["prototype"]
            _current_scale = cfg["PROTOTYPE_SCALE"]
            _current_lr_ml = cfg["JOINT_LR_ML"]
            _current_lr_pr = cfg["JOINT_LR_PR"]
            _current_patch_grid_ml = cfg["PATCH_GRID_ML"]
            _current_patch_dim_ml = cfg["PATCH_DIM_ML"]
            _current_patch_grid_pr = cfg["PATCH_GRID_PR"]
            _current_patch_dim_pr = cfg["PATCH_DIM_PR"]
            _current_mom_ml, _current_mom_pr = cfg["JOINT_MOMENTUM_ML"], cfg["JOINT_MOMENTUM_PR"]
            _current_wd_ml, _current_wd_pr = cfg["WEIGHT_DECAY_ML"], cfg["WEIGHT_DECAY_PR"]
            _current_lrp_ml, _current_lrp_pr = cfg["JOINT_LR_POWER_ML"], cfg["JOINT_LR_POWER_PR"]
            _current_mri_ml, _current_mri_pr = cfg["MIN_REL_IMPROVE_ML"], cfg["MIN_REL_IMPROVE_PR"]
            _current_lrec_ml, _current_lrec_pr = cfg["LAMBDA_REC_ML"], cfg["LAMBDA_REC_PR"]
            if (_saved_ml == _current_ml and _saved_pr == _current_pr and _saved_scale == _current_scale
                    and _saved_lr_ml == _current_lr_ml and _saved_lr_pr == _current_lr_pr
                    and _saved_patch_grid_ml == _current_patch_grid_ml
                    and _saved_patch_dim_ml == _current_patch_dim_ml
                    and _saved_patch_grid_pr == _current_patch_grid_pr
                    and _saved_patch_dim_pr == _current_patch_dim_pr
                    and _saved_mom_ml == _current_mom_ml and _saved_mom_pr == _current_mom_pr
                    and _saved_wd_ml == _current_wd_ml and _saved_wd_pr == _current_wd_pr
                    and _saved_lrp_ml == _current_lrp_ml and _saved_lrp_pr == _current_lrp_pr
                    and _saved_mri_ml == _current_mri_ml and _saved_mri_pr == _current_mri_pr
                    and _saved_lrec_ml == _current_lrec_ml and _saved_lrec_pr == _current_lrec_pr):
                resume_ok = True
            else:
                print(f"\n--- Production repetition {rep+1}/{cfg['N_RUNS']} (seed={rep_seed}) -- "
                      f"saved repetition was trained under different settings -- retraining ---")

        if resume_ok:
            print(f"\n--- Production repetition {rep+1}/{cfg['N_RUNS']} (seed={rep_seed}) -- "
                  f"already complete with the current settings, loading from disk ---")
            p1_result = load_phase1(cfg, device, run_idx=rep_run_idx)
            # All repetitions share one set of loaders, so loaded repetitions do not
            # each keep their own persistent worker processes alive.
            p1_result["loaders"] = loaders
        else:
            set_seed(rep_seed)
            print(f"\n--- Production repetition {rep+1}/{cfg['N_RUNS']} (seed={rep_seed}) ---")
            p1_result = phase1(
                cfg, device, loaders, num_classes, nsym,
                lambda_sem_max_logit=analysis["winning_lambda_sem"]["max_logit"],
                lambda_sem_prototype=analysis["winning_lambda_sem"]["prototype"],
                patience=analysis["derived_patience"],
                run_idx=rep_run_idx)
            # Record the repetition's seed in its metadata.
            if rep_meta_path.exists():
                with open(rep_meta_path) as f:
                    _meta = json.load(f)
                _meta["seed"] = rep_seed
                _meta["seed_source"] = "SEEDS"
                with open(rep_meta_path, "w") as f:
                    json.dump(_meta, f, indent=2)
        final_runs.append(p1_result)

    return {"final_runs": final_runs, "grid_analysis": analysis,
            "loaders": loaders, "num_classes": num_classes, "nsym": nsym, "out": out}
