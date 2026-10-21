"""Phase III: compression study at the three compression ratios of
cfg["CHANNEL_SYMBOLS"] (low / medium / high), with the seeds of cfg["SEEDS"].

At DEFAULT_CR ("medium") the Phase I production models are reused. At each other
compression ratio, lambda_sem is first selected by its own small grid search
(same criterion as Phase I, five selection seeds), then both systems are trained
for every seed and swept over SNR on AWGN. Each (CR, seed) result is stored in
run_0/phase3_multiseed_state.json with a fingerprint of its settings, so an
interrupted study resumes; phase3.json holds the mean +/- SD per CR and SNR.
"""
import gc
import json
import tempfile
from pathlib import Path
import torch
from src.models import SwinJSCCSystem, MaxLogitDetector, PrototypeDetector
from src.evaluate import eval_both_systems, eval_noise_seed
from src.train import train_joint_system
from src.metrics import calibrate_tau, calibrate_sdop_threshold
from src.config import set_seed, training_fingerprint, evaluation_fingerprint, fingerprint_key
from phases.phase2_sweep import _aggregate_metric_dicts


def _local_tmp_dir(name):
    """Directory on the VM's local disk for temporary training checkpoints (deleted
    after each model is evaluated). Results and resume files stay in OUTPUT_DIR; files
    deleted from a mounted Drive can keep using its quota.
    """
    base = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    d = base / "jscc_tmp_checkpoints" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _grid_search_lambda_sem_for_cr(cfg, device, p1_default, cr_name, nsym, cr_out_dir):
    """lambda_sem of both detectors at one non-default compression ratio.

    Every value of cfg["LAMBDA_SEM_GRID_MAXLOGIT"] / ["LAMBDA_SEM_GRID_PROTOTYPE"] is
    trained on SELECTION_SEEDS[:5] and scored by pixel AUROC on the calibration
    images at DEFAULT_SNR_DB; the highest mean wins. Per-seed results and the final
    choice are cached with a fingerprint of the settings they depend on. Returns
    (lam_sem_ml, lam_sem_pr).
    """
    import statistics

    verify_seeds = cfg["SELECTION_SEEDS"][:5]
    _fp_det = {d: fingerprint_key({"train": training_fingerprint(cfg, d), "nsym": nsym,
                                   "patience": cfg["GRID_SEARCH_PATIENCE"],
                                   "eval": evaluation_fingerprint(cfg)})
               for d in ("max_logit", "prototype")}
    _grid_fp = fingerprint_key({"det": _fp_det, "seeds": list(verify_seeds),
                                "grid_ml": list(cfg["LAMBDA_SEM_GRID_MAXLOGIT"]),
                                "grid_pr": list(cfg["LAMBDA_SEM_GRID_PROTOTYPE"])})
    cache_path = cr_out_dir / f"lambda_sem_grid_{cr_name}.json"
    if cache_path.exists():
        cached = json.load(open(cache_path))
        _cache_matches = cached.get("_fp") == _grid_fp
        if _cache_matches:
            print(f"    lambda_sem for CR={cr_name}: using cached grid result -- "
                  f"ML={cached['lam_sem_ml']}, PR={cached['lam_sem_pr']}")
            return cached["lam_sem_ml"], cached["lam_sem_pr"]
        else:
            print(f"    lambda_sem for CR={cr_name}: cached result was produced under different "
                  f"settings -- re-running.")

    state_path = cr_out_dir / f"lambda_sem_grid_{cr_name}_state.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    if state:
        n_done = sum(len(v) for v in state.values())
        print(f"    lambda_sem for CR={cr_name}: RESUMING -- found {n_done} already-completed "
              f"(detector, lambda_sem, seed) result(s)")
    else:
        n_grid = len(cfg["LAMBDA_SEM_GRID_MAXLOGIT"]) + len(cfg["LAMBDA_SEM_GRID_PROTOTYPE"])
        print(f"    lambda_sem for CR={cr_name}: grid search at this CR "
              f"({n_grid} combinations x N=5 seeds = {n_grid*5} runs)...")

    grid_dir = _local_tmp_dir(f"PHASE3_{cr_name}_lambda_sem_grid")
    detector_classes = {"max_logit": MaxLogitDetector, "prototype": PrototypeDetector}
    grids = {"max_logit": cfg["LAMBDA_SEM_GRID_MAXLOGIT"], "prototype": cfg["LAMBDA_SEM_GRID_PROTOTYPE"]}

    for det_name, grid_values in grids.items():
        for lam in grid_values:
            combo_key = f"{det_name}_{lam}"
            state.setdefault(combo_key, {})
            for seed in verify_seeds:
                seed_key = str(seed)
                _cached = state[combo_key].get(seed_key)
                # A per-seed result is reused only if trained under the current settings.
                _cache_valid = isinstance(_cached, dict) and _cached.get("_fp") == _fp_det[det_name]
                if _cache_valid:
                    continue
                set_seed(seed)
                _grid = cfg["PATCH_GRID_ML"] if det_name == "max_logit" else cfg["PATCH_GRID_PR"]
                _dim = cfg["PATCH_DIM_ML"] if det_name == "max_logit" else cfg["PATCH_DIM_PR"]
                jscc = SwinJSCCSystem(nsym, pretrained=True,
                                       patch_grid=_grid, patch_dim=_dim,
                                       stage_idx=cfg.get("BACKBONE_STAGE_ML" if det_name == "max_logit"
                                                         else "BACKBONE_STAGE_PR", 2)).to(device)
                extra = (cfg["PROTOTYPE_SCALE"],) if det_name == "prototype" else ()
                det = detector_classes[det_name](_dim, _grid,
                                                  p1_default["num_classes"], *extra).to(device)
                tag = f"gridcr_{cr_name}_{det_name}_sem{lam}_seed{seed}"
                _lr = cfg["JOINT_LR_ML"] if det_name == "max_logit" else cfg["JOINT_LR_PR"]
                _mom = cfg["JOINT_MOMENTUM_ML"] if det_name == "max_logit" else cfg["JOINT_MOMENTUM_PR"]
                _wd = cfg["WEIGHT_DECAY_ML"] if det_name == "max_logit" else cfg["WEIGHT_DECAY_PR"]
                _lrp = cfg["JOINT_LR_POWER_ML"] if det_name == "max_logit" else cfg["JOINT_LR_POWER_PR"]
                _mri = cfg["MIN_REL_IMPROVE_ML"] if det_name == "max_logit" else cfg["MIN_REL_IMPROVE_PR"]
                _lrec = cfg["LAMBDA_REC_ML"] if det_name == "max_logit" else cfg["LAMBDA_REC_PR"]
                r = train_joint_system(jscc, det, p1_default["loaders"]["cls_train"], p1_default["loaders"]["cls_val"],
                                        cfg, device, p1_default["num_classes"],
                                        jscc_save_path=grid_dir / f"{tag}_jscc.pt",
                                        detector_save_path=grid_dir / f"{tag}_detector.pt",
                                        lambda_rec=_lrec, lambda_sem=lam,
                                        joint_lr=_lr, joint_momentum=_mom, weight_decay=_wd, joint_lr_power=_lrp,
                                        patience=cfg["GRID_SEARCH_PATIENCE"], min_rel_improve=_mri,
                                        max_epochs=cfg["JOINT_MAX_EPOCHS"])
                # Training sees no anomalies; AUROC is measured on the calibration images.
                from src.evaluate import _eval_one_system
                jscc.eval(); det.eval()
                tau = calibrate_tau(jscc, det, p1_default["loaders"]["val_mixed"], device,
                                     target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
                sdop = calibrate_sdop_threshold(jscc, p1_default["loaders"]["val_mixed"], device,
                                                 snr_db=cfg["CLS_FIT_SNR"],
                                                 percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
                r_eval = _eval_one_system(jscc, det, tau, sdop, p1_default["loaders"]["val_mixed"],
                                           cfg["DEFAULT_SNR_DB"], cfg, device, "awgn",
                                           noise_seed=eval_noise_seed(cfg, cfg["DEFAULT_SNR_DB"], "awgn"))
                state[combo_key][seed_key] = {
                    "loss": r["loss"], "auc": float(r_eval["auc_roc_pixel"]),
                    "_patch_grid": _grid, "_patch_dim": _dim,
                    "_joint_lr": _lr, "_joint_momentum": _mom, "_weight_decay": _wd,
                    "_joint_lr_power": _lrp, "_min_rel_improve": _mri, "_lambda_rec": _lrec,
                    "_prototype_scale": cfg["PROTOTYPE_SCALE"],
                    "_fp": _fp_det[det_name],
                }
                with open(state_path, "w") as f:
                    json.dump(state, f, indent=2)
                for suffix in ("_jscc.pt", "_detector.pt"):
                    (grid_dir / f"{tag}{suffix}").unlink(missing_ok=True)
                # Free GPU memory before the next model is built.
                del jscc, det
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"      {det_name}, lambda_sem={lam}, seed={seed}: "
                      f"val_task_loss={r['loss']:.4f}, AUC={r_eval['auc_roc_pixel']:.4f}, "
                      f"checkpoints deleted")

    results_by_detector = {}
    for det_name, grid_values in grids.items():
        results_by_detector[det_name] = []
        for lam in grid_values:
            combo_key = f"{det_name}_{lam}"
            aucs = [state[combo_key][str(s)]["auc"] for s in verify_seeds]
            results_by_detector[det_name].append({"lambda_sem": lam, "aucs": aucs,
                                                    "mean_auc": statistics.mean(aucs)})
            print(f"    CR={cr_name}, {det_name}, lambda_sem={lam}: "
                  f"mean_auc={statistics.mean(aucs):.4f}, std={statistics.stdev(aucs):.4f} (N=5)")

    lam_sem_ml = max(results_by_detector["max_logit"], key=lambda r: r["mean_auc"])["lambda_sem"]
    lam_sem_pr = max(results_by_detector["prototype"], key=lambda r: r["mean_auc"])["lambda_sem"]
    print(f"    CR={cr_name} winners (N=5): lambda_sem_ml={lam_sem_ml}, lambda_sem_pr={lam_sem_pr}")

    with open(cache_path, "w") as f:
        json.dump({"lam_sem_ml": lam_sem_ml, "lam_sem_pr": lam_sem_pr,
                    "prototype_scale_at_search": cfg["PROTOTYPE_SCALE"],
                    "patch_grid_ml_at_search": cfg["PATCH_GRID_ML"],
                    "patch_dim_ml_at_search": cfg["PATCH_DIM_ML"],
                    "patch_grid_pr_at_search": cfg["PATCH_GRID_PR"],
                    "patch_dim_pr_at_search": cfg["PATCH_DIM_PR"],
                    "joint_lr_ml_at_search": cfg["JOINT_LR_ML"],
                    "joint_lr_pr_at_search": cfg["JOINT_LR_PR"],
                    "_fp": _grid_fp,
                    "raw_results": results_by_detector}, f, indent=2)
    return lam_sem_ml, lam_sem_pr


def _train_one_cr_rep(cfg, device, p1, cr_name, nsym, seed, lam_sem_ml, lam_sem_pr,
                       patience, out_dir):
    """Train both systems for one seed at one compression ratio (nsym symbols).
    Seeded once, then Max Logit followed by Prototype, as in Phase I production.
    """
    set_seed(seed)
    jscc_a = SwinJSCCSystem(nsym, pretrained=True,
                             patch_grid=cfg["PATCH_GRID_ML"], patch_dim=cfg["PATCH_DIM_ML"],
                             stage_idx=cfg.get("BACKBONE_STAGE_ML", 2)).to(device)
    max_logit = MaxLogitDetector(cfg["PATCH_DIM_ML"], cfg["PATCH_GRID_ML"], p1["num_classes"]).to(device)
    train_joint_system(jscc_a, max_logit, p1["loaders"]["cls_train"], p1["loaders"]["cls_val"],
                        cfg, device, p1["num_classes"],
                        jscc_save_path=out_dir / f"jscc_a_{cr_name}_seed{seed}.pt",
                        detector_save_path=out_dir / f"max_logit_{cr_name}_seed{seed}.pt",
                        lambda_rec=cfg["LAMBDA_REC_ML"], lambda_sem=lam_sem_ml,
                        joint_lr=cfg["JOINT_LR_ML"], joint_momentum=cfg["JOINT_MOMENTUM_ML"],
                        weight_decay=cfg["WEIGHT_DECAY_ML"], joint_lr_power=cfg["JOINT_LR_POWER_ML"],
                        patience=patience, min_rel_improve=cfg["MIN_REL_IMPROVE_ML"],
                        max_epochs=cfg["JOINT_MAX_EPOCHS"])

    # No re-seeding between the two systems (as in phase1_train.phase1).
    jscc_b = SwinJSCCSystem(nsym, pretrained=True,
                             patch_grid=cfg["PATCH_GRID_PR"], patch_dim=cfg["PATCH_DIM_PR"],
                             stage_idx=cfg.get("BACKBONE_STAGE_PR", 2)).to(device)
    prototype = PrototypeDetector(cfg["PATCH_DIM_PR"], cfg["PATCH_GRID_PR"], p1["num_classes"],
                                   cfg["PROTOTYPE_SCALE"]).to(device)
    train_joint_system(jscc_b, prototype, p1["loaders"]["cls_train"], p1["loaders"]["cls_val"],
                        cfg, device, p1["num_classes"],
                        jscc_save_path=out_dir / f"jscc_b_{cr_name}_seed{seed}.pt",
                        detector_save_path=out_dir / f"prototype_{cr_name}_seed{seed}.pt",
                        lambda_rec=cfg["LAMBDA_REC_PR"], lambda_sem=lam_sem_pr,
                        joint_lr=cfg["JOINT_LR_PR"], joint_momentum=cfg["JOINT_MOMENTUM_PR"],
                        weight_decay=cfg["WEIGHT_DECAY_PR"], joint_lr_power=cfg["JOINT_LR_POWER_PR"],
                        patience=patience, min_rel_improve=cfg["MIN_REL_IMPROVE_PR"],
                        max_epochs=cfg["JOINT_MAX_EPOCHS"])
    return jscc_a, max_logit, jscc_b, prototype


def phase3(p1_grid, cfg, device):
    """Compression study over every compression ratio and seed (see the module
    docstring). Returns {"by_cr", "snr_by_cr", "n_runs"}: by_cr aggregates each CR at
    DEFAULT_SNR_DB, snr_by_cr each CR over the SNR sweep.
    """
    print(f"\n{'='*62}\n  PHASE III -- Compression Study "
          f"(N={len(cfg['SEEDS'])} seeds per CR)\n{'='*62}")
    lam_sem_ml_default = p1_grid["grid_analysis"]["winning_lambda_sem"]["max_logit"]
    lam_sem_pr_default = p1_grid["grid_analysis"]["winning_lambda_sem"]["prototype"]
    patience = p1_grid["grid_analysis"]["derived_patience"]
    seeds = cfg["SEEDS"]

    p1_default = p1_grid["final_runs"][0]
    # Phase III results are written next to Phase II's (run_0/).
    out = p1_grid["out"]
    ldr = p1_default["loaders"]["test"]
    npx = 3 * cfg["IMG_H"] * cfg["IMG_W"]

    state_path = Path(out) / "phase3_multiseed_state.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    if state:
        n_done = sum(len(v) for v in state.values())
        print(f"  RESUMING -- found {n_done} already-completed (cr, seed) result(s) at {state_path}")

    by_cr = {}
    snr_by_cr = {}
    output_path = Path(out) / "phase3.json"

    for cr_name, nsym in cfg["CHANNEL_SYMBOLS"].items():
        cr_ratio = nsym / npx
        state.setdefault(cr_name, {})
        print(f"\n  CR={cr_name} ({nsym} symbols, ratio={cr_ratio:.5f})")

        cr_out_dir = Path(out) / f"PHASE3_{cr_name}"
        cr_out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = _local_tmp_dir(f"PHASE3_{cr_name}")
        if cr_name != cfg["DEFAULT_CR"]:
            lam_sem_ml, lam_sem_pr = _grid_search_lambda_sem_for_cr(
                cfg, device, p1_default, cr_name, nsym, cr_out_dir)
        else:
            lam_sem_ml, lam_sem_pr = lam_sem_ml_default, lam_sem_pr_default

        for idx, seed in enumerate(seeds):
            seed_key = str(seed)
            reuse_from_p1 = cr_name == cfg["DEFAULT_CR"] and idx < len(p1_grid["final_runs"])
            _fp = {"max_logit": training_fingerprint(cfg, "max_logit"),
                   "prototype": training_fingerprint(cfg, "prototype"),
                   "eval": evaluation_fingerprint(cfg), "nsym": nsym, "patience": patience,
                   "lambda_sem_ml": lam_sem_ml, "lambda_sem_pr": lam_sem_pr}
            if reuse_from_p1:
                # For reused models the thresholds identify the production models.
                _fp["tau_max_logit"] = p1_grid["final_runs"][idx]["tau_max_logit"]
                _fp["tau_prototype"] = p1_grid["final_runs"][idx]["tau_prototype"]
            _fp = fingerprint_key(_fp)
            if state[cr_name].get(seed_key, {}).get("_fp") == _fp:
                print(f"    seed={seed}: SKIP, already done with the current settings (resumed)")
                continue
            if seed_key in state[cr_name]:
                print(f"    seed={seed}: stored result was produced under different settings -- re-running")

            if reuse_from_p1:
                print(f"    seed={seed}: reusing Phase I production repetition "
                      f"{idx+1}/{len(p1_grid['final_runs'])}.")
                p1_rep = p1_grid["final_runs"][idx]
                jscc_a, max_logit = p1_rep["jscc_a"], p1_rep["max_logit_det"]
                jscc_b, prototype = p1_rep["jscc_b"], p1_rep["prototype_det"]
                tau_ml, tau_pr = p1_rep["tau_max_logit"], p1_rep["tau_prototype"]
                sdop_ml, sdop_pr = p1_rep["sdop_threshold_a"], p1_rep["sdop_threshold_b"]
                ckpt_paths = None
            else:
                print(f"    seed={seed}: training both systems at this CR "
                      f"(lambda_sem_ml={lam_sem_ml}, lambda_sem_pr={lam_sem_pr}, "
                      f"patience={patience}, "
                      f"{'CR-specific grid winner' if cr_name != cfg['DEFAULT_CR'] else 'grid-derived'})...")
                jscc_a, max_logit, jscc_b, prototype = _train_one_cr_rep(
                    cfg, device, p1_default, cr_name, nsym, seed,
                    lam_sem_ml, lam_sem_pr, patience, ckpt_dir)
                jscc_a.eval(); max_logit.eval(); jscc_b.eval(); prototype.eval()
                tau_ml = calibrate_tau(jscc_a, max_logit, p1_default["loaders"]["val_mixed"], device,
                                        target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
                tau_pr = calibrate_tau(jscc_b, prototype, p1_default["loaders"]["val_mixed"], device,
                                        target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
                sdop_ml = calibrate_sdop_threshold(jscc_a, p1_default["loaders"]["val_mixed"], device,
                                                    snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
                sdop_pr = calibrate_sdop_threshold(jscc_b, p1_default["loaders"]["val_mixed"], device,
                                                    snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
                ckpt_paths = [ckpt_dir / f"jscc_a_{cr_name}_seed{seed}.pt",
                              ckpt_dir / f"max_logit_{cr_name}_seed{seed}.pt",
                              ckpt_dir / f"jscc_b_{cr_name}_seed{seed}.pt",
                              ckpt_dir / f"prototype_{cr_name}_seed{seed}.pt"]

            sweep = {}
            for snr in cfg["SNR_RANGE_DB"]:
                sweep[str(snr)] = eval_both_systems(jscc_a, max_logit, tau_ml, sdop_ml,
                                                     jscc_b, prototype, tau_pr, sdop_pr,
                                                     ldr, snr, cfg, device, "awgn",
                                                     noise_seed=eval_noise_seed(cfg, snr, "awgn"))
            # The DEFAULT_SNR_DB result is the sweep point itself (same seeded noise).
            if str(cfg["DEFAULT_SNR_DB"]) in sweep:
                r0 = sweep[str(cfg["DEFAULT_SNR_DB"])]
            else:
                r0 = eval_both_systems(jscc_a, max_logit, tau_ml, sdop_ml, jscc_b, prototype, tau_pr, sdop_pr,
                                       ldr, cfg["DEFAULT_SNR_DB"], cfg, device, "awgn",
                                       noise_seed=eval_noise_seed(cfg, cfg["DEFAULT_SNR_DB"], "awgn"))

            state[cr_name][seed_key] = {"r0": r0, "sweep": sweep, "_fp": _fp,
                                         "lambda_sem_ml": lam_sem_ml, "lambda_sem_pr": lam_sem_pr}
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)

            # Free GPU memory of models trained here (reused Phase I models stay referenced
            # by p1_grid).
            del jscc_a, max_logit, jscc_b, prototype
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            deleted_ok = True
            if ckpt_paths is not None:
                for p in ckpt_paths:
                    p.unlink(missing_ok=True)
                    if p.exists():
                        deleted_ok = False
                total_bytes = sum(f.stat().st_size for f in ckpt_dir.glob("*") if f.is_file())
                print(f"    seed={seed}: done, ML_AUC={r0['max_logit_auc_roc_pixel']:.4f}, "
                      f"PR_AUC={r0['prototype_auc_roc_pixel']:.4f}, "
                      f"checkpoints {'deleted OK' if deleted_ok else 'DELETION FAILED'}, "
                      f"local checkpoint folder now {total_bytes/1e9:.2f}GB")
            else:
                print(f"    seed={seed}: done, ML_AUC={r0['max_logit_auc_roc_pixel']:.4f}, "
                      f"PR_AUC={r0['prototype_auc_roc_pixel']:.4f} (reused, no checkpoints to delete)")

        by_cr_reps = [state[cr_name][str(s)]["r0"] for s in seeds]
        by_cr_agg = _aggregate_metric_dicts(by_cr_reps)
        by_cr_agg.update({"cr_name": cr_name, "n_symbols": nsym, "cr_ratio": cr_ratio,
                           "lambda_sem_ml": lam_sem_ml, "lambda_sem_pr": lam_sem_pr})
        by_cr[cr_name] = by_cr_agg
        print(f"    SNR={cfg['DEFAULT_SNR_DB']:+d}dB  "
              f"MaxLogit AUC={by_cr_agg.get('max_logit_auc_roc_pixel_mean', float('nan')):.4f} "
              f"+/- {by_cr_agg.get('max_logit_auc_roc_pixel_std', 0.0):.4f}  "
              f"Prototype AUC={by_cr_agg.get('prototype_auc_roc_pixel_mean', float('nan')):.4f} "
              f"+/- {by_cr_agg.get('prototype_auc_roc_pixel_std', 0.0):.4f}  "
              f"(N={len(by_cr_reps)} seeds)")

        sweep_agg = []
        for snr in cfg["SNR_RANGE_DB"]:
            sweep_reps = [state[cr_name][str(s)]["sweep"][str(snr)] for s in seeds]
            agg_point = _aggregate_metric_dicts(sweep_reps)
            agg_point.update({"cr_name": cr_name, "cr_ratio": cr_ratio})
            sweep_agg.append(agg_point)
        snr_by_cr[cr_name] = sweep_agg

        with open(output_path, "w") as f:
            json.dump({"by_cr": by_cr, "snr_by_cr": snr_by_cr, "n_runs": len(seeds)}, f, indent=2)

    output = {"by_cr": by_cr, "snr_by_cr": snr_by_cr, "n_runs": len(seeds)}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Phase III] Done (N={len(seeds)} seeds/CR, mean +/- std, "
          f"per-CR lambda_sem) -> {output_path}")
    return output
