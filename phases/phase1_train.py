"""Phase I building blocks: train one system, train both production systems of one
repetition, and reload a finished repetition from its checkpoints.

System A: Swin-Tiny Deep JSCC + Max Logit detector (Hendrycks et al., 2022)
System B: Swin-Tiny Deep JSCC + Prototype detector (Fontanel et al., 2021)
Each system is trained end-to-end with its own encoder (src/train.py).
"""
from pathlib import Path
import json
import torch
from src.dataset import build_dataloaders
from src.config import LOADER_GENERATORS
from src.models import SwinJSCCSystem, MaxLogitDetector, PrototypeDetector
from src.train import train_joint_system
from src.metrics import calibrate_tau, calibrate_sdop_threshold

_DETECTOR_CLASSES = {"max_logit": MaxLogitDetector, "prototype": PrototypeDetector}
_DETECTOR_LABELS = {"max_logit": "Max Logit detector (Hendrycks et al., 2022)",
                     "prototype": "Prototype detector (Fontanel et al., 2021)"}


def train_one_system(cfg, device, loaders, num_classes, nsym, detector_type,
                      lambda_sem, patience, save_dir, tag):
    """Build and train one fresh (jscc, detector) system with the settings of
    detector_type ("max_logit" or "prototype") in cfg: backbone stage, patch grid and
    dimension, optimiser settings, LAMBDA_REC and, for Prototype, PROTOTYPE_SCALE.

    lambda_sem and patience are passed explicitly. Checkpoints are written to
    save_dir/{tag}_jscc.pt and {tag}_detector.pt. Returns train_joint_system's result
    with the trained "jscc" and "detector" added.
    """
    _sfx = "ML" if detector_type == "max_logit" else "PR"
    jscc = SwinJSCCSystem(nsym, pretrained=True,
                           patch_grid=cfg[f"PATCH_GRID_{_sfx}"], patch_dim=cfg[f"PATCH_DIM_{_sfx}"],
                           stage_idx=cfg.get(f"BACKBONE_STAGE_{_sfx}", 2)).to(device)
    if detector_type == "max_logit":
        detector = MaxLogitDetector(cfg[f"PATCH_DIM_{_sfx}"], cfg[f"PATCH_GRID_{_sfx}"], num_classes).to(device)
    elif detector_type == "prototype":
        detector = PrototypeDetector(cfg[f"PATCH_DIM_{_sfx}"], cfg[f"PATCH_GRID_{_sfx}"], num_classes,
                                      cfg["PROTOTYPE_SCALE"]).to(device)
    else:
        raise ValueError(f"Unknown detector_type: {detector_type}")

    result = train_joint_system(
        jscc, detector, loaders["cls_train"], loaders["cls_val"],
        cfg, device, num_classes,
        jscc_save_path=save_dir / f"{tag}_jscc.pt",
        detector_save_path=save_dir / f"{tag}_detector.pt",
        lambda_rec=cfg[f"LAMBDA_REC_{_sfx}"], lambda_sem=lambda_sem,
        joint_lr=cfg[f"JOINT_LR_{_sfx}"], joint_momentum=cfg[f"JOINT_MOMENTUM_{_sfx}"],
        weight_decay=cfg[f"WEIGHT_DECAY_{_sfx}"], joint_lr_power=cfg[f"JOINT_LR_POWER_{_sfx}"],
        patience=patience, min_rel_improve=cfg[f"MIN_REL_IMPROVE_{_sfx}"],
        max_epochs=cfg["JOINT_MAX_EPOCHS"])
    result["jscc"] = jscc
    result["detector"] = detector
    result["detector_type"] = detector_type
    result["tag"] = tag
    return result


def phase1(cfg, device, loaders, num_classes, nsym,
           lambda_sem_max_logit, lambda_sem_prototype, patience, run_idx=0):
    """Train both systems of one production repetition, calibrate each system's
    threshold tau and SDOP threshold on the calibration images at CLS_FIT_SNR, and
    write run_{run_idx}/phase1_meta.json (settings, thresholds, training summary) and
    config_snapshot.json. Returns the trained models, thresholds and loaders.
    """
    out = Path(cfg["OUTPUT_DIR"]) / f"run_{run_idx}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/2] System A: JSCC + {_DETECTOR_LABELS['max_logit']} -- "
          f"JOINT end-to-end training, {num_classes} known classes, "
          f"lambda_sem={lambda_sem_max_logit} (grid-search winner), "
          f"patience={patience} (grid-derived)...")
    m1 = train_one_system(cfg, device, loaders, num_classes, nsym, "max_logit",
                           lambda_sem_max_logit, patience, out, "jscc_a_max_logit")
    jscc_a, max_logit_det = m1["jscc"], m1["detector"]
    print(f"  Done: {m1['epochs_run']} epochs, final val_task_loss={m1['loss']:.4f}")

    print(f"\n[2/2] System B: JSCC + {_DETECTOR_LABELS['prototype']} -- "
          f"JOINT end-to-end training, INDEPENDENT from System A, "
          f"lambda_sem={lambda_sem_prototype} (grid-search winner), "
          f"patience={patience} (grid-derived)...")
    m2 = train_one_system(cfg, device, loaders, num_classes, nsym, "prototype",
                           lambda_sem_prototype, patience, out, "jscc_b_prototype")
    jscc_b, prototype_det = m2["jscc"], m2["detector"]
    print(f"  Done: {m2['epochs_run']} epochs, final val_task_loss={m2['loss']:.4f}")

    print("\nCalibrating tau separately for each system (pixel-level scores on the "
          "calibration images)...")
    tau_max_logit = calibrate_tau(jscc_a, max_logit_det, loaders["val_mixed"], device,
                                   target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
    tau_prototype  = calibrate_tau(jscc_b, prototype_det, loaders["val_mixed"], device,
                                   target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])

    print("\nCalibrating the SDOP threshold separately for each system...")
    sdop_threshold_a = calibrate_sdop_threshold(jscc_a, loaders["val_mixed"], device,
                                                 snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
    sdop_threshold_b = calibrate_sdop_threshold(jscc_b, loaders["val_mixed"], device,
                                                 snr_db=cfg["CLS_FIT_SNR"], percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])

    meta = {"tau_max_logit": tau_max_logit, "tau_prototype": tau_prototype,
            "sdop_threshold_a": sdop_threshold_a, "sdop_threshold_b": sdop_threshold_b,
            "n_symbols": nsym,
            "lambda_sem_max_logit": lambda_sem_max_logit, "lambda_sem_prototype": lambda_sem_prototype,
            "prototype_scale": cfg["PROTOTYPE_SCALE"],
            "joint_lr_ml": cfg["JOINT_LR_ML"],
            "joint_lr_pr": cfg["JOINT_LR_PR"],
            "joint_momentum_ml": cfg["JOINT_MOMENTUM_ML"], "joint_momentum_pr": cfg["JOINT_MOMENTUM_PR"],
            "weight_decay_ml": cfg["WEIGHT_DECAY_ML"], "weight_decay_pr": cfg["WEIGHT_DECAY_PR"],
            "joint_lr_power_ml": cfg["JOINT_LR_POWER_ML"], "joint_lr_power_pr": cfg["JOINT_LR_POWER_PR"],
            "min_rel_improve_ml": cfg["MIN_REL_IMPROVE_ML"], "min_rel_improve_pr": cfg["MIN_REL_IMPROVE_PR"],
            "lambda_rec_ml": cfg["LAMBDA_REC_ML"], "lambda_rec_pr": cfg["LAMBDA_REC_PR"],
            "patience_used": patience,
            "max_logit_epochs": m1["epochs_run"], "max_logit_val_task_loss": m1["loss"],
            "prototype_epochs": m2["epochs_run"], "prototype_val_task_loss": m2["loss"],
            "num_known_classes": num_classes, "known_class_ids": loaders["known_class_ids"],
            "cal_fallback": loaders.get("_cal_fallback", False),
            "patch_grid_ml": cfg["PATCH_GRID_ML"], "patch_dim_ml": cfg["PATCH_DIM_ML"],
            "patch_grid_pr": cfg["PATCH_GRID_PR"], "patch_dim_pr": cfg["PATCH_DIM_PR"],
            "architecture": "joint_end_to_end_two_independent_systems"}
    with open(out / "phase1_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out / "config_snapshot.json", "w") as f:
        json.dump({k: v for k, v in cfg.items()}, f, indent=2, default=str)

    print(f"\n[Phase I] Done. tau_max_logit={tau_max_logit:.5f}  tau_prototype={tau_prototype:.5f}")
    return {"jscc_a": jscc_a, "jscc_b": jscc_b,
            "max_logit_det": max_logit_det, "prototype_det": prototype_det,
            "tau_max_logit": tau_max_logit, "tau_prototype": tau_prototype,
            "sdop_threshold_a": sdop_threshold_a, "sdop_threshold_b": sdop_threshold_b,
            "num_classes": num_classes, "loaders": loaders, "out": out}


def load_phase1(cfg, device, run_idx=0):
    """Rebuild a finished repetition from run_{run_idx}/ without retraining: both
    systems at the stage and patch settings in cfg, their checkpoints, and the
    thresholds from phase1_meta.json.
    """
    out = Path(cfg["OUTPUT_DIR"]) / f"run_{run_idx}"
    nsym = cfg["CHANNEL_SYMBOLS"][cfg["DEFAULT_CR"]]

    meta = json.load(open(out / "phase1_meta.json"))
    num_classes = meta["num_known_classes"]

    jscc_a = SwinJSCCSystem(nsym, pretrained=False,
                             patch_grid=cfg["PATCH_GRID_ML"], patch_dim=cfg["PATCH_DIM_ML"],
                             stage_idx=cfg.get("BACKBONE_STAGE_ML", 2)).to(device)
    jscc_a.load_state_dict(torch.load(out / "jscc_a_max_logit_jscc.pt", map_location=device)["model"])
    jscc_a.eval()

    jscc_b = SwinJSCCSystem(nsym, pretrained=False,
                             patch_grid=cfg["PATCH_GRID_PR"], patch_dim=cfg["PATCH_DIM_PR"],
                             stage_idx=cfg.get("BACKBONE_STAGE_PR", 2)).to(device)
    jscc_b.load_state_dict(torch.load(out / "jscc_b_prototype_jscc.pt", map_location=device)["model"])
    jscc_b.eval()

    max_logit_det = MaxLogitDetector(cfg["PATCH_DIM_ML"], cfg["PATCH_GRID_ML"], num_classes).to(device)
    max_logit_det.load_state_dict(torch.load(out / "jscc_a_max_logit_detector.pt", map_location=device))
    max_logit_det.eval()

    prototype_det = PrototypeDetector(cfg["PATCH_DIM_PR"], cfg["PATCH_GRID_PR"], num_classes,
                                       cfg["PROTOTYPE_SCALE"]).to(device)
    prototype_det.load_state_dict(torch.load(out / "jscc_b_prototype_detector.pt", map_location=device))
    prototype_det.eval()

    # build_dataloaders replaces the registered loader generators; keep the ones
    # registered before this call, so the loaders of repetitions still to be trained
    # keep following their seed.
    _registered = list(LOADER_GENERATORS)
    loaders = build_dataloaders(cfg)
    LOADER_GENERATORS.extend(g for g in _registered if all(g is not h for h in LOADER_GENERATORS))

    print(f"[load_phase1] Loaded run_{run_idx}: tau_max_logit={meta['tau_max_logit']:.5f}  "
          f"tau_prototype={meta['tau_prototype']:.5f}  num_classes={num_classes}")
    return {"jscc_a": jscc_a, "jscc_b": jscc_b,
            "max_logit_det": max_logit_det, "prototype_det": prototype_det,
            "tau_max_logit": meta["tau_max_logit"], "tau_prototype": meta["tau_prototype"],
            "sdop_threshold_a": meta["sdop_threshold_a"], "sdop_threshold_b": meta["sdop_threshold_b"],
            "num_classes": num_classes, "loaders": loaders, "out": out}
