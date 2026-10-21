"""Detector x architecture control (dissertation Section 3.1.9 and Appendix F.3): is
the Prototype system's lower SNR50 due to the detector or to the architecture it
was selected with?

The production systems differ in both. Max Logit uses architecture A_ML (Swin
stage BACKBONE_STAGE_ML, latent PATCH_GRID_ML x PATCH_GRID_ML x PATCH_DIM_ML) and
Prototype uses A_PR (BACKBONE_STAGE_PR, PATCH_GRID_PR x PATCH_GRID_PR x
PATCH_DIM_PR); both latents hold 1,568 values sent over 2,048 channel symbols.
This check trains the two missing cells of a 2x2 design for the first
ARCH2X2_N_SEEDS production seeds:

    Max Logit on A_PR   (Max Logit's Phase I training recipe)
    Prototype on A_ML   (Prototype's Phase I training recipe)

and pairs them by seed with the production cells (Max Logit on A_ML, Prototype
on A_PR: Phase I repetitions 0..N-1). Training at 40 dB, calibration and
evaluation follow Phase I and Phase II (same sweep and seeded channel noise, AWGN
and Rayleigh). Nothing is re-tuned for the swapped architectures.

Effects per seed on SNR_X (Decision Retention, src/decision_retention.py):
    detector effect on A_ML     = PR(A_ML) - ML(A_ML)
    detector effect on A_PR     = PR(A_PR) - ML(A_PR)
    architecture effect, ML     = ML(A_PR) - ML(A_ML)
    architecture effect, PR     = PR(A_PR) - PR(A_ML)
    main detector effect        = mean of the two detector effects
    main architecture effect    = mean of the two architecture effects
    interaction                 = detector effect on A_PR - detector effect on A_ML
Negative = lower SNR_X (decisions kept down to a lower SNR). Each effect is
summarised by its mean, SD, sign counts, exact Wilcoxon signed-rank p (at n = 5
the smallest two-sided p is 0.0625) and one-sample t-test p.

Collapse guard: each new model is checked for input-independent output as in
phases/multisnr_check.collapse_flags; flagged seeds are reported and the effects
are repeated without them.

Outputs: run_0/arch2x2/ (resumable per seed), tables/arch2x2_check.md and
plots/13_arch2x2_*.
"""
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from src.models import SwinJSCCSystem, MaxLogitDetector, PrototypeDetector
from src.evaluate import eval_both_systems, eval_noise_seed
from src.train import train_joint_system
from src.metrics import calibrate_tau, calibrate_sdop_threshold
from src.config import set_seed, training_fingerprint, evaluation_fingerprint, fingerprint_key
from src.decision_retention import decision_retention_analysis
from phases.location_prior_check import input_dependence as _input_dependence
from phases.multisnr_check import collapse_flags

CELLS = (("ml_aml", "Max Logit", "A_ML"), ("pr_aml", "Prototype", "A_ML"),
         ("ml_apr", "Max Logit", "A_PR"), ("pr_apr", "Prototype", "A_PR"))


def architectures(cfg):
    """The two production architectures, read from the selected configuration."""
    return {"A_ML": {"stage": cfg.get("BACKBONE_STAGE_ML", 2), "grid": cfg["PATCH_GRID_ML"], "dim": cfg["PATCH_DIM_ML"]},
            "A_PR": {"stage": cfg.get("BACKBONE_STAGE_PR", 2), "grid": cfg["PATCH_GRID_PR"], "dim": cfg["PATCH_DIM_PR"]}}


def arch_label(a):
    """Short text label of an architecture, e.g. 'stage 2, 14x14x8'."""
    return f"stage {a['stage']}, {a['grid']}x{a['grid']}x{a['dim']}"


def cross_cfgs(cfg):
    """Configurations of the two new cells: each detector's settings with the other
    production architecture. Used to build the models and for the fingerprints
    (training_fingerprint reads stage, grid and dim per detector).
    """
    A = architectures(cfg)
    c_ml = copy.deepcopy(cfg)
    c_ml["BACKBONE_STAGE_ML"], c_ml["PATCH_GRID_ML"], c_ml["PATCH_DIM_ML"] = A["A_PR"]["stage"], A["A_PR"]["grid"], A["A_PR"]["dim"]
    c_pr = copy.deepcopy(cfg)
    c_pr["BACKBONE_STAGE_PR"], c_pr["PATCH_GRID_PR"], c_pr["PATCH_DIM_PR"] = A["A_ML"]["stage"], A["A_ML"]["grid"], A["A_ML"]["dim"]
    return c_ml, c_pr


def _train_cross_pair(cfg, device, p1, seed, lam_ml, lam_pr, patience, nsym, out_dir):
    """Train Max Logit on A_PR, then Prototype on A_ML, seeded once as in Phase I
    production; each uses its own Phase I recipe and the 40 dB training channel.
    """
    A = architectures(cfg)
    set_seed(seed)
    loaders = p1["loaders"]
    a = A["A_PR"]
    jscc_ml = SwinJSCCSystem(nsym, pretrained=True, patch_grid=a["grid"], patch_dim=a["dim"],
                             stage_idx=a["stage"]).to(device)
    det_ml = MaxLogitDetector(a["dim"], a["grid"], p1["num_classes"]).to(device)
    r_ml = train_joint_system(
        jscc_ml, det_ml, loaders["cls_train"], loaders["cls_val"], cfg, device, p1["num_classes"],
        jscc_save_path=out_dir / f"jscc_ml_apr_seed{seed}.pt", detector_save_path=out_dir / f"det_ml_apr_seed{seed}.pt",
        lambda_rec=cfg["LAMBDA_REC_ML"], lambda_sem=lam_ml,
        joint_lr=cfg["JOINT_LR_ML"], joint_momentum=cfg["JOINT_MOMENTUM_ML"],
        weight_decay=cfg["WEIGHT_DECAY_ML"], joint_lr_power=cfg["JOINT_LR_POWER_ML"],
        patience=patience, min_rel_improve=cfg["MIN_REL_IMPROVE_ML"], max_epochs=cfg["JOINT_MAX_EPOCHS"])
    a = A["A_ML"]
    jscc_pr = SwinJSCCSystem(nsym, pretrained=True, patch_grid=a["grid"], patch_dim=a["dim"],
                             stage_idx=a["stage"]).to(device)
    det_pr = PrototypeDetector(a["dim"], a["grid"], p1["num_classes"], cfg["PROTOTYPE_SCALE"]).to(device)
    r_pr = train_joint_system(
        jscc_pr, det_pr, loaders["cls_train"], loaders["cls_val"], cfg, device, p1["num_classes"],
        jscc_save_path=out_dir / f"jscc_pr_aml_seed{seed}.pt", detector_save_path=out_dir / f"det_pr_aml_seed{seed}.pt",
        lambda_rec=cfg["LAMBDA_REC_PR"], lambda_sem=lam_pr,
        joint_lr=cfg["JOINT_LR_PR"], joint_momentum=cfg["JOINT_MOMENTUM_PR"],
        weight_decay=cfg["WEIGHT_DECAY_PR"], joint_lr_power=cfg["JOINT_LR_POWER_PR"],
        patience=patience, min_rel_improve=cfg["MIN_REL_IMPROVE_PR"], max_epochs=cfg["JOINT_MAX_EPOCHS"])
    return jscc_ml, det_ml, jscc_pr, det_pr, r_ml, r_pr


def _effect_summary(vals):
    """Mean, SD, sign counts, Wilcoxon and t-test p of per-seed effects (None values skipped)."""
    d = np.array([v for v in vals if v is not None], dtype=float)
    out = {"n": int(d.size), "mean_db": float(d.mean()) if d.size else None,
           "sd_db": float(d.std(ddof=1)) if d.size > 1 else None,
           "n_negative": int((d < 0).sum()), "n_positive": int((d > 0).sum()),
           "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if d.size >= 2 and np.any(d != 0) else None,
           "ttest_p": (float(stats.ttest_1samp(d, 0.0).pvalue)
                       if d.size >= 2 and np.std(d, ddof=1) > 0 else None),
           "values": d.tolist()}
    return out


def _cell_snrx(dr_prod, dr_cross, ch, ref, lv):
    """Per-seed SNR_X of the four cells (None = censored)."""
    k = str(lv)
    def get(dr, det):
        blk = dr["channels"].get(ch, {}).get("auc", {}).get(ref)
        return None if blk is None else blk["detectors"][det]["levels"][k]["per_seed_snr_db"]
    def status(dr, det):
        blk = dr["channels"].get(ch, {}).get("auc", {}).get(ref)
        return None if blk is None else blk["detectors"][det]["levels"][k]["per_seed_status"]
    vals = {"ml_aml": get(dr_prod, "max_logit"), "pr_apr": get(dr_prod, "prototype"),
            "ml_apr": get(dr_cross, "max_logit"), "pr_aml": get(dr_cross, "prototype")}
    stat = {"ml_aml": status(dr_prod, "max_logit"), "pr_apr": status(dr_prod, "prototype"),
            "ml_apr": status(dr_cross, "max_logit"), "pr_aml": status(dr_cross, "prototype")}
    return vals, stat


def arch2x2_analysis(prod_per_seed, cross_per_seed, snrs, cfg, seeds):
    """Per-cell SNR_X and 2x2 effects (no GPU).

    prod_per_seed: Phase II per-seed results of the first len(seeds) repetitions
    (max_logit_* = ML on A_ML, prototype_* = PR on A_PR); cross_per_seed: the new
    cells (max_logit_* = ML on A_PR, prototype_* = PR on A_ML).
    """
    dr_prod = decision_retention_analysis(prod_per_seed, snrs, cfg)
    dr_cross = decision_retention_analysis(cross_per_seed, snrs, cfg)
    out = {"seeds": list(seeds), "levels": list(cfg["DR_HEADLINE_LEVELS"]),
           "architectures": {k: dict(v, label=arch_label(v)) for k, v in architectures(cfg).items()},
           "dr_production": dr_prod, "dr_cross": dr_cross, "results": {}}
    snr_f = [float(s) for s in snrs]
    i40 = snr_f.index(float(cfg["CLS_FIT_SNR"]))
    clean = {}
    for cell, src, det in (("ml_aml", prod_per_seed, "max_logit"), ("pr_apr", prod_per_seed, "prototype"),
                           ("ml_apr", cross_per_seed, "max_logit"), ("pr_aml", cross_per_seed, "prototype")):
        v = [run[i40][f"{det}_auc_roc_pixel"] for run in src["awgn"]]
        rng = [max(pt[f"{det}_auc_roc_pixel"] for j, pt in enumerate(run) if j != i40)
               - min(pt[f"{det}_auc_roc_pixel"] for j, pt in enumerate(run) if j != i40) for run in src["awgn"]]
        clean[cell] = {"auc_40db_mean": float(np.mean(v)), "auc_40db_sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                       "auc_range_over_sweep_min": float(np.min(rng))}
    out["clean_auroc"] = clean
    out["guard_auroc_range"] = float(cfg["COLLAPSE_GUARD_AUROC_RANGE"])
    for ch in ("awgn", "rayleigh"):
        for ref in ("ref", "peak"):
            for lv in cfg["DR_HEADLINE_LEVELS"]:
                vals, stat = _cell_snrx(dr_prod, dr_cross, ch, ref, lv)
                if any(v is None for v in vals.values()):
                    continue
                n = len(seeds)
                def comb(f):
                    res = []
                    for i in range(n):
                        x = {c: vals[c][i] for c in vals}
                        res.append(None if any(v is None for v in x.values()) else f(x))
                    return res
                eff = {
                    "detector_on_A_ML": comb(lambda x: x["pr_aml"] - x["ml_aml"]),
                    "detector_on_A_PR": comb(lambda x: x["pr_apr"] - x["ml_apr"]),
                    "architecture_ML": comb(lambda x: x["ml_apr"] - x["ml_aml"]),
                    "architecture_PR": comb(lambda x: x["pr_apr"] - x["pr_aml"]),
                    "main_detector": comb(lambda x: ((x["pr_aml"] - x["ml_aml"]) + (x["pr_apr"] - x["ml_apr"])) / 2),
                    "main_architecture": comb(lambda x: ((x["ml_apr"] - x["ml_aml"]) + (x["pr_apr"] - x["pr_aml"])) / 2),
                    "interaction": comb(lambda x: (x["pr_apr"] - x["ml_apr"]) - (x["pr_aml"] - x["ml_aml"])),
                    "production_gap": comb(lambda x: x["pr_apr"] - x["ml_aml"]),
                }
                cells = {}
                for c in vals:
                    v = [x for x in vals[c] if x is not None]
                    cells[c] = {"per_seed_snr_db": vals[c], "per_seed_status": stat[c], "n": len(v),
                                "n_left_censored": sum(x == "left_censored" for x in stat[c]),
                                "n_not_reached": sum(x == "not_reached" for x in stat[c]),
                                "n_undefined": sum(x == "undefined" for x in stat[c]),
                                "mean_db": float(np.mean(v)) if v else None,
                                "sd_db": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                out["results"].setdefault(ch, {}).setdefault(ref, {})[str(lv)] = {
                    "cells": cells, "effects": {k: _effect_summary(v) for k, v in eff.items()}}
    return out


EFFECT_NAMES = (("detector_on_A_ML", "detector effect on A_ML  (PR - ML)"),
                ("detector_on_A_PR", "detector effect on A_PR  (PR - ML)"),
                ("architecture_ML", "architecture effect, ML (A_PR - A_ML)"),
                ("architecture_PR", "architecture effect, PR (A_PR - A_ML)"),
                ("main_detector", "main detector effect"),
                ("main_architecture", "main architecture effect"),
                ("interaction", "interaction"),
                ("production_gap", "production gap (PR on A_PR - ML on A_ML)"))


def _f(v, nd=2, sign=True):
    return "--" if v is None else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def _report(an):
    A = an["architectures"]
    L = [f"  A_ML = {A['A_ML']['label']} (Max Logit's production architecture); "
         f"A_PR = {A['A_PR']['label']} (Prototype's)",
         f"  Seeds {an['seeds']}; Max Logit on A_ML and Prototype on A_PR are the production repetitions "
         f"with these seeds.",
         "  Clean AUROC at 40 dB (mean +/- SD) and smallest AUROC range over the -20..+20 dB sweep:"]
    for c, name, arch in CELLS:
        r = an["clean_auroc"][c]
        L.append(f"    {name:<9} on {arch}: {r['auc_40db_mean']:.4f} +/- {_f(r['auc_40db_sd'], 4, False)};  "
                 f"range {r['auc_range_over_sweep_min']:.3f}"
                 + ("   <- NEARLY CONSTANT OVER SNR: check this cell trained properly"
                    if r["auc_range_over_sweep_min"] < an["guard_auroc_range"] else ""))
    for ch, by_ref in an["results"].items():
        for ref, by_lv in by_ref.items():
            refname = "40 dB reference" if ref == "ref" else "each seed's best swept value as reference"
            for lv, blk in by_lv.items():
                L.append(f"\n  {ch.upper()}, DR_AUC SNR{int(round(float(lv)*100))} ({refname}), dB:")
                for c, name, arch in CELLS:
                    x = blk["cells"][c]
                    L.append(f"    {name:<9} on {arch}: {_f(x['mean_db'])} +/- {_f(x['sd_db'], 2, False)} "
                             f"(n={x['n']} crossed; {x['n_left_censored']} L, {x['n_not_reached']} NR, "
                             f"{x['n_undefined']} undef)")
                for k, name in EFFECT_NAMES:
                    e = blk["effects"][k]
                    L.append(f"    {name:<40} {_f(e['mean_db'])} +/- {_f(e['sd_db'], 2, False)} dB; "
                             f"negative in {e['n_negative']}/{e['n']}; Wilcoxon p="
                             f"{'--' if e['wilcoxon_p'] is None else format(e['wilcoxon_p'], '.3g')}, t-test p="
                             f"{'--' if e['ttest_p'] is None else format(e['ttest_p'], '.3g')}")
    L.append("\n  Negative effect = decisions kept down to a lower SNR. Exact Wilcoxon p cannot go below 0.0625")
    L.append("  at n = 5; read consistency from the sign counts, size from the means. Effects use the seeds")
    L.append("  where all four cells crossed; 'L' = left-censored, 'NR' = level not reached.")
    L.append("  Identity, per seed: production gap = main detector effect + main architecture effect.")
    if an.get("guard"):
        L.append("\n  Collapse guard of the two new cells (input-independent output), per seed:")
        for s, g in an["guard"].items():
            for det, name, arch in (("max_logit", "Max Logit", "A_PR"), ("prototype", "Prototype", "A_ML")):
                x = g[det]
                mms = "--" if x["matched_minus_shuffled"] is None else f"{x['matched_minus_shuffled']:.3f}"
                ep = an.get("epochs", {}).get(s, {}).get("ml_apr" if det == "max_logit" else "pr_aml", "--")
                L.append(f"    seed {s:>5} {name} on {arch}: AUROC range {x['auroc_range']:.3f}, matched - shuffled "
                         f"{mms}, epochs {ep}" + (f"   <- FLAGGED: {'; '.join(x['reasons'])}" if x["flagged"] else ""))
        L.append(f"    Flagged seeds: {an.get('flagged_seeds') or 'none'}")
    return "\n".join(L)


def _plots(an, out_root, prod, cross, cfg):
    import matplotlib.pyplot as plt
    pdir = Path(out_root) / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    A = an["architectures"]
    blk = an["results"]["awgn"]["ref"]
    fig, axes = plt.subplots(1, len(blk), figsize=(6 * len(blk), 4.5), squeeze=False)
    for ax, (lv, b) in zip(axes[0], blk.items()):
        for det, name, color, mk, cells in (("ml", "Max Logit", "#1f77b4", "o", ("ml_aml", "ml_apr")),
                                            ("pr", "Prototype", "#2ca02c", "D", ("pr_aml", "pr_apr"))):
            v = np.array([[np.nan if x is None else x for x in b["cells"][c]["per_seed_snr_db"]] for c in cells])
            for s in range(v.shape[1]):
                ax.plot([0, 1], v[:, s], color=color, alpha=.25, lw=1)
            ax.plot([0, 1], np.nanmean(v, axis=1), marker=mk, color=color, lw=2.5, label=name)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"A_ML\n({A['A_ML']['label']})", f"A_PR\n({A['A_PR']['label']})"])
        ax.set(xlim=(-0.3, 1.3), ylabel=f"SNR{int(round(float(lv)*100))} (dB), AWGN", xlabel="Architecture")
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(pdir / f"13_arch2x2_snrx.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    snrs = [float(s) for s in cfg["SNR_RANGE_DB"]]
    keep = [i for i, s in enumerate(snrs) if s != float(cfg["CLS_FIT_SNR"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    for src, det, name, arch, color, ls in ((prod, "max_logit", "Max Logit", "A_ML", "#1f77b4", "-"),
                                            (cross, "max_logit", "Max Logit", "A_PR", "#1f77b4", "--"),
                                            (cross, "prototype", "Prototype", "A_ML", "#2ca02c", "--"),
                                            (prod, "prototype", "Prototype", "A_PR", "#2ca02c", "-")):
        a = np.array([[run[i][f"{det}_auc_roc_pixel"] for i in keep] for run in src["awgn"]], dtype=float)
        ax.plot([snrs[i] for i in keep], a.mean(0), ls=ls, color=color, lw=2,
                label=f"{name} on {arch} ({A[arch]['label']})")
    ax.axhline(0.5, color="black", ls=":", lw=.8, alpha=.4, label="Chance")
    ax.set(xlabel="SNR (dB)", ylabel="Pixel AUROC (AWGN, mean over seeds)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(pdir / f"13b_arch2x2_auroc.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[2x2 check] plots -> {pdir}/13_arch2x2_snrx.png, 13b_arch2x2_auroc.png (+ .pdf)")


def run_arch2x2_check(p1_grid, cfg, device):
    """Train the two new cells (resumable per seed), evaluate as in Phase II,
    analyse and plot. Needs p1_grid (Phase I: data loaders and selected settings)
    and run_0/phase2_per_seed.json; the Phase I models are not used.
    """
    seeds = list(cfg["SEEDS"][:cfg["ARCH2X2_N_SEEDS"]])
    p1 = p1_grid["final_runs"][0]
    lam_ml = p1_grid["grid_analysis"]["winning_lambda_sem"]["max_logit"]
    lam_pr = p1_grid["grid_analysis"]["winning_lambda_sem"]["prototype"]
    patience = p1_grid["grid_analysis"]["derived_patience"]
    nsym = cfg["CHANNEL_SYMBOLS"][cfg["DEFAULT_CR"]]
    out_root = Path(p1_grid["out"])
    res_dir = out_root / "arch2x2"; res_dir.mkdir(parents=True, exist_ok=True)
    state_path = res_dir / "arch2x2_state.json"
    from phases.phase3_compression import _local_tmp_dir
    ckpt_dir = _local_tmp_dir("ARCH2X2")

    per_seed_path = out_root / "phase2_per_seed.json"
    if not per_seed_path.exists():
        raise FileNotFoundError(f"{per_seed_path} not found -- run Phase II first.")
    prod_all = json.load(open(per_seed_path))
    snrs = [float(s) for s in cfg["SNR_RANGE_DB"]]
    if [float(s) for s in prod_all["snr_range_db"]] != snrs:
        raise ValueError("phase2_per_seed.json SNR grid differs from SNR_RANGE_DB -- stale Phase II file.")
    if len(prod_all["awgn"]) < len(seeds):
        raise ValueError("phase2_per_seed.json has fewer repetitions than ARCH2X2_N_SEEDS.")

    A = architectures(cfg)
    if A["A_ML"] == A["A_PR"]:
        raise ValueError("Both detectors use the same architecture -- the 2x2 design is not defined.")
    c_ml, c_pr = cross_cfgs(cfg)
    fp = fingerprint_key({"ml_apr": training_fingerprint(c_ml, "max_logit"),
                          "pr_aml": training_fingerprint(c_pr, "prototype"),
                          "eval": evaluation_fingerprint(cfg), "nsym": nsym, "patience": patience,
                          "lam_ml": lam_ml, "lam_pr": lam_pr, "train_snr": cfg["CLS_FIT_SNR"], "version": 1})
    state = json.load(open(state_path)) if state_path.exists() else {}
    print(f"\n{'='*62}\n  2x2 DETECTOR x ARCHITECTURE CHECK -- {len(seeds)} seeds {seeds}\n"
          f"  A_ML = {arch_label(A['A_ML'])}, A_PR = {arch_label(A['A_PR'])}\n{'='*62}")
    for i, seed in enumerate(seeds):
        key = str(seed)
        if state.get(key, {}).get("_fp") == fp:
            print(f"  seed={seed}: SKIP, already done with the current settings (resumed)")
            continue
        print(f"  seed={seed} ({i+1}/{len(seeds)}): training Max Logit on A_PR, then Prototype on A_ML...")
        jscc_ml, det_ml, jscc_pr, det_pr, r_ml, r_pr = _train_cross_pair(cfg, device, p1, seed, lam_ml, lam_pr,
                                                                         patience, nsym, ckpt_dir)
        tau_ml = calibrate_tau(jscc_ml, det_ml, p1["loaders"]["val_mixed"], device,
                               target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
        tau_pr = calibrate_tau(jscc_pr, det_pr, p1["loaders"]["val_mixed"], device,
                               target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
        sdop_ml = calibrate_sdop_threshold(jscc_ml, p1["loaders"]["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"],
                                           percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
        sdop_pr = calibrate_sdop_threshold(jscc_pr, p1["loaders"]["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"],
                                           percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
        sweep = {"awgn": [], "rayleigh": []}
        for snr in cfg["SNR_RANGE_DB"]:
            r = eval_both_systems(jscc_ml, det_ml, tau_ml, sdop_ml, jscc_pr, det_pr, tau_pr, sdop_pr,
                                  p1["loaders"]["test"], snr, cfg, device, "awgn",
                                  noise_seed=eval_noise_seed(cfg, snr, "awgn"))
            sweep["awgn"].append(r)
            msg = (f"    SNR={snr:+4d}dB  Max Logit on A_PR AUC={r['max_logit_auc_roc_pixel']:.3f}  "
                   f"Prototype on A_ML AUC={r['prototype_auc_roc_pixel']:.3f}")
            if cfg["USE_RAYLEIGH"]:
                rr = eval_both_systems(jscc_ml, det_ml, tau_ml, sdop_ml, jscc_pr, det_pr, tau_pr, sdop_pr,
                                       p1["loaders"]["test"], snr, cfg, device, "rayleigh",
                                       noise_seed=eval_noise_seed(cfg, snr, "rayleigh"))
                sweep["rayleigh"].append(rr)
                msg += f"  | Rayleigh {rr['max_logit_auc_roc_pixel']:.3f} / {rr['prototype_auc_roc_pixel']:.3f}"
            print(msg)
        dep = {"max_logit": _input_dependence(jscc_ml, det_ml, p1["loaders"]["test"], cfg["CLS_FIT_SNR"], cfg, device),
               "prototype": _input_dependence(jscc_pr, det_pr, p1["loaders"]["test"], cfg["CLS_FIT_SNR"], cfg, device)}
        guard = collapse_flags(sweep["awgn"], dep, cfg)
        state[key] = {"_fp": fp, "sweep": sweep, "tau_ml_apr": tau_ml, "tau_pr_aml": tau_pr,
                      "epochs_ml_apr": r_ml["epochs_run"], "epochs_pr_aml": r_pr["epochs_run"],
                      "val_task_loss_ml_apr": r_ml["loss"], "val_task_loss_pr_aml": r_pr["loss"],
                      "input_dependence_40db": dep, "guard": guard}
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        del jscc_ml, det_ml, jscc_pr, det_pr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for p in ckpt_dir.glob(f"*_seed{seed}.pt"):
            p.unlink(missing_ok=True)
        flag_txt = "; ".join(f"{d}: {', '.join(g['reasons'])}" for d, g in guard.items() if g["flagged"])
        print(f"  seed={seed}: done (Max Logit on A_PR {r_ml['epochs_run']} epochs, Prototype on A_ML "
              f"{r_pr['epochs_run']} epochs), saved -> {state_path}, local checkpoints deleted"
              + (f"\n  seed={seed}: COLLAPSE GUARD FLAGGED -- {flag_txt}" if flag_txt else ""))

    cross = {"awgn": [state[str(s)]["sweep"]["awgn"] for s in seeds],
             "rayleigh": ([state[str(s)]["sweep"]["rayleigh"] for s in seeds] if cfg["USE_RAYLEIGH"] else []),
             "snr_range_db": cfg["SNR_RANGE_DB"], "n_seeds": len(seeds)}
    # Phase I repetition i used SEEDS[i].
    idx = [list(cfg["SEEDS"]).index(s) for s in seeds]
    prod = {"awgn": [prod_all["awgn"][i] for i in idx],
            "rayleigh": ([prod_all["rayleigh"][i] for i in idx] if prod_all.get("rayleigh") and cfg["USE_RAYLEIGH"] else []),
            "snr_range_db": prod_all["snr_range_db"], "n_seeds": len(seeds)}
    an = arch2x2_analysis(prod, cross, cfg["SNR_RANGE_DB"], cfg, seeds)
    an["epochs"] = {str(s): {"ml_apr": state[str(s)]["epochs_ml_apr"], "pr_aml": state[str(s)]["epochs_pr_aml"]}
                    for s in seeds}
    an["guard"] = {str(s): state[str(s)]["guard"] for s in seeds}
    flagged = [s for s in seeds if any(g["flagged"] for g in an["guard"][str(s)].values())]
    an["flagged_seeds"] = flagged
    rep = _report(an)
    if flagged:
        keep = [k for k, s in enumerate(seeds) if s not in flagged]
        if len(keep) >= 2:
            sub = lambda d: {"awgn": [d["awgn"][k] for k in keep],
                             "rayleigh": [d["rayleigh"][k] for k in keep] if d.get("rayleigh") else [],
                             "snr_range_db": d["snr_range_db"], "n_seeds": len(keep)}
            an_ok = arch2x2_analysis(sub(prod), sub(cross), cfg["SNR_RANGE_DB"], cfg, [seeds[k] for k in keep])
            an["excluding_flagged"] = an_ok
            rep += (f"\n\n  SAME EFFECTS WITHOUT THE FLAGGED SEEDS {flagged} (n = {len(keep)}; read with caution):\n"
                    + _report(an_ok))
        else:
            rep += "\n\n  Fewer than 2 unflagged seeds -- no analysis without the flagged seeds."
    with open(res_dir / "arch2x2_summary.json", "w") as f:
        json.dump(an, f, indent=2)
    tdir = out_root / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "arch2x2_check.md", "w") as f:
        f.write(f"# 2x2 detector x architecture check ({len(seeds)} seeds {seeds})\n\n```\n{rep}\n```\n")
    print(rep)
    _plots(an, out_root, prod, cross, cfg)
    print(f"[2x2 check] summary -> {res_dir / 'arch2x2_summary.json'}, table -> {tdir / 'arch2x2_check.md'}")
    return an
