"""Training over a range of SNRs (dissertation Section 3.1.9, Appendix F.4 and
Appendix G): does the Max Logit vs Prototype difference remain when training and
test SNRs match?

The production models are trained at CLS_FIT_SNR = 40 dB and tested from -20 to
+20 dB. This check trains MULTISNR_N_SEEDS new pairs of systems (the first
MULTISNR_N_SEEDS of SEEDS, so each pair matches the production repetition with
the same seed) with a per-image SNR drawn uniformly from MULTISNR_TRAIN_RANGE_DB,
and evaluates them as in Phase II (same sweep, seeded channel noise and metrics).
Training otherwise follows production: train_joint_system, the Phase I
hyperparameters (not re-tuned for multi-SNR training), the grid-derived patience,
and tau and the SDOP threshold calibrated at 40 dB on the calibration subset.

Two variants, chosen by run_multisnr_check(warm_start=...) (default
MULTISNR_WARM_START):

  cold start   fresh models (ImageNet-pretrained Swin, random JSCC and
               detector). Outputs: run_0/multisnr/, tables/multisnr_check.md,
               plots 11 and 11b.
  warm start   each seed starts from its own production checkpoints
               (run_0_rep{i}/) and is fine-tuned with the same recipe and
               stopping rule (the learning-rate schedule restarts). Outputs:
               run_0/multisnr_ft/, tables/multisnr_finetune.md, plots 11c and 11d.
               The fine-tuned models receive more epochs than the 40 dB models;
               both detectors are fine-tuned in the same way.

Collapse guard: a new model is flagged as input-independent if its AWGN AUROC
range over the sweep is below COLLAPSE_GUARD_AUROC_RANGE, or if matched minus
shuffled AUROC at 40 dB (phases/location_prior_check.py) is below
COLLAPSE_GUARD_MIN_DEPENDENCE. Flagged seeds are reported and the comparison is
repeated without them.

The random-SNR channel is installed only on the new models during their training
call and replaced by the 40 dB channel before calibration and evaluation.
Production checkpoints are only read; temporary checkpoints go to local disk and
are deleted after evaluation.
"""
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from src.models import SwinJSCCSystem, MaxLogitDetector, PrototypeDetector
from src.evaluate import eval_both_systems, eval_noise_seed
from src.train import train_joint_system
from src.metrics import calibrate_tau, calibrate_sdop_threshold
from src.config import set_seed, training_fingerprint, evaluation_fingerprint, fingerprint_key
from src.decision_retention import decision_retention_analysis
from phases.location_prior_check import input_dependence as _input_dependence

# Production checkpoint files in each Phase I repetition folder (see phase1_train.load_phase1).
PRODUCTION_CKPTS = {"max_logit": ("jscc_a_max_logit_jscc.pt", "jscc_a_max_logit_detector.pt"),
                    "prototype": ("jscc_b_prototype_jscc.pt", "jscc_b_prototype_detector.pt")}


class RandomSNRAWGNChannel(nn.Module):
    """AWGN with a per-image SNR drawn uniformly (in dB) from [lo_db, hi_db].

    Same noise convention as src/channels.py AWGNChannel (noise SD from the
    whole-tensor mean signal power); only the SNR differs per image. Training mode
    draws from the global RNG (seeded by set_seed). Evaluation mode, used by the
    validation pass of train_joint_system, uses a private generator re-seeded each
    time the module enters evaluation mode, so every epoch is validated on the same
    SNR and noise draws.
    """

    def __init__(self, lo_db, hi_db, eval_seed):
        super().__init__()
        if not lo_db < hi_db:
            raise ValueError(f"MULTISNR_TRAIN_RANGE_DB must be increasing, got {lo_db}, {hi_db}")
        self.lo_db, self.hi_db, self.eval_seed = float(lo_db), float(hi_db), int(eval_seed)
        self._gen = None

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            # Re-seeded at the first forward pass of each evaluation pass.
            self._gen = None
        return self

    def forward(self, X):
        shape = (X.shape[0],) + (1,) * (X.dim() - 1)
        if self.training:
            u = torch.rand(shape, device=X.device)
            noise = torch.randn_like(X)
        else:
            if self._gen is None:
                self._gen = torch.Generator(device=X.device)
                self._gen.manual_seed(self.eval_seed)
            u = torch.rand(shape, device=X.device, generator=self._gen)
            noise = torch.randn(X.shape, device=X.device, dtype=X.dtype, generator=self._gen)
        snr_db = self.lo_db + (self.hi_db - self.lo_db) * u
        sig = X.pow(2).mean()
        std = (sig / (10.0 ** (snr_db / 10.0) + 1e-12)).sqrt().to(X.dtype)
        return X + noise * std


def _train_with_channel(jscc, detector, channel, **kwargs):
    """Run train_joint_system with `channel` in place of its 40 dB AWGN channel.

    train_joint_system sets its training channel with one call,
    jscc.set_channel("awgn", CLS_FIT_SNR). During that call set_channel is
    overridden on this instance so that `channel` is installed instead; the
    override is removed afterwards. The number of calls is checked, so a change to
    train_joint_system raises an error instead of silently training at 40 dB.
    """
    calls = []

    def _install(ctype, snr_db):
        calls.append((ctype, snr_db))
        jscc.channel = channel.to(next(jscc.parameters()).device)

    jscc.set_channel = _install
    try:
        result = train_joint_system(jscc, detector, **kwargs)
    finally:
        del jscc.set_channel
    if len(calls) != 1:
        raise RuntimeError(f"expected train_joint_system to set the channel once, saw {calls}")
    return result


def load_production(jscc, detector, rep_dir, det, device):
    """Load one production system's weights (Phase I repetition folder rep_dir)."""
    rep_dir = Path(rep_dir)
    jf, dfile = PRODUCTION_CKPTS[det]
    for f in (jf, dfile):
        if not (rep_dir / f).exists():
            raise FileNotFoundError(f"production checkpoint {rep_dir / f} not found")
    jscc.load_state_dict(torch.load(rep_dir / jf, map_location=device)["model"])
    detector.load_state_dict(torch.load(rep_dir / dfile, map_location=device))


def _train_pair(cfg, device, p1, seed, lam_ml, lam_pr, patience, nsym, out_dir, init_dir=None):
    """Train both systems for one seed in the Phase I production order (seed
    once, Max Logit then Prototype); only the training channel differs.
    init_dir: the Phase I repetition folder of the same seed; if given, each
    system starts from its production weights (warm start).
    """
    lo, hi = cfg["MULTISNR_TRAIN_RANGE_DB"]
    fresh = init_dir is None
    if not fresh:                                  # check every file before training
        for det in PRODUCTION_CKPTS:
            for f in PRODUCTION_CKPTS[det]:
                if not (Path(init_dir) / f).exists():
                    raise FileNotFoundError(f"production checkpoint {Path(init_dir) / f} not found")
    set_seed(seed)
    loaders = p1["loaders"]
    jscc_a = SwinJSCCSystem(nsym, pretrained=fresh, patch_grid=cfg["PATCH_GRID_ML"],
                            patch_dim=cfg["PATCH_DIM_ML"],
                            stage_idx=cfg.get("BACKBONE_STAGE_ML", 2)).to(device)
    max_logit = MaxLogitDetector(cfg["PATCH_DIM_ML"], cfg["PATCH_GRID_ML"], p1["num_classes"]).to(device)
    if not fresh:
        load_production(jscc_a, max_logit, init_dir, "max_logit", device)
    r_ml = _train_with_channel(
        jscc_a, max_logit, RandomSNRAWGNChannel(lo, hi, cfg["MULTISNR_VAL_SEED"]),
        loader=loaders["cls_train"], val_loader=loaders["cls_val"], cfg=cfg, device=device,
        num_classes=p1["num_classes"],
        jscc_save_path=out_dir / f"jscc_a_multisnr_seed{seed}.pt",
        detector_save_path=out_dir / f"max_logit_multisnr_seed{seed}.pt",
        lambda_rec=cfg["LAMBDA_REC_ML"], lambda_sem=lam_ml,
        joint_lr=cfg["JOINT_LR_ML"], joint_momentum=cfg["JOINT_MOMENTUM_ML"],
        weight_decay=cfg["WEIGHT_DECAY_ML"], joint_lr_power=cfg["JOINT_LR_POWER_ML"],
        patience=patience, min_rel_improve=cfg["MIN_REL_IMPROVE_ML"], max_epochs=cfg["JOINT_MAX_EPOCHS"])
    jscc_b = SwinJSCCSystem(nsym, pretrained=fresh, patch_grid=cfg["PATCH_GRID_PR"],
                            patch_dim=cfg["PATCH_DIM_PR"],
                            stage_idx=cfg.get("BACKBONE_STAGE_PR", 2)).to(device)
    prototype = PrototypeDetector(cfg["PATCH_DIM_PR"], cfg["PATCH_GRID_PR"], p1["num_classes"],
                                  cfg["PROTOTYPE_SCALE"]).to(device)
    if not fresh:
        load_production(jscc_b, prototype, init_dir, "prototype", device)
    r_pr = _train_with_channel(
        jscc_b, prototype, RandomSNRAWGNChannel(lo, hi, cfg["MULTISNR_VAL_SEED"]),
        loader=loaders["cls_train"], val_loader=loaders["cls_val"], cfg=cfg, device=device,
        num_classes=p1["num_classes"],
        jscc_save_path=out_dir / f"jscc_b_multisnr_seed{seed}.pt",
        detector_save_path=out_dir / f"prototype_multisnr_seed{seed}.pt",
        lambda_rec=cfg["LAMBDA_REC_PR"], lambda_sem=lam_pr,
        joint_lr=cfg["JOINT_LR_PR"], joint_momentum=cfg["JOINT_MOMENTUM_PR"],
        weight_decay=cfg["WEIGHT_DECAY_PR"], joint_lr_power=cfg["JOINT_LR_POWER_PR"],
        patience=patience, min_rel_improve=cfg["MIN_REL_IMPROVE_PR"], max_epochs=cfg["JOINT_MAX_EPOCHS"])
    for m in (jscc_a, jscc_b):
        # Replace the random-SNR channel with the 40 dB channel.
        m.set_channel("awgn", cfg["CLS_FIT_SNR"])
    return jscc_a, max_logit, jscc_b, prototype, r_ml, r_pr


def _paired_summary(a, b):
    """Paired differences a - b over seeds where both values exist (None = level
    not crossed), with the exact Wilcoxon signed-rank p. At n = 5 the smallest
    two-sided p is 0.0625, so consistency is read from the sign count.
    """
    d = np.array([x - y for x, y in zip(a, b) if x is not None and y is not None], dtype=float)
    p = float(stats.wilcoxon(d).pvalue) if d.size >= 2 and np.any(d != 0) else None
    return {"n": int(d.size), "mean_diff_db": float(d.mean()) if d.size else None,
            "sd_diff_db": float(d.std(ddof=1)) if d.size > 1 else None,
            "n_first_lower": int((d < 0).sum()), "wilcoxon_p": p, "diffs": d.tolist()}


def multisnr_analysis(multi_per_seed, fixed_per_seed, snrs, cfg, seeds):
    """Decision Retention of the multi-SNR-trained systems against the 40 dB-trained
    production systems of the same seeds (no GPU).
    """
    dr_multi = decision_retention_analysis(multi_per_seed, snrs, cfg)
    dr_fixed = decision_retention_analysis(fixed_per_seed, snrs, cfg)
    out = {"seeds": list(seeds), "levels": list(cfg["DR_HEADLINE_LEVELS"]),
           "train_range_db": list(cfg["MULTISNR_TRAIN_RANGE_DB"]),
           "dr_multi": dr_multi, "dr_fixed": dr_fixed, "comparisons": {}}
    snr_f = [float(s) for s in snrs]
    for ch in ("awgn", "rayleigh"):
        if ch not in dr_multi["channels"] or ch not in dr_fixed["channels"]:
            continue
        bm = dr_multi["channels"][ch]["auc"]["ref"]["detectors"]
        bf = dr_fixed["channels"][ch]["auc"]["ref"]["detectors"]
        comp = {}
        for lv in cfg["DR_HEADLINE_LEVELS"]:
            k = str(lv)
            pm = bm["prototype"]["levels"][k]["per_seed_snr_db"]; mm = bm["max_logit"]["levels"][k]["per_seed_snr_db"]
            pf = bf["prototype"]["levels"][k]["per_seed_snr_db"]; mf = bf["max_logit"]["levels"][k]["per_seed_snr_db"]
            gap_m = [None if (x is None or y is None) else x - y for x, y in zip(pm, mm)]
            gap_f = [None if (x is None or y is None) else x - y for x, y in zip(pf, mf)]
            comp[k] = {
                "gap_multi_prototype_minus_max_logit": _paired_summary(pm, mm),
                "gap_fixed_prototype_minus_max_logit": _paired_summary(pf, mf),
                "gap_change_multi_minus_fixed": _paired_summary(gap_m, gap_f),
                "max_logit_multi_minus_fixed": _paired_summary(mm, mf),
                "prototype_multi_minus_fixed": _paired_summary(pm, pf),
            }
        absolute = {}
        for regime, ps in (("multi", multi_per_seed), ("fixed", fixed_per_seed)):
            absolute[regime] = {}
            for det in ("max_logit", "prototype"):
                a = np.array([[pt[f"{det}_auc_roc_pixel"] for pt in run] for run in ps[ch]], dtype=float)
                absolute[regime][det] = {"snr_db": snr_f, "mean_auc": a.mean(0).tolist(),
                                         "sd_auc": a.std(0, ddof=1).tolist() if a.shape[0] > 1 else None}
        comp["absolute_auroc"] = absolute
        out["comparisons"][ch] = comp
    return out


def collapse_flags(sweep_awgn, dependence, cfg):
    """Input-independence check of one seed's two new models.

    sweep_awgn: the AWGN sweep (eval_both_systems dicts over SNR_RANGE_DB);
    dependence: per detector, input_dependence() at CLS_FIT_SNR, or None.
    A model is flagged if its AUROC range over the swept SNRs (CLS_FIT_SNR
    excluded) is below COLLAPSE_GUARD_AUROC_RANGE, or if matched - shuffled AUROC
    at CLS_FIT_SNR is below COLLAPSE_GUARD_MIN_DEPENDENCE.
    """
    snrs = [float(s) for s in cfg["SNR_RANGE_DB"]]
    keep = [i for i, s in enumerate(snrs) if s != float(cfg["CLS_FIT_SNR"])]
    out = {}
    for det in ("max_logit", "prototype"):
        a = [sweep_awgn[i][f"{det}_auc_roc_pixel"] for i in keep]
        rng = float(max(a) - min(a))
        dep = (dependence or {}).get(det)
        mms = None if dep is None else dep.get("matched_minus_shuffled")
        reasons = []
        if rng < cfg["COLLAPSE_GUARD_AUROC_RANGE"]:
            reasons.append(f"AUROC range over the sweep {rng:.3f} < {cfg['COLLAPSE_GUARD_AUROC_RANGE']}")
        if mms is not None and mms < cfg["COLLAPSE_GUARD_MIN_DEPENDENCE"]:
            reasons.append(f"matched - shuffled AUROC at {cfg['CLS_FIT_SNR']:g} dB {mms:.3f} < "
                           f"{cfg['COLLAPSE_GUARD_MIN_DEPENDENCE']}")
        out[det] = {"auroc_range": rng, "matched_minus_shuffled": mms, "flagged": bool(reasons), "reasons": reasons}
    return out


def _fmt(v, nd=2):
    return "--" if v is None else f"{v:+.{nd}f}"


def _censor(dr, ch, det, lv):
    x = dr["channels"][ch]["auc"]["ref"]["detectors"][det]["levels"][str(lv)]
    n = len(x["per_seed_status"])
    return (f"{n - x['n_left_censored'] - x['n_not_reached'] - x['n_undefined']} crossed, "
            f"{x['n_left_censored']} L, {x['n_not_reached']} NR, {x['n_undefined']} undef")


def _report(an, regime_name="trained over SNRs"):
    lines = []
    for ch, comp in an["comparisons"].items():
        lines.append(f"\n  {ch.upper()} -- SNR at which DR_AUC first stays >= level (mean over seeds, dB)")
        for lv in an["levels"]:
            c = comp[str(lv)]
            gm, gf, gc_ = (c["gap_multi_prototype_minus_max_logit"], c["gap_fixed_prototype_minus_max_logit"],
                           c["gap_change_multi_minus_fixed"])
            lines.append(f"    {float(lv):.0%}: Prototype - Max Logit gap: trained at 40 dB {_fmt(gf['mean_diff_db'])} dB "
                         f"(Prototype lower in {gf['n_first_lower']}/{gf['n']}), {regime_name} "
                         f"{_fmt(gm['mean_diff_db'])} dB (Prototype lower in {gm['n_first_lower']}/{gm['n']}); "
                         f"change {_fmt(gc_['mean_diff_db'])} dB")
            for det, name in (("max_logit", "Max Logit"), ("prototype", "Prototype")):
                d = c[f"{det}_multi_minus_fixed"]
                lines.append(f"         {name}: {regime_name} minus 40 dB training {_fmt(d['mean_diff_db'])} dB "
                             f"(lower in {d['n_first_lower']}/{d['n']}, exact Wilcoxon p="
                             f"{'--' if d['wilcoxon_p'] is None else format(d['wilcoxon_p'], '.4g')}); "
                             f"seeds: {regime_name} {_censor(an['dr_multi'], ch, det, lv)} | "
                             f"40 dB {_censor(an['dr_fixed'], ch, det, lv)}")
        ab = comp["absolute_auroc"]
        snr = ab["multi"]["max_logit"]["snr_db"]
        pick = [s for s in (-20.0, -10.0, 0.0, 10.0, 20.0, 40.0) if s in snr]
        lines.append(f"    Absolute AUROC (mean)  SNR: " + "  ".join(f"{int(s):+d}" for s in pick))
        for regime, rname in (("fixed", "40 dB"), ("multi", "multi")):
            for det, name in (("max_logit", "Max Logit"), ("prototype", "Prototype")):
                m = ab[regime][det]["mean_auc"]
                lines.append(f"      {name:>9} ({rname:>5}):     " + "  ".join(f"{m[snr.index(s)]:.3f}" for s in pick))
    lines.append("\n  'L' = DR already >= level at the lowest swept SNR (left-censored), 'NR' = level not reached;")
    lines.append("  censored seeds are counted, not averaged. Exact Wilcoxon p cannot go below 0.0625 at n = 5.")
    if an.get("guard"):
        lines.append("\n  Collapse guard (input-independent output), per seed:")
        for s, g in an["guard"].items():
            for det, name in (("max_logit", "Max Logit"), ("prototype", "Prototype")):
                x = g[det]
                mms = "--" if x["matched_minus_shuffled"] is None else f"{x['matched_minus_shuffled']:.3f}"
                lines.append(f"    seed {s:>5} {name:<9}: AUROC range {x['auroc_range']:.3f}, matched - shuffled "
                             f"{mms}, epochs {an['epochs'][s][0 if det == 'max_logit' else 1]}"
                             + (f"   <- FLAGGED: {'; '.join(x['reasons'])}" if x["flagged"] else ""))
        fl = an.get("flagged_seeds", [])
        lines.append(f"    Flagged seeds: {fl if fl else 'none'}")
    return "\n".join(lines)


def _plots(an, out_root, cfg, names=("11_multisnr_dr_auc", "11b_multisnr_auroc"), regime_name="trained over SNRs"):
    import matplotlib.pyplot as plt
    pdir = Path(out_root) / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    style = (("max_logit", "Max Logit", "#1f77b4", "o"), ("prototype", "Prototype", "#2ca02c", "D"))
    for key, fname, ylabel in (("dr", names[0], r"Decision Retention $DR_{AUC}$"),
                               ("auc", names[1], "Pixel AUROC")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for regime, ls, label in (("fixed", "--", "trained at 40 dB"), ("multi", "-", regime_name)):
            dr = an[f"dr_{regime}"]
            x = dr["snr_db"]
            for det, name, color, mk in style:
                if key == "dr":
                    d = dr["channels"]["awgn"]["auc"]["ref"]["detectors"][det]
                    m = np.array([np.nan if v is None else v for v in d["mean_curve"]], dtype=float)
                    h = np.array([np.nan if v is None else v for v in d["ci95_half_width_curve"]], dtype=float)
                    xs = x
                else:
                    ab = an["comparisons"]["awgn"]["absolute_auroc"][regime][det]
                    keep = [i for i, s in enumerate(ab["snr_db"]) if s != float(cfg["CLS_FIT_SNR"])]
                    m = np.array([ab["mean_auc"][i] for i in keep]); xs = [ab["snr_db"][i] for i in keep]
                    sd = ab["sd_auc"]; n = len(an["seeds"])
                    h = (np.array([sd[i] for i in keep]) * stats.t.ppf(0.975, n - 1) / np.sqrt(n)
                         if sd is not None and n > 1 else np.full(len(keep), np.nan))
                ax.plot(xs, m, ls=ls, marker=mk, markersize=4, color=color, linewidth=2 if regime == "multi" else 1.5,
                        alpha=1.0 if regime == "multi" else .6, label=f"{name}, {label}")
                ax.fill_between(xs, m - h, m + h, color=color, alpha=.12 if regime == "multi" else .06, linewidth=0)
        if key == "dr":
            for lv in cfg["DR_HEADLINE_LEVELS"]:
                ax.axhline(lv, color="grey", ls=":", lw=.8, alpha=.6)
            ax.axhline(1.0, color="black", lw=.8, alpha=.35); ax.axhline(0.0, color="black", ls=":", lw=.8, alpha=.4)
        else:
            ax.axhline(0.5, color="black", ls=":", lw=.8, alpha=.4, label="Chance")
        ax.set(xlabel="SNR (dB)", ylabel=ylabel)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(str(pdir / f"{fname}.{ext}"), dpi=600, bbox_inches="tight")
        plt.close(fig)
    print(f"[Multi-SNR check] plots -> {pdir}/{names[0]}.png, {names[1]}.png (+ .pdf)")


def _subset(per_seed, keep_idx):
    return {"awgn": [per_seed["awgn"][i] for i in keep_idx],
            "rayleigh": [per_seed["rayleigh"][i] for i in keep_idx] if per_seed.get("rayleigh") else [],
            "snr_range_db": per_seed["snr_range_db"], "n_seeds": len(keep_idx)}


def run_multisnr_check(p1_grid, cfg, device, warm_start=None):
    """Train, evaluate (resumable per seed), analyse and plot.

    Needs p1_grid (Phase I) and run_0/phase2_per_seed.json (Phase II); the warm
    start also needs the production checkpoints in run_0_rep{i}/.
    warm_start=None reads cfg["MULTISNR_WARM_START"].
    """
    if warm_start is None:
        warm_start = bool(cfg.get("MULTISNR_WARM_START", True))
    tag = "multisnr_ft" if warm_start else "multisnr"
    table_name = "multisnr_finetune.md" if warm_start else "multisnr_check.md"
    plot_names = (("11c_multisnr_ft_dr_auc", "11d_multisnr_ft_auroc") if warm_start
                  else ("11_multisnr_dr_auc", "11b_multisnr_auroc"))
    regime_name = "fine-tuned over SNRs" if warm_start else "trained over SNRs"
    seeds = list(cfg["SEEDS"][:cfg["MULTISNR_N_SEEDS"]])
    p1 = p1_grid["final_runs"][0]
    lam_ml = p1_grid["grid_analysis"]["winning_lambda_sem"]["max_logit"]
    lam_pr = p1_grid["grid_analysis"]["winning_lambda_sem"]["prototype"]
    patience = p1_grid["grid_analysis"]["derived_patience"]
    nsym = cfg["CHANNEL_SYMBOLS"][cfg["DEFAULT_CR"]]
    out_root = Path(p1_grid["out"])
    res_dir = out_root / tag; res_dir.mkdir(parents=True, exist_ok=True)
    state_path = res_dir / f"{tag}_state.json"
    from phases.phase3_compression import _local_tmp_dir
    ckpt_dir = _local_tmp_dir(tag.upper())

    per_seed_path = out_root / "phase2_per_seed.json"
    if not per_seed_path.exists():
        raise FileNotFoundError(f"{per_seed_path} not found -- run Phase II first.")
    fixed_all = json.load(open(per_seed_path))
    snrs = [float(s) for s in cfg["SNR_RANGE_DB"]]
    if [float(s) for s in fixed_all["snr_range_db"]] != snrs:
        raise ValueError("phase2_per_seed.json SNR grid differs from SNR_RANGE_DB -- stale Phase II file.")
    if len(fixed_all["awgn"]) < len(seeds):
        raise ValueError("phase2_per_seed.json has fewer repetitions than MULTISNR_N_SEEDS.")
    # Phase I repetition i used SEEDS[i].
    idx = [list(cfg["SEEDS"]).index(s) for s in seeds]

    fp_obj = {"ml": training_fingerprint(cfg, "max_logit"), "pr": training_fingerprint(cfg, "prototype"),
              "eval": evaluation_fingerprint(cfg), "nsym": nsym, "patience": patience,
              "lam_ml": lam_ml, "lam_pr": lam_pr, "range": list(cfg["MULTISNR_TRAIN_RANGE_DB"]),
              "val_seed": cfg["MULTISNR_VAL_SEED"], "version": 1}
    init_dirs = {}
    if warm_start:
        runs = p1_grid["final_runs"]
        if len(runs) <= max(idx):
            raise ValueError(f"warm start needs Phase I repetitions 0..{max(idx)}; p1_grid has {len(runs)}.")
        for s, i in zip(seeds, idx):
            init_dirs[s] = Path(runs[i]["out"])
            for det in PRODUCTION_CKPTS:
                for f in PRODUCTION_CKPTS[det]:
                    if not (init_dirs[s] / f).exists():
                        raise FileNotFoundError(f"production checkpoint {init_dirs[s] / f} not found")
        # Link the cached results to the production models they were fine-tuned from.
        fp_obj.update({"version": 2, "warm_start": True,
                       "init_taus": [[runs[i].get("tau_max_logit"), runs[i].get("tau_prototype")] for i in idx]})
    fp = fingerprint_key(fp_obj)
    state = json.load(open(state_path)) if state_path.exists() else {}
    print(f"\n{'='*62}\n  MULTI-SNR TRAINING CHECK ({'warm start from the production checkpoints' if warm_start else 'cold start'})"
          f" -- {len(seeds)} seeds {seeds}, training SNR uniform in {cfg['MULTISNR_TRAIN_RANGE_DB']} dB\n{'='*62}")
    for i, seed in enumerate(seeds):
        key = str(seed)
        if state.get(key, {}).get("_fp") == fp:
            print(f"  seed={seed}: SKIP, already done with the current settings (resumed)")
            continue
        print(f"  seed={seed} ({i+1}/{len(seeds)}): {'fine-tuning' if warm_start else 'training'} both systems over SNRs"
              + (f" from {init_dirs[seed]}" if warm_start else "") + "...")
        kw = {"init_dir": init_dirs[seed]} if warm_start else {}
        jscc_a, ml, jscc_b, pr, r_ml, r_pr = _train_pair(cfg, device, p1, seed, lam_ml, lam_pr,
                                                          patience, nsym, ckpt_dir, **kw)
        tau_ml = calibrate_tau(jscc_a, ml, p1["loaders"]["val_mixed"], device,
                               target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
        tau_pr = calibrate_tau(jscc_b, pr, p1["loaders"]["val_mixed"], device,
                               target_tpr=cfg["TARGET_TPR"], snr_db=cfg["CLS_FIT_SNR"])
        sdop_ml = calibrate_sdop_threshold(jscc_a, p1["loaders"]["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"],
                                           percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
        sdop_pr = calibrate_sdop_threshold(jscc_b, p1["loaders"]["val_mixed"], device, snr_db=cfg["CLS_FIT_SNR"],
                                           percentile=cfg["SDOP_CALIBRATION_PERCENTILE"])
        sweep = {"awgn": [], "rayleigh": []}
        for snr in cfg["SNR_RANGE_DB"]:
            r = eval_both_systems(jscc_a, ml, tau_ml, sdop_ml, jscc_b, pr, tau_pr, sdop_pr,
                                  p1["loaders"]["test"], snr, cfg, device, "awgn",
                                  noise_seed=eval_noise_seed(cfg, snr, "awgn"))
            sweep["awgn"].append(r)
            msg = (f"    SNR={snr:+4d}dB  MaxLogit AUC={r['max_logit_auc_roc_pixel']:.3f}  "
                   f"Prototype AUC={r['prototype_auc_roc_pixel']:.3f}")
            if cfg["USE_RAYLEIGH"]:
                rr = eval_both_systems(jscc_a, ml, tau_ml, sdop_ml, jscc_b, pr, tau_pr, sdop_pr,
                                       p1["loaders"]["test"], snr, cfg, device, "rayleigh",
                                       noise_seed=eval_noise_seed(cfg, snr, "rayleigh"))
                sweep["rayleigh"].append(rr)
                msg += (f"  | Rayleigh MaxLogit={rr['max_logit_auc_roc_pixel']:.3f} "
                        f"Prototype={rr['prototype_auc_roc_pixel']:.3f}")
            print(msg)
        dep = {"max_logit": _input_dependence(jscc_a, ml, p1["loaders"]["test"], cfg["CLS_FIT_SNR"], cfg, device),
               "prototype": _input_dependence(jscc_b, pr, p1["loaders"]["test"], cfg["CLS_FIT_SNR"], cfg, device)}
        guard = collapse_flags(sweep["awgn"], dep, cfg)
        state[key] = {"_fp": fp, "sweep": sweep, "tau_max_logit": tau_ml, "tau_prototype": tau_pr,
                      "epochs_max_logit": r_ml["epochs_run"], "epochs_prototype": r_pr["epochs_run"],
                      "val_task_loss_max_logit": r_ml["loss"], "val_task_loss_prototype": r_pr["loss"],
                      "input_dependence_40db": dep, "guard": guard}
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        del jscc_a, ml, jscc_b, pr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for p in ckpt_dir.glob(f"*_multisnr_seed{seed}.pt"):
            p.unlink(missing_ok=True)
        flag_txt = "; ".join(f"{d}: {', '.join(g['reasons'])}" for d, g in guard.items() if g["flagged"])
        print(f"  seed={seed}: done (Max Logit {r_ml['epochs_run']} epochs, Prototype {r_pr['epochs_run']} "
              f"epochs), saved -> {state_path}, local checkpoints deleted"
              + (f"\n  seed={seed}: COLLAPSE GUARD FLAGGED -- {flag_txt}" if flag_txt else ""))

    multi = {"awgn": [state[str(s)]["sweep"]["awgn"] for s in seeds],
             "rayleigh": ([state[str(s)]["sweep"]["rayleigh"] for s in seeds] if cfg["USE_RAYLEIGH"] else []),
             "snr_range_db": cfg["SNR_RANGE_DB"], "n_seeds": len(seeds)}
    fixed = {"awgn": [fixed_all["awgn"][i] for i in idx],
             "rayleigh": ([fixed_all["rayleigh"][i] for i in idx] if fixed_all.get("rayleigh") else []),
             "snr_range_db": fixed_all["snr_range_db"], "n_seeds": len(seeds)}
    an = multisnr_analysis(multi, fixed, cfg["SNR_RANGE_DB"], cfg, seeds)
    an["warm_start"] = bool(warm_start)
    an["epochs"] = {str(s): [state[str(s)]["epochs_max_logit"], state[str(s)]["epochs_prototype"]] for s in seeds}
    an["guard"] = {str(s): (state[str(s)].get("guard")
                            or collapse_flags(state[str(s)]["sweep"]["awgn"], None, cfg)) for s in seeds}
    flagged = [s for s in seeds if any(g["flagged"] for g in an["guard"][str(s)].values())]
    an["flagged_seeds"] = flagged
    rep = _report(an, regime_name)
    if flagged:
        keep = [k for k, s in enumerate(seeds) if s not in flagged]
        if len(keep) >= 2:
            an_ok = multisnr_analysis(_subset(multi, keep), _subset(fixed, keep), cfg["SNR_RANGE_DB"], cfg,
                                      [seeds[k] for k in keep])
            an["excluding_flagged"] = an_ok
            rep += (f"\n\n  SAME COMPARISON WITHOUT THE FLAGGED SEEDS {flagged} "
                    f"(n = {len(keep)}; read with caution):" + _report(an_ok, regime_name))
        else:
            rep += f"\n\n  Fewer than 2 unflagged seeds -- no comparison without the flagged seeds."
    with open(res_dir / f"{tag}_summary.json", "w") as f:
        json.dump(an, f, indent=2)
    tdir = out_root / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / table_name, "w") as f:
        f.write(f"# Multi-SNR training check, {'warm start (fine-tuned from the production models)' if warm_start else 'cold start'}"
                f" ({len(seeds)} seeds {seeds}; training SNR uniform in {cfg['MULTISNR_TRAIN_RANGE_DB']} dB)\n\n```\n{rep}\n```\n")
    print(rep)
    _plots(an, out_root, cfg, plot_names, regime_name)
    print(f"[Multi-SNR check] summary -> {res_dir / (tag + '_summary.json')}, table -> {tdir / table_name}")
    return an
