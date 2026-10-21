"""Ideal digital baseline (dissertation Section 3.1.9 and Appendix F.6): the trained
systems with the analogue channel replaced by a quantised, error-free digital link.

Each production repetition is reused unchanged. The latent z is quantised with a
uniform scalar quantiser of b bits per value and sent with an ideal
(capacity-achieving) channel code over the same number of complex channel uses as
the analogue system (CHANNEL_SYMBOLS[DEFAULT_CR] = 2,048):

  rate      R_b = latent_dim * b / n_symbols   bits per complex channel use
  AWGN      error-free if log2(1 + SNR) >= R_b, i.e. SNR >= SNR_b = 2^R_b - 1;
            otherwise nothing is received (outage)

Above SNR_b the receiver runs the trained chain on the quantised latent (mapper ->
JSCC decoder -> z' -> detector), so quantisation is the only distortion. Below
SNR_b detection is at chance (AUROC 0.5, DR = 0). This is the best case for any
digital scheme at this bit depth; AWGN only.

The quantiser range is set per latent channel from the DIGITAL_QUANT_PERCENTILES of
the latent values on the calibration subset; values outside it are clipped.
Evaluation uses src/evaluate.py _eval_one_system (same test set, pixel subsample
and FPR95 as Phase II). Outputs: run_0/digital_baseline/ (resumable per
repetition), plot 12_digital_vs_analogue_auroc and tables/digital_baseline.md.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.evaluate import _eval_one_system
from src.config import training_fingerprint, evaluation_fingerprint, fingerprint_key


def digital_threshold_db(bits, latent_dim, n_symbols):
    """Minimum AWGN SNR (dB) at which b bits per latent value fit through
    n_symbols complex channel uses with an ideal code."""
    rate = latent_dim * bits / n_symbols
    return 10.0 * math.log10(2.0 ** rate - 1.0)


class LatentQuantiser:
    """Uniform mid-point quantiser, 2^bits levels per latent channel over [lo, hi]
    (one range per latent channel c; flat index j belongs to channel j // (P*P),
    matching SwinEncoder's z = reduced.flatten(1) of a [B, patch_dim, P, P] map)."""

    def __init__(self, lo, hi, bits, patch_dim, patch_grid):
        self.bits = bits
        self.levels = 2 ** bits
        rep = patch_grid * patch_grid
        self.lo = torch.as_tensor(lo, dtype=torch.float32).repeat_interleave(rep)
        self.hi = torch.as_tensor(hi, dtype=torch.float32).repeat_interleave(rep)
        if self.lo.numel() != patch_dim * rep or not torch.all(self.hi > self.lo):
            raise ValueError("quantiser range must have one increasing [lo, hi] per latent channel")

    def __call__(self, z):
        lo = self.lo.to(z.device, z.dtype); hi = self.hi.to(z.device, z.dtype)
        step = (hi - lo) / self.levels
        idx = torch.clamp(torch.floor((z - lo) / step), 0, self.levels - 1)
        return lo + (idx + 0.5) * step


@torch.no_grad()
def calibrate_quantiser_range(jscc, loader, device, percentiles):
    """Per-latent-channel [lo, hi] from the calibration subset (pre-channel z)."""
    jscc.eval()
    vals = []
    for b in loader:
        z, _ = jscc.encoder(b["image"].to(device))
        vals.append(z.float().view(z.shape[0], jscc.patch_dim, -1).permute(1, 0, 2).reshape(jscc.patch_dim, -1).cpu())
    v = torch.cat(vals, dim=1)
    q = torch.tensor([percentiles[0] / 100.0, percentiles[1] / 100.0])
    lo, hi = [], []
    for c in range(v.shape[0]):
        a, bq = torch.quantile(v[c], q).tolist()
        # Constant channel: use a minimal non-zero range.
        if not bq > a:
            a, bq = a - 1e-6, a + 1e-6
        lo.append(a); hi.append(bq)
    return lo, hi


class DigitalLink(nn.Module):
    """A trained SwinJSCCSystem with the analogue channel replaced by an error-free
    digital link carrying the quantised latent (quantiser=None: unquantised).
    It has the same forward signature as SwinJSCCSystem, so _eval_one_system
    evaluates it with the Phase II protocol; set_channel does nothing.
    """

    def __init__(self, jscc, quantiser=None):
        super().__init__()
        self.jscc = jscc
        self.quantiser = quantiser

    def set_channel(self, ctype, snr_db):
        pass

    def forward(self, x):
        j = self.jscc
        z, _ = j.encoder(x)
        zq = self.quantiser(z) if self.quantiser is not None else z
        z_prime = j.jscc_dec(j.mapper(zq))
        x_hat = j.img_dec(z_prime.view(-1, j.patch_dim, j.patch_grid, j.patch_grid))
        return x_hat, z, z_prime


def digital_analysis(state, analogue_per_seed, cfg, latent_dims, n_symbols):
    """Summarise the digital results against the analogue sweep (no GPU).
    state[rep][det][str(bits) or "none"] holds the quantised-link evaluation;
    analogue_per_seed is phase2_per_seed.json.
    """
    snrs = [float(s) for s in analogue_per_seed["snr_range_db"]]
    ref_i = snrs.index(float(cfg["CLS_FIT_SNR"]))
    sweep_i = [i for i, s in enumerate(snrs) if s != float(cfg["CLS_FIT_SNR"])]
    reps = sorted(state, key=int)
    out = {"snr_db": [snrs[i] for i in sweep_i], "n_symbols": n_symbols, "bits": list(cfg["DIGITAL_BITS"]),
           "n_reps": len(reps), "detectors": {}}
    from src.decision_retention import dr_values, retention_snr
    for det in ("max_logit", "prototype"):
        key = f"{det}_auc_roc_pixel"
        an = np.array([[analogue_per_seed["awgn"][int(r)][i][key] for i in range(len(snrs))] for r in reps])
        ref = an[:, ref_i]
        d = {"latent_dim": latent_dims[det], "analogue": {}, "unquantised": {}, "digital": {}}
        d["analogue"]["mean_auc_sweep"] = an[:, sweep_i].mean(0).tolist()
        d["analogue"]["auc_40db"] = {"mean": float(ref.mean()), "sd": float(ref.std(ddof=1)) if len(reps) > 1 else None}
        s50 = [retention_snr(np.array(out["snr_db"]), dr_values(an[k, sweep_i], ref[k], "auc"), 0.5) for k in range(len(reps))]
        d["analogue"]["snr50_per_rep"] = [x for x, _ in s50]
        un = np.array([state[r][det]["none"]["auc_roc_pixel"] for r in reps])
        d["unquantised"] = {"mean_auc": float(un.mean()), "mean_abs_diff_from_40db": float(np.abs(un - ref).mean())}
        for b in cfg["DIGITAL_BITS"]:
            aq = np.array([state[r][det][str(b)]["auc_roc_pixel"] for r in reps])
            fq = np.array([state[r][det][str(b)]["fpr95"] for r in reps])
            dq = np.array([state[r][det][str(b)]["dsem"] for r in reps])
            thr = digital_threshold_db(b, latent_dims[det], n_symbols)
            drq = (aq - 0.5) / (ref - 0.5)
            # Analogue minus digital SNR50: negative means the analogue system keeps half
            # of its clean ability down to a lower SNR than the digital threshold.
            a50 = np.array([np.nan if x is None else x for x in d["analogue"]["snr50_per_rep"]])
            dig50 = np.where(drq >= 0.5, thr, np.nan)
            diff = a50 - dig50
            d["digital"][str(b)] = {
                "rate_bits_per_symbol": latent_dims[det] * b / n_symbols, "threshold_snr_db": thr,
                "auc_mean": float(aq.mean()), "auc_sd": float(aq.std(ddof=1)) if len(reps) > 1 else None,
                "fpr95_mean": float(fq.mean()), "dsem_mean": float(dq.mean()),
                "dr_above_threshold_mean": float(drq.mean()),
                "reps_reaching_50pct": int((drq >= 0.5).sum()),
                "analogue_snr50_minus_threshold_db_mean": (float(np.nanmean(diff)) if np.any(np.isfinite(diff)) else None),
                # AUROC at each swept SNR (the plot draws the cliff exactly at threshold_snr_db).
                "curve_auc_mean": [float(aq.mean()) if s >= thr else 0.5 for s in out["snr_db"]],
            }
        out["detectors"][det] = d
    return out


def _report(an):
    lines = [f"  Ideal digital link over {an['n_symbols']} complex channel uses (AWGN), N={an['n_reps']} repetitions"]
    for det, name in (("max_logit", "Max Logit"), ("prototype", "Prototype")):
        d = an["detectors"][det]
        a50 = [x for x in d["analogue"]["snr50_per_rep"] if x is not None]
        lines.append(f"\n  {name}: analogue AUROC at 40 dB {d['analogue']['auc_40db']['mean']:.4f}; "
                     f"unquantised noise-free link {d['unquantised']['mean_auc']:.4f} "
                     f"(mean |diff| from 40 dB {d['unquantised']['mean_abs_diff_from_40db']:.4f}); "
                     f"analogue SNR50 {np.mean(a50):+.2f} dB")
        lines.append(f"    {'bits':>4} {'rate':>6} {'threshold':>10} {'AUROC above':>12} {'DR above':>9} "
                     f"{'analogue SNR50 - threshold':>27}")
        for b, v in d["digital"].items():
            diff = v["analogue_snr50_minus_threshold_db_mean"]
            lines.append(f"    {b:>4} {v['rate_bits_per_symbol']:>6.3f} {v['threshold_snr_db']:>+8.2f}dB "
                         f"{v['auc_mean']:>12.4f} {v['dr_above_threshold_mean']:>9.3f} "
                         f"{'--' if diff is None else f'{diff:+.2f} dB':>27}")
    return "\n".join(lines)


def digital_step_curve(x_min, x_max, threshold_db, auc):
    """Points of the ideal digital AUROC curve with the cliff drawn exactly at the
    threshold SNR (not at the next point of the SNR grid): chance below, the
    quantised link's AUROC from the threshold upwards."""
    if threshold_db <= x_min:
        return [x_min, x_max], [auc, auc]
    if threshold_db > x_max:
        return [x_min, x_max], [0.5, 0.5]
    return [x_min, threshold_db, threshold_db, x_max], [0.5, 0.5, auc, auc]


def _plot(an, analogue_per_seed, out_root, cfg):
    import matplotlib.pyplot as plt
    pdir = Path(out_root) / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = an["snr_db"]
    greys = ["#4d4d4d", "#7f7f7f", "#a6a6a6", "#c8c8c8", "#e0e0e0"]
    for ax, (det, name, color) in zip(axes, (("max_logit", "Max Logit", "#1f77b4"),
                                            ("prototype", "Prototype", "#2ca02c"))):
        d = an["detectors"][det]
        ax.plot(x, d["analogue"]["mean_auc_sweep"], "-o", color=color, linewidth=2, markersize=4,
                label="Deep JSCC (analogue), mean")
        for k, (b, v) in enumerate(d["digital"].items()):
            sx, sy = digital_step_curve(x[0], x[-1], v["threshold_snr_db"], v["auc_mean"])
            ax.plot(sx, sy, color=greys[k % len(greys)], linewidth=1.6,
                    ls=["--", "-.", ":", (0, (5, 1)), (0, (1, 1))][k % 5],
                    label=f"Ideal digital, {b} bit(s)/value (threshold {v['threshold_snr_db']:+.1f} dB)")
        ax.axhline(0.5, color="black", ls=":", lw=.8, alpha=.4, label="Chance")
        ax.set(xlabel="SNR (dB)", title=name)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Pixel AUROC")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(pdir / f"12_digital_vs_analogue_auroc.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[Digital baseline] plot -> {pdir}/12_digital_vs_analogue_auroc.png/.pdf")


def run_digital_baseline(p1_grid, cfg, device):
    """Evaluate every production repetition, both systems, at each bit depth in
    DIGITAL_BITS and with the unquantised link (no training). Resumable per
    repetition. Needs p1_grid (Phase I) and phase2_per_seed.json.
    """
    out_root = Path(p1_grid["out"])
    res_dir = out_root / "digital_baseline"; res_dir.mkdir(parents=True, exist_ok=True)
    state_path = res_dir / "digital_state.json"
    per_seed_path = out_root / "phase2_per_seed.json"
    if not per_seed_path.exists():
        raise FileNotFoundError(f"{per_seed_path} not found -- run Phase II first.")
    analogue = json.load(open(per_seed_path))
    n_symbols = cfg["CHANNEL_SYMBOLS"][cfg["DEFAULT_CR"]]
    fp = fingerprint_key({"ml": training_fingerprint(cfg, "max_logit"), "pr": training_fingerprint(cfg, "prototype"),
                          "eval": evaluation_fingerprint(cfg), "bits": list(cfg["DIGITAL_BITS"]),
                          "pct": list(cfg["DIGITAL_QUANT_PERCENTILES"]), "nsym": n_symbols,
                          "taus": [[r["tau_max_logit"], r["tau_prototype"]] for r in p1_grid["final_runs"]],
                          "version": 1})
    saved = json.load(open(state_path)) if state_path.exists() else {}
    state = saved.get("reps", {}) if saved.get("_fp") == fp else {}
    print(f"\n{'='*62}\n  IDEAL DIGITAL BASELINE -- bits {cfg['DIGITAL_BITS']}, "
          f"{len(p1_grid['final_runs'])} repetitions, evaluation only\n{'='*62}")
    latent_dims = {}
    for rep_i, p1 in enumerate(p1_grid["final_runs"]):
        for det, jk, dk, tk, sk in (("max_logit", "jscc_a", "max_logit_det", "tau_max_logit", "sdop_threshold_a"),
                                    ("prototype", "jscc_b", "prototype_det", "tau_prototype", "sdop_threshold_b")):
            latent_dims[det] = p1[jk].patch_dim * p1[jk].patch_grid ** 2
        if str(rep_i) in state:
            print(f"  repetition {rep_i+1}: SKIP, already done (resumed)")
            continue
        rec = {}
        for det, jk, dk, tk, sk in (("max_logit", "jscc_a", "max_logit_det", "tau_max_logit", "sdop_threshold_a"),
                                    ("prototype", "jscc_b", "prototype_det", "tau_prototype", "sdop_threshold_b")):
            jscc = p1[jk]
            lo, hi = calibrate_quantiser_range(jscc, p1["loaders"]["val_mixed"], device,
                                               cfg["DIGITAL_QUANT_PERCENTILES"])
            rec[det] = {"quantiser_lo": lo, "quantiser_hi": hi}
            for b in [None] + list(cfg["DIGITAL_BITS"]):
                q = None if b is None else LatentQuantiser(lo, hi, b, jscc.patch_dim, jscc.patch_grid)
                r = _eval_one_system(DigitalLink(jscc, q), p1[dk], p1[tk], p1[sk], p1["loaders"]["test"],
                                     None, cfg, device, "awgn", noise_seed=None)
                rec[det]["none" if b is None else str(b)] = {k: r[k] for k in ("auc_roc_pixel", "fpr95", "dsem", "psnr")}
        state[str(rep_i)] = rec
        with open(state_path, "w") as f:
            json.dump({"_fp": fp, "reps": state}, f, indent=2)
        def _fmt_rec(r):
            return ", ".join(f"{'unquantised' if k == 'none' else k + ' bit'}={v['auc_roc_pixel']:.3f}"
                             for k, v in r.items() if k not in ("quantiser_lo", "quantiser_hi"))
        print(f"  repetition {rep_i+1}: Max Logit AUROC {_fmt_rec(rec['max_logit'])} | "
              f"Prototype AUROC {_fmt_rec(rec['prototype'])}")
    an = digital_analysis(state, analogue, cfg, latent_dims, n_symbols)
    with open(res_dir / "digital_summary.json", "w") as f:
        json.dump(an, f, indent=2)
    tdir = out_root / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    rep = _report(an)
    with open(tdir / "digital_baseline.md", "w") as f:
        f.write(f"# Ideal digital baseline (AWGN)\n\n```\n{rep}\n```\n")
    print(rep)
    _plot(an, analogue, out_root, cfg)
    return an
