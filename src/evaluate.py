"""Evaluation of a trained system at one (SNR, channel) point.

Pixel AUROC, FPR95, the rates at the frozen threshold tau, D_sem, SDOP, PSNR and
SRI are computed over the 1,200 evaluation images. Pixel AUROC uses a fixed
200,000-pixel subsample (the same pixels at every SNR and for every system).
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from src.metrics import (metric_fpr95, metric_sri_auc, metric_sri_op,
                          metric_sdop, metric_tpr_fpr_at_tau)
from src.dataset import denorm_to_tanh_range
from sklearn.metrics import roc_auc_score

def eval_noise_seed(cfg, snr_db, channel_type):
    """Seed of the channel noise at one evaluation point. It depends only on the
    channel and the SNR, so every repetition and both systems see the same noise
    draws at a given point, and results can be reproduced from the checkpoints."""
    base = int(cfg.get("EVAL_NOISE_SEED", 20261009))
    ch = {"awgn": 0, "rayleigh": 1}[channel_type]
    return base + 1000 * ch + int(round(float(snr_db))) + 100


def _as_list_dtype(a):
    """Cast to the dtype np.array(a.tolist()) would give (float64 / int64 / bool)."""
    if a.dtype.kind == "f":
        return a.astype(np.float64)
    if a.dtype.kind in "iu":
        return a.astype(np.int64)
    return a


@torch.no_grad()
def _eval_one_system(model, detector, tau, sdop_threshold, loader, snr_db, cfg, device, channel_type,
                      pixel_auc_sample=200_000, noise_seed=None):
    """Evaluate one system (jscc + detector) at one SNR and channel.

    tau (frozen anomaly threshold) and sdop_threshold were calibrated once in
    Phase I. With noise_seed, the channel noise is drawn from that seed and the
    global RNG state is restored afterwards. Returns a dict of metrics; sri_op is
    None when FPR95 came from the fallback path (95% TPR never reached).
    """
    if noise_seed is not None:
        # Seeded channel noise for this evaluation only; the RNG state is restored on exit.
        _devs = ([device.index if device.index is not None else torch.cuda.current_device()]
                 if getattr(device, "type", str(device)) == "cuda" else [])
        with torch.random.fork_rng(devices=_devs):
            torch.manual_seed(int(noise_seed))
            return _eval_one_system(model, detector, tau, sdop_threshold, loader, snr_db, cfg, device,
                                    channel_type, pixel_auc_sample=pixel_auc_sample, noise_seed=None)
    model.eval(); model.set_channel(channel_type, snr_db)
    img_lbl=[]; px_lbl=[]; dsem_l=[]
    img_sc=[]; px_sc=[]
    sse_sum=0.0; sse_n=0
    for b in loader:
        imgs=b["image"].to(device, non_blocking=True); masks=b["mask"].to(device, non_blocking=True)
        has_a=b["has_anomaly"].cpu().numpy()
        xh,z,zp = model(imgs)
        imgs_target = denorm_to_tanh_range(imgs)
        # Per-image d_sem = (1 - cos(z, z'))/2; averaged over images below.
        per_img_dsem = ((1. - F.cosine_similarity(z.detach(), zp.detach(), dim=1)) / 2)
        dsem_l.extend(per_img_dsem.cpu().tolist())
        # PSNR from the squared error pooled over the whole evaluation set.
        x01 = (imgs_target.detach() + 1.0) / 2.0
        xhat01 = (xh.detach() + 1.0) / 2.0
        sse_sum += float(((xhat01 - x01) ** 2).sum())
        sse_n += x01.numel()

        smap, isc = detector.anomaly_score(zp, img_size=imgs.shape[-1])

        img_lbl.extend(has_a.tolist())
        # Pixel labels and scores are kept as one numpy array per batch (about 60M
        # pixels in total; Python lists would need several GB of RAM).
        px_lbl.append(masks.cpu().numpy().ravel())
        img_sc.extend(isc.cpu().numpy().tolist())
        px_sc.append(smap.cpu().numpy().ravel())

    lbl=np.array(img_lbl); plbl=_as_list_dtype(np.concatenate(px_lbl)); dv=np.array(dsem_l)
    dm=float(dv.mean()); eps=cfg["SRI_EPSILON"]
    sc = np.array(img_sc); ps = _as_list_dtype(np.concatenate(px_sc))
    pooled_mse = sse_sum / sse_n
    psnr_pooled = 10.*math.log10(1./pooled_mse) if pooled_mse>1e-10 else float("inf")

    auc = (float(roc_auc_score(lbl,sc)) if lbl.sum()>0 and lbl.sum()<len(lbl) else float("nan"))
    fpr, fpr95_is_fallback = metric_fpr95(plbl, ps)
    # Rates at the frozen threshold tau, on the same pixels as FPR95.
    tpr_at_tau, fpr_at_tau = metric_tpr_fpr_at_tau(plbl, ps, tau)
    if plbl.sum()>0 and plbl.sum()<len(plbl):
        n = len(plbl)
        # Fixed subsample: the same pixels at every SNR point and repetition.
        idx = np.random.RandomState(42).choice(n, size=min(pixel_auc_sample, n), replace=False)
        auc_pixel = float(roc_auc_score(plbl[idx], ps[idx]))
    else:
        auc_pixel = float("nan")

    sri_op_value = None if fpr95_is_fallback else metric_sri_op(fpr,dm,eps)

    return {"snr_db":snr_db, "channel":channel_type, "dsem":dm,
            "sdop":metric_sdop(dv,sdop_threshold), "psnr":psnr_pooled,
            "auc_roc": auc, "auc_roc_pixel": auc_pixel, "fpr95": fpr,
            "fpr95_is_fallback": fpr95_is_fallback,
            "tpr_at_tau": tpr_at_tau, "fpr_at_tau": fpr_at_tau,
            # auc_roc is image-level and nearly degenerate here (almost every test
            # image contains an anomaly), so SRI uses the pixel AUROC.
            "sri_auc": metric_sri_auc(auc_pixel,dm,eps), "sri_op": sri_op_value}


def eval_both_systems(jscc_a, max_logit_det, tau_max_logit, sdop_threshold_a,
                       jscc_b, prototype_det, tau_prototype, sdop_threshold_b,
                       loader, snr_db, cfg, device, channel_type, noise_seed=None):
    """Evaluate both systems at one point; keys are prefixed max_logit_ / prototype_."""
    a = _eval_one_system(jscc_a, max_logit_det, tau_max_logit, sdop_threshold_a,
                          loader, snr_db, cfg, device, channel_type, noise_seed=noise_seed)
    b = _eval_one_system(jscc_b, prototype_det, tau_prototype, sdop_threshold_b,
                          loader, snr_db, cfg, device, channel_type, noise_seed=noise_seed)
    result = {"snr_db": snr_db, "channel": channel_type}
    result.update({f"max_logit_{k}": v for k, v in a.items() if k not in ("snr_db", "channel")})
    result.update({f"prototype_{k}": v for k, v in b.items() if k not in ("snr_db", "channel")})
    return result
