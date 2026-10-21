"""Detection and semantic-distortion metrics, and the two thresholds calibrated in Phase I.

Pixel AUROC is computed in src/evaluate.py; this module holds FPR95, the rates
at the frozen threshold tau, SRI, SDOP and the calibration of tau and of the
SDOP threshold.
"""
import torch
import numpy as np
from sklearn.metrics import roc_curve

def metric_fpr95(labels, scores):
    """False-positive rate at 95% true-positive rate, from this point's own ROC curve.

    The threshold is re-derived at every evaluation point (threshold-adaptive);
    metric_tpr_fpr_at_tau gives the rates at the frozen threshold instead.
    Returns (fpr95, is_fallback). is_fallback is True when the ROC curve never
    reaches 95% TPR and the last point of the curve is returned instead.
    """
    n_pos = labels.sum()
    if n_pos==0 or n_pos==len(labels): return float("nan"), False
    fprs, tprs, _ = roc_curve(labels, scores)
    for fpr, tpr in zip(fprs, tprs):
        if tpr >= 0.95: return float(fpr), False
    return float(fprs[-1]), True

def metric_tpr_fpr_at_tau(labels, scores, tau):
    """TPR and FPR at the fixed threshold tau calibrated once in Phase I.

    Shows how a receiver that cannot re-calibrate behaves as the channel
    degrades. Returns (tpr, fpr); a rate is NaN when its class is absent.
    """
    labels = np.asarray(labels); scores = np.asarray(scores)
    # ">=" as in roc_curve, from which tau is taken. Scores are constant over each
    # latent cell, so a whole block of pixels can sit exactly at tau.
    predicted_positive = scores >= tau
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    tpr = float((predicted_positive & (labels==1)).sum() / n_pos) if n_pos > 0 else float("nan")
    fpr = float((predicted_positive & (labels==0)).sum() / n_neg) if n_neg > 0 else float("nan")
    return tpr, fpr

def metric_sri_auc(auc, dsem, eps=1e-6):
    """SRI_AUC = AUROC / (D_sem + eps)."""
    return auc / (dsem + eps)

def metric_sri_op(fpr95, dsem, eps=1e-6):
    """SRI_op = (1 - FPR95) / (D_sem + eps)."""
    return (1.0 - fpr95) / (dsem + eps)

def metric_sdop(dsem_arr, threshold=0.10):
    """Semantic Distortion Outage Probability, P(d_sem > threshold) over images."""
    return float((dsem_arr > threshold).mean())


def calibrate_sdop_threshold(jscc, cal_loader, device, snr_db=40.0, percentile=90):
    """SDOP threshold: the given percentile of the per-image d_sem of the
    calibration images at the clean reference SNR.

    With percentile=90, SDOP is about 10% at the reference condition, and its
    rise as the SNR falls is read against that baseline. Calibrated once per
    system and held fixed over all SNRs.
    """
    import numpy as np
    import torch.nn.functional as F
    jscc.eval()
    jscc.set_channel("awgn", snr_db)
    dsem_vals = []
    with torch.no_grad():
        for b in cal_loader:
            imgs = b["image"].to(device, non_blocking=True)
            _, z, zp = jscc(imgs)
            per_sample_dsem = (1. - F.cosine_similarity(z.detach(), zp.detach(), dim=1)) / 2
            dsem_vals.extend(per_sample_dsem.cpu().tolist())
    threshold = float(np.percentile(dsem_vals, percentile))
    print(f"[SDOP] Calibrated threshold={threshold:.6f} ({percentile}th percentile of "
          f"{len(dsem_vals)} calibration D_sem values at {snr_db}dB)")
    return threshold


def calibrate_tau(jscc, detector, cal_loader, device, target_tpr=0.95, snr_db=40.0):
    """Anomaly threshold tau: the first ROC threshold reaching target_tpr on the
    pixel labels of the calibration images (both classes present).

    The detector scores the received latent at snr_db, as in evaluation. tau is
    calibrated once and held fixed at every evaluated SNR and channel.
    """
    jscc.eval(); detector.eval()
    jscc.set_channel("awgn", snr_db)
    labels, scores = [], []
    with torch.no_grad():
        for b in cal_loader:
            imgs = b["image"].to(device, non_blocking=True)
            masks = b["mask"].to(device, non_blocking=True)
            _, _, z_prime = jscc(imgs)
            smap, _ = detector.anomaly_score(z_prime, img_size=imgs.shape[-1])
            labels.extend(masks.cpu().numpy().flatten().tolist())
            scores.extend(smap.cpu().numpy().flatten().tolist())
    labels = np.array(labels); scores = np.array(scores)
    if labels.sum() == 0:
        tau = float(np.median(scores))
        print(f"[tau] WARNING: no anomalies in calibration set -> tau={tau:.5f} (fallback)")
        return tau
    fprs, tprs, thresholds = roc_curve(labels, scores)
    for fpr, tpr, thr in zip(fprs, tprs, thresholds):
        if tpr >= target_tpr:
            print(f"[tau] Calibrated tau={thr:.5f} (TPR={tpr:.3f}, FPR={fpr:.3f})")
            return float(thr)
    tau = float(thresholds[-1])
    print(f"[tau] Target TPR={target_tpr} not reached -> tau={tau:.5f}")
    return tau
