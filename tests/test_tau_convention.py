"""Fixed-threshold rates (src/metrics.py metric_tpr_fpr_at_tau) use the threshold
convention of calibrate_tau, which takes tau from sklearn's roc_curve
(score >= threshold is positive). With one score per latent cell, whole blocks
of pixels share a score; at a calibrated tau the fixed-threshold rates must
equal the (TPR, FPR) that roc_curve reports for that threshold.

Usage:  python3 tests/test_tau_convention.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.metrics import roc_curve
from src.metrics import metric_tpr_fpr_at_tau

rng = np.random.default_rng(0)
ok = []
for trial in range(50):
    # 300 latent cells, each repeated over a 16 x 16 block of pixels.
    cell_scores = rng.normal(size=300).astype(np.float32)
    scores = np.repeat(cell_scores, 256)
    labels = (rng.random(scores.size) < 0.02 + 0.3 * (scores > 1.0)).astype(int)
    fprs, tprs, thr = roc_curve(labels, scores)
    # The operating point calibrate_tau selects.
    i = int(np.argmax(tprs >= 0.95))
    tpr, fpr = metric_tpr_fpr_at_tau(labels, scores, float(thr[i]))
    ok.append(abs(tpr - tprs[i]) < 1e-12 and abs(fpr - fprs[i]) < 1e-12)
print(f"  [{'PASS' if all(ok) else 'FAIL'}] fixed-tau rates == roc_curve operating point at tau "
      f"({sum(ok)}/{len(ok)} random block-tied cases)")
if not all(ok):
    sys.exit(1)
print("All tau-convention checks passed.")
