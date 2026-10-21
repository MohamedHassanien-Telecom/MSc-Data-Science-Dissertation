"""Pooled quantities of src/evaluate.py _eval_one_system with batches of unequal
size (a stub model returns fixed latents and reconstructions per image):
  1. D_sem is the mean over images of (1 - cos(z, z'))/2, not the mean of batch
     means;
  2. PSNR is computed from the squared error pooled over all pixels, not the mean
     of batch PSNRs;
  3. both differ from the batch-mean versions for these batches, so the test
     distinguishes them.

Usage:  python3 tests/test_eval_pooling.py
"""
import os, sys, copy, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from src.config import CONFIG
from src.dataset import denorm_to_tanh_range
from src.evaluate import _eval_one_system

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

H = W = 8
g = torch.Generator().manual_seed(0)


class StubModel:
    """Returns, per image, a latent z, a received latent z' and a reconstruction
    x_hat looked up from the image's first pixel (its index)."""
    def __init__(self, n):
        self.z = torch.randn(n, 16, generator=g)
        self.zp = self.z + torch.randn(n, 16, generator=g) * torch.linspace(0.1, 2.0, n)[:, None]
        self.err = torch.linspace(0.01, 0.4, n)
    def eval(self): return self
    def set_channel(self, *a): pass
    def __call__(self, imgs):
        i = imgs[:, 0, 0, 0].round().long()
        return self.xhat_for(imgs, i), self.z[i], self.zp[i]
    def xhat_for(self, imgs, i):
        target = denorm_to_tanh_range(imgs)
        return (target + 2.0 * self.err[i][:, None, None, None]).clamp(-1, 1)


class StubDetector:
    """Pixel scores: each image's first latent value, broadcast over the image."""
    def anomaly_score(self, zp, img_size=224):
        b = zp.shape[0]
        smap = zp[:, :1, None].expand(b, 1, img_size * img_size).reshape(b, 1, img_size, img_size)
        return smap, zp[:, 0]


n_img = 7
sizes = [5, 2]                                  # unequal batches
model = StubModel(n_img)
batches, k = [], 0
for bs in sizes:
    imgs = torch.rand(bs, 3, H, W, generator=g)
    imgs[:, 0, 0, 0] = torch.arange(k, k + bs, dtype=torch.float32)     # image index in the first pixel
    masks = (torch.rand(bs, H, W, generator=g) < 0.3).long()
    batches.append({"image": imgs, "mask": masks, "has_anomaly": torch.ones(bs, dtype=torch.long)})
    k += bs

cfg = copy.deepcopy(CONFIG)
r = _eval_one_system(model, StubDetector(), 0.0, 0.1, batches, 0.0, cfg, torch.device("cpu"), "awgn",
                     pixel_auc_sample=10)

# Reference values computed directly.
per_img = ((1 - torch.nn.functional.cosine_similarity(model.z, model.zp, dim=1)) / 2).numpy()
batch_means, start = [], 0
for bs in sizes:
    batch_means.append(per_img[start:start + bs].mean()); start += bs
check(f"D_sem = mean over images ({r['dsem']:.6f} vs {per_img.mean():.6f})", abs(r["dsem"] - per_img.mean()) < 1e-6)
check(f"the mean of batch means differs ({np.mean(batch_means):.6f})", abs(np.mean(batch_means) - per_img.mean()) > 1e-4)

sse, n, batch_psnr = 0.0, 0, []
for b in batches:
    imgs = b["image"]; i = imgs[:, 0, 0, 0].round().long()
    x01 = (denorm_to_tanh_range(imgs) + 1) / 2
    xh01 = (model.xhat_for(imgs, i) + 1) / 2
    e = float(((xh01 - x01) ** 2).sum()); sse += e; n += x01.numel()
    batch_psnr.append(10 * math.log10(1 / (e / x01.numel())))
pooled = 10 * math.log10(1 / (sse / n))
check(f"PSNR from the pooled squared error ({r['psnr']:.4f} vs {pooled:.4f} dB)", abs(r["psnr"] - pooled) < 1e-4)
check(f"the mean of batch PSNRs differs ({np.mean(batch_psnr):.4f} dB)", abs(np.mean(batch_psnr) - pooled) > 1e-3)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All evaluation-pooling checks passed.")
