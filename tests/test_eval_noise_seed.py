"""Seeded channel noise in evaluation (src/evaluate.py eval_noise_seed and the
noise_seed argument of _eval_one_system / eval_both_systems), with real models
(random weights) on CPU:
  1. the same (SNR, channel) seed gives identical results;
  2. a seeded evaluation leaves the global torch RNG unchanged;
  3. a different seed gives different noise (different D_sem);
  4. eval_both_systems gives both systems the same noise draws;
  5. eval_noise_seed differs for every (channel, SNR) point of the sweep.

Usage:  python3 tests/test_eval_noise_seed.py
"""
import os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.config import CONFIG
from src.models import SwinJSCCSystem, MaxLogitDetector
from src.evaluate import _eval_one_system, eval_both_systems, eval_noise_seed

cfg = copy.deepcopy(CONFIG)
dev = torch.device("cpu")
torch.manual_seed(0)
jscc = SwinJSCCSystem(256, pretrained=False, patch_grid=14, patch_dim=8, stage_idx=2).eval()
det = MaxLogitDetector(8, 14, 13).eval()

g = torch.Generator().manual_seed(1)
loader = []
for _ in range(2):
    mask = (torch.rand(2, 1, 224, 224, generator=g) > 0.97).float()
    loader.append({"image": torch.randn(2, 3, 224, 224, generator=g), "mask": mask,
                   "has_anomaly": torch.tensor([1, 1])})

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

def same(a, b):
    """Dict equality that treats NaN == NaN (image-level AUROC is NaN when every image is anomalous)."""
    return a.keys() == b.keys() and all(
        (isinstance(a[k], float) and isinstance(b[k], float) and a[k] != a[k] and b[k] != b[k]) or a[k] == b[k]
        for k in a)

def ev(seed, snr=0.0):
    return _eval_one_system(jscc, det, 0.0, 0.1, loader, snr, cfg, dev, "awgn", noise_seed=seed)

s0 = eval_noise_seed(cfg, 0, "awgn")
r1, r2 = ev(s0), ev(s0)
check("same seed -> identical results", same(r1, r2))

torch.manual_seed(777); before = torch.get_rng_state().clone()
ev(s0)
check("global torch RNG state unchanged by a seeded evaluation", torch.equal(before, torch.get_rng_state()))

r3 = ev(s0 + 1)
check("different seed -> different noise (D_sem differs)", r3["dsem"] != r1["dsem"])

both = eval_both_systems(jscc, det, 0.0, 0.1, jscc, det, 0.0, 0.1, loader, 0.0, cfg, dev, "awgn", noise_seed=s0)
check("eval_both_systems: both systems see the same noise",
      both["max_logit_dsem"] == both["prototype_dsem"] == r1["dsem"]
      and both["max_logit_auc_roc_pixel"] == both["prototype_auc_roc_pixel"] == r1["auc_roc_pixel"])

seeds = [eval_noise_seed(cfg, s, ch) for ch in ("awgn", "rayleigh") for s in cfg["SNR_RANGE_DB"]]
check(f"eval_noise_seed distinct for all {len(seeds)} (channel, SNR) points", len(set(seeds)) == len(seeds))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All eval noise seed checks passed.")
