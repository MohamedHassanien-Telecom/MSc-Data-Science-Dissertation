"""src/train.py train_joint_system returns models without gradient tensors, so the
production repetitions kept in memory do not also hold a gradient copy of their
parameters. Real models with random weights, one epoch on CPU, synthetic batches.

Usage:  python3 tests/test_train_frees_gradients.py
"""
import os, sys, copy, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.config import CONFIG, set_seed
from src.models import SwinJSCCSystem, MaxLogitDetector, PrototypeDetector
from src.train import train_joint_system

cfg = copy.deepcopy(CONFIG)
cfg["USE_TORCH_COMPILE"] = False
K = 13
g = torch.Generator().manual_seed(0)
batches = [{"image": torch.randn(2, 3, 224, 224, generator=g),
            "class_map": torch.randint(0, K, (2, 1, 224, 224), generator=g)} for _ in range(2)]

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

with tempfile.TemporaryDirectory() as d:
    for name, st, gr, dim, det in (("max_logit", 2, 14, 8, MaxLogitDetector(8, 14, K)),
                                   ("prototype", 3, 7, 32, PrototypeDetector(32, 7, K, 4.0))):
        set_seed(1)
        jscc = SwinJSCCSystem(256, pretrained=False, patch_grid=gr, patch_dim=dim, stage_idx=st)
        train_joint_system(jscc, det, batches, batches, cfg, torch.device("cpu"), K,
                           jscc_save_path=os.path.join(d, f"{name}_j.pt"),
                           detector_save_path=os.path.join(d, f"{name}_d.pt"),
                           lambda_rec=0.5, lambda_sem=2.0, patience=15, joint_lr=0.003,
                           joint_momentum=0.9, weight_decay=1e-4, joint_lr_power=1.0,
                           min_rel_improve=0.01, max_epochs=1)
        held = [n for n, p in list(jscc.named_parameters()) + list(det.named_parameters()) if p.grad is not None]
        check(f"{name}: no gradient tensors left after training ({len(held)} held)", not held)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All gradient-freeing checks passed.")
