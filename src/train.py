"""Joint end-to-end training of one Deep JSCC system with its detector.

    L = L_task + lambda_rec * L_rec + lambda_sem * L_sem

L_task is the cell-level cross-entropy of the detector head on the known classes
(each latent cell labelled with the majority known class of its pixels). The task
gradient flows through the channel into the encoder (Shao, Mao and Zhang, 2022).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses import composite_loss
from src.dataset import denorm_to_tanh_range, pool_class_map_to_patches
from src.gpu_utils import get_amp_dtype
from pathlib import Path
from datetime import datetime
import time


def train_joint_system(jscc, detector, loader, val_loader, cfg, device, num_classes,
                        jscc_save_path, detector_save_path, lambda_rec, lambda_sem,
                        patience, joint_lr=None, joint_momentum=None, weight_decay=None,
                        joint_lr_power=None, min_rel_improve=0.01, max_epochs=None):
    """Train jscc and detector together on an AWGN channel at cfg["CLS_FIT_SNR"].

    Optimiser: SGD with momentum and weight decay; polynomial learning-rate decay
    lr = joint_lr * (1 - (epoch - 1) / max_epochs) ** joint_lr_power; gradient
    norm clipped at 1. Mixed precision uses bf16 where the GPU supports it and
    fp16 with a gradient scaler otherwise.

    Stopping: after every epoch the validation task loss is computed; both models
    are checkpointed whenever it improves. Training stops when the relative
    improvement over the last `patience` epochs falls below min_rel_improve, or
    at max_epochs (default cfg["JOINT_MAX_EPOCHS"]). The best checkpoint is
    reloaded before returning.

    The optimiser settings are detector-specific (cfg["*_ML"] / cfg["*_PR"]) and
    must be passed explicitly. Returns a dict with the best validation loss, the
    loss histories, the number of epochs and the epoch of the best checkpoint.
    """
    _missing = [name for name, val in [("joint_lr", joint_lr), ("joint_momentum", joint_momentum),
                                        ("weight_decay", weight_decay), ("joint_lr_power", joint_lr_power)]
                if val is None]
    if _missing:
        raise ValueError(
            f"train_joint_system() requires explicit values for {_missing}: pass the "
            f"detector-specific cfg['*_ML'] or cfg['*_PR'] value for each.")
    if max_epochs is None:
        max_epochs = cfg["JOINT_MAX_EPOCHS"]

    jscc.train(); detector.train()
    jscc.set_channel("awgn", cfg["CLS_FIT_SNR"])

    # torch.compile speeds up training without changing what is computed; if
    # compilation fails, execution falls back to eager mode.
    if cfg.get("USE_TORCH_COMPILE", True) and hasattr(torch, "compile"):
        try:
            from torch import _dynamo
            _dynamo.config.suppress_errors = True
            jscc = torch.compile(jscc)
            detector = torch.compile(detector)
        except Exception as e:
            print(f"[train] torch.compile() unavailable ({e}); continuing uncompiled.")

    params = list(jscc.parameters()) + list(detector.parameters())
    opt = torch.optim.SGD(params, lr=joint_lr, momentum=joint_momentum,
                           weight_decay=weight_decay)
    base_lr = joint_lr; lr_power = joint_lr_power

    amp_dtype = get_amp_dtype()
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    def _eval_loss(dl):
        jscc.eval(); detector.eval()
        tot_task = 0.0; n = 0
        with torch.no_grad():
            for b in dl:
                imgs = b["image"].to(device, non_blocking=True)
                class_map = b["class_map"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                     dtype=amp_dtype, enabled=amp_dtype is not None):
                    _, _, zp = jscc(imgs)
                    logits = detector(zp)
                labels = pool_class_map_to_patches(class_map, detector.patch_grid, num_classes)
                tot_task += F.cross_entropy(logits.reshape(-1, num_classes), labels.reshape(-1),
                                            ignore_index=-1).item()
                n += 1
        jscc.train(); detector.train()
        return tot_task / n

    history = []
    train_history = []
    best_val = float("inf")
    for ep in range(1, max_epochs + 1):
        current_lr = base_lr * (1 - (ep - 1) / max_epochs) ** lr_power
        for pg in opt.param_groups:
            pg["lr"] = current_lr

        jscc.train(); detector.train()
        tot_task = tot_rec = tot_sem = 0.0; n = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()   # the epoch's peak GPU memory is logged below
        for b in loader:
            imgs = b["image"].to(device, non_blocking=True)
            class_map = b["class_map"].to(device, non_blocking=True)
            x_target = denorm_to_tanh_range(imgs)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                 dtype=amp_dtype, enabled=amp_dtype is not None):
                xh, z, zp = jscc(imgs)
                logits = detector(zp)
                labels = pool_class_map_to_patches(class_map, detector.patch_grid, num_classes)
                l_task = F.cross_entropy(logits.reshape(-1, num_classes), labels.reshape(-1), ignore_index=-1)
                l_rec_sem, lr_val, ls_val = composite_loss(x_target, xh, z, zp,
                                                            lam1=lambda_rec, lam2=lambda_sem)
                loss = l_task + l_rec_sem

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            tot_task += l_task.item(); tot_rec += lr_val; tot_sem += ls_val; n += 1

        train_task_avg = tot_task / n
        val_task_avg = _eval_loss(val_loader)
        history.append(val_task_avg)
        train_history.append(train_task_avg)
        if val_task_avg < best_val:
            best_val = val_task_avg
            save_ckpt(jscc, opt, ep, val_task_avg, jscc_save_path)
            torch.save(_unwrap_for_saving(detector).state_dict(), detector_save_path)
        if ep == 1 or ep % 5 == 0:
            marker = " <- best so far, checkpoint saved" if val_task_avg == best_val else ""
            peak_str = ""
            if device.type == "cuda":
                epoch_peak_gb = torch.cuda.max_memory_allocated() / 1e9
                peak_str = f" [epoch peak: {epoch_peak_gb:.2f} GB]"
            print(f"  Ep{ep:03d} lr={current_lr:.5f} lambda_sem={lambda_sem} task_loss={train_task_avg:.4f} "
                  f"val_task_loss={val_task_avg:.4f} rec={tot_rec/n:.4f} sem={tot_sem/n:.4f}{marker}{peak_str}")
        if len(history) > patience:
            recent = history[-patience:]
            rel_improve = (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-8)
            if rel_improve < min_rel_improve:
                print(f"  Converged at ep{ep:03d} (validation task-loss relative improvement over "
                      f"last {patience} epochs = {rel_improve*100:.2f}% < {min_rel_improve*100:.2f}%)")
                break
    else:
        print(f"  Reached max_epochs={max_epochs} without meeting the convergence criterion "
              f"-- final val_task_loss={history[-1]:.4f}")

    load_ckpt(jscc, jscc_save_path, device=device)
    _unwrap_for_saving(detector).load_state_dict(torch.load(detector_save_path, map_location=device))
    jscc.eval(); detector.eval()
    # Release the gradients so that trained models kept in memory do not hold them.
    for p in params:
        p.grad = None
    best_epoch_idx = history.index(best_val)
    epochs_since_best = len(history) - 1 - best_epoch_idx
    return {"loss": best_val, "epochs_run": len(history), "history": history,
            "train_history": train_history,
            "best_epoch_idx": best_epoch_idx, "epochs_since_best_at_stop": epochs_since_best,
            "lambda_sem": lambda_sem}


def _unwrap_for_saving(model):
    """The original module behind a torch.compile wrapper (or `model` itself), so
    saved state_dicts have the same keys whether or not compilation was used."""
    return getattr(model, "_orig_mod", model)


def save_ckpt(model, opt, epoch, loss, path, extra=None):
    """Save the model weights with the epoch and loss. The optimiser state is not
    saved (nothing resumes from it); `opt` is accepted for interface stability."""
    state = {"epoch":epoch,"loss":loss,"model":_unwrap_for_saving(model).state_dict(),
             "ts":datetime.now().isoformat()}
    if extra: state.update(extra)
    torch.save(state, str(path))

def load_ckpt(model, path, opt=None, device=None):
    """Load weights saved by save_ckpt into `model`. The read is retried up to five
    times with back-off, because a file just written to a mounted Google Drive can
    briefly be unavailable."""
    last_err = None
    for attempt in range(5):
        try:
            ck = torch.load(str(path), map_location=device or "cpu")
            break
        except FileNotFoundError as e:
            last_err = e
            if attempt < 4:
                print(f"  [ckpt] {Path(path).name} not found on attempt "
                      f"{attempt + 1}/5 -- retrying in {2 ** attempt}s")
                time.sleep(2 ** attempt)
    else:
        raise last_err
    _unwrap_for_saving(model).load_state_dict(ck["model"])
    if opt and "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
    print(f"  [ckpt] ep{ck['epoch']}  loss={ck['loss']:.4f}  <- {Path(path).name}")
    return ck
