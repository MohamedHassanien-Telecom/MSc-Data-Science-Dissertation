"""GPU-dependent settings: mixed-precision dtype and the batch-size probe.

The probe runs only when CONFIG["BATCH_SIZE_OVERRIDE"] is None; the default
configuration fixes the batch size at 90 instead.
"""
import torch


def _try_forward_backward(model_builder_fn, device, img_size, bs, amp_dtype):
    """One forward and backward pass at batch size bs under the training autocast
    dtype. Returns (True, peak GPU memory in GB) or (False, None) on out-of-memory;
    the models and tensors are always released."""
    jscc = detector = dummy = xh = z = zp = logits = loss = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        jscc, detector = model_builder_fn()
        jscc.to(device); detector.to(device)
        dummy = torch.randn(bs, 3, img_size, img_size, device=device)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                             dtype=amp_dtype, enabled=amp_dtype is not None):
            xh, z, zp = jscc(dummy)
            logits = detector(zp)
            loss = logits.float().pow(2).mean() + xh.float().pow(2).mean()
        loss.backward()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return True, peak_gb
    except torch.cuda.OutOfMemoryError:
        return False, None
    finally:
        del jscc, detector, dummy, xh, z, zp, logits, loss
        torch.cuda.empty_cache()


def detect_max_batch_size(model_builder_fn, device, img_size=224, start=2,
                            safety_factor=0.75, warn_above=256):
    """Largest training batch that fits on the current GPU, times safety_factor.

    model_builder_fn returns a fresh (jscc, detector) pair. The batch size is
    doubled from `start` until out-of-memory, the gap is closed by binary search,
    safety_factor leaves room for the optimiser state, and the returned size is
    checked with one more pass (backing off in 10% steps if needed). A note is
    printed if the result exceeds warn_above, since the learning-rate candidates
    were not tested at such sizes. Returns 2 when no GPU is available.
    """
    if not torch.cuda.is_available():
        print("[gpu_utils] No CUDA device -- using batch_size=2 on CPU.")
        return 2

    amp_dtype = get_amp_dtype()

    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_before_gb, _ = torch.cuda.mem_get_info()
    free_before_gb = free_before_gb / 1e9
    print(f"[gpu_utils] Detected GPU: {device_name} ({total_mem_gb:.1f} GB total memory, "
          f"{free_before_gb:.1f} GB free)")
    print("[gpu_utils] Probing the largest feasible batch size...")

    bs = start
    last_good = start
    first_fail = None
    while True:
        ok, peak_gb = _try_forward_backward(model_builder_fn, device, img_size, bs, amp_dtype)
        if ok:
            print(f"[gpu_utils]   batch={bs:4d} OK (peak {peak_gb:.2f} GB)")
            last_good = bs
            bs *= 2
        else:
            print(f"[gpu_utils]   batch={bs:4d} out of memory -- end of doubling phase")
            first_fail = bs
            break

    if first_fail is not None and first_fail - last_good > 1:
        print(f"[gpu_utils] Binary search between {last_good} (fits) and {first_fail} (does not fit)...")
        lo, hi = last_good, first_fail
        while hi - lo > 1:
            mid = (lo + hi) // 2
            ok, peak_gb = _try_forward_backward(model_builder_fn, device, img_size, mid, amp_dtype)
            if ok:
                print(f"[gpu_utils]   batch={mid:4d} OK (peak {peak_gb:.2f} GB)")
                lo = mid
            else:
                print(f"[gpu_utils]   batch={mid:4d} out of memory")
                hi = mid
        print(f"[gpu_utils] Largest batch that fits: {lo}")
        last_good = lo

    safe_bs = max(1, int(last_good * safety_factor))

    # Check the returned size itself; if it fails (e.g. memory fragmentation),
    # back off in 10% steps.
    MAX_BACKOFF_ATTEMPTS = 8
    verified = False
    for attempt in range(MAX_BACKOFF_ATTEMPTS):
        ok, verified_peak_gb = _try_forward_backward(model_builder_fn, device, img_size, safe_bs, amp_dtype)
        if ok:
            pct = verified_peak_gb / total_mem_gb * 100
            if attempt == 0:
                print(f"[gpu_utils] Verified: batch_size={safe_bs} uses {verified_peak_gb:.2f} GB "
                      f"of {total_mem_gb:.1f} GB ({pct:.0f}%)")
            else:
                print(f"[gpu_utils] Verified after {attempt} back-off step(s): batch_size={safe_bs} "
                      f"uses {verified_peak_gb:.2f} GB of {total_mem_gb:.1f} GB ({pct:.0f}%)")
            verified = True
            break
        else:
            next_bs = max(1, int(safe_bs * 0.9))
            print(f"[gpu_utils]   verification at batch_size={safe_bs} ran out of memory -- "
                  f"backing off to {next_bs} (attempt {attempt + 1}/{MAX_BACKOFF_ATTEMPTS})")
            if next_bs == safe_bs:
                safe_bs = max(1, safe_bs - 1)
            else:
                safe_bs = next_bs

    if not verified:
        print(f"[gpu_utils] WARNING: no batch size could be verified after {MAX_BACKOFF_ATTEMPTS} "
              f"back-off steps (another process may be using the GPU). Using batch_size={safe_bs} "
              f"unverified.")

    print(f"[gpu_utils] Largest batch that fit: {last_good}; with the {safety_factor:.0%} "
          f"safety factor -> batch_size={safe_bs}")
    if safe_bs > warn_above:
        print(f"[gpu_utils] NOTE: batch_size={safe_bs} is above {warn_above}; the learning-rate "
              f"candidates were not tested at this size (they are re-verified in Step B).")
    return safe_bs


def get_amp_dtype():
    """Mixed-precision dtype: bf16 where the GPU supports it, otherwise fp16 (with
    a gradient scaler in training); None without a GPU."""
    if not torch.cuda.is_available():
        return None
    if torch.cuda.is_bf16_supported():
        print("[gpu_utils] GPU supports bf16 -- using it for mixed precision.")
        return torch.bfloat16
    else:
        print("[gpu_utils] GPU does not support bf16 -- using fp16 with a gradient scaler.")
        return torch.float16
