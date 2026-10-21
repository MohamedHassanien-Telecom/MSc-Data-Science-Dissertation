"""Location audit (dissertation Appendix F.1): how much of the pixel AUROC comes from
where anomalies usually appear in the frame rather than from the received image?

Evaluation only, every production repetition, both systems, AWGN at 40 / 0 / -10 /
-20 dB:

  matched    the Phase II pixel AUROC (same model, noise seed, loader order and
             pixel subsample as src/evaluate.py; checked against
             phase2_per_seed.json)
  shuffled   each image's scores against the mask of a partner image from the
             other test town (t5 <-> t6), at the same pixel positions: what the
             score maps achieve from location alone
  location   a model-free map: for each image, the fraction of the other town's
             images that are anomalous at each pixel

Without town folders the partner is the next image in the test order and the map
leaves out only the image itself; the report states the rule used. Outputs:
run_0/location_prior/ (resumable per repetition) and tables/location_prior_check.md.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.evaluate import eval_noise_seed, _as_list_dtype
from src.config import training_fingerprint, evaluation_fingerprint, fingerprint_key

LOCATION_SNRS_DB = (40, 0, -10, -20)
SYSTEMS = (("max_logit", "Max Logit", "jscc_a", "max_logit_det"),
           ("prototype", "Prototype", "jscc_b", "prototype_det"))


@torch.no_grad()
def score_maps(model, detector, loader, snr_db, device, channel_type="awgn", noise_seed=None):
    """Per-pixel anomaly scores and labels for every test image, concatenated in
    loader order exactly as src/evaluate.py _eval_one_system builds them (same
    forward pass, same seeded channel noise). Returns (labels, scores, hw) with
    labels/scores as 1-D arrays over all pixels and hw = pixels per image."""
    if noise_seed is not None:
        _devs = ([device.index if device.index is not None else torch.cuda.current_device()]
                 if getattr(device, "type", str(device)) == "cuda" else [])
        with torch.random.fork_rng(devices=_devs):
            torch.manual_seed(int(noise_seed))
            return score_maps(model, detector, loader, snr_db, device, channel_type, noise_seed=None)
    model.eval(); model.set_channel(channel_type, snr_db)
    px_lbl, px_sc, hw = [], [], None
    for b in loader:
        imgs = b["image"].to(device, non_blocking=True); masks = b["mask"].to(device, non_blocking=True)
        _, _, zp = model(imgs)
        smap, _ = detector.anomaly_score(zp, img_size=imgs.shape[-1])
        hw = int(np.prod(masks.shape[1:]))
        px_lbl.append(masks.cpu().numpy().ravel())
        px_sc.append(smap.cpu().numpy().ravel())
    return _as_list_dtype(np.concatenate(px_lbl)), _as_list_dtype(np.concatenate(px_sc)), hw


@lru_cache(maxsize=4)
def _subsample_cached(n, pixel_auc_sample):
    idx = np.random.RandomState(42).choice(n, size=min(pixel_auc_sample, n), replace=False)
    idx.setflags(write=False)
    return idx


def pixel_subsample(n, pixel_auc_sample=200_000):
    """The pixel subsample src/evaluate.py uses (seeded, identical at every SNR)."""
    return _subsample_cached(int(n), int(pixel_auc_sample))


def test_groups(loader):
    """Town folder (t5 / t6) of every test image, in loader order; None if the
    dataset does not expose its image paths."""
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return None
    base, ids = (ds.dataset, list(ds.indices)) if hasattr(ds, "indices") else (ds, list(range(len(ds))))
    paths = getattr(base, "img_paths", None)
    if paths is None:
        return None
    return [Path(paths[j]).parent.name for j in ids]


def _usable_groups(groups, n_img):
    return groups is not None and len(groups) == n_img and len(set(groups)) >= 2


def partner_images(groups, n_img):
    """Partner image of every image: an image of a different town (cyclic within
    that town's images, in loader order). With fewer than two towns: the next image
    in loader order. Returns (partner indices, rule description)."""
    if not _usable_groups(groups, n_img):
        return (np.arange(n_img) + 1) % n_img, "next image in the (seeded random) test order"
    g = np.asarray(groups)
    names = sorted(set(groups))
    pos = {name: np.flatnonzero(g == name) for name in names}
    partner = np.empty(n_img, dtype=np.int64)
    for k, name in enumerate(names):
        mine, other = pos[name], pos[names[(k + 1) % len(names)]]
        partner[mine] = other[np.arange(mine.size) % other.size]
    return partner, "image from the other town"


def shuffled_labels(labels, idx, hw, partner):
    """Labels at the subsample positions idx, taken from image partner[i] instead
    of image i (same pixel position within the image)."""
    img, pix = np.divmod(np.asarray(idx), hw)
    return labels[partner[img] * hw + pix]


def location_baseline_scores(labels, idx, hw, groups=None):
    """Model-free location score at the subsample positions: for image i, the
    fraction of the images of the other town(s) that are anomalous at the same
    pixel position. With fewer than two towns: the fraction of the other N-1
    images (leave-one-out)."""
    n_img = labels.size // hw
    L = labels.reshape(n_img, hw)
    img, pix = np.divmod(np.asarray(idx), hw)
    total = L.sum(axis=0, dtype=np.float64)
    if not _usable_groups(groups, n_img):
        return (total[pix] - labels[idx]) / (n_img - 1)
    g = np.asarray(groups)
    out = np.empty(img.size, dtype=np.float64)
    for name in set(groups):
        rows = g == name
        freq = (total - L[rows].sum(axis=0, dtype=np.float64)) / (n_img - int(rows.sum()))
        sel = rows[img]
        out[sel] = freq[pix[sel]]
    return out


def location_aucs(labels, scores, hw, partner, pixel_auc_sample=200_000):
    """matched and shuffled AUROC of one score set, on the evaluate.py subsample."""
    idx = pixel_subsample(labels.size, pixel_auc_sample)
    y = labels[idx]
    out = {"matched": float(roc_auc_score(y, scores[idx]))}
    ys = shuffled_labels(labels, idx, hw, partner)
    out["shuffled"] = float(roc_auc_score(ys, scores[idx])) if 0 < ys.sum() < ys.size else None
    return out


def input_dependence(model, detector, loader, snr_db, cfg, device, channel_type="awgn"):
    """matched and shuffled AUROC of one system at one SNR (Phase II noise seed),
    and their difference. A system whose output does not depend on the image has
    matched ~= shuffled. Used as the collapse guard of the retraining checks."""
    labels, scores, hw = score_maps(model, detector, loader, snr_db, device, channel_type,
                                    noise_seed=eval_noise_seed(cfg, snr_db, channel_type))
    partner, rule = partner_images(test_groups(loader), labels.size // hw)
    a = location_aucs(labels, scores, hw, partner)
    a["matched_minus_shuffled"] = None if a["shuffled"] is None else a["matched"] - a["shuffled"]
    a["partner_rule"] = rule
    return a


def _summ(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()) if v.size else None,
            "sd": float(v.std(ddof=1)) if v.size > 1 else None}


def location_analysis(state, reps, snrs):
    """Means over repetitions of matched / shuffled AUROC and of the share of the
    above-chance AUROC that location alone reproduces, (shuffled - 0.5) / (matched - 0.5)."""
    out = {"snr_db": [float(s) for s in snrs], "n_reps": len(reps), "systems": {}}
    for det, name, _, _ in SYSTEMS:
        rec = {}
        for s in snrs:
            k = str(float(s))
            m = [state[str(r)]["results"][det][k]["matched"] for r in reps]
            sh = [state[str(r)]["results"][det][k]["shuffled"] for r in reps]
            share = [None if (b is None or a <= 0.5) else (b - 0.5) / (a - 0.5) for a, b in zip(m, sh)]
            rec[k] = {"matched": _summ(m), "shuffled": _summ(sh), "location_share": _summ(share),
                      "matched_minus_shuffled": _summ([None if b is None else a - b for a, b in zip(m, sh)])}
        out["systems"][det] = rec
    loc = [state[str(r)]["location_prior_auc"] for r in reps]
    out["location_prior_auc"] = loc[0] if loc else None
    out["max_abs_diff_from_phase2"] = float(np.nanmax([state[str(r)]["max_abs_diff_from_phase2"] for r in reps]))
    out["partner_rule"] = state.get("_partner_rule")
    out["town_counts"] = state.get("_town_counts")
    return out


def _report(an):
    L = [f"  Pixel AUROC, N={an['n_reps']} repetitions, mean +/- SD (AWGN)",
         f"  Partner / location map rule: {an['partner_rule']}; test images per town: {an['town_counts']}",
         f"  Model-free location baseline (anomaly-frequency map): AUROC {an['location_prior_auc']:.4f}",
         f"  Matched AUROC reproduces Phase II: max |difference| {an['max_abs_diff_from_phase2']:.2e}",
         "    SNR   system      matched           shuffled          matched - shuffled   location share"]
    f = lambda d, nd=4: "--" if d["mean"] is None else (f"{d['mean']:.{nd}f} +/- {d['sd']:.{nd}f}"
                                                         if d["sd"] is not None else f"{d['mean']:.{nd}f}")
    for s in an["snr_db"]:
        for det, name, _, _ in SYSTEMS:
            r = an["systems"][det][str(s)]
            L.append(f"    {s:+4.0f}  {name:<10}  {f(r['matched'])}   {f(r['shuffled'])}   "
                     f"{f(r['matched_minus_shuffled'])}      {f(r['location_share'], 2)}")
    L.append("  'shuffled': each image's scores against the mask of a partner image (location only).")
    L.append("  'location share' = (shuffled - 0.5) / (matched - 0.5): the part of the above-chance AUROC")
    L.append("  that the system's score maps reproduce without matching image and labels.")
    return "\n".join(L)


def run_location_prior_check(p1_grid, cfg, device, snrs=LOCATION_SNRS_DB):
    """Evaluate every production repetition (both systems) at `snrs` (AWGN),
    resumable per repetition. Needs the Phase I models in p1_grid."""
    out_root = Path(p1_grid["out"])
    res_dir = out_root / "location_prior"; res_dir.mkdir(parents=True, exist_ok=True)
    state_path = res_dir / "location_state.json"
    per_seed_path = out_root / "phase2_per_seed.json"
    if not per_seed_path.exists():
        raise FileNotFoundError(f"{per_seed_path} not found -- run Phase II first.")
    p2s = json.load(open(per_seed_path))
    grid = [float(x) for x in p2s["snr_range_db"]]
    for s in snrs:
        if float(s) not in grid:
            raise ValueError(f"SNR {s} dB is not in the Phase II grid {grid}")
    runs = p1_grid["final_runs"]
    if len(p2s["awgn"]) < len(runs):
        raise ValueError("phase2_per_seed.json has fewer repetitions than Phase I -- stale Phase II file.")
    if any(r.get("jscc_a") is None or r.get("jscc_b") is None for r in runs):
        raise RuntimeError("The Phase I models are not in memory (released by an earlier cell). Restart the "
                           "session, re-run the setup cells and the Phase I cell, then this cell.")
    loader = runs[0]["loaders"]["test"]
    groups = test_groups(loader)
    n_img = len(loader.dataset)
    partner, rule = partner_images(groups, n_img)
    fp = fingerprint_key({"ml": training_fingerprint(cfg, "max_logit"), "pr": training_fingerprint(cfg, "prototype"),
                          "eval": evaluation_fingerprint(cfg), "snrs": [float(s) for s in snrs],
                          "taus": [[r["tau_max_logit"], r["tau_prototype"]] for r in runs],
                          "partner_rule": rule, "version": 2})
    state = json.load(open(state_path)) if state_path.exists() else {}
    if state.get("_fp") not in (None, fp):
        print("  stored results were produced under different settings -- recomputing")
        state = {}
    state["_fp"] = fp
    state["_partner_rule"] = rule
    state["_town_counts"] = ({g: int(sum(1 for x in groups if x == g)) for g in sorted(set(groups))}
                             if groups is not None else None)
    print(f"\n{'='*62}\n  LOCATION-PRIOR CHECK -- {len(runs)} repetitions, SNRs {list(snrs)} dB (AWGN)\n"
          f"  partner rule: {rule}; towns {state['_town_counts']}\n{'='*62}")
    for r_i, rep in enumerate(runs):
        key = str(r_i)
        if key in state and "results" in state[key]:
            print(f"  repetition {r_i+1}: SKIP, already done (resumed)")
            continue
        res = {det: {} for det, *_ in SYSTEMS}
        diffs, loc_auc = [], None
        for s in snrs:
            for det, name, jk, dk in SYSTEMS:
                labels, scores, hw = score_maps(rep[jk], rep[dk], loader, s, device, "awgn",
                                                noise_seed=eval_noise_seed(cfg, s, "awgn"))
                if labels.size != n_img * hw:
                    raise RuntimeError(f"pixel count {labels.size} != {n_img} images x {hw} pixels")
                a = location_aucs(labels, scores, hw, partner)
                ref = p2s["awgn"][r_i][grid.index(float(s))][f"{det}_auc_roc_pixel"]
                diffs.append(abs(a["matched"] - ref))
                res[det][str(float(s))] = a
                if loc_auc is None:
                    idx = pixel_subsample(labels.size)
                    loc_auc = float(roc_auc_score(labels[idx], location_baseline_scores(labels, idx, hw, groups)))
        state[key] = {"results": res, "location_prior_auc": loc_auc, "max_abs_diff_from_phase2": float(max(diffs))}
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        m0 = res["max_logit"][str(float(snrs[0]))]; p0 = res["prototype"][str(float(snrs[0]))]
        print(f"  repetition {r_i+1}: at {snrs[0]:+d} dB Max Logit matched {m0['matched']:.4f} shuffled "
              f"{m0['shuffled']:.4f} | Prototype matched {p0['matched']:.4f} shuffled {p0['shuffled']:.4f} "
              f"| location baseline {loc_auc:.4f} | max |diff| from Phase II {max(diffs):.1e}")
    reps = list(range(len(runs)))
    an = location_analysis(state, reps, snrs)
    with open(res_dir / "location_summary.json", "w") as f:
        json.dump(an, f, indent=2)
    tdir = out_root / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    rep_txt = _report(an)
    with open(tdir / "location_prior_check.md", "w") as f:
        f.write(f"# Location-prior check (production systems, AWGN)\n\n```\n{rep_txt}\n```\n")
    print(rep_txt)
    if an["max_abs_diff_from_phase2"] > 1e-6:
        print(f"  WARNING: matched AUROC differs from Phase II by up to {an['max_abs_diff_from_phase2']:.2e} -- "
              f"the Phase II evaluation is not reproduced exactly.")
    print(f"[Location-prior check] summary -> {res_dir / 'location_summary.json'}, "
          f"table -> {tdir / 'location_prior_check.md'}")
    return an
