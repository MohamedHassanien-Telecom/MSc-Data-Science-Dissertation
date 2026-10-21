"""Region-balanced LS-AUROC and town split (dissertation Appendix F.8): two checks of
the location-stratified evaluation in phases/stratified_eval.py, evaluation only.

  region-balanced LS-AUROC  pooled LS-AUROC weights each region by its number of
        (anomalous x normal) pixel pairs; the region-balanced variant gives every
        region containing both classes the same weight (the mean of the
        per-region AUROCs). Same 14 x 14 regions and pixels.
  town split  LS-AUROC (14 x 14 regions) and plain pixel AUROC computed
        separately on the Town 5 and Town 6 images of the test subset.

For each variant, Decision Retention is computed per repetition (40 dB and
best-swept-value references; src/decision_retention.py) with the paired
Prototype - Max Logit comparison, as for the primary LS-AUROC.

Models, seeded channel noise, loader order and the 2,000,000-pixel subsample are
those of run_stratified_sweep. The pooled LS-AUROC computed here is checked
against stratified/stratified_state.json when that file exists, and the plain
AUROC against phase2_per_seed.json.

Outputs: run_0/stratified_supp/ (resumable per repetition and channel) and
tables/stratified_supplement.md. Needs the Phase I models in memory.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluate import eval_noise_seed
from src.decision_retention import decision_retention_analysis
from phases import location_prior_check as LP
from phases import stratified_eval as SE

SUPP_GRID = 14
SUPP_CHANNELS = ("awgn",)


def auc_by_region(y, s, reg):
    """Within-region Mann-Whitney statistic (ties 1/2) and pair count of every
    region present: returns (region ids, U, pairs). sum(U) / sum(pairs) equals
    stratified_eval.stratified_auc(y, s, reg).
    """
    y = np.asarray(y).astype(np.float64)
    s = np.asarray(s, dtype=np.float64)
    reg = np.asarray(reg, dtype=np.int64)
    if y.size == 0:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    order = np.lexsort((s, reg))
    r, v, t = reg[order], s[order], y[order]
    new = np.empty(r.size, dtype=bool)
    new[0] = True
    new[1:] = (r[1:] != r[:-1]) | (v[1:] != v[:-1])
    gid = np.cumsum(new) - 1
    pos = np.bincount(gid, weights=t)
    neg = np.bincount(gid).astype(np.float64) - pos
    g_reg = r[new]
    reg_start = np.empty(g_reg.size, dtype=bool)
    reg_start[0] = True
    reg_start[1:] = g_reg[1:] != g_reg[:-1]
    rid = np.cumsum(reg_start) - 1
    cneg = np.cumsum(neg)
    before = cneg - neg
    neg_below = before - before[reg_start][rid]
    U = np.bincount(rid, weights=pos * (neg_below + 0.5 * neg))
    P = np.bincount(rid, weights=pos)
    N = np.bincount(rid, weights=neg)
    return g_reg[reg_start], U, P * N


def region_balanced_auc(y, s, reg):
    """Mean of the per-region AUROCs over regions containing both classes, and the
    number of such regions."""
    _, U, pairs = auc_by_region(y, s, reg)
    keep = pairs > 0
    return (float(np.mean(U[keep] / pairs[keep])) if keep.any() else float("nan")), int(keep.sum())


def town_index(groups, n_img):
    """Town number (0, 1, ...) of every image and the sorted town names. Without
    usable town folders the images are split into a first and a second half by
    loader position.
    """
    if not LP._usable_groups(groups, n_img):
        groups = ["half_a" if i < n_img // 2 else "half_b" for i in range(n_img)]
    names = sorted(set(groups))
    return np.array([names.index(g) for g in groups], dtype=np.int64), names


def supp_metrics(labels, scores, hw, town_img, town_names, cfg):
    """Metrics of one set of score maps: plain AUROC (full set, src/evaluate.py
    subsample), pooled and region-balanced LS-AUROC, and plain AUROC and LS-AUROC
    per town.
    """
    H, W = int(cfg["IMG_H"]), int(cfg["IMG_W"])
    if H * W != hw:
        raise ValueError(f"pixels per image {hw} != IMG_H x IMG_W = {H} x {W}")
    idx_small = LP.pixel_subsample(labels.size)
    out = {"matched": float(roc_auc_score(labels[idx_small], scores[idx_small]))}
    idx = SE.strat_subsample(labels.size, cfg)
    y, s = labels[idx], scores[idx]
    reg = SE.region_index(idx % hw, H, W, SUPP_GRID)
    regs, U, pairs = auc_by_region(y, s, reg)
    out["ls"] = float(U.sum() / pairs.sum()) if pairs.sum() > 0 else float("nan")
    keep = pairs > 0
    out["rb"] = float(np.mean(U[keep] / pairs[keep])) if keep.any() else float("nan")
    out["rb_n_regions"] = int(keep.sum())
    t_small, t_big = town_img[idx_small // hw], town_img[idx // hw]
    out["town"] = {}
    for k, name in enumerate(town_names):
        m_s, m_b = t_small == k, t_big == k
        ys = labels[idx_small][m_s]
        plain = float(roc_auc_score(ys, scores[idx_small][m_s])) if 0 < ys.sum() < ys.size else float("nan")
        ls_t = SE.stratified_auc(y[m_b], s[m_b], reg[m_b])[0]
        out["town"][name] = {"plain": plain, "ls": float(ls_t)}
    return out


# --- Analysis ---
def _variants(town_names):
    v = {"ls": ("LS-AUROC, pooled (as the primary analysis)", lambda e: e["ls"]),
         "rb": ("Region-balanced LS-AUROC", lambda e: e["rb"])}
    for t in town_names:
        v[f"ls_{t}"] = (f"LS-AUROC, {t} only", lambda e, t=t: e["town"][t]["ls"])
        v[f"plain_{t}"] = (f"Plain pixel AUROC, {t} only", lambda e, t=t: e["town"][t]["plain"])
    return v


def supp_analysis(reps_state, p2s, cfg, town_names, n_img_per_town=None, stored_ls=None):
    """Per-SNR means and Decision Retention for each variant (no GPU), with the
    reproduction checks against Phase II and stratified_state.json.
    """
    snrs = [float(s) for s in cfg["SNR_RANGE_DB"]]
    reps = sorted(reps_state, key=int)
    chans = [ch for ch in SUPP_CHANNELS if all(ch in reps_state[r] for r in reps)]
    out = {"snr_db": snrs, "reps": reps, "channels": chans, "towns": town_names,
           "n_images_per_town": n_img_per_town, "per_snr": {}, "dr": {}}
    diffs, ls_diffs = [], []
    for ch in chans:
        for det, *_ in SE.DETS:
            for i in range(len(snrs)):
                for r in reps:
                    e = reps_state[r][ch][det][i]
                    diffs.append(abs(e["matched"] - p2s[ch][int(r)][i][f"{det}_auc_roc_pixel"]))
                    if stored_ls is not None and r in stored_ls and ch in stored_ls[r]:
                        ls_diffs.append(abs(e["ls"] - stored_ls[r][ch][det][i]["ls"][str(SUPP_GRID)]))
    out["max_abs_diff_from_phase2"] = float(max(diffs)) if diffs else None
    out["max_abs_diff_from_stratified_ls"] = float(max(ls_diffs)) if ls_diffs else None
    variants = _variants(town_names)
    for ch in chans:
        rows = []
        for i, s in enumerate(snrs):
            row = {"snr_db": s}
            for key, (_, fn) in variants.items():
                ml = np.array([fn(reps_state[r][ch]["max_logit"][i]) for r in reps], dtype=float)
                pr = np.array([fn(reps_state[r][ch]["prototype"][i]) for r in reps], dtype=float)
                d = pr - ml
                row[key] = {"max_logit": SE._ms(ml), "prototype": SE._ms(pr),
                            "n_prototype_higher": int(np.sum(d > 0)), "n": int(np.isfinite(d).sum())}
            rows.append(row)
        out["per_snr"][ch] = rows
    for key, (label, fn) in variants.items():
        ps = SE._variant_per_seed(reps_state, p2s, reps, snrs, chans, fn)
        dr = decision_retention_analysis(ps, snrs, cfg, channels=tuple(chans))
        out["dr"][key] = {"label": label, "channels": {ch: dr["channels"][ch]["auc"] for ch in dr["channels"]},
                          "snr_sweep_db": dr["snr_db"]}
    return out


def _report(an):
    L = [f"  N = {len(an['reps'])} production repetitions; channels {an['channels']}; {SUPP_GRID}x{SUPP_GRID} regions; "
         f"towns {an['towns']} (images {an['n_images_per_town']})",
         f"  Plain pixel AUROC (full set) reproduces Phase II: max |difference| {an['max_abs_diff_from_phase2']:.2e}"]
    if an.get("max_abs_diff_from_stratified_ls") is not None:
        L.append(f"  Pooled LS-AUROC reproduces stratified_state.json: max |difference| "
                 f"{an['max_abs_diff_from_stratified_ls']:.2e}")
    keys = list(an["dr"])
    for ch in an["channels"]:
        L.append(f"\n  {ch.upper()} -- mean over repetitions (Max Logit / Prototype; repetitions with Prototype higher)")
        L.append("    SNR  " + "  ".join(f"{k:>22}" for k in keys))
        for row in an["per_snr"][ch]:
            if row["snr_db"] not in (-20.0, -10.0, 0.0, 10.0, 20.0, 40.0):
                continue
            cells = []
            for k in keys:
                c = row[k]
                ml, pr = c["max_logit"]["mean"], c["prototype"]["mean"]
                cells.append(f"{ml:.4f}/{pr:.4f} {c['n_prototype_higher']:>2}/{c['n']}" if ml is not None and pr is not None
                             else "--")
            L.append(f"    {row['snr_db']:>+4.0f} " + "  ".join(f"{x:>22}" for x in cells))
    for key, blk in an["dr"].items():
        for ch, by_ref in blk["channels"].items():
            for ref, b in by_ref.items():
                refname = "40 dB reference" if ref == "ref" else "each seed's best swept value"
                L.append(f"\n  DR on {blk['label']} -- {ch.upper()}, {refname}")
                for lv in ("0.5", "0.9"):
                    if lv not in b["detectors"]["max_logit"]["levels"]:
                        continue
                    ml = b["detectors"]["max_logit"]["levels"][lv]
                    pr = b["detectors"]["prototype"]["levels"][lv]
                    p = b["paired_prototype_minus_max_logit"][lv]
                    md = "--" if p["mean_diff_db"] is None else f"{p['mean_diff_db']:+.2f}"
                    wp = "--" if p["wilcoxon_p"] is None else f"{p['wilcoxon_p']:.3g}"
                    L.append(f"    SNR{int(round(float(lv)*100))}: Max Logit {SE._snrx(ml)} | Prototype {SE._snrx(pr)}")
                    L.append(f"           Prototype - Max Logit {md} dB (n={p['n_both_crossed']}); Prototype lower in "
                             f"{p['prototype_lower_count']}/{p['n_order_determinable']}; Wilcoxon p={wp}")
    L.append("\n  Region-balanced: mean of the per-region AUROCs (regions with both classes), instead of pooling pairs.")
    L.append("  'L' = DR already >= level at the lowest swept SNR, 'NR' = level not reached; censored seeds are counted.")
    return "\n".join(L)


# --- Run ---
def run_stratified_supplement(p1_grid, cfg, device):
    """Every production repetition, both systems, every SNR, AWGN. Resumable per
    (repetition, channel). Needs the Phase I models in memory.
    """
    out_root, p2s, runs = SE._check_inputs(p1_grid, cfg)
    res_dir = out_root / "stratified_supp"; res_dir.mkdir(parents=True, exist_ok=True)
    state_path = res_dir / "stratified_supp_state.json"
    loader = runs[0]["loaders"]["test"]
    groups = LP.test_groups(loader)
    n_img = len(loader.dataset)
    town_img, town_names = town_index(groups, n_img)
    n_per_town = {t: int((town_img == k).sum()) for k, t in enumerate(town_names)}
    fp = SE._fp(cfg, runs, "town split", {"version": 1, "supp_grid": SUPP_GRID, "towns": town_names})
    saved = json.load(open(state_path)) if state_path.exists() else {}
    if saved.get("_fp") not in (None, fp):
        print("  stored results were produced under different settings -- recomputing")
        saved = {}
    state = {"_fp": fp, "reps": saved.get("reps", {})}
    snrs = list(cfg["SNR_RANGE_DB"])
    print(f"\n{'='*62}\n  REGION-BALANCED LS-AUROC AND TOWN SPLIT -- {len(runs)} repetitions x {len(snrs)} SNRs x "
          f"{list(SUPP_CHANNELS)}\n  towns {n_per_town}\n{'='*62}")
    for r_i, rep in enumerate(runs):
        rk = str(r_i)
        state["reps"].setdefault(rk, {})
        for ch in SUPP_CHANNELS:
            if ch in state["reps"][rk]:
                print(f"  repetition {r_i+1}, {ch}: SKIP, already done (resumed)")
                continue
            rec = {det: [] for det, *_ in SE.DETS}
            worst = 0.0
            for i, snr in enumerate(snrs):
                for det, name, jk, dk in SE.DETS:
                    labels, scores, hw = LP.score_maps(rep[jk], rep[dk], loader, snr, device, ch,
                                                       noise_seed=eval_noise_seed(cfg, snr, ch))
                    if labels.size != n_img * hw:
                        raise RuntimeError(f"pixel count {labels.size} != {n_img} images x {hw} pixels")
                    m = supp_metrics(labels, scores, hw, town_img, town_names, cfg)
                    m["snr_db"] = float(snr)
                    worst = max(worst, abs(m["matched"] - p2s[ch][r_i][i][f"{det}_auc_roc_pixel"]))
                    rec[det].append(m)
            state["reps"][rk][ch] = rec
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            i0 = snrs.index(0) if 0 in snrs else 0
            print(f"  repetition {r_i+1}, {ch}: done; at {snrs[i0]:+d} dB region-balanced LS-AUROC Max Logit "
                  f"{rec['max_logit'][i0]['rb']:.4f}, Prototype {rec['prototype'][i0]['rb']:.4f}; "
                  f"max |diff| from Phase II {worst:.1e}")
    reps_state = {k: v for k, v in state["reps"].items() if all(ch in v for ch in SUPP_CHANNELS)}
    strat_path = out_root / "stratified" / "stratified_state.json"
    stored = json.load(open(strat_path)).get("reps") if strat_path.exists() else None
    an = supp_analysis(reps_state, p2s, cfg, town_names, n_per_town, stored)
    with open(res_dir / "stratified_supp_summary.json", "w") as f:
        json.dump(an, f, indent=2)
    tdir = out_root / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    txt = _report(an)
    with open(tdir / "stratified_supplement.md", "w") as f:
        f.write(f"# Region-balanced LS-AUROC and town split (production systems)\n\n```\n{txt}\n```\n")
    print(txt)
    for k, lim in (("max_abs_diff_from_phase2", 1e-6), ("max_abs_diff_from_stratified_ls", 1e-9)):
        if an.get(k) is not None and an[k] > lim:
            print(f"  WARNING: {k} = {an[k]:.2e} -- the earlier evaluation is not reproduced exactly; "
                  f"check the inputs before using these numbers.")
    print(f"[Stratified supplement] summary -> {res_dir / 'stratified_supp_summary.json'}, "
          f"table -> {tdir / 'stratified_supplement.md'}")
    return an
