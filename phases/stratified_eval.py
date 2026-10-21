"""Location-stratified pixel AUROC (LS-AUROC; dissertation Section 3.1.9 and
Appendix F.2): pixel AUROC with the image-independent location prior removed.

Plain pixel AUROC compares anomalous and normal pixels from any two positions, so
a score map that is high where anomalies usually appear already ranks most
anomalous pixels above most normal ones (phases/location_prior_check.py).
LS-AUROC compares pixels only within the same image region:

    LS-AUROC = P( s(anomalous pixel) > s(normal pixel) | both in the same region ),
               ties counted 1/2,

pooled over regions: the within-region Mann-Whitney U statistics are summed and
divided by the summed within-region (anomalous x normal) pair counts. The image
(IMG_H x IMG_W) is divided into G x G square regions. The primary grid is G = 14
(16 x 16-pixel regions at 224 x 224), the finest grid on which both systems'
score maps (Max Logit 14 x 14, Prototype 7 x 7, nearest-upsampled) are constant
inside every region; G = 7 and G = 28 are sensitivity grids.

Any score map that does not depend on the image and is constant inside each
region scores exactly 0.5, so Decision Retention anchored at 0.5 and the SRI
numerator measure what the received signal adds beyond location. The location
information left inside the regions is measured by the LS-AUROC of the
model-free anomaly-frequency map (location_prior_check.location_baseline_scores).

Pixels: a fixed subsample of STRAT_PIXEL_SAMPLE pixels (RandomState(42),
2,000,000 by default), the same at every SNR, channel, repetition and system.
Plain pixel AUROC is computed as in src/evaluate.py (its 200,000-pixel
subsample) and checked against phase2_per_seed.json. The shuffled-label AUROC
(scores against the mask of an image from the other test town) is kept as a
second location check.

Two runs (evaluation only; both need the Phase I models in memory):
  run_stratified_sweep    every production repetition, both systems, every SNR
                          in SNR_RANGE_DB, AWGN and Rayleigh (seeded noise as in
                          Phase II); resumable per (repetition, channel). Gives
                          Decision Retention per seed on LS-AUROC (40 dB and
                          best-swept-value references), the paired Prototype -
                          Max Logit comparison and the SRI on LS-AUROC.
  run_digital_stratified  the ideal digital link of phases/digital_baseline.py
                          scored with LS-AUROC. Below its threshold the receiver
                          outputs an image-independent map, scored at the best
                          value such a map reaches: max(0.5, LS-AUROC of the
                          model-free location map).

Outputs: run_0/stratified/, tables/stratified_eval.md, tables/digital_stratified.md
and plots 14_*, 14b_*, 14c_*.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.evaluate import eval_noise_seed
from src.config import training_fingerprint, evaluation_fingerprint, fingerprint_key
from src.decision_retention import decision_retention_analysis
from phases import location_prior_check as LP

DETS = (("max_logit", "Max Logit", "jscc_a", "max_logit_det"),
        ("prototype", "Prototype", "jscc_b", "prototype_det"))
# Default region grids; the first is the primary grid.
STRAT_GRIDS = (14, 7, 28)
STRAT_PIXEL_SAMPLE = 2_000_000


def grids(cfg):
    """Region grids G, primary first (cfg STRAT_GRIDS, default STRAT_GRIDS)."""
    return tuple(int(g) for g in cfg.get("STRAT_GRIDS", STRAT_GRIDS))


def region_index(pix, H, W, G):
    """Region (0 .. G*G-1) of within-image pixel positions pix (row-major, H x W)."""
    pix = np.asarray(pix)
    row, col = np.divmod(pix, W)
    return (row * G // H) * G + (col * G // W)


def stratified_auc(y, s, reg):
    """Pooled within-region AUROC: sum over regions of the Mann-Whitney U of the
    region's anomalous vs normal pixels (ties 1/2), divided by the summed
    within-region pair counts. Returns (auroc, n_pairs); auroc is NaN if no region
    contains both classes."""
    y = np.asarray(y).astype(np.float64)
    s = np.asarray(s, dtype=np.float64)
    reg = np.asarray(reg, dtype=np.int64)
    if y.size == 0:
        return float("nan"), 0.0
    order = np.lexsort((s, reg))
    r, v, t = reg[order], s[order], y[order]
    new = np.empty(r.size, dtype=bool)
    new[0] = True
    new[1:] = (r[1:] != r[:-1]) | (v[1:] != v[:-1])
    # Group = (region, tied score value).
    gid = np.cumsum(new) - 1
    pos = np.bincount(gid, weights=t)
    neg = np.bincount(gid).astype(np.float64) - pos
    g_reg = r[new]
    reg_start = np.empty(g_reg.size, dtype=bool)
    reg_start[0] = True
    reg_start[1:] = g_reg[1:] != g_reg[:-1]
    rid = np.cumsum(reg_start) - 1
    cneg = np.cumsum(neg)
    # Normal pixels in earlier groups, then restricted to the same region.
    before = cneg - neg
    neg_below = before - before[reg_start][rid]
    U = float(np.sum(pos * (neg_below + 0.5 * neg)))
    P = np.bincount(rid, weights=pos)
    N = np.bincount(rid, weights=neg)
    pairs = float(np.sum(P * N))
    return (U / pairs if pairs > 0 else float("nan")), pairs


@lru_cache(maxsize=4)
def _big_subsample(n, size):
    idx = np.random.RandomState(42).choice(n, size=min(size, n), replace=False)
    idx.setflags(write=False)
    return idx


def strat_subsample(n, cfg):
    """Fixed seeded subsample of pixel indices used for every LS-AUROC."""
    return _big_subsample(int(n), int(cfg.get("STRAT_PIXEL_SAMPLE", STRAT_PIXEL_SAMPLE)))


def map_metrics(labels, scores, hw, partner, cfg):
    """Metrics of one set of score maps: plain pixel AUROC (src/evaluate.py
    subsample), shuffled-label AUROC (same subsample, partner images) and LS-AUROC
    per grid (STRAT_PIXEL_SAMPLE subsample).
    """
    H, W = int(cfg["IMG_H"]), int(cfg["IMG_W"])
    if H * W != hw:
        raise ValueError(f"pixels per image {hw} != IMG_H x IMG_W = {H} x {W}")
    a = LP.location_aucs(labels, scores, hw, partner)
    idx = strat_subsample(labels.size, cfg)
    pix = idx % hw
    y, s = labels[idx], scores[idx]
    a["ls"] = {str(g): stratified_auc(y, s, region_index(pix, H, W, g))[0] for g in grids(cfg)}
    return a


def baseline_metrics(labels, hw, groups, cfg):
    """Plain AUROC and LS-AUROC of the model-free anomaly-frequency map: the
    location information left inside the regions.

    Each image's map is the anomaly frequency of the other town, as in the location
    audit. For LS-AUROC, pixel pairs are also restricted to the same town: within a
    town every image gets the same map, so the value measures only within-region
    variation of the prior. Pairing across towns would compare maps estimated from
    each other, which anti-correlates map and labels at a position and biases the
    value below 0.5. Without town folders the test images are split into two
    alternating halves and used in the same way.
    """
    H, W = int(cfg["IMG_H"]), int(cfg["IMG_W"])
    n_img = labels.size // hw
    if not LP._usable_groups(groups, n_img):
        groups = ["a" if i % 2 == 0 else "b" for i in range(n_img)]
    idx_small = LP.pixel_subsample(labels.size)
    raw = float(roc_auc_score(labels[idx_small], LP.location_baseline_scores(labels, idx_small, hw, groups)))
    idx = strat_subsample(labels.size, cfg)
    sc = LP.location_baseline_scores(labels, idx, hw, groups)
    names = sorted(set(groups))
    town = np.array([names.index(x) for x in groups], dtype=np.int64)[idx // hw]
    ls = {}
    for g in grids(cfg):
        reg = town * (g * g) + region_index(idx % hw, H, W, g)
        ls[str(g)] = stratified_auc(labels[idx], sc, reg)[0]
    return {"raw": raw, "ls": ls}


# --- Analysis ---
def _channels(cfg):
    return ("awgn", "rayleigh") if cfg["USE_RAYLEIGH"] else ("awgn",)


def _variant_per_seed(reps_state, p2s, reps, snrs, chans, value):
    """Per-seed results in the phase2_per_seed.json layout with the AUROC entry
    replaced by value(entry); FPR95 and D_sem are taken from Phase II.
    """
    out = {"snr_range_db": snrs}
    for ch in chans:
        runs = []
        for r in reps:
            run = []
            for i, s in enumerate(snrs):
                pt = {"snr_db": s, "channel": ch}
                for det, *_ in DETS:
                    p2 = p2s[ch][int(r)][i]
                    pt[f"{det}_auc_roc_pixel"] = value(reps_state[r][ch][det][i])
                    pt[f"{det}_fpr95"] = p2.get(f"{det}_fpr95")
                    pt[f"{det}_dsem"] = p2.get(f"{det}_dsem")
                run.append(pt)
            runs.append(run)
        out[ch] = runs
    return out


def _variants(cfg):
    """AUROC variants analysed with Decision Retention: plain, LS-AUROC per grid and
    the image-specific gain (0.5 + matched - shuffled).
    """
    g0 = str(grids(cfg)[0])
    v = {"raw": ("plain pixel AUROC (= Phase II)", lambda e: e["matched"]),
         f"ls{g0}": (f"LS-AUROC, {g0}x{g0} regions (PRIMARY)", lambda e, g=g0: e["ls"][g])}
    for g in grids(cfg)[1:]:
        v[f"ls{g}"] = (f"LS-AUROC, {g}x{g} regions (sensitivity)", lambda e, g=str(g): e["ls"][g])
    v["gain"] = ("matched - shuffled AUROC (DR = gain / gain at 40 dB)",
                 lambda e: None if e["shuffled"] is None else 0.5 + e["matched"] - e["shuffled"])
    return v


def _ms(a):
    """Mean, SD and count of the finite values."""
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()) if a.size else None,
            "sd": float(a.std(ddof=1)) if a.size > 1 else None}


def stratified_analysis(reps_state, p2s, cfg, baseline=None):
    """Per-SNR means, Decision Retention per AUROC variant and SRI on LS-AUROC
    (no GPU). reps_state: per-repetition map_metrics; p2s: phase2_per_seed.json.
    """
    snrs = [float(s) for s in cfg["SNR_RANGE_DB"]]
    reps = sorted(reps_state, key=int)
    chans = [ch for ch in _channels(cfg) if all(ch in reps_state[r] for r in reps)]
    g0 = str(grids(cfg)[0])
    eps = float(cfg["SRI_EPSILON"])
    out = {"snr_db": snrs, "reps": reps, "channels": chans, "grids": list(grids(cfg)),
           "primary": f"ls{g0}", "baseline": baseline, "per_snr": {}, "dr": {}, "sri": {}}
    diffs = []
    for ch in chans:
        for det, *_ in DETS:
            for i in range(len(snrs)):
                for r in reps:
                    diffs.append(abs(reps_state[r][ch][det][i]["matched"] - p2s[ch][int(r)][i][f"{det}_auc_roc_pixel"]))
    out["max_abs_diff_from_phase2"] = float(max(diffs)) if diffs else None
    for ch in chans:
        rows = []
        for i, s in enumerate(snrs):
            row = {"snr_db": s}
            for det, *_ in DETS:
                es = [reps_state[r][ch][det][i] for r in reps]
                row[det] = {"raw": _ms([e["matched"] for e in es]),
                            "ls": {g: _ms([e["ls"][g] for e in es]) for g in es[0]["ls"]},
                            "shuffled": _ms([e["shuffled"] for e in es]),
                            "gain": _ms([None if e["shuffled"] is None else e["matched"] - e["shuffled"] for e in es])}
            d = np.array([reps_state[r][ch]["prototype"][i]["ls"][g0] - reps_state[r][ch]["max_logit"][i]["ls"][g0]
                          for r in reps], dtype=float)
            row["ls_prototype_minus_max_logit"] = {
                "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if d.size > 1 else None,
                "n_prototype_higher": int((d > 0).sum()), "n": int(d.size),
                "wilcoxon_p": float(stats.wilcoxon(d).pvalue) if d.size >= 2 and np.any(d != 0) else None}
            rows.append(row)
        out["per_snr"][ch] = rows
        sri = []
        for i, s in enumerate(snrs):
            row = {"snr_db": s}
            for det, *_ in DETS:
                ls = np.array([reps_state[r][ch][det][i]["ls"][g0] for r in reps], dtype=float)
                ds = np.array([p2s[ch][int(r)][i][f"{det}_dsem"] for r in reps], dtype=float)
                row[det] = {"sri_ls": _ms(ls / (ds + eps)), "excess_sri_ls": _ms((ls - 0.5) / (ds + eps))}
            sri.append(row)
        out["sri"][ch] = sri
    for key, (label, fn) in _variants(cfg).items():
        ps = _variant_per_seed(reps_state, p2s, reps, snrs, chans, fn)
        dr = decision_retention_analysis(ps, snrs, cfg, channels=tuple(chans))
        out["dr"][key] = {"label": label, "channels": {ch: dr["channels"][ch]["auc"] for ch in dr["channels"]},
                          "snr_sweep_db": dr["snr_db"]}
    return out


# --- Report ---
def _f(d, nd=4):
    if d is None or d.get("mean") is None:
        return "--"
    return f"{d['mean']:.{nd}f} +/- {d['sd']:.{nd}f}" if d.get("sd") is not None else f"{d['mean']:.{nd}f}"


def _snrx(lvl):
    s = lvl["snr_db"]
    m = "--" if s.get("mean") is None else f"{s['mean']:+.2f} +/- {(s.get('sd') or 0):.2f}"
    return (f"{m} (n={s.get('n', 0)}; L {lvl['n_left_censored']}, NR {lvl['n_not_reached']}, "
            f"undef {lvl['n_undefined']})")


def _report(an):
    g0 = an["primary"][2:]
    L = [f"  N = {len(an['reps'])} production repetitions; channels {an['channels']}; LS-AUROC grids "
         f"{an['grids']} (primary {g0}x{g0})",
         f"  Plain pixel AUROC reproduces Phase II: max |difference| {an['max_abs_diff_from_phase2']:.2e}"]
    if an.get("baseline"):
        b = an["baseline"]
        L.append(f"  Model-free location map (other town): plain AUROC {b['raw']:.4f}; LS-AUROC "
                 + ", ".join(f"{g}x{g} {v:.4f}" for g, v in b["ls"].items())
                 + "  (0.5 = no location information left inside the regions)")
    for ch in an["channels"]:
        L.append(f"\n  {ch.upper()} -- mean +/- SD over repetitions")
        L.append(f"    {'SNR':>4}  {'Max Logit plain':>16} {'Max Logit LS':>17}  {'Prototype plain':>16} "
                 f"{'Prototype LS':>17}  {'LS: PR - ML':>12} {'PR higher':>9}")
        for row in an["per_snr"][ch]:
            d = row["ls_prototype_minus_max_logit"]
            L.append(f"    {row['snr_db']:>+4.0f}  {row['max_logit']['raw']['mean']:>16.4f} {_f(row['max_logit']['ls'][g0]):>17}"
                     f"  {row['prototype']['raw']['mean']:>16.4f} {_f(row['prototype']['ls'][g0]):>17}"
                     f"  {d['mean']:>+12.4f} {d['n_prototype_higher']:>5}/{d['n']}")
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
                    L.append(f"    SNR{int(round(float(lv)*100))}: Max Logit {_snrx(ml)} | Prototype {_snrx(pr)}")
                    L.append(f"           Prototype - Max Logit {md} dB (n={p['n_both_crossed']}); Prototype lower in "
                             f"{p['prototype_lower_count']}/{p['n_order_determinable']}; Wilcoxon p={wp}")
    L.append(f"\n  SRI on LS-AUROC ({g0}x{g0}), AWGN: SRI_LS = LS/(D_sem+eps); excess SRI_LS = (LS-0.5)/(D_sem+eps)")
    for row in an["sri"].get("awgn", []):
        if row["snr_db"] in (-20.0, -10.0, 0.0, 10.0, 20.0, 40.0):
            L.append(f"    {row['snr_db']:>+4.0f}  Max Logit {_f(row['max_logit']['sri_ls'], 2)} / "
                     f"{_f(row['max_logit']['excess_sri_ls'], 2)}   Prototype {_f(row['prototype']['sri_ls'], 2)} / "
                     f"{_f(row['prototype']['excess_sri_ls'], 2)}")
    L.append("\n  LS-AUROC: anomalous vs normal pixels compared only within the same image region (ties 1/2);")
    L.append("  0.5 for any map that does not depend on the image. 'L' = DR already >= level at the lowest swept")
    L.append("  SNR, 'NR' = level not reached; censored seeds are counted, not averaged.")
    return "\n".join(L)


def _plots(an, out_root, cfg):
    import matplotlib.pyplot as plt
    pdir = Path(out_root) / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    g0 = an["primary"][2:]
    keep = [i for i, s in enumerate(an["snr_db"]) if s != float(cfg["CLS_FIT_SNR"])]
    xs = [an["snr_db"][i] for i in keep]
    n = len(an["reps"])
    tq = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    style = (("max_logit", "Max Logit", "#1f77b4", "o"), ("prototype", "Prototype", "#2ca02c", "D"))
    fig, axes = plt.subplots(1, len(an["channels"]), figsize=(6.5 * len(an["channels"]), 4.8), squeeze=False, sharey=True)
    for ax, ch in zip(axes[0], an["channels"]):
        rows = an["per_snr"][ch]
        for det, name, color, mk in style:
            raw = np.array([rows[i][det]["raw"]["mean"] for i in keep])
            m = np.array([rows[i][det]["ls"][g0]["mean"] for i in keep])
            sd = np.array([rows[i][det]["ls"][g0]["sd"] or 0.0 for i in keep])
            h = sd * tq / np.sqrt(n)
            ax.plot(xs, m, "-", marker=mk, markersize=4, color=color, lw=2, label=f"{name}, LS-AUROC ({g0}x{g0})")
            ax.fill_between(xs, m - h, m + h, color=color, alpha=.15, lw=0)
            ax.plot(xs, raw, "--", color=color, lw=1.2, alpha=.7, label=f"{name}, plain pixel AUROC")
        if an.get("baseline"):
            ax.axhline(an["baseline"]["raw"], color="grey", ls="-.", lw=1, label="Location map, plain AUROC")
        ax.axhline(0.5, color="black", ls=":", lw=.8, alpha=.5, label="Chance / any image-independent map (LS)")
        ax.set(xlabel="SNR (dB)", title=ch.upper())
        ax.legend(fontsize=7, loc="lower right")
    axes[0][0].set_ylabel("AUROC")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(pdir / f"14_stratified_auroc.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, ls_, alpha in ((an["primary"], "-", 1.0), ("raw", "--", .55)):
        b = an["dr"][key]["channels"]["awgn"]["ref"]["detectors"]
        x = an["dr"][key]["snr_sweep_db"]
        for det, name, color, mk in style:
            m = np.array([np.nan if v is None else v for v in b[det]["mean_curve"]], dtype=float)
            ax.plot(x, m, ls=ls_, marker=mk if ls_ == "-" else None, markersize=4, color=color, alpha=alpha,
                    label=f"{name}, DR on {'LS-AUROC' if key != 'raw' else 'plain AUROC'}")
            if ls_ == "-":
                h = np.array([np.nan if v is None else v for v in b[det]["ci95_half_width_curve"]], dtype=float)
                ax.fill_between(x, m - h, m + h, color=color, alpha=.12, lw=0)
    for lv in cfg["DR_HEADLINE_LEVELS"]:
        ax.axhline(lv, color="grey", ls=":", lw=.8, alpha=.6)
    ax.axhline(1.0, color="black", lw=.8, alpha=.35); ax.axhline(0.0, color="black", ls=":", lw=.8, alpha=.4)
    ax.set(xlabel="SNR (dB)", ylabel="Decision Retention (AWGN, 40 dB reference)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(pdir / f"14b_stratified_dr.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[Stratified] plots -> {pdir}/14_stratified_auroc.png, 14b_stratified_dr.png (+ .pdf)")


# --- Runs ---
def _check_inputs(p1_grid, cfg):
    """Load phase2_per_seed.json and check it matches the current SNR grid and
    the Phase I repetitions in memory.
    """
    out_root = Path(p1_grid["out"])
    per_seed_path = out_root / "phase2_per_seed.json"
    if not per_seed_path.exists():
        raise FileNotFoundError(f"{per_seed_path} not found -- run Phase II first.")
    p2s = json.load(open(per_seed_path))
    if [float(s) for s in p2s["snr_range_db"]] != [float(s) for s in cfg["SNR_RANGE_DB"]]:
        raise ValueError("phase2_per_seed.json SNR grid differs from SNR_RANGE_DB -- stale Phase II file.")
    runs = p1_grid["final_runs"]
    if len(p2s["awgn"]) < len(runs):
        raise ValueError("phase2_per_seed.json has fewer repetitions than Phase I -- stale Phase II file.")
    if any(r.get("jscc_a") is None or r.get("jscc_b") is None for r in runs):
        raise RuntimeError("The Phase I models are not in memory (released by an earlier cell). Restart the "
                           "session, re-run the setup cells and the Phase I cell, then this cell.")
    return out_root, p2s, runs


def _fp(cfg, runs, rule, extra):
    """Fingerprint of the settings that determine the stored results."""
    return fingerprint_key({"ml": training_fingerprint(cfg, "max_logit"), "pr": training_fingerprint(cfg, "prototype"),
                            "eval": evaluation_fingerprint(cfg), "grids": list(grids(cfg)),
                            "strat_sample": int(cfg.get("STRAT_PIXEL_SAMPLE", STRAT_PIXEL_SAMPLE)),
                            "img": [cfg["IMG_H"], cfg["IMG_W"]], "partner_rule": rule,
                            "taus": [[r["tau_max_logit"], r["tau_prototype"]] for r in runs], **extra})


def run_stratified_sweep(p1_grid, cfg, device):
    """Every production repetition, both systems, every SNR, AWGN and Rayleigh.
    Resumable per (repetition, channel). Needs the Phase I models in memory.
    """
    out_root, p2s, runs = _check_inputs(p1_grid, cfg)
    res_dir = out_root / "stratified"; res_dir.mkdir(parents=True, exist_ok=True)
    state_path = res_dir / "stratified_state.json"
    loader = runs[0]["loaders"]["test"]
    groups = LP.test_groups(loader)
    n_img = len(loader.dataset)
    partner, rule = LP.partner_images(groups, n_img)
    fp = _fp(cfg, runs, rule, {"version": 1})
    saved = json.load(open(state_path)) if state_path.exists() else {}
    if saved.get("_fp") not in (None, fp):
        print("  stored results were produced under different settings -- recomputing")
        saved = {}
    state = {"_fp": fp, "reps": saved.get("reps", {}), "baseline": saved.get("baseline")}
    snrs = list(cfg["SNR_RANGE_DB"])
    chans = _channels(cfg)
    print(f"\n{'='*62}\n  LOCATION-STRATIFIED EVALUATION -- {len(runs)} repetitions x {len(snrs)} SNRs x "
          f"{list(chans)}, grids {list(grids(cfg))}\n  partner rule: {rule}\n{'='*62}")
    for r_i, rep in enumerate(runs):
        rk = str(r_i)
        state["reps"].setdefault(rk, {})
        for ch in chans:
            if ch in state["reps"][rk]:
                print(f"  repetition {r_i+1}, {ch}: SKIP, already done (resumed)")
                continue
            rec = {det: [] for det, *_ in DETS}
            worst = 0.0
            for i, snr in enumerate(snrs):
                for det, name, jk, dk in DETS:
                    labels, scores, hw = LP.score_maps(rep[jk], rep[dk], loader, snr, device, ch,
                                                       noise_seed=eval_noise_seed(cfg, snr, ch))
                    if labels.size != n_img * hw:
                        raise RuntimeError(f"pixel count {labels.size} != {n_img} images x {hw} pixels")
                    if state["baseline"] is None:
                        state["baseline"] = baseline_metrics(labels, hw, groups, cfg)
                    m = map_metrics(labels, scores, hw, partner, cfg)
                    m["snr_db"] = float(snr)
                    worst = max(worst, abs(m["matched"] - p2s[ch][r_i][i][f"{det}_auc_roc_pixel"]))
                    rec[det].append(m)
            state["reps"][rk][ch] = rec
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            g0 = str(grids(cfg)[0])
            i0 = snrs.index(0) if 0 in snrs else 0
            print(f"  repetition {r_i+1}, {ch}: done; at {snrs[i0]:+d} dB LS-AUROC Max Logit "
                  f"{rec['max_logit'][i0]['ls'][g0]:.4f}, Prototype {rec['prototype'][i0]['ls'][g0]:.4f} "
                  f"(plain {rec['max_logit'][i0]['matched']:.4f} / {rec['prototype'][i0]['matched']:.4f}); "
                  f"max |diff| from Phase II {worst:.1e}")
    reps_state = {k: v for k, v in state["reps"].items() if all(ch in v for ch in chans)}
    an = stratified_analysis(reps_state, p2s, cfg, state["baseline"])
    with open(res_dir / "stratified_summary.json", "w") as f:
        json.dump(an, f, indent=2)
    tdir = out_root / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    txt = _report(an)
    with open(tdir / "stratified_eval.md", "w") as f:
        f.write(f"# Location-stratified evaluation (production systems)\n\n```\n{txt}\n```\n")
    print(txt)
    if an["max_abs_diff_from_phase2"] is not None and an["max_abs_diff_from_phase2"] > 1e-6:
        print(f"  WARNING: plain AUROC differs from Phase II by up to {an['max_abs_diff_from_phase2']:.2e} -- "
              f"the Phase II evaluation is not reproduced exactly.")
    _plots(an, out_root, cfg)
    print(f"[Stratified] summary -> {res_dir / 'stratified_summary.json'}, table -> {tdir / 'stratified_eval.md'}")
    return an


# --- Ideal digital link on LS-AUROC ---
def digital_stratified_analysis(dstate, sweep_an, ref_ls, cfg, latent_dims, n_symbols, baseline):
    """Ideal digital link on LS-AUROC against the analogue systems (no GPU).

    dstate[rep][det][str(bits) or "none"]: map_metrics of the digital link;
    sweep_an: stratified_analysis output (analogue SNR50 on LS-AUROC per seed);
    ref_ls[det][rep]: analogue LS-AUROC at CLS_FIT_SNR (AWGN).
    """
    from phases.digital_baseline import digital_threshold_db
    g0 = str(grids(cfg)[0])
    key = f"ls{g0}"
    reps = sorted(dstate, key=int)
    # Best image-independent output: a map constant inside each region scores 0.5;
    # the model-free location map can score slightly more (within-region residual).
    fallback = max(0.5, float(baseline["ls"][g0]))
    out = {"bits": list(cfg["DIGITAL_BITS"]), "n_symbols": n_symbols, "grid": int(g0), "reps": reps,
           "fallback_ls_auroc": fallback, "detectors": {}}
    for det, name, *_ in DETS:
        ref = np.array([ref_ls[det][r] for r in reps], dtype=float)
        lvl = sweep_an["dr"][key]["channels"]["awgn"]["ref"]["detectors"][det]["levels"]["0.5"]
        a50_all = dict(zip(sweep_an["reps"], lvl["per_seed_snr_db"]))
        a50 = np.array([np.nan if a50_all.get(r) is None else a50_all[r] for r in reps], dtype=float)
        un = np.array([dstate[r][det]["none"]["ls"][g0] for r in reps], dtype=float)
        d = {"analogue_ls_40db": _ms(ref), "analogue_snr50_ls": _ms(a50), "unquantised_ls": _ms(un),
             "fallback_dr": _ms((fallback - 0.5) / (ref - 0.5)), "digital": {}}
        for b in cfg["DIGITAL_BITS"]:
            ls = np.array([dstate[r][det][str(b)]["ls"][g0] for r in reps], dtype=float)
            raw = np.array([dstate[r][det][str(b)]["matched"] for r in reps], dtype=float)
            thr = digital_threshold_db(b, latent_dims[det], n_symbols)
            dr = (ls - 0.5) / (ref - 0.5)
            dig50 = np.where(dr >= 0.5, thr, np.nan)
            diff = a50 - dig50
            d["digital"][str(b)] = {"threshold_snr_db": thr, "ls_auroc": _ms(ls), "plain_auroc": _ms(raw),
                                    "dr_above_threshold": _ms(dr), "reps_reaching_50pct": int((dr >= 0.5).sum()),
                                    "analogue_snr50_minus_threshold_db": _ms(diff)}
        out["detectors"][det] = d
    return out


def _digital_report(an):
    g0 = an["grid"]
    L = [f"  Ideal digital link over {an['n_symbols']} complex channel uses (AWGN), N={len(an['reps'])} repetitions,"
         f" scored with LS-AUROC ({g0}x{g0} regions)",
         f"  Below the threshold the receiver has no latent; it outputs an image-independent map, scored at"
         f" max(0.5, LS-AUROC of the model-free location map) = {an['fallback_ls_auroc']:.4f}"]
    for det, name, *_ in DETS:
        d = an["detectors"][det]
        L.append(f"\n  {name}: analogue LS-AUROC at 40 dB {_f(d['analogue_ls_40db'])}; unquantised noise-free link "
                 f"{_f(d['unquantised_ls'])}; analogue SNR50 (LS) {_f(d['analogue_snr50_ls'], 2)} dB; DR of the "
                 f"fallback map {_f(d['fallback_dr'], 3)}")
        L.append(f"    {'bits':>4} {'threshold':>10} {'LS above':>17} {'DR above':>15} {'reach 50%':>9} "
                 f"{'analogue SNR50 - threshold':>27}")
        for b, v in d["digital"].items():
            df = v["analogue_snr50_minus_threshold_db"]
            L.append(f"    {b:>4} {v['threshold_snr_db']:>+8.2f}dB {_f(v['ls_auroc']):>17} {_f(v['dr_above_threshold'], 3):>15} "
                     f"{v['reps_reaching_50pct']:>5}/{len(an['reps'])} "
                     f"{('--' if df['mean'] is None else format(df['mean'], '+.2f') + ' dB'):>27}")
    return "\n".join(L)


def digital_ls_step(x_min, x_max, threshold_db, value, fallback):
    """Digital LS-AUROC curve with the cliff exactly at the threshold: the fallback
    map's value below it, the quantised link's value from it upwards."""
    if threshold_db <= x_min:
        return [x_min, x_max], [value, value]
    if threshold_db > x_max:
        return [x_min, x_max], [fallback, fallback]
    return [x_min, threshold_db, threshold_db, x_max], [fallback, fallback, value, value]


def _digital_plot(an, sweep_an, out_root, cfg):
    import matplotlib.pyplot as plt
    pdir = Path(out_root) / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    g0 = str(an["grid"])
    keep = [i for i, s in enumerate(sweep_an["snr_db"]) if s != float(cfg["CLS_FIT_SNR"])]
    x = [sweep_an["snr_db"][i] for i in keep]
    greys = ["#4d4d4d", "#7f7f7f", "#a6a6a6", "#c8c8c8"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (det, name, color) in zip(axes, (("max_logit", "Max Logit", "#1f77b4"), ("prototype", "Prototype", "#2ca02c"))):
        rows = sweep_an["per_snr"]["awgn"]
        ax.plot(x, [rows[i][det]["ls"][g0]["mean"] for i in keep], "-o", color=color, lw=2, markersize=4,
                label="Deep JSCC (analogue), LS-AUROC, mean")
        for k, (b, v) in enumerate(an["detectors"][det]["digital"].items()):
            sx, sy = digital_ls_step(x[0], x[-1], v["threshold_snr_db"], v["ls_auroc"]["mean"],
                                     an["fallback_ls_auroc"])
            ax.plot(sx, sy, color=greys[k % 4], lw=1.6, ls=["--", "-.", ":", (0, (5, 1))][k % 4],
                    label=f"Ideal digital, {b} bit(s)/value (threshold {v['threshold_snr_db']:+.1f} dB)")
        ax.axhline(0.5, color="black", ls=":", lw=.8, alpha=.4, label="Chance / image-independent map")
        ax.set(xlabel="SNR (dB)", title=name)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel(f"LS-AUROC ({g0}x{g0} regions)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(pdir / f"14c_digital_stratified.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[Digital, stratified] plot -> {pdir}/14c_digital_stratified.png/.pdf")


def run_digital_stratified(p1_grid, cfg, device):
    """The ideal digital link scored with LS-AUROC. Needs the Phase I models and a
    finished run_stratified_sweep. Resumable per repetition.
    """
    from phases.digital_baseline import LatentQuantiser, DigitalLink, calibrate_quantiser_range
    out_root, p2s, runs = _check_inputs(p1_grid, cfg)
    res_dir = out_root / "stratified"
    sweep_path = res_dir / "stratified_state.json"
    if not sweep_path.exists():
        raise FileNotFoundError(f"{sweep_path} not found -- run run_stratified_sweep first.")
    sweep = json.load(open(sweep_path))
    loader = runs[0]["loaders"]["test"]
    groups = LP.test_groups(loader)
    partner, rule = LP.partner_images(groups, len(loader.dataset))
    if sweep.get("_fp") != _fp(cfg, runs, rule, {"version": 1}):
        raise RuntimeError("stratified_state.json was produced under different settings -- re-run run_stratified_sweep.")
    reps_state = {k: v for k, v in sweep["reps"].items() if "awgn" in v}
    if len(reps_state) < len(runs):
        raise RuntimeError(f"run_stratified_sweep has finished {len(reps_state)}/{len(runs)} repetitions -- finish it first.")
    n_symbols = cfg["CHANNEL_SYMBOLS"][cfg["DEFAULT_CR"]]
    state_path = res_dir / "digital_stratified_state.json"
    fp = _fp(cfg, runs, rule, {"bits": list(cfg["DIGITAL_BITS"]), "pct": list(cfg["DIGITAL_QUANT_PERCENTILES"]),
                               "nsym": n_symbols, "version": 1})
    saved = json.load(open(state_path)) if state_path.exists() else {}
    dstate = saved.get("reps", {}) if saved.get("_fp") == fp else {}
    print(f"\n{'='*62}\n  IDEAL DIGITAL BASELINE, LOCATION-STRATIFIED -- bits {cfg['DIGITAL_BITS']}, "
          f"{len(runs)} repetitions\n{'='*62}")
    latent_dims = {}
    for r_i, rep in enumerate(runs):
        for det, name, jk, dk in DETS:
            latent_dims[det] = rep[jk].patch_dim * rep[jk].patch_grid ** 2
        if str(r_i) in dstate:
            print(f"  repetition {r_i+1}: SKIP, already done (resumed)")
            continue
        rec = {}
        for det, name, jk, dk in DETS:
            jscc = rep[jk]
            lo, hi = calibrate_quantiser_range(jscc, rep["loaders"]["val_mixed"], device, cfg["DIGITAL_QUANT_PERCENTILES"])
            rec[det] = {}
            for b in [None] + list(cfg["DIGITAL_BITS"]):
                q = None if b is None else LatentQuantiser(lo, hi, b, jscc.patch_dim, jscc.patch_grid)
                labels, scores, hw = LP.score_maps(DigitalLink(jscc, q), rep[dk], loader, None, device, "awgn",
                                                   noise_seed=None)
                rec[det]["none" if b is None else str(b)] = map_metrics(labels, scores, hw, partner, cfg)
        dstate[str(r_i)] = rec
        with open(state_path, "w") as f:
            json.dump({"_fp": fp, "reps": dstate}, f, indent=2)
        g0 = str(grids(cfg)[0])
        print(f"  repetition {r_i+1}: LS-AUROC Max Logit " + ", ".join(
            f"{k}={v['ls'][g0]:.3f}" for k, v in rec["max_logit"].items()) + " | Prototype " + ", ".join(
            f"{k}={v['ls'][g0]:.3f}" for k, v in rec["prototype"].items()))
    sweep_an = stratified_analysis(reps_state, p2s, cfg, sweep["baseline"])
    g0 = str(grids(cfg)[0])
    i40 = [float(s) for s in cfg["SNR_RANGE_DB"]].index(float(cfg["CLS_FIT_SNR"]))
    ref_ls = {det: {r: reps_state[r]["awgn"][det][i40]["ls"][g0] for r in reps_state} for det, *_ in DETS}
    an = digital_stratified_analysis(dstate, sweep_an, ref_ls, cfg, latent_dims, n_symbols, sweep["baseline"])
    with open(res_dir / "digital_stratified_summary.json", "w") as f:
        json.dump(an, f, indent=2)
    tdir = out_root / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    txt = _digital_report(an)
    with open(tdir / "digital_stratified.md", "w") as f:
        f.write(f"# Ideal digital baseline, location-stratified (AWGN)\n\n```\n{txt}\n```\n")
    print(txt)
    _digital_plot(an, sweep_an, out_root, cfg)
    return an
