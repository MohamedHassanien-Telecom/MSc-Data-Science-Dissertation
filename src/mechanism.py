"""Scoring-rule check (dissertation Appendix F.5): does sensitivity to the received
latent's magnitude explain the Max Logit - Prototype retention gap?

Decision Retention of three scorings, evaluated by the notebook's scoring-rule cell
on every production repetition at its SNRs:

  ml      Max Logit (its score grows with the received cell's magnitude)
  pr      Prototype (cosine to the class prototypes; magnitude discarded)
  pr_mag  the same trained Prototype weights scored by
          models.PrototypeDetectorMagnitudeAware (no retraining)

If magnitude sensitivity made Max Logit lose its decisions at a higher SNR, then
pr -> pr_mag should move Prototype's SNR_50 up towards Max Logit's. pr_mag - pr is
paired by repetition (same weights, same channel noise), so it isolates the
scoring rule.

References: "ref" is each scoring's AUROC at the highest SNR of the cell's grid
(40 dB is not evaluated there); "peak" is each repetition's best AUROC on that
grid. Plain Prototype scoring peaks mid-range, so part of pr_mag - pr under "ref"
is a reference effect, which "peak" removes; the scoring rule's contribution is
read from "peak". The grid is coarser than Phase II's, so these SNR_X values are
compared with each other only.
"""
import numpy as np
from scipy import stats

from src.decision_retention import dr_values, retention_snr

VARIANTS = (("ml", "Max Logit"), ("pr", "Prototype"), ("pr_mag", "Prototype-mag"))
PAIRS = (("pr_mag", "pr"), ("pr_mag", "ml"), ("pr", "ml"))
REFERENCES = (("ref", "each scoring's own AUROC at the highest explainer SNR"),
              ("peak", "each repetition's best AUROC over the explainer SNRs"))


def _retention_block(auc_by_variant, snr_sorted, levels, reference):
    """Per-scoring SNR_X summaries and the paired differences of PAIRS, for one reference."""
    n = next(iter(auc_by_variant.values())).shape[0]
    block = {"variants": {}, "pairs": {}}
    per = {}
    for v, _ in VARIANTS:
        auc = auc_by_variant[v]
        ref = auc[:, -1] if reference == "ref" else np.nanmax(auc, axis=1)
        per[v] = {}
        rec = {"auc_mean": auc.mean(0).tolist(), "auc_ref_mean": float(ref.mean()), "levels": {}}
        for lv in levels:
            xs = [retention_snr(snr_sorted, dr_values(auc[r], ref[r], "auc"), lv) for r in range(n)]
            per[v][lv] = xs
            vals = [x for x, s in xs if s == "crossed"]
            rec["levels"][str(lv)] = {
                "per_rep_snr_db": [x for x, _ in xs], "per_rep_status": [s for _, s in xs],
                "mean": float(np.mean(vals)) if vals else None,
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "n": len(vals)}
        block["variants"][v] = rec
    for a, b in PAIRS:
        block["pairs"][f"{a}_minus_{b}"] = {}
        for lv in levels:
            d = np.array([xa - xb for (xa, sa), (xb, sb) in zip(per[a][lv], per[b][lv])
                          if sa == "crossed" and sb == "crossed"], dtype=float)
            p = float(stats.wilcoxon(d).pvalue) if d.size >= 2 and np.any(d != 0) else None
            block["pairs"][f"{a}_minus_{b}"][str(lv)] = {
                "n": int(d.size), "mean_diff_db": float(d.mean()) if d.size else None,
                "sd_diff_db": float(d.std(ddof=1)) if d.size > 1 else None,
                "n_first_higher": int((d > 0).sum()), "wilcoxon_p": p}
    return block


def mechanism_retention(mag_ablation, snrs, levels):
    """Decision Retention of the three scorings.

    mag_ablation[snr][scoring] is a list over repetitions of pixel AUROC. Returns
    the "ref" results at the top level ("variants", "pairs"), the "peak" results
    under "peak", and under "share_of_gap" the share of the Prototype - Max Logit
    gap accounted for by the scoring rule, for each reference."""
    snr_sorted = sorted(float(s) for s in snrs)
    key_of = {float(s): s for s in snrs}
    ref_snr = snr_sorted[-1]
    n = len(mag_ablation[key_of[ref_snr]]["pr"])
    auc_by_variant = {v: np.array([[mag_ablation[key_of[s]][v][r] for s in snr_sorted] for r in range(n)],
                                  dtype=float) for v, _ in VARIANTS}
    out = {"snr_db": snr_sorted, "reference_snr_db": ref_snr, "levels": list(levels), "n_reps": n,
           "references": {k: d for k, d in REFERENCES}}
    out.update(_retention_block(auc_by_variant, snr_sorted, levels, "ref"))
    out["peak"] = _retention_block(auc_by_variant, snr_sorted, levels, "peak")
    out["share_of_gap"] = {}
    for refk, blk in (("ref", out), ("peak", out["peak"])):
        out["share_of_gap"][refk] = {}
        for lv in levels:
            num = blk["pairs"]["pr_mag_minus_pr"][str(lv)]["mean_diff_db"]
            gap = blk["pairs"]["pr_minus_ml"][str(lv)]["mean_diff_db"]
            out["share_of_gap"][refk][str(lv)] = (float(num / -gap) if num is not None and gap not in (None, 0)
                                                 else None)
    return out


def _block_lines(blk, levels, indent="    "):
    L = []
    for v, name in VARIANTS:
        rec = blk["variants"][v]
        cells = []
        for lv in levels:
            r = rec["levels"][str(lv)]
            cells.append(f"SNR{int(round(lv*100))} = "
                         + ("--" if r["mean"] is None else f"{r['mean']:+.2f}") + " +/- "
                         + ("--" if r["sd"] is None else f"{r['sd']:.2f}") + f" dB (n={r['n']})")
        L.append(f"{indent}{name:>14}: AUROC at reference {rec['auc_ref_mean']:.4f};  " + ";  ".join(cells))
    L.append(f"{indent}Paired differences (first minus second; positive = first keeps its decisions only at a higher SNR):")
    for key, by_lv in blk["pairs"].items():
        a, b = key.split("_minus_")
        an = dict(VARIANTS)[a]; bn = dict(VARIANTS)[b]
        for lv, p in by_lv.items():
            L.append(f"{indent}  {an} - {bn}, SNR{int(round(float(lv)*100))}: "
                     + ("--" if p["mean_diff_db"] is None else f"{p['mean_diff_db']:+.2f}")
                     + " +/- " + ("--" if p["sd_diff_db"] is None else f"{p['sd_diff_db']:.2f}")
                     + f" dB, higher in {p['n_first_higher']}/{p['n']}, Wilcoxon p="
                     + ("--" if p["wilcoxon_p"] is None else format(p["wilcoxon_p"], ".4g")))
    return L


def mechanism_report(res):
    """Printable summary of mechanism_retention's result."""
    lv = res["levels"]
    L = [f"  Decision Retention on the explainer grid (N={res['n_reps']} repetitions)",
         f"  Reference 'ref' = {res['references']['ref']} ({res['reference_snr_db']:+.0f} dB):"]
    L += _block_lines(res, lv)
    if "peak" in res:
        L.append(f"  Reference 'peak' = {res['references']['peak']}:")
        L += _block_lines(res["peak"], lv)
    if "share_of_gap" in res:
        L.append("  Share of the Prototype - Max Logit gap accounted for by the scoring rule "
                 "((Prototype-mag - Prototype) / (Max Logit - Prototype)):")
        for refk, by_lv in res["share_of_gap"].items():
            L.append(f"    {refk:>4}: " + ";  ".join(
                f"SNR{int(round(float(k)*100))} " + ("--" if v is None else f"{100*v:.0f}%")
                for k, v in by_lv.items()))
    L.append("  Reading: the sign of Prototype-mag - Prototype says whether adding magnitude makes Prototype")
    L.append("  lose its decisions at a higher SNR; its size relative to the Prototype - Max Logit gap, under")
    L.append("  the 'peak' reference, says how much of the gap the scoring rule accounts for. A small share")
    L.append("  means the gap comes from the trained system (encoder, latent layout, training objective),")
    L.append("  which this check does not separate.")
    return "\n".join(L)
