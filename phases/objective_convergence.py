"""Stopping rule of the Phase I hyperparameter search (dissertation Appendix B.5).

Within every step the candidate with the highest mean AUROC over the paired seeds
is selected. At the end of each round r >= 1, each detector's complete round-end
configuration is compared with the previous round's on the same seeds: the
per-seed AUROCs of the Step F (LAMBDA_REC) winner measure it, because Step F runs
last and trains its candidates under every other final setting of the round. The
search stops when neither detector improved significantly (paired Wilcoxon test
with the small-N rank-biserial guard, phases.phase1_grid._tie_test). Each detector
then keeps whichever of the two configurations has the higher mean AUROC.

The Step F winner's mean is the maximum of several candidates and is therefore
slightly optimistic; both configurations are selected the same way, and the final
configuration is re-measured on the production seeds in Phase II.
"""
import json
from pathlib import Path

DET_SUFFIX = {"max_logit": "ml", "prototype": "pr"}
MIN_COMMON_SEEDS = 3


def round_end_scores(state_path, winners):
    """Per-seed AUROCs of a round's complete configuration.

    state_path: the round's LAMBDA_REC state file (Step F).
    winners:    {"max_logit": value, "prototype": value}, the LAMBDA_REC selected
                for each detector in that round.
    Returns {detector: {seed (int): AUROC}}.
    """
    with open(state_path) as f:
        state = json.load(f)
    out = {}
    for det, val in winners.items():
        entry = state.get(str(val))
        if not entry:
            raise KeyError(f"{Path(state_path).name}: no results for LAMBDA_REC={val} ({det})")
        out[det] = {int(s): float(v[det]) for s, v in entry.items() if det in v}
    return out


def objective_decision(prev_scores, this_scores, tie_fn, alpha, seed_order):
    """Compare this round's configuration with the previous round's, per detector,
    on their common seeds.

    tie_fn(values_this, values_prev, alpha) -> (tied, p_value, effect_r).
    Returns (stop, per_detector); stop is True only if no detector improved
    significantly and every detector had at least MIN_COMMON_SEEDS common seeds.
    """
    per_det, any_improved, any_untestable = {}, False, False
    order = {s: i for i, s in enumerate(seed_order)}
    for det in this_scores:
        common = sorted(set(this_scores[det]) & set(prev_scores[det]), key=lambda s: order.get(s, 10**9))
        if len(common) < MIN_COMMON_SEEDS:
            per_det[det] = {"n": len(common), "testable": False, "improved": False, "keep": "this"}
            any_untestable = True
            continue
        a = [this_scores[det][s] for s in common]
        b = [prev_scores[det][s] for s in common]
        tied, p, r = tie_fn(a, b, alpha)
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        improved = (not tied) and ma > mb
        per_det[det] = {"n": len(common), "testable": True, "seeds": common,
                        "mean_this": ma, "mean_prev": mb, "diff": ma - mb,
                        "p_value": p, "rank_biserial_r": r, "tied": bool(tied),
                        "improved": bool(improved), "keep": "this" if ma >= mb else "prev"}
        any_improved = any_improved or improved
    return (not any_improved and not any_untestable), per_det


def revert_detector_config(cfg, analysis, this_snap, prev_snap, det):
    """Restore one detector's complete configuration (cfg values and its lambda_sem
    in `analysis`) to the previous round's. Returns False, changing nothing, if the
    patch configuration differs (that would need new loaders and batch size)."""
    sfx = DET_SUFFIX[det]
    pk = (f"patch_grid_{sfx}", f"patch_dim_{sfx}")
    if any(this_snap[k] != prev_snap[k] for k in pk):
        return False
    for k in prev_snap:
        if k.endswith(f"_{sfx}"):
            cfg[k.upper()] = prev_snap[k]
    if det == "prototype":
        cfg["PROTOTYPE_SCALE"] = prev_snap["prototype_scale"]
    analysis["winning_lambda_sem"][det] = prev_snap[f"lambda_sem_{det}"]
    return True
