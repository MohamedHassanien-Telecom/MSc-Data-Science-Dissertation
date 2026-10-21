"""Phase IV: figures, tables and the collapse report from the Phase II and Phase III results.

phase4() writes the figures to OUT/plots (PNG and PDF), the tables to OUT/tables
and collapse_report.json to OUT (OUT = run_0/). The figure title is not drawn on
the image; it is appended to the file name.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import csv
from src.collapse import detect_collapse, bootstrap_elbow_stability, structural_break_test
from src.decision_retention import decision_retention_analysis, phase3_retention_by_cr


def _elbow_marker(coll, stage34_det):
    """How Plots 1, 3 and 3b mark the Stage-1 elbow of one system.

    "supported": a headline elbow confirmed by Stage 2 and not rejected by Stage 4
                 (smooth_vs_break verdict other than "smooth");
    "candidate": a headline elbow that fails either condition;
    "tie":       tied candidates, no headline elbow;
    None:        nothing to mark.
    Plots 1 and 3 draw only "supported" elbows; Plot 3b also shows candidates and
    ties, labelled as not confirmed."""
    brk = (stage34_det or {}).get("structural_break") or {}
    verdict = (brk.get("smooth_vs_break") or {}).get("verdict")
    if coll.get("elbow_snr") is not None:
        return "supported" if coll.get("elbow_confirmed") and verdict != "smooth" else "candidate"
    if coll.get("elbow_tie_detected") and [v for v in (coll.get("elbow_tied_values") or []) if v is not None]:
        return "tie"
    return None


def _run_collapse_stages_3_4(out_root, snrs, ml_auc, pr_auc, cfg, keep_idx=None, all_snrs=None):
    """Collapse Stages 3 (bootstrap stability) and 4 (smooth vs break) for both systems.

    Uses each production seed's trajectory from phase2_per_seed.json, restricted to
    the swept SNRs (keep_idx; the 40 dB reference point is excluded), together
    with the mean trajectories already used for Stages 1-2. Raises if the file's
    SNR grid or per-seed means do not match the Phase II aggregate; returns None
    results with a warning if the file is missing.
    """
    per_seed_path = Path(out_root) / "phase2_per_seed.json"
    if not per_seed_path.exists():
        print(f"[Phase IV] WARNING: {per_seed_path} not found -- Stage 3 (bootstrap "
              f"stability) and Stage 4 (structural-break test) skipped. Re-run Phase II "
              f"(phase2_aggregated) to produce this file if these results are needed.")
        return {"max_logit": None, "prototype": None}

    with open(per_seed_path) as f:
        per_seed = json.load(f)

    if all_snrs is None:
        all_snrs = list(snrs)
    if keep_idx is None:
        keep_idx = list(range(len(all_snrs)))
    file_snrs = [float(x) for x in (per_seed.get("snr_range_db")
                                    or [pt["snr_db"] for pt in per_seed["awgn"][0]])]
    if file_snrs != [float(s) for s in all_snrs]:
        raise ValueError(f"{per_seed_path}: SNR grid {file_snrs} != Phase II aggregate grid "
                         f"{[float(s) for s in all_snrs]} -- stale file; re-run phase2_aggregated().")

    results = {}
    for det_name, auc_key, dsem_key, auc_mean in [
        ("max_logit", "max_logit_auc_roc_pixel", "max_logit_dsem", ml_auc),
        ("prototype", "prototype_auc_roc_pixel", "prototype_dsem", pr_auc),
    ]:
        per_seed_auc = [[seed_run[i][auc_key] for i in keep_idx] for seed_run in per_seed["awgn"]]
        per_seed_dsem = [[seed_run[i][dsem_key] for i in keep_idx] for seed_run in per_seed["awgn"]]
        seed_mean = np.nanmean(np.asarray(per_seed_auc, dtype=float), axis=0)
        if not np.allclose(seed_mean, np.asarray(auc_mean, dtype=float), atol=1e-9, equal_nan=True):
            raise ValueError(f"{per_seed_path}: per-seed mean AUC of {det_name} differs from the "
                             f"Phase II aggregate used for Stages 1-2 -- the two files are from "
                             f"different runs; re-run phase2_aggregated().")
        print(f"\n[Phase IV] Stage 3/4 -- {det_name} ({len(per_seed_auc)} production seeds, "
              f"{len(keep_idx)} SNR points)")
        boot = bootstrap_elbow_stability(per_seed_auc, per_seed_dsem, snrs, cfg)
        brk = structural_break_test(np.asarray(snrs, dtype=float), np.asarray(auc_mean, dtype=float), cfg,
                                    per_seed_auc=per_seed_auc)
        results[det_name] = {"bootstrap": boot, "structural_break": brk}
    return results


def _export_snr_tables(awgn, out_root, cfg):
    """SNR-sweep results (AWGN, mean +/- std) as tables/phase2_snr_sweep.csv and .md."""
    tables_dir = Path(out_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = ["auc_roc_pixel", "fpr95", "dsem", "sri_auc", "sri_op"]
    metric_labels = {"auc_roc_pixel": "Pixel AUC-ROC", "fpr95": "FPR95",
                      "dsem": "D_sem", "sri_auc": "SRI_AUC", "sri_op": "SRI_op"}

    def _fmt(res, k):
        mean = res.get(f"{k}_mean", res.get(k, float("nan")))
        std = res.get(f"{k}_std")
        if std is None:
            return f"{mean:.4f}"
        return f"{mean:.4f} +/- {std:.4f}"

    # CSV: one row per (SNR, detector) with mean and std columns.
    csv_path = tables_dir / "phase2_snr_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["snr_db", "detector"]
        for k in metric_keys:
            header.extend([f"{k}_mean", f"{k}_std"])
        writer.writerow(header)
        for r in awgn:
            for detector_prefix, detector_name in [("max_logit", "Max Logit"), ("prototype", "Prototype")]:
                row = [r["snr_db"], detector_name]
                for k in metric_keys:
                    full_key = f"{detector_prefix}_{k}"
                    row.append(r.get(f"{full_key}_mean", r.get(full_key, "")))
                    row.append(r.get(f"{full_key}_std", ""))
                writer.writerow(row)
    print(f"[Phase IV] CSV table -> {csv_path}")

    # Markdown: one table per metric, both detectors.
    md_path = tables_dir / "phase2_snr_sweep.md"
    with open(md_path, "w") as f:
        f.write(f"# Phase II SNR Sweep -- mean +/- std across N_RUNS={cfg['N_RUNS']} repeated runs\n\n")
        for k in metric_keys:
            f.write(f"## {metric_labels[k]}\n\n")
            f.write("| SNR (dB) | Max Logit | Prototype |\n")
            f.write("|---|---|---|\n")
            for r in awgn:
                ml_val = _fmt(r, f"max_logit_{k}")
                pr_val = _fmt(r, f"prototype_{k}")
                f.write(f"| {r['snr_db']} | {ml_val} | {pr_val} |\n")
            f.write("\n")
    print(f"[Phase IV] Markdown table -> {md_path}")
    return {"csv": str(csv_path), "markdown": str(md_path)}


def _load_per_seed(out_root, all_snrs):
    """phase2_per_seed.json, checked against the aggregate SNR grid; None if absent."""
    per_seed_path = Path(out_root) / "phase2_per_seed.json"
    if not per_seed_path.exists():
        return None
    with open(per_seed_path) as f:
        per_seed = json.load(f)
    file_snrs = [float(x) for x in (per_seed.get("snr_range_db")
                                    or [pt["snr_db"] for pt in per_seed["awgn"][0]])]
    if file_snrs != [float(s) for s in all_snrs]:
        raise ValueError(f"{per_seed_path}: SNR grid {file_snrs} != Phase II aggregate grid "
                         f"{[float(s) for s in all_snrs]} -- stale file; re-run phase2_aggregated().")
    return per_seed


_DET_STYLE = (("max_logit", "Max Logit", "#1f77b4", "o"), ("prototype", "Prototype", "#2ca02c", "D"))


def _fmt_num(v, nd=2):
    return "--" if v is None else f"{v:.{nd}f}"


def _export_dr_tables(dr, dr_cr, out_root):
    """Decision Retention tables: decision_retention.csv (every number) and .md."""
    tables_dir = Path(out_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "decision_retention.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "metric", "reference", "row", "level",
                    "n_crossed", "n_left_censored", "n_not_reached", "n_undefined",
                    "snr_mean_db", "snr_sd_db", "snr_ci95_half_width_db",
                    "dsem_at_snr_mean", "dsem_at_snr_sd",
                    "paired_n_both_crossed", "paired_mean_diff_db", "paired_sd_diff_db",
                    "paired_wilcoxon_p", "unpaired_mannwhitney_p", "prototype_lower_count",
                    "n_order_determinable"])
        for ch, by_m in dr["channels"].items():
            for metric, by_ref in by_m.items():
                for ref, block in by_ref.items():
                    for det, d in block["detectors"].items():
                        for lv, L in d["levels"].items():
                            w.writerow([ch, metric, ref, det, lv, L["snr_db"]["n"], L["n_left_censored"],
                                        L["n_not_reached"], L["n_undefined"], L["snr_db"]["mean"],
                                        L["snr_db"]["sd"], L["snr_db"]["ci95_half_width"],
                                        L["dsem_at_snr"]["mean"], L["dsem_at_snr"]["sd"]] + [""] * 7)
                    for lv, P in block["paired_prototype_minus_max_logit"].items():
                        w.writerow([ch, metric, ref, "prototype_minus_max_logit", lv] + [""] * 9
                                   + [P["n_both_crossed"], P["mean_diff_db"], P["sd_diff_db"],
                                      P["wilcoxon_p"], P["mannwhitney_p"], P["prototype_lower_count"],
                                      P["n_order_determinable"]])
        if dr_cr:
            for cr, res in dr_cr["by_cr"].items():
                if res is None:
                    continue
                for det in ("max_logit", "prototype"):
                    for lv, L in res[det].items():
                        st = L["per_seed_status"]
                        w.writerow([f"awgn_cr_{cr}", "auc", "ref", det, lv, L["snr_db"]["n"],
                                    st.count("left_censored"), st.count("not_reached"), st.count("undefined"),
                                    L["snr_db"]["mean"], L["snr_db"]["sd"], L["snr_db"]["ci95_half_width"],
                                    "", ""] + [""] * 7)
                for lv, P in res["paired_prototype_minus_max_logit"].items():
                    w.writerow([f"awgn_cr_{cr}", "auc", "ref", "prototype_minus_max_logit", lv] + [""] * 9
                               + [P["n_both_crossed"], P["mean_diff_db"], P["sd_diff_db"],
                                  P["wilcoxon_p"], P["mannwhitney_p"], P["prototype_lower_count"],
                                  P["n_order_determinable"]])
    print(f"[Phase IV] CSV table -> {csv_path}")

    md_path = tables_dir / "decision_retention.md"
    names = {"auc": "DR_AUC", "op": "DR_op"}
    refs = {"ref": f"reference = {dr['reference_snr_db']:g} dB (AWGN)",
            "peak": "reference = each seed's best swept value (sensitivity)"}
    with open(md_path, "w") as f:
        f.write(f"# Decision Retention -- SNR at which DR first stays >= level "
                f"(N={dr['n_seeds']} seeds; mean +/- SD [95% CI half-width])\n\n")
        f.write("Censored seeds are counted, not averaged: 'L' = DR already >= level at the "
                "lowest swept SNR, 'NR' = level not reached at the highest swept SNR.\n\n")
        for ch, by_m in dr["channels"].items():
            for metric, by_ref in by_m.items():
                for ref, block in by_ref.items():
                    f.write(f"## {ch.upper()} -- {names[metric]} -- {refs[ref]}\n\n")
                    f.write("| Level | Max Logit SNR (dB) | Prototype SNR (dB) | Prototype - Max Logit (dB) "
                            "| Prototype lower | Wilcoxon p (paired) | Mann-Whitney p (unpaired) "
                            "| D_sem at SNR: ML / PR |\n|---|---|---|---|---|---|---|---|\n")
                    for lv in block["paired_prototype_minus_max_logit"]:
                        cells = []
                        for det in ("max_logit", "prototype"):
                            L = block["detectors"][det]["levels"][lv]
                            c = (f"{_fmt_num(L['snr_db']['mean'])} +/- {_fmt_num(L['snr_db']['sd'])} "
                                 f"[{_fmt_num(L['snr_db']['ci95_half_width'])}], n={L['snr_db']['n']}")
                            if L["n_left_censored"] or L["n_not_reached"] or L["n_undefined"]:
                                c += f" (L {L['n_left_censored']}, NR {L['n_not_reached']}, undef {L['n_undefined']})"
                            cells.append(c)
                        P = block["paired_prototype_minus_max_logit"][lv]
                        dml = block["detectors"]["max_logit"]["levels"][lv]["dsem_at_snr"]["mean"]
                        dpr = block["detectors"]["prototype"]["levels"][lv]["dsem_at_snr"]["mean"]
                        pstr = "--" if P["wilcoxon_p"] is None else format(P["wilcoxon_p"], ".4g")
                        pstr += " | " + ("--" if P["mannwhitney_p"] is None else format(P["mannwhitney_p"], ".4g"))
                        diff = f"{_fmt_num(P['mean_diff_db'])} +/- {_fmt_num(P['sd_diff_db'])} (n={P['n_both_crossed']})"
                        f.write(f"| {float(lv):.0%} | {cells[0]} | {cells[1]} | {diff} | "
                                f"{P['prototype_lower_count']}/{P['n_order_determinable']} | {pstr} | "
                                f"{_fmt_num(dml, 3)} / {_fmt_num(dpr, 3)} |\n")
                    f.write("\n")
        if dr_cr and any(v is not None for v in dr_cr["by_cr"].values()):
            f.write("## Phase III -- DR_AUC per compression ratio (each CR's own 40 dB reference)\n\n")
            f.write("| CR | Level | Max Logit SNR (dB) | Prototype SNR (dB) | Prototype - Max Logit (dB) | Prototype lower |\n"
                    "|---|---|---|---|---|---|\n")
            for cr, res in dr_cr["by_cr"].items():
                if res is None:
                    f.write(f"| {cr} | -- | per-seed sweeps incomplete | | | |\n")
                    continue
                for lv in res["paired_prototype_minus_max_logit"]:
                    a, b = res["max_logit"][lv]["snr_db"], res["prototype"][lv]["snr_db"]
                    P = res["paired_prototype_minus_max_logit"][lv]
                    f.write(f"| {cr} | {float(lv):.0%} | {_fmt_num(a['mean'])} +/- {_fmt_num(a['sd'])} (n={a['n']}) | "
                            f"{_fmt_num(b['mean'])} +/- {_fmt_num(b['sd'])} (n={b['n']}) | "
                            f"{_fmt_num(P['mean_diff_db'])} | {P['prototype_lower_count']}/{P['n_order_determinable']} |\n")
            f.write("\n")
    print(f"[Phase IV] Markdown table -> {md_path}")
    return {"csv": str(csv_path), "markdown": str(md_path)}


def _decision_retention_section(out_root, snrs, p3, cfg, save):
    """Decision Retention analysis, Plots 10-10d and the DR tables.

    Needs phase2_per_seed.json (skipped with a warning if absent); the per-CR
    analysis also needs phase3_multiseed_state.json."""
    per_seed = _load_per_seed(out_root, snrs)
    if per_seed is None:
        print("[Phase IV] WARNING: phase2_per_seed.json not found -- Decision Retention skipped.")
        return None, None
    dr = decision_retention_analysis(per_seed, snrs, cfg)
    x = dr["snr_db"]
    head = [lv for lv in cfg["DR_HEADLINE_LEVELS"]]

    # Plots 10 / 10b: mean DR curve +/- 95% CI, AWGN solid, Rayleigh dashed.
    for metric, fname, ylab in (("auc", "10_dr_auc_vs_snr", r"Decision Retention $DR_{AUC}$"),
                                ("op", "10b_dr_op_vs_snr", r"Decision Retention $DR_{op}$")):
        fig, ax = plt.subplots(figsize=(8, 5))
        lo_y, hi_y = -0.1, 1.15
        for ch, ls, alpha_line, alpha_band in (("awgn", "-", 1.0, .15), ("rayleigh", "--", .55, .07)):
            block = dr["channels"].get(ch, {}).get(metric, {}).get("ref")
            if not block:
                continue
            for det, name, color, mk in _DET_STYLE:
                d = block["detectors"][det]
                m = np.array([np.nan if v is None else v for v in d["mean_curve"]], dtype=float)
                h = np.array([np.nan if v is None else v for v in d["ci95_half_width_curve"]], dtype=float)
                if not np.any(np.isfinite(m)):
                    continue
                ax.plot(x, m, ls=ls, marker=mk, markersize=5 if ch == "awgn" else 4, color=color,
                        alpha=alpha_line, linewidth=2 if ch == "awgn" else 1.5,
                        label=f"{name} ({'AWGN' if ch == 'awgn' else 'Rayleigh'}, mean)")
                ax.fill_between(x, m - h, m + h, color=color, alpha=alpha_band, linewidth=0)
                lo_y, hi_y = min(lo_y, np.nanmin(m - h) - .05), max(hi_y, np.nanmax(m + h) + .05)
                if ch == "awgn":
                    L50 = d["levels"].get(str(0.5), {}).get("snr_db", {})
                    if L50.get("mean") is not None:
                        ax.axvline(L50["mean"], color=color, ls=":", alpha=.7,
                                   label=f"{name} SNR50 = {L50['mean']:.1f} dB (AWGN)")
        ax.axhline(1.0, color="black", ls="-", lw=.8, alpha=.35,
                   label=f"Clean-channel reference ({dr['reference_snr_db']:g} dB) = 1")
        for lv in head:
            ax.axhline(lv, color="grey", ls=":", lw=.8, alpha=.6)
            ax.text(x[0], lv + .01, f"{lv:.0%}", va="bottom", ha="left", fontsize=7, color="dimgrey")
        ax.axhline(0.0, color="black", ls=":", lw=.8, alpha=.4, label="Chance = 0")
        ax.set(xlabel="SNR (dB)", ylabel=ylab, ylim=(lo_y, hi_y))
        ax.legend(fontsize=7, loc="lower right")
        save(fig, fname, title=f"Decision Retention vs Channel Quality ({'AUROC' if metric == 'auc' else 'FPR95'}; "
                               f"mean +/- 95% CI, N={dr['n_seeds']} seeds)")

    # Plot 10c: per-seed SNR50 / SNR90 of DR_AUC (AWGN), paired by seed.
    block = dr["channels"]["awgn"]["auc"]["ref"]
    fig, axes = plt.subplots(1, len(head), figsize=(4.5 * len(head), 4.5), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, lv in zip(axes, head):
        vals = {det: block["detectors"][det]["levels"][str(lv)]["per_seed_snr_db"] for det, *_ in _DET_STYLE}
        for a, b in zip(vals["max_logit"], vals["prototype"]):
            if a is not None and b is not None:
                ax.plot([0, 1], [a, b], color="grey", alpha=.35, lw=1, zorder=1)
        for i, (det, name, color, mk) in enumerate(_DET_STYLE):
            v = [u for u in vals[det] if u is not None]
            ax.scatter([i] * len(v), v, color=color, marker=mk, s=40, zorder=3, label=name)
            S = block["detectors"][det]["levels"][str(lv)]["snr_db"]
            if S["mean"] is not None and S["ci95_half_width"] is not None:
                ax.errorbar(i + .18, S["mean"], yerr=S["ci95_half_width"], fmt="_", color="black",
                            capsize=4, markersize=14, zorder=4)
        P = block["paired_prototype_minus_max_logit"][str(lv)]
        ax.set_xticks([0, 1]); ax.set_xticklabels([n for _, n, _, _ in _DET_STYLE])
        ax.set_xlim(-.5, 1.6)
        ax.set_ylabel(f"SNR{int(round(lv * 100))} (dB)")
        ax.set_title(f"SNR at {lv:.0%} retention", fontsize=10)
        ax.text(.02, .02, f"Prototype lower in {P['prototype_lower_count']}/{P['n_order_determinable']} seeds",
                transform=ax.transAxes, fontsize=7)
    axes[0].legend(fontsize=7, loc="upper right")
    save(fig, "10c_dr_snr50_snr90_paired",
         title=f"Per-seed SNR at 50% and 90% Decision Retention (AUROC, AWGN, black = mean +/- 95% CI)")

    # Plot 10d: DR_AUC per compression ratio (needs the Phase III per-seed state).
    dr_cr = None
    state_path = Path(out_root) / "phase3_multiseed_state.json"
    if p3 and p3.get("snr_by_cr") and state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        cr_order = [c for c in ("low", "medium", "high") if c in p3["snr_by_cr"]]
        dr_cr = phase3_retention_by_cr(state, cr_order, cfg["SEEDS"], snrs, cfg)
        ok = [c for c in cr_order if dr_cr["by_cr"].get(c) is not None]
        if ok:
            fig, axes = plt.subplots(1, len(head), figsize=(4.5 * len(head), 4.5))
            axes = np.atleast_1d(axes)
            for ax, lv in zip(axes, head):
                for k, (det, name, color, mk) in enumerate(_DET_STYLE):
                    m = [dr_cr["by_cr"][c][det][str(lv)]["snr_db"]["mean"] for c in ok]
                    e = [dr_cr["by_cr"][c][det][str(lv)]["snr_db"]["ci95_half_width"] for c in ok]
                    xs = np.arange(len(ok)) + (k - .5) * .12
                    ax.errorbar(xs, [np.nan if v is None else v for v in m],
                                yerr=[0 if v is None else v for v in e], fmt=f"{mk}-", color=color,
                                capsize=3, label=name)
                ax.set_xticks(range(len(ok))); ax.set_xticklabels(ok)
                ax.set(xlabel="Compression ratio", ylabel=f"SNR{int(round(lv * 100))} (dB)")
                ax.set_title(f"SNR at {lv:.0%} retention", fontsize=10)
            axes[0].legend(fontsize=8)
            save(fig, "10d_dr_snr_vs_cr",
                 title="SNR at 50% and 90% Decision Retention vs Compression Ratio (mean +/- 95% CI)")
    elif p3 and p3.get("snr_by_cr"):
        print(f"[Phase IV] {state_path} not found -- Decision Retention per CR skipped.")

    _export_dr_tables(dr, dr_cr, out_root)
    for ch in ("awgn", "rayleigh"):
        b = dr["channels"].get(ch, {}).get("auc", {}).get("ref")
        if not b:
            continue
        for lv in head:
            ml = b["detectors"]["max_logit"]["levels"][str(lv)]["snr_db"]
            pr = b["detectors"]["prototype"]["levels"][str(lv)]["snr_db"]
            P = b["paired_prototype_minus_max_logit"][str(lv)]
            print(f"[Phase IV] DR_AUC {ch.upper()} SNR{int(round(lv*100))}: Max Logit "
                  f"{_fmt_num(ml['mean'])} +/- {_fmt_num(ml['sd'])} dB (n={ml['n']}), Prototype "
                  f"{_fmt_num(pr['mean'])} +/- {_fmt_num(pr['sd'])} dB (n={pr['n']}); Prototype lower in "
                  f"{P['prototype_lower_count']}/{P['n_order_determinable']} seeds, Wilcoxon p="
                  f"{'--' if P['wilcoxon_p'] is None else format(P['wilcoxon_p'], '.4g')}")
    return dr, dr_cr


def phase4(p2, p3, out_root, cfg):
    """All Phase IV figures and tables, and collapse_report.json.

    p2: Phase II aggregate (phase2_aggregated.json: per SNR, *_mean / *_std for
    each metric and system); p3: Phase III result (phase3.json) or None. Bands
    show +/- 1 SD across the production repetitions unless stated otherwise.
    """
    pd = Path(out_root) / "plots"; pd.mkdir(parents=True, exist_ok=True)
    awgn=p2["awgn"]; ray=p2.get("rayleigh",[])
    snrs=[r["snr_db"] for r in awgn]

    def _get(res,k): return [r.get(f"{k}_mean", r.get(k, float("nan"))) for r in res]
    def _get_std(res,k): return [r.get(f"{k}_std") or 0.0 for r in res]

    import re
    def _slugify(title):
        """Short, file-system-safe form of a plot title (LaTeX removed, lower case)."""
        s = re.sub(r"\$[^$]*\$", "", title)          # drop $...$ math spans
        s = re.sub(r"[\\{}$]", "", s)                  # drop remaining LaTeX characters
        s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
        s = re.sub(r"_+", "_", s).lower()
        return s[:80]

    def _save(fig, n, title=None):
        """Save as PNG and PDF (600 dpi); the title goes into the file name, not the image."""
        fig.tight_layout()
        fname = n if title is None else f"{n}__{_slugify(title)}"
        for ext in ("png","pdf"): fig.savefig(str(pd/f"{fname}.{ext}"),dpi=600,bbox_inches="tight")
        plt.close(fig)

    ml_dsem = _get(awgn,"max_logit_dsem")
    pr_dsem = _get(awgn,"prototype_dsem")
    # Pixel AUROC (the image-level AUROC is nearly degenerate on this test set).
    ml_auc = _get(awgn,"max_logit_auc_roc_pixel")
    pr_auc = _get(awgn,"prototype_auc_roc_pixel")
    ml_auc_std = _get_std(awgn,"max_logit_auc_roc_pixel")
    pr_auc_std = _get_std(awgn,"prototype_auc_roc_pixel")
    ml_dsem_std = _get_std(awgn,"max_logit_dsem")
    pr_dsem_std = _get_std(awgn,"prototype_dsem")
    ml_sri_auc_std = _get_std(awgn,"max_logit_sri_auc")
    pr_sri_auc_std = _get_std(awgn,"prototype_sri_auc")
    ml_fpr95 = _get(awgn,"max_logit_fpr95"); ml_fpr95_std = _get_std(awgn,"max_logit_fpr95")
    pr_fpr95 = _get(awgn,"prototype_fpr95"); pr_fpr95_std = _get_std(awgn,"prototype_fpr95")

    _export_snr_tables(awgn, out_root, cfg)

    # Collapse analysis per system on its own mean AUROC and D_sem. The 40 dB
    # reference point lies outside the evenly spaced sweep and is excluded from
    # Stages 1-4 (it is still plotted).
    coll_idx = [i for i, s in enumerate(snrs) if float(s) != float(cfg["CLS_FIT_SNR"])]
    c_snrs = [snrs[i] for i in coll_idx]
    def _sel(v): return [v[i] for i in coll_idx]
    collapse_ml = detect_collapse(c_snrs, _sel(ml_auc), _sel(ml_dsem), cfg)
    collapse_pr = detect_collapse(c_snrs, _sel(pr_auc), _sel(pr_dsem), cfg)
    elbow_ml, elbow_pr = collapse_ml["elbow_snr"], collapse_pr["elbow_snr"]
    for _c in (collapse_ml, collapse_pr):
        _c["excluded_snr_db"] = [float(s) for s in snrs if float(s) == float(cfg["CLS_FIT_SNR"])]

    # Stages 3-4 on the same SNR points.
    stages_3_4 = _run_collapse_stages_3_4(out_root, c_snrs, _sel(ml_auc), _sel(pr_auc), cfg,
                                          keep_idx=coll_idx, all_snrs=snrs)
    marker_ml = _elbow_marker(collapse_ml, stages_3_4.get("max_logit"))
    marker_pr = _elbow_marker(collapse_pr, stages_3_4.get("prototype"))

    # Plot 1: pixel AUROC vs SNR (AWGN solid, Rayleigh dashed).
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(snrs, ml_auc, "o-", color="#1f77b4", linewidth=2, label="Max Logit (AWGN, mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(ml_auc,ml_auc_std)],
                     [m+s for m,s in zip(ml_auc,ml_auc_std)], color="#1f77b4", alpha=.15)
    ax.plot(snrs, pr_auc, "D-", color="#2ca02c", linewidth=2, label="Prototype (AWGN, mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(pr_auc,pr_auc_std)],
                     [m+s for m,s in zip(pr_auc,pr_auc_std)], color="#2ca02c", alpha=.15)
    if ray:
        ml_ray_auc = _get(ray,"max_logit_auc_roc_pixel"); ml_ray_auc_std = _get_std(ray,"max_logit_auc_roc_pixel")
        pr_ray_auc = _get(ray,"prototype_auc_roc_pixel"); pr_ray_auc_std = _get_std(ray,"prototype_auc_roc_pixel")
        ax.plot(snrs,ml_ray_auc,"o--",color="#1f77b4",alpha=.5,label="Max Logit (Rayleigh, mean)")
        ax.fill_between(snrs, [m-s for m,s in zip(ml_ray_auc,ml_ray_auc_std)],
                         [m+s for m,s in zip(ml_ray_auc,ml_ray_auc_std)], color="#1f77b4", alpha=.08)
        ax.plot(snrs,pr_ray_auc,"D--",color="#2ca02c",alpha=.5,label="Prototype (Rayleigh, mean)")
        ax.fill_between(snrs, [m-s for m,s in zip(pr_ray_auc,pr_ray_auc_std)],
                         [m+s for m,s in zip(pr_ray_auc,pr_ray_auc_std)], color="#2ca02c", alpha=.08)
    # Published no-channel values are not drawn: they use a different protocol
    # (dissertation Section 4.3).
    ax.axhline(0.5,ls=":",color="black",alpha=.4,label="Chance level")
    # Elbow drawn only where supported (_elbow_marker); candidates appear in Plot 3b.
    for _coll, _mk, _name, _color in ((collapse_ml, marker_ml, "Max Logit", "#1f77b4"),
                                      (collapse_pr, marker_pr, "Prototype", "#2ca02c")):
        if _mk == "supported":
            ax.axvline(_coll["elbow_snr"], color=_color, ls="--", alpha=.5,
                       label=f"{_name} elbow (SNR={_coll['elbow_snr']}dB)")
    ax.set(xlabel="SNR (dB)",ylabel="AUC-ROC",ylim=(0,1.05))
    ax.legend(fontsize=7)
    _save(fig,"1_snr_vs_auc",title=f"Detection Performance vs Channel Quality (mean +/- std, N={cfg['N_RUNS']} runs)")

    # Plot 1b: FPR95 vs SNR, as Plot 1.
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(snrs, ml_fpr95, "o-", color="#1f77b4", linewidth=2, label="Max Logit (AWGN, mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(ml_fpr95,ml_fpr95_std)],
                     [m+s for m,s in zip(ml_fpr95,ml_fpr95_std)], color="#1f77b4", alpha=.15)
    ax.plot(snrs, pr_fpr95, "D-", color="#2ca02c", linewidth=2, label="Prototype (AWGN, mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(pr_fpr95,pr_fpr95_std)],
                     [m+s for m,s in zip(pr_fpr95,pr_fpr95_std)], color="#2ca02c", alpha=.15)
    if ray:
        ml_ray_fpr = _get(ray,"max_logit_fpr95"); ml_ray_fpr_std = _get_std(ray,"max_logit_fpr95")
        pr_ray_fpr = _get(ray,"prototype_fpr95"); pr_ray_fpr_std = _get_std(ray,"prototype_fpr95")
        ax.plot(snrs,ml_ray_fpr,"o--",color="#1f77b4",alpha=.5,label="Max Logit (Rayleigh, mean)")
        ax.fill_between(snrs, [m-s for m,s in zip(ml_ray_fpr,ml_ray_fpr_std)],
                         [m+s for m,s in zip(ml_ray_fpr,ml_ray_fpr_std)], color="#1f77b4", alpha=.08)
        ax.plot(snrs,pr_ray_fpr,"D--",color="#2ca02c",alpha=.5,label="Prototype (Rayleigh, mean)")
        ax.fill_between(snrs, [m-s for m,s in zip(pr_ray_fpr,pr_ray_fpr_std)],
                         [m+s for m,s in zip(pr_ray_fpr,pr_ray_fpr_std)], color="#2ca02c", alpha=.08)
    ax.axhline(0.95,ls=":",color="black",alpha=.4,label="Chance level (FPR95 = 0.95)")
    ax.set(xlabel="SNR (dB)",ylabel="FPR95",ylim=(0,1.05))
    ax.legend(fontsize=7)
    _save(fig,"1b_snr_vs_fpr95",title=f"False Positive Rate at 95% TPR vs Channel Quality (mean +/- std, N={cfg['N_RUNS']} runs)")

    # Plot 2: D_sem vs SNR, one curve per system (each in its own latent space).
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(snrs,ml_dsem,"o-",color="#1f77b4",label="System A / Max Logit's JSCC (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(ml_dsem,ml_dsem_std)],
                     [m+s for m,s in zip(ml_dsem,ml_dsem_std)], color="#1f77b4", alpha=.15)
    ax.plot(snrs,pr_dsem,"D-",color="#2ca02c",label="System B / Prototype's JSCC (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(pr_dsem,pr_dsem_std)],
                     [m+s for m,s in zip(pr_dsem,pr_dsem_std)], color="#2ca02c", alpha=.15)
    ax.set(xlabel="SNR (dB)",ylabel=r"$D_{sem}$")
    ax.legend(fontsize=9)
    _save(fig,"2_snr_vs_dsem",title=f"Semantic Distortion vs Channel Quality (mean +/- std, N={cfg['N_RUNS']} runs)")

    # Plot 3: AUROC vs D_sem, one panel per system, error bars +/- 1 SD on both axes.
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,dsem_v,auc_v,dsem_s,auc_s,coll,mk,name,color in [
            (axes[0],ml_dsem,ml_auc,ml_dsem_std,ml_auc_std,collapse_ml,marker_ml,"Max Logit","#1f77b4"),
            (axes[1],pr_dsem,pr_auc,pr_dsem_std,pr_auc_std,collapse_pr,marker_pr,"Prototype","#2ca02c")]:
        ax.errorbar(dsem_v, auc_v, xerr=dsem_s, yerr=auc_s, fmt="none",
                     ecolor=color, alpha=.3, zorder=1, capsize=2)
        sc=ax.scatter(dsem_v,auc_v,c=snrs,cmap="RdYlGn_r",s=90,zorder=3)
        ax.plot(dsem_v,auc_v,"-",color=color,alpha=.4,zorder=2)
        if mk == "supported":
            ax.axvline(dsem_v[snrs.index(coll["elbow_snr"])],color="darkred",ls="--",alpha=.6,
                       label=f"Elbow SNR={coll['elbow_snr']}dB")
        # Panel titles name the system.
        ax.set(xlabel=r"$D_{sem}$",ylabel="AUC-ROC",title=name,ylim=(0,1.05))
        if ax.get_legend_handles_labels()[0]:   # no empty legend box
            ax.legend(fontsize=8)
    # Colour bar attached to the right panel only, so it does not overlap it.
    fig.colorbar(sc,ax=axes[1],label="SNR (dB)")
    _save(fig,"3_dsem_vs_auc",
          title=f"Semantic Collapse: Dsem vs Detection Performance (mean +/- std, each system's own Dsem, N={cfg['N_RUNS']} runs)")

    # Plot 3b: first and second derivative curves used by Stage 1.
    fig,axes=plt.subplots(2,1,figsize=(9,8),sharex=True)
    for coll,mk,name,color in [(collapse_ml,marker_ml,"Max Logit","#1f77b4"),
                               (collapse_pr,marker_pr,"Prototype","#2ca02c")]:
        axes[0].plot(coll["snr_db"],coll["first_derivative"],"o-",color=color,label=f"{name} dAUC/dDsem")
        axes[1].plot(coll["snr_db"],coll["second_derivative"],"o-",color=color,label=f"{name} d2AUC/dDsem2")
        # Candidates and ties are shown here, labelled by status.
        if mk == "supported":
            axes[1].axvline(coll["elbow_snr"],color=color,ls="--",alpha=.5,
                             label=f"{name} elbow (SNR={coll['elbow_snr']}dB)")
        elif mk == "candidate":
            axes[1].axvline(coll["elbow_snr"],color=color,ls="--",alpha=.35,
                             label=f"{name} elbow candidate, not confirmed (SNR={coll['elbow_snr']}dB)")
        elif mk == "tie":
            _tied = [tv for tv in coll["elbow_tied_values"] if tv is not None]
            for _j, tv in enumerate(_tied):
                axes[1].axvline(tv,color=color,ls=":",alpha=.35,
                                 label=(f"{name} tied candidates, not confirmed ({_tied} dB)" if _j == 0 else None))
    axes[0].axhline(0,ls=":",color="black",alpha=.3)
    axes[1].axhline(0,ls=":",color="black",alpha=.3)
    axes[0].set(ylabel="First derivative (dAUC/dDsem)")
    axes[1].set(xlabel="SNR (dB)",ylabel="Second derivative (d2AUC/dDsem2)")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    _save(fig,"3b_collapse_derivatives",title="Semantic Collapse Detection: Derivative Curves Underlying the Elbow Point")

    # Plot 4: SRI_AUC and SRI_op vs SNR (band on SRI_AUC only); 4b: log scale.
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(snrs,_get(awgn,"max_logit_sri_auc"),"o-",color="#1f77b4",label=r"$SRI_{AUC}$ (Max Logit, mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(_get(awgn,"max_logit_sri_auc"),ml_sri_auc_std)],
                     [m+s for m,s in zip(_get(awgn,"max_logit_sri_auc"),ml_sri_auc_std)],
                     color="#1f77b4", alpha=.15)
    ax.plot(snrs,_get(awgn,"max_logit_sri_op"), "o--",color="#1f77b4",alpha=.6,label=r"$SRI_{op}$ (Max Logit, mean)")
    ax.plot(snrs,_get(awgn,"prototype_sri_auc"),"D-",color="#2ca02c",label=r"$SRI_{AUC}$ (Prototype, mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(_get(awgn,"prototype_sri_auc"),pr_sri_auc_std)],
                     [m+s for m,s in zip(_get(awgn,"prototype_sri_auc"),pr_sri_auc_std)],
                     color="#2ca02c", alpha=.15)
    ax.plot(snrs,_get(awgn,"prototype_sri_op"), "D--",color="#2ca02c",alpha=.6,label=r"$SRI_{op}$ (Prototype, mean)")
    ax.set(xlabel="SNR (dB)",ylabel="Semantic Robustness Index")
    ax.legend(fontsize=8)
    _save(fig,"4_sri_vs_snr",title=f"SRI_AUC and SRI_op vs Channel Quality (mean +/- std on SRI_AUC, N={cfg['N_RUNS']} runs)")

    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(snrs,_get(awgn,"max_logit_sri_auc"),"o-",color="#1f77b4",label=r"$SRI_{AUC}$ (Max Logit)")
    ax.plot(snrs,_get(awgn,"max_logit_sri_op"), "o--",color="#1f77b4",alpha=.6,label=r"$SRI_{op}$ (Max Logit)")
    ax.plot(snrs,_get(awgn,"prototype_sri_auc"),"D-",color="#2ca02c",label=r"$SRI_{AUC}$ (Prototype)")
    ax.plot(snrs,_get(awgn,"prototype_sri_op"), "D--",color="#2ca02c",alpha=.6,label=r"$SRI_{op}$ (Prototype)")
    ax.set_yscale("log")
    ax.set(xlabel="SNR (dB)",ylabel="SRI (log scale)")
    ax.legend(fontsize=8)
    _save(fig,"4b_sri_vs_snr_log",title="SRI_AUC and SRI_op vs Channel Quality (log scale)")

    # Plot 5: SRI_AUC vs compression ratio (Phase III).
    if p3 and p3.get("by_cr"):
        crs=list(p3["by_cr"].keys())
        ml_m=[p3["by_cr"][c].get("max_logit_sri_auc_mean", float("nan")) for c in crs]
        ml_s=[p3["by_cr"][c].get("max_logit_sri_auc_std", 0.0) for c in crs]
        pr_m=[p3["by_cr"][c].get("prototype_sri_auc_mean", float("nan")) for c in crs]
        pr_s=[p3["by_cr"][c].get("prototype_sri_auc_std", 0.0) for c in crs]
        fig,ax=plt.subplots(figsize=(7,5))
        ax.errorbar(crs,ml_m,yerr=ml_s,fmt="o-",color="#1f77b4",label="Max Logit (mean +/- std)",capsize=3)
        ax.errorbar(crs,pr_m,yerr=pr_s,fmt="D-",color="#2ca02c",label="Prototype (mean +/- std)",capsize=3)
        ax.set(xlabel="Compression Ratio",ylabel=r"$SRI_{AUC}$")
        ax.legend(fontsize=9)
        _save(fig,"5_sri_vs_cr",title=f"SRI_AUC vs Compression Ratio (mean +/- std, N={p3.get('n_runs', cfg['N_RUNS'])} runs)")

    # Plot 6: SNR x compression-ratio heatmaps, one per system and metric (AUROC,
    # FPR95, SRI_AUC, SRI_op); number format per metric.
    if p3 and p3.get("snr_by_cr"):
        keys=list(p3["snr_by_cr"])
        _heatmap_metrics = [
            ("auc_roc_pixel", "AUC", ".2f"),
            ("fpr95", "FPR95", ".2f"),
            ("sri_auc", "SRI_AUC", ".0f"),
            ("sri_op", "SRI_op", ".1f"),
        ]
        for metric_base, metric_label, fmt in _heatmap_metrics:
            for det, det_label, fname_prefix in [("max_logit", "Max Logit", "6"),
                                                   ("prototype", "Prototype", "6")]:
                metric_key = f"{det}_{metric_base}_mean"
                mat=np.array([[r.get(metric_key, float("nan")) for r in p3["snr_by_cr"][k]] for k in keys])
                fig,ax=plt.subplots(figsize=(14,4))
                sns.heatmap(mat,xticklabels=snrs,yticklabels=keys,annot=True,fmt=fmt,
                            annot_kws={"size":8},cmap="YlGnBu",ax=ax,cbar_kws={"label":metric_label})
                ax.set(xlabel="SNR (dB)",ylabel="CR")
                fname=f"{fname_prefix}_2d_heatmap_{det}_{metric_base}"
                _save(fig,fname,title=f"2D Robustness Map: {det_label} {metric_label} "
                                       f"(mean, N={p3.get('n_runs', cfg['N_RUNS'])} runs)")

    # Plot 5b: AUROC vs compression ratio at -20, -10, 0, +10 and +20 dB.
    if p3 and p3.get("snr_by_cr"):
        _anchor_snrs = [s for s in [-20, -10, 0, 10, 20] if s in snrs]
        crs=list(p3["snr_by_cr"].keys())
        fig, axes = plt.subplots(1, len(_anchor_snrs), figsize=(4*len(_anchor_snrs), 4), sharey=True)
        if len(_anchor_snrs) == 1:
            axes = [axes]
        for ax, snr_point in zip(axes, _anchor_snrs):
            snr_idx = snrs.index(snr_point)
            ml_vals = [p3["snr_by_cr"][cr][snr_idx].get("max_logit_auc_roc_pixel_mean", float("nan")) for cr in crs]
            ml_stds = [p3["snr_by_cr"][cr][snr_idx].get("max_logit_auc_roc_pixel_std", 0.0) for cr in crs]
            pr_vals = [p3["snr_by_cr"][cr][snr_idx].get("prototype_auc_roc_pixel_mean", float("nan")) for cr in crs]
            pr_stds = [p3["snr_by_cr"][cr][snr_idx].get("prototype_auc_roc_pixel_std", 0.0) for cr in crs]
            ax.errorbar(crs, ml_vals, yerr=ml_stds, fmt="o-", color="#1f77b4", label="Max Logit", capsize=3)
            ax.errorbar(crs, pr_vals, yerr=pr_stds, fmt="D-", color="#2ca02c", label="Prototype", capsize=3)
            ax.set_title(f"{snr_point:+d}dB")
            ax.set_xlabel("Compression Ratio")
        axes[0].set_ylabel("Pixel AUC-ROC")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        _save(fig, "5b_auc_vs_cr_multi_snr",
              title=f"AUC vs Compression Ratio at Multiple SNR Points (mean +/- std, "
                    f"N={p3.get('n_runs', cfg['N_RUNS'])} runs)")

    # Plot 7: SDOP vs SNR.
    ml_sdop = _get(awgn,"max_logit_sdop"); ml_sdop_std = _get_std(awgn,"max_logit_sdop")
    pr_sdop = _get(awgn,"prototype_sdop"); pr_sdop_std = _get_std(awgn,"prototype_sdop")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(snrs,ml_sdop,"o-",color="#1f77b4",label="Max Logit's JSCC (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(ml_sdop,ml_sdop_std)],
                     [m+s for m,s in zip(ml_sdop,ml_sdop_std)], color="#1f77b4", alpha=.15)
    ax.plot(snrs,pr_sdop,"D-",color="#2ca02c",label="Prototype's JSCC (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(pr_sdop,pr_sdop_std)],
                     [m+s for m,s in zip(pr_sdop,pr_sdop_std)], color="#2ca02c", alpha=.15)
    # The SDOP threshold is a percentile of the calibration images' d_sem at 40 dB,
    # so the clean-channel SDOP is (100 - percentile)% by construction.
    _sdop_clean = 1.0 - cfg["SDOP_CALIBRATION_PERCENTILE"] / 100.0
    ax.axhline(_sdop_clean,ls=":",color="grey",alpha=.7,
               label=f"Clean-channel level ({_sdop_clean:.0%}, set by calibration)")
    ax.set(xlabel="SNR (dB)",ylabel="SDOP")
    ax.legend(fontsize=9, loc="center right")
    _save(fig,"7_sdop_vs_snr",title=f"Semantic Distortion Outage Probability vs SNR (mean +/- std, N={cfg['N_RUNS']} runs)")

    # Plot 8: PSNR vs SNR (reference only).
    ml_psnr = _get(awgn,"max_logit_psnr"); ml_psnr_std = _get_std(awgn,"max_logit_psnr")
    pr_psnr = _get(awgn,"prototype_psnr"); pr_psnr_std = _get_std(awgn,"prototype_psnr")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(snrs,ml_psnr,"o-",color="#1f77b4",label="Max Logit's JSCC (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(ml_psnr,ml_psnr_std)],
                     [m+s for m,s in zip(ml_psnr,ml_psnr_std)], color="#1f77b4", alpha=.15)
    ax.plot(snrs,pr_psnr,"D-",color="#2ca02c",label="Prototype's JSCC (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(pr_psnr,pr_psnr_std)],
                     [m+s for m,s in zip(pr_psnr,pr_psnr_std)], color="#2ca02c", alpha=.15)
    ax.set(xlabel="SNR (dB)",ylabel="PSNR (dB)")
    ax.legend(fontsize=9)
    _save(fig,"8_psnr_vs_snr",title=f"Signal-Level PSNR (reference only -- task performance != PSNR) (mean +/- std, N={cfg['N_RUNS']} runs)")

    # Plot 9: TPR and FPR at the frozen threshold tau vs SNR.
    ml_tpr_tau = _get(awgn,"max_logit_tpr_at_tau"); ml_tpr_tau_std = _get_std(awgn,"max_logit_tpr_at_tau")
    ml_fpr_tau = _get(awgn,"max_logit_fpr_at_tau"); ml_fpr_tau_std = _get_std(awgn,"max_logit_fpr_at_tau")
    pr_tpr_tau = _get(awgn,"prototype_tpr_at_tau"); pr_tpr_tau_std = _get_std(awgn,"prototype_tpr_at_tau")
    pr_fpr_tau = _get(awgn,"prototype_fpr_at_tau"); pr_fpr_tau_std = _get_std(awgn,"prototype_fpr_at_tau")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(snrs,ml_tpr_tau,"o-",color="#1f77b4",label="Max Logit TPR@tau (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(ml_tpr_tau,ml_tpr_tau_std)],
                     [m+s for m,s in zip(ml_tpr_tau,ml_tpr_tau_std)], color="#1f77b4", alpha=.12)
    ax.plot(snrs,ml_fpr_tau,"o--",color="#1f77b4",alpha=.6,label="Max Logit FPR@tau (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(ml_fpr_tau,ml_fpr_tau_std)],
                     [m+s for m,s in zip(ml_fpr_tau,ml_fpr_tau_std)], color="#1f77b4", alpha=.12)
    ax.plot(snrs,pr_tpr_tau,"D-",color="#2ca02c",label="Prototype TPR@tau (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(pr_tpr_tau,pr_tpr_tau_std)],
                     [m+s for m,s in zip(pr_tpr_tau,pr_tpr_tau_std)], color="#2ca02c", alpha=.12)
    ax.plot(snrs,pr_fpr_tau,"D--",color="#2ca02c",alpha=.6,label="Prototype FPR@tau (mean)")
    ax.fill_between(snrs, [m-s for m,s in zip(pr_fpr_tau,pr_fpr_tau_std)],
                     [m+s for m,s in zip(pr_fpr_tau,pr_fpr_tau_std)], color="#2ca02c", alpha=.12)
    ax.axhline(0.95,ls=":",color="grey",alpha=.5,label="Calibration target (95% TPR at clean channel)")
    ax.set(xlabel="SNR (dB)",ylabel="Rate",ylim=(0,1.05))
    ax.legend(fontsize=7)
    _save(fig,"9_tpr_fpr_at_tau_vs_snr",title=f"Fixed-Threshold Operational Performance (tau calibrated once, mean +/- std, N={cfg['N_RUNS']} runs)")

    # Plots 10-10d and tables: Decision Retention.
    dr, dr_cr = _decision_retention_section(out_root, snrs, p3, cfg, _save)

    report = {
        "max_logit": collapse_ml, "prototype": collapse_pr,
        "decision_retention": dr, "decision_retention_by_cr": dr_cr,
        "stage3_bootstrap": {"max_logit": (stages_3_4["max_logit"] or {}).get("bootstrap"),
                              "prototype": (stages_3_4["prototype"] or {}).get("bootstrap")},
        "stage4_structural_break": {"max_logit": (stages_3_4["max_logit"] or {}).get("structural_break"),
                                     "prototype": (stages_3_4["prototype"] or {}).get("structural_break")},
    }
    with open(Path(out_root)/"collapse_report.json","w") as f:
        json.dump(report, f, indent=2)
    print(f"[Phase IV] Done -> {pd}")
    return report
