"""Location prior of an anomaly benchmark (dissertation Appendix F.9): the pixel AUROC
reached by a map of where anomalies usually appear, without looking at any image.

The model-free location map of Section 3.1.9 and Appendix F.1-F.2, applied to any
anomaly-segmentation benchmark that releases its label masks. Only the masks are
read.

  1. Every mask is resized to SIZE x SIZE by nearest-neighbour interpolation
     (aspect ratio not preserved, as in the study) and mapped to
     1 = anomalous, 0 = normal, -1 = void (ignored), using --anomaly / --void.
  2. The images are split into K folds: a seeded permutation by default; by
     parent folder with --group-by-parent (e.g. the StreetHazards test towns); or
     with --group-regex, so that frames of the same recording share a fold. The
     map for an image of fold f is the anomaly frequency at each pixel over the
     images of all other folds (void pixels excluded), so no image contributes
     to its own score.
  3. Plain pixel AUROC of that map over all non-void pixels, and LS-AUROC at
     14x14, 7x7 and 28x28 regions (phases.stratified_eval.stratified_auc), with
     the regions also split by fold so that every pair is compared under one map
     (as in Appendix F.2). Both use one seeded subsample of at most --max-pixels
     non-void pixels (RandomState(42)).

A plain AUROC well above 0.5 means that location alone ranks anomalous pixels
first on that benchmark; LS-AUROC near 0.5 means that little of this prior
remains inside the regions. The map uses the benchmark's own labels, so it
measures the strength of the prior; it is not a deployable detector.

Usage (the label encodings are those documented by each benchmark):
  python3 tools/benchmark_location_map.py --name RoadAnomaly \
      --masks "RoadAnomaly/frames/*.labels/labels_semantic.png" --anomaly 2 --out out_dir
  python3 tools/benchmark_location_map.py --name FS_LostAndFound \
      --masks "fishyscapes_lostandfound/*_labels.png" --anomaly 1 --void 255 --out out_dir
Writes <out>/<name>_location_map.json and rewrites <out>/location_map_benchmarks.md
(one row per benchmark).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phases.stratified_eval import region_index, stratified_auc   # noqa: E402

GRIDS = (14, 7, 28)


def load_masks(files, size, anomaly_values, void_values):
    """(n, size*size) int8: 1 anomalous, 0 normal, -1 void."""
    from PIL import Image
    out = np.zeros((len(files), size * size), dtype=np.int8)
    for i, f in enumerate(files):
        m = np.array(Image.open(f))
        # Colour-coded masks: use the first channel.
        if m.ndim == 3:
            m = m[..., 0]
        m = np.array(Image.fromarray(m).resize((size, size), resample=Image.NEAREST))
        lab = np.zeros(m.shape, dtype=np.int8)
        lab[np.isin(m, anomaly_values)] = 1
        lab[np.isin(m, void_values)] = -1
        out[i] = lab.reshape(-1)
    return out


def fold_ids(files, k, group_by_parent, group_regex=None):
    """Fold of every image.

    --group-by-parent: one fold per parent folder. --group-regex: images whose file
    names give the same first capture group (e.g. the recording sequence) share a
    fold; groups are assigned largest first, each to the fold with the fewest
    images (deterministic). Otherwise: a seeded random split into k folds.
    """
    import re
    if group_by_parent:
        names = [Path(f).parent.name for f in files]
        uniq = sorted(set(names))
        if len(uniq) < 2:
            raise ValueError("--group-by-parent needs at least two parent folders")
        return np.array([uniq.index(x) for x in names], dtype=np.int64), uniq
    n = len(files)
    if group_regex:
        rx = re.compile(group_regex)
        keys = []
        for f in files:
            m = rx.search(Path(f).name)
            if not m:
                raise ValueError(f"--group-regex {group_regex!r} does not match {Path(f).name}")
            keys.append(m.group(1))
        groups = sorted(set(keys))
        if len(groups) < 2:
            raise ValueError(f"--group-regex gives only {len(groups)} group(s)")
        k = min(k, len(groups))
        sizes = {g: keys.count(g) for g in groups}
        load, assign = [0] * k, {}
        for g in sorted(groups, key=lambda g: (-sizes[g], g)):
            j = int(np.argmin(load)); assign[g] = j; load[j] += sizes[g]
        return np.array([assign[x] for x in keys], dtype=np.int64), [f"fold{j}" for j in range(k)]
    if n < k:
        raise ValueError(f"{n} images cannot be split into {k} folds")
    perm = np.random.RandomState(42).permutation(n)
    f = np.empty(n, dtype=np.int64)
    f[perm] = np.arange(n) % k
    return f, [f"fold{j}" for j in range(k)]


def fold_maps(labels, folds):
    """(K, hw) anomaly frequency per pixel over the images not in each fold."""
    K = int(folds.max()) + 1
    anom = np.stack([(labels[folds == j] == 1).sum(axis=0) for j in range(K)]).astype(np.float64)
    valid = np.stack([(labels[folds == j] != -1).sum(axis=0) for j in range(K)]).astype(np.float64)
    a_out = anom.sum(axis=0)[None] - anom
    v_out = valid.sum(axis=0)[None] - valid
    return np.divide(a_out, v_out, out=np.zeros_like(a_out), where=v_out > 0)


def evaluate(labels, folds, size, max_pixels=2_000_000):
    """Plain AUROC and LS-AUROC per grid of the out-of-fold location map."""
    n, hw = labels.shape
    maps = fold_maps(labels, folds)
    flat = labels.reshape(-1)
    valid_idx = np.flatnonzero(flat != -1)
    if valid_idx.size > max_pixels:
        valid_idx = np.sort(np.random.RandomState(42).choice(valid_idx, size=max_pixels, replace=False))
    img, pix = np.divmod(valid_idx, hw)
    y = flat[valid_idx].astype(np.int64)
    s = maps[folds[img], pix]
    res = {"n_images": int(n), "n_images_with_anomaly": int(((labels == 1).sum(axis=1) > 0).sum()),
           "anomalous_pixel_fraction": float((labels == 1).sum() / max((labels != -1).sum(), 1)),
           "n_pixels_scored": int(valid_idx.size), "n_folds": int(folds.max() + 1)}
    if y.min() == y.max():
        raise ValueError("the scored pixels contain only one class")
    res["plain_auroc"] = float(roc_auc_score(y, s))
    for g in GRIDS:
        reg = folds[img] * (g * g) + region_index(pix, size, size, g)
        res[f"ls_auroc_{g}"] = stratified_auc(y, s, reg)[0]
    return res


def write_table(out):
    """Rewrite <out>/location_map_benchmarks.md from every *_location_map.json in out
    (one row per benchmark, so re-running a benchmark replaces its row).
    """
    rows = ["| Benchmark | Images | Anomalous pixels | Plain AUROC | LS 14x14 | LS 7x7 | LS 28x28 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for p in sorted(Path(out).glob("*_location_map.json")):
        r = json.load(open(p))
        rows.append(f"| {r['name']} | {r['n_images']} | {100 * r['anomalous_pixel_fraction']:.2f}% | {r['plain_auroc']:.4f} | "
                    f"{r['ls_auroc_14']:.4f} | {r['ls_auroc_7']:.4f} | {r['ls_auroc_28']:.4f} |")
    (Path(out) / "location_map_benchmarks.md").write_text("\n".join(rows) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--masks", required=True, help="glob of label mask files")
    ap.add_argument("--anomaly", required=True, help="comma-separated mask values meaning 'anomalous'")
    ap.add_argument("--void", default="", help="comma-separated mask values to ignore")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--group-by-parent", action="store_true", help="use parent folders as folds")
    ap.add_argument("--group-regex", default=None,
                    help="regex with one capture group on the file name; images with the same group share a fold")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max-pixels", type=int, default=2_000_000)
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(a.masks, recursive=True))
    if not files:
        sys.exit(f"no files match {a.masks}")
    parse = lambda t: [int(x) for x in t.split(",") if x.strip() != ""]
    labels = load_masks(files, a.size, parse(a.anomaly), parse(a.void))
    folds, fold_names = fold_ids(files, a.folds, a.group_by_parent, a.group_regex)
    res = evaluate(labels, folds, a.size, a.max_pixels)
    res.update({"name": a.name, "masks": a.masks, "anomaly_values": parse(a.anomaly), "void_values": parse(a.void),
                "folds": fold_names, "size": a.size, "group_regex": a.group_regex,
                "n_groups": (len(set(__import__("re").compile(a.group_regex).search(Path(f).name).group(1) for f in files))
                             if a.group_regex else None)})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{a.name}_location_map.json", "w"), indent=2)
    write_table(out)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
