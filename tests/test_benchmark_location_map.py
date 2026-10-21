"""tools/benchmark_location_map.py on synthetic label masks written to disk:
  1. masks are read, resized by nearest neighbour and encoded (anomaly / void /
     normal);
  2. no image contributes to its own map (an anomaly at a position no other image
     uses scores 0), and void pixels are excluded from the counts and the scoring;
  3. a location-biased benchmark gives a high plain AUROC; anomalous pixels drawn
     independently of position give about 0.5;
  4. LS-AUROC is computed within fold x region (equal to stratified_auc over the
     same groups);
  5. --group-by-parent uses the parent folders as folds; the CLI writes its outputs
     and replaces a benchmark's table row on a re-run;
  6. --group-regex keeps the frames of one sequence in one fold, so near-duplicate
     frames do not inflate the score.

Usage:  python3 tests/test_benchmark_location_map.py
"""
import os, sys, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pathlib import Path
from PIL import Image
import tools.benchmark_location_map as B
from phases.stratified_eval import stratified_auc, region_index

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

rng = np.random.default_rng(0)

def write(d, masks, sub=None):
    files = []
    for i, m in enumerate(masks):
        p = Path(d) / (sub[i] if sub else "") / f"m{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(m.astype(np.uint8)).save(p); files.append(str(p))
    return files

with tempfile.TemporaryDirectory() as d:
    # 1. Encoding and nearest-neighbour resizing (448 -> 224).
    m = np.zeros((448, 448), np.uint8); m[:224, :] = 2; m[400:, :] = 255
    f = write(Path(d) / "enc", [m])
    lab = B.load_masks(f, 224, [2], [255])[0].reshape(224, 224)
    check("encoding: anomaly value -> 1, void -> -1, rest -> 0, nearest resize",
          (lab[:112] == 1).all() and (lab[200:] == -1).all() and (lab[112:200] == 0).all())

    # 2. Own-image exclusion and void handling.
    labels = np.zeros((6, 16), np.int8)
    # Only image 0 is anomalous at pixel 3; image 0 is void at pixel 5.
    labels[0, 3] = 1
    labels[1:, 5] = 1; labels[0, 5] = -1
    folds = np.arange(6)
    maps = B.fold_maps(labels, folds)
    check("own image excluded: image 0's map is 0 at the pixel only it uses", maps[0, 3] == 0.0)
    check("void excluded from counts: image 1's map at pixel 5 is 4/4 (image 0 void)", abs(maps[1, 5] - 1.0) < 1e-12)

    # 3. Location-biased and position-independent benchmarks (120 images).
    def rect_masks(n, biased):
        out = []
        for _ in range(n):
            mm = np.zeros((224, 224), np.uint8)
            h, w = rng.integers(10, 40, 2)
            if biased:
                r = int(np.clip(rng.normal(160, 12), h, 223 - h)); c = int(np.clip(rng.normal(112, 20), w, 223 - w))
            else:
                r = int(rng.integers(h, 224 - h)); c = int(rng.integers(w, 224 - w))
            mm[r - h // 2:r + h // 2, c - w // 2:c + w // 2] = 1
            out.append(mm)
        return out
    fb = write(Path(d) / "biased", rect_masks(120, True))
    fu = write(Path(d) / "uniform", [(rng.random((224, 224)) < 0.05).astype(np.uint8) for _ in range(120)])
    rb = B.evaluate(B.load_masks(fb, 224, [1], []), B.fold_ids(fb, 5, False)[0], 224)
    ru = B.evaluate(B.load_masks(fu, 224, [1], []), B.fold_ids(fu, 5, False)[0], 224)
    check(f"location-biased benchmark: plain AUROC {rb['plain_auroc']:.3f} > 0.85", rb["plain_auroc"] > 0.85)
    check(f"anomalies independent of position: plain AUROC {ru['plain_auroc']:.3f} within 0.02 of 0.5", abs(ru["plain_auroc"] - 0.5) < 0.02)
    check(f"biased benchmark: LS 14x14 {rb['ls_auroc_14']:.3f} far below plain {rb['plain_auroc']:.3f}",
          rb["ls_auroc_14"] < rb["plain_auroc"] - 0.2)

    # 4. LS-AUROC = stratified_auc over fold x region.
    L = B.load_masks(fb, 224, [1], []); fo = B.fold_ids(fb, 5, False)[0]
    mp = B.fold_maps(L, fo)
    hw = 224 * 224
    flat = L.reshape(-1); idx = np.arange(flat.size); img, pix = np.divmod(idx, hw)
    y = flat.astype(np.int64); s = mp[fo[img], pix]
    reg = fo[img] * 196 + region_index(pix, 224, 224, 14)
    direct = stratified_auc(y, s, reg)[0]
    r_all = B.evaluate(L, fo, 224, max_pixels=10 ** 9)
    check(f"LS 14x14 = stratified_auc over fold x region ({r_all['ls_auroc_14']:.6f} vs {direct:.6f})",
          abs(r_all["ls_auroc_14"] - direct) < 1e-12)

    # 5. --group-by-parent and the command-line interface.
    sub = ["townA"] * 60 + ["townB"] * 60
    fg = write(Path(d) / "grp", rect_masks(120, True), sub)
    g, names = B.fold_ids(fg, 5, True)
    check("group-by-parent: two folds named after the folders", names == ["townA", "townB"] and set(g.tolist()) == {0, 1})
    out = Path(d) / "out"
    res = B.main(["--name", "synthetic", "--masks", str(Path(d) / "grp" / "*" / "*.png"), "--anomaly", "1",
                  "--group-by-parent", "--out", str(out)])
    check("CLI writes the JSON and the markdown table",
          (out / "synthetic_location_map.json").exists() and "synthetic" in (out / "location_map_benchmarks.md").read_text()
          and json.load(open(out / "synthetic_location_map.json"))["n_images"] == 120)
    B.main(["--name", "synthetic", "--masks", str(Path(d) / "grp" / "*" / "*.png"), "--anomaly", "1",
            "--group-by-parent", "--out", str(out)])
    check("re-running a benchmark replaces its table row instead of adding one",
          (out / "location_map_benchmarks.md").read_text().count("| synthetic |") == 1)

with tempfile.TemporaryDirectory() as d:
    # 6. --group-regex: 12 sequences of 5 near-identical frames.
    masks, names = [], []
    for g in range(12):
        r, c = rng.integers(20, 200, 2)
        for i in range(5):
            mm = np.zeros((224, 224), np.uint8); mm[r - 8:r + 8, c - 8:c + 8] = 1
            masks.append(mm); names.append(f"0{g:02d}_seq{g:02d}_{i:06d}_labels.png")
    files = []
    for m_, n_ in zip(masks, names):
        p = Path(d) / n_; Image.fromarray(m_).save(p); files.append(str(p))
    files = sorted(files)
    g, _ = B.fold_ids(files, 5, False, r"_(seq\d+)_")
    seq = [Path(f).name.split("_")[1] for f in files]
    same = all(len({int(g[i]) for i in range(len(files)) if seq[i] == s_}) == 1 for s_ in set(seq))
    check("group-regex: every frame of a sequence is in one fold", same and len(set(g.tolist())) == 5)
    L = B.load_masks(files, 224, [1], [])
    leak = B.evaluate(L, B.fold_ids(files, 5, False)[0], 224)["plain_auroc"]
    grouped = B.evaluate(L, g, 224)["plain_auroc"]
    check(f"near-duplicate frames inflate a random split ({leak:.3f}) but not a grouped one ({grouped:.3f})",
          leak > 0.9 and grouped < leak - 0.2)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
if not all(ok):
    sys.exit(1)
print("All benchmark location-map checks passed.")
