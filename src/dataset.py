"""StreetHazards data loading (Hendrycks et al., 2022).

Expected layout under CONFIG["STREETHAZARDS_ROOT"] (official release, both tar
files extracted under one root):

    train/images/training/...      towns 1-3, 5,125 images (training)
    train/images/validation/...    town 4, 1,031 images, no anomalies (validation)
    test/images/test/{t5,t6}/...   towns 5-6, 1,500 images with anomalies
    (annotations/... mirrors images/...)

build_dataloaders returns the loaders used everywhere: cls_train and cls_val for
training and early stopping (known classes only), val_mixed (300 test images) for
threshold calibration, and test (the other 1,200 test images) for evaluation.
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
import torchvision.transforms as T
from PIL import Image
from src.config import LOADER_GENERATORS
from pathlib import Path


def detect_ram_cache_budget_gb(fraction=0.65, override_gb=None):
    """RAM available for caching resized images: `fraction` of MemAvailable in
    /proc/meminfo, in GB. Returns override_gb if given, and 0.0 (caching off) when
    /proc/meminfo cannot be read."""
    if override_gb is not None:
        print(f"[dataset] RAM cache budget: {override_gb:.2f} GB (explicit override, "
              f"not auto-detected)")
        return override_gb

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    available_gb = available_kb / 1e6
                    budget_gb = available_gb * fraction
                    print(f"[dataset] RAM cache budget: {available_gb:.1f} GB currently "
                          f"available (directly queried from /proc/meminfo) x {fraction:.0%} "
                          f"safety fraction = {budget_gb:.2f} GB")
                    return budget_gb
        print("[dataset] WARNING: /proc/meminfo exists but has no MemAvailable line -- "
              "RAM caching disabled rather than guessed.")
        return 0.0
    except FileNotFoundError:
        print("[dataset] /proc/meminfo not available (non-Linux environment?) -- "
              "RAM caching disabled rather than guessed.")
        return 0.0

# ImageNet statistics of the input normalisation (needed by the pretrained Swin
# encoder); denorm_to_tanh_range inverts it.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def denorm_to_tanh_range(x_imagenet: torch.Tensor) -> torch.Tensor:
    """Undo the ImageNet normalisation and rescale to [-1, 1], the output range of
    the reconstruction decoder (tanh). Used for reconstruction targets and PSNR."""
    mean = x_imagenet.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = x_imagenet.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x01 = (x_imagenet * std + mean).clamp(0.0, 1.0)
    return 2.0 * x01 - 1.0


class StreetHazardsDataset(Dataset):
    """One StreetHazards split ("train", "validation" or "test").

    Each item is a dict with the normalised image, the binary anomaly mask
    (pixels equal to anomaly_value), has_anomaly, class_map (known-class index
    per pixel; -1 for anomalies and unlabelled pixels, ignored by the loss) and
    the file path. Images and annotations are resized to img_size when read and
    can be kept in RAM up to ram_cache_limit_gb; augmentation (if enabled) is
    applied after the cache.
    """

    def __init__(self, root, split="train", img_size=(224,224), augment=False, id_to_idx=None,
                 anomaly_value=None, ram_cache_limit_gb=None):
        self.root = Path(root); self.split = split; self.img_size = img_size
        self.id_to_idx = id_to_idx
        self.anomaly_value = anomaly_value
        self._cache = {}
        self._cache_bytes = 0
        self._cache_limit_bytes = (ram_cache_limit_gb * 1e9) if ram_cache_limit_gb else 0
        # Bytes per cached item: RGB image (3 bytes/pixel) + annotation (1 byte/pixel).
        self._cache_item_bytes = self.img_size[0] * self.img_size[1] * 4
        self._img_resize = T.Resize(img_size)
        self._ann_resize = T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST)
        _dirs = {
            "train":      ("train/images/training",    "train/annotations/training"),
            "validation": ("train/images/validation",   "train/annotations/validation"),
            "test":       ("test/images/test",          "test/annotations/test"),
        }
        if split not in _dirs:
            raise ValueError(f"Unknown split '{split}'")
        img_rel, ann_rel = _dirs[split]
        img_dir = self.root / img_rel
        ann_dir = (self.root / ann_rel) if ann_rel else None

        if not img_dir.exists():
            raise FileNotFoundError(
                f"[Dataset] '{split}' not found at {img_dir}\n"
                f"Download: https://github.com/hendrycks/anomaly-seg\n"
                f"Set STREETHAZARDS_ROOT in CONFIG.")
        # Images sit in town subfolders (t1-t6); search recursively if needed.
        self.img_paths = sorted(img_dir.glob("*.png"))
        if not self.img_paths:
            self.img_paths = sorted(img_dir.rglob("*.png"))
        if not self.img_paths:
            raise FileNotFoundError(f"No .png files in {img_dir}")

        self.ann_paths = []
        for p in self.img_paths:
            if ann_dir and ann_dir.exists():
                a = ann_dir / p.relative_to(img_dir)
                if not a.exists(): a = ann_dir / p.name
                self.ann_paths.append(a if a.exists() else None)
            else:
                self.ann_paths.append(None)
        if id_to_idx is not None and all(a is None for a in self.ann_paths):
            raise FileNotFoundError(
                f"[Dataset] id_to_idx was supplied but '{split}' has no "
                f"annotations under {self.root} -- this split cannot supply "
                f"classifier training labels.")

        norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        if augment:
            self.img_tf = T.Compose([T.RandomHorizontalFlip(0.5),
                                     T.ColorJitter(0.2,0.2,0.1), T.ToTensor(), norm])
        else:
            self.img_tf = T.Compose([T.ToTensor(), norm])
        self.ann_tf = T.ToTensor()

    def __len__(self): return len(self.img_paths)

    def _load_raw(self, idx):
        """(image, annotation or None) as PIL images resized to img_size, from the
        RAM cache when present; newly read items are cached while room remains."""
        if idx in self._cache:
            return self._cache[idx]

        img_pil = self._img_resize(Image.open(self.img_paths[idx]).convert("RGB"))
        ann_p = self.ann_paths[idx]
        ann_pil = self._ann_resize(Image.open(ann_p)) if (ann_p and ann_p.exists()) else None

        if self._cache_limit_bytes > 0:
            if self._cache_bytes + self._cache_item_bytes <= self._cache_limit_bytes:
                self._cache[idx] = (img_pil, ann_pil)
                self._cache_bytes += self._cache_item_bytes

        return img_pil, ann_pil

    def __getitem__(self, idx):
        img_pil, ann_pil = self._load_raw(idx)
        img = self.img_tf(img_pil)
        class_map = torch.full((1, *self.img_size), -1, dtype=torch.long)
        if ann_pil is not None:
            raw = self.ann_tf(ann_pil)
            pixel_vals = (raw * 255).round()
            if self.anomaly_value is None:
                raise ValueError(
                    "[StreetHazardsDataset] anomaly_value was never set -- pass it in "
                    "from detect_anomaly_value() via build_dataloaders.")
            mask = (pixel_vals == self.anomaly_value).float()
            has_anom = (mask.sum() > 0).float()
            if self.id_to_idx is not None:
                ids = pixel_vals.round().long()
                remap = torch.full((256,), -1, dtype=torch.long)
                for cid, i in self.id_to_idx.items():
                    remap[cid] = i
                class_map = remap[ids.clamp(0, 255)]
        else:
            mask = torch.zeros(1, *self.img_size); has_anom = torch.tensor(0.0)
        return {"image": img, "mask": mask, "has_anomaly": has_anom,
                "class_map": class_map, "path": str(self.img_paths[idx])}


# In-process caches of the two label scans below; both depend only on their arguments.
_ANOMALY_VALUE_CACHE = {}


def detect_anomaly_value(root, sample_size=300):
    """Label value of the anomaly class, found from the first sample_size test
    annotations: among values present in at least 95% of images, the one that
    covers the smallest share of pixels (anomalies are small objects inserted
    into nearly every test image). Prints the per-value statistics."""
    _cache_key = (str(root), sample_size)
    if _cache_key in _ANOMALY_VALUE_CACHE:
        _cached = _ANOMALY_VALUE_CACHE[_cache_key]
        print(f"[detect_anomaly_value] Using cached result for root={root}, "
              f"sample_size={sample_size} (already scanned earlier this run) "
              f"-> anomaly_value={_cached}")
        return _cached

    import numpy as np
    ann_dir = Path(root) / "test" / "annotations" / "test"
    ann_paths = sorted(ann_dir.glob("*.png")) or sorted(ann_dir.rglob("*.png"))
    ann_paths = ann_paths[:sample_size]
    if not ann_paths:
        raise FileNotFoundError(f"[detect_anomaly_value] No files found under {ann_dir}")

    image_count = {}; pixel_count = {}; total_px = 0
    for p in ann_paths:
        arr = np.array(Image.open(p))
        total_px += arr.size
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            image_count[v] = image_count.get(v, 0) + 1
            pixel_count[v] = pixel_count.get(v, 0) + c

    n_images = len(ann_paths)
    stats = [(v, image_count[v]/n_images, pixel_count[v]/total_px) for v in sorted(image_count)]

    print(f"[detect_anomaly_value] Statistical signature per value, {n_images} test images:")
    print(f"  {'value':>6} {'in % of images':>16} {'% of all pixels':>18}")
    for v, img_frac, px_frac in stats:
        print(f"  {v:>6} {img_frac*100:>15.1f}% {px_frac*100:>17.4f}%")

    HIGH_IMG_COVERAGE_THRESHOLD = 0.95
    candidates = [t for t in stats if t[1] >= HIGH_IMG_COVERAGE_THRESHOLD]
    if not candidates:
        raise ValueError(
            f"[detect_anomaly_value] No class found with image-coverage >= "
            f"{HIGH_IMG_COVERAGE_THRESHOLD*100:.0f}% -- cannot identify the "
            f"anomaly class. Inspect the table above manually.")

    by_pixel_frac = sorted(candidates, key=lambda t: t[2])
    detected_tuple = by_pixel_frac[0]
    detected = detected_tuple[0]

    if len(by_pixel_frac) > 1:
        gap = by_pixel_frac[1][2] - detected_tuple[2]
        rel_gap = gap / max(detected_tuple[2], 1e-8)
        print(f"[detect_anomaly_value] Detected anomaly value = {detected} "
              f"(pixel-coverage={detected_tuple[2]*100:.4f}%, next-lowest candidate "
              f"is {by_pixel_frac[1][0]} at {by_pixel_frac[1][2]*100:.4f}%, "
              f"relative gap={rel_gap*100:.0f}%)")
        if rel_gap < 0.10:
            print(f"[detect_anomaly_value] WARNING: relative gap to the next-lowest "
                  f"candidate is small ({rel_gap*100:.0f}%) -- inspect the table above "
                  f"manually before trusting this result.")
    else:
        print(f"[detect_anomaly_value] Detected anomaly value = {detected} "
              f"(pixel-coverage={detected_tuple[2]*100:.4f}%, only candidate "
              f"above the image-coverage threshold)")

    _ANOMALY_VALUE_CACHE[_cache_key] = detected
    return detected


_KNOWN_CLASSES_CACHE = {}


def discover_known_classes(root, anomaly_value, ann_split="train"):
    """Known class ids: every label value in the ann_split annotations except
    anomaly_value. Returns (sorted ids, {id: class index})."""
    _cache_key = (str(root), anomaly_value, ann_split)
    if _cache_key in _KNOWN_CLASSES_CACHE:
        _cached_known, _cached_id_to_idx = _KNOWN_CLASSES_CACHE[_cache_key]
        print(f"[Dataset] Using cached known-class discovery for root={root}, "
              f"ann_split={ann_split} (already scanned earlier this run): "
              f"{len(_cached_known)} known classes ids={_cached_known}")
        return _cached_known, _cached_id_to_idx

    import numpy as np
    split_ann_dirs = {
        "train":      Path(root) / "train" / "annotations" / "training",
        "validation": Path(root) / "train" / "annotations" / "validation",
        "test":       Path(root) / "test"  / "annotations" / "test",
    }
    ann_dir = split_ann_dirs[ann_split]
    if not ann_dir.exists():
        raise FileNotFoundError(f"[Dataset] {ann_split} annotations not found at {ann_dir}")
    ann_paths = sorted(ann_dir.glob("*.png")) or sorted(ann_dir.rglob("*.png"))
    if not ann_paths:
        raise FileNotFoundError(f"No annotation files found in {ann_dir}")
    seen = set()
    for p in ann_paths:
        arr = np.array(Image.open(p))
        seen.update(np.unique(arr).tolist())
    seen.discard(anomaly_value)
    known = sorted(seen)
    id_to_idx = {cid: i for i, cid in enumerate(known)}
    print(f"[Dataset] Discovered {len(known)} known classes from "
          f"{len(ann_paths)} {ann_split} annotations: ids={known}")
    _KNOWN_CLASSES_CACHE[_cache_key] = (known, id_to_idx)
    return known, id_to_idx


def build_dataloaders(cfg):
    """All DataLoaders, as a dict with keys cls_train, cls_val, val_mixed, test,
    known_class_ids, id_to_idx and _cal_fallback.

    The 1,500 test images are split once with a generator seeded by cfg["SEED"]:
    20% (300) for calibration (val_mixed) and 80% (1,200) for evaluation (test).
    The RAM-cache budget is divided over the three datasets and the worker
    processes (each worker holds its own copy).
    """
    root = cfg["STREETHAZARDS_ROOT"]
    sz   = (cfg["IMG_H"], cfg["IMG_W"])
    bs   = cfg["BATCH_SIZE"]; nw = cfg["NUM_WORKERS"]

    ram_cache_total_gb = detect_ram_cache_budget_gb(
        fraction=cfg["RAM_CACHE_AVAILABLE_FRACTION"],
        override_gb=cfg.get("RAM_CACHE_TOTAL_GB_OVERRIDE"))
    _n_dataset_instances = 3
    _cache_divisor = _n_dataset_instances * max(1, nw)
    per_instance_cache_gb = ram_cache_total_gb / _cache_divisor
    if per_instance_cache_gb > 0:
        print(f"[Data] RAM caching enabled: {ram_cache_total_gb:.2f} GB total budget / "
              f"({_n_dataset_instances} dataset instances x {max(1, nw)} worker(s)) = "
              f"{per_instance_cache_gb:.2f} GB per instance-worker pair")

    anomaly_value = detect_anomaly_value(root, sample_size=cfg["ANOMALY_VALUE_SAMPLE_SIZE"])

    # Every loader owns private torch.Generators, re-seeded by config.set_seed, so
    # shuffle order and worker seeds do not depend on the global RNG. The sampler
    # and the DataLoader get separate generators because the DataLoader draws the
    # worker base seed from its own generator when its iterator is created.
    LOADER_GENERATORS.clear()
    def _gen():
        g = torch.Generator(); g.manual_seed(cfg["SEED"]); LOADER_GENERATORS.append(g); return g

    loaders = {}

    known_class_ids, id_to_idx = discover_known_classes(root, anomaly_value, ann_split="train")
    loaders["known_class_ids"] = known_class_ids
    loaders["id_to_idx"] = id_to_idx

    _cls_train_ds = StreetHazardsDataset(root, "train", sz, augment=False, id_to_idx=id_to_idx,
                                          anomaly_value=anomaly_value, ram_cache_limit_gb=per_instance_cache_gb)
    loaders["cls_train"] = DataLoader(
        _cls_train_ds,
        bs, sampler=RandomSampler(_cls_train_ds, generator=_gen()), generator=_gen(), num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=(nw > 0), prefetch_factor=(4 if nw > 0 else None))
    loaders["cls_val"] = DataLoader(
        StreetHazardsDataset(root, "validation", sz, augment=False, id_to_idx=id_to_idx, anomaly_value=anomaly_value,
                              ram_cache_limit_gb=per_instance_cache_gb),
        bs, shuffle=False, generator=_gen(), num_workers=nw, pin_memory=True, persistent_workers=(nw > 0), prefetch_factor=(4 if nw > 0 else None))

    full_test = StreetHazardsDataset(root, "test", sz, anomaly_value=anomaly_value,
                                       ram_cache_limit_gb=per_instance_cache_gb)
    n_cal = max(2, int(len(full_test) * 0.20))
    rng = torch.Generator().manual_seed(cfg["SEED"])
    perm = torch.randperm(len(full_test), generator=rng).tolist()
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
    loaders["val_mixed"] = DataLoader(Subset(full_test, cal_idx), bs, shuffle=False, generator=_gen(), num_workers=nw,
                                        pin_memory=True, persistent_workers=(nw > 0), prefetch_factor=(4 if nw > 0 else None))
    loaders["test"] = DataLoader(Subset(full_test, eval_idx), bs, shuffle=False, generator=_gen(), num_workers=nw,
                                   pin_memory=True, persistent_workers=(nw > 0), prefetch_factor=(4 if nw > 0 else None))
    loaders["_cal_fallback"] = True

    print(f"[Data] Cls train: {len(loaders['cls_train'].dataset)} | "
          f"Cls val: {len(loaders['cls_val'].dataset)} | Calibration: {n_cal} | "
          f"Test: {len(full_test)-n_cal} | {len(known_class_ids)} known classes")
    return loaders


def pool_class_map_to_patches(class_map, patch_grid, num_classes):
    """Per-cell labels: the majority known class of the pixels in each of the
    patch_grid x patch_grid cells; -1 where a cell has no known-class pixel."""
    B, _, H, W = class_map.shape
    valid = (class_map >= 0).float()
    safe = class_map.clamp(min=0)
    onehot = torch.nn.functional.one_hot(safe.squeeze(1), num_classes).permute(0, 3, 1, 2).float()
    onehot = onehot * valid
    pooled = torch.nn.functional.adaptive_avg_pool2d(onehot, (patch_grid, patch_grid))
    counts = pooled.sum(dim=1)
    majority = pooled.argmax(dim=1)
    return torch.where(counts > 0, majority, torch.full_like(majority, -1))
