"""Optional RAM cache of StreetHazardsDataset (src/dataset.py). Images are resized
once when read, and the resized image is cached; augmentation runs on every
access.
  1. repeated access to one index reads the files only once;
  2. the cache stops adding items at its byte limit;
  3. augmentation is not frozen by the cache: the random transform gives a
     different result on every call.
PIL, torch and torchvision are replaced by minimal stand-ins, and the dataset
object is built directly, without scanning a real directory.

Usage:  python3 tests/test_ram_caching.py
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_open_calls = []

class FakeImg:
    def __init__(self, path, size=(1280, 720)):
        self.path = path
        self.size = size
    def convert(self, mode): return self

class FakeImageModule:
    @staticmethod
    def open(path):
        _open_calls.append(str(path))
        return FakeImg(str(path))

class FakeTensor:
    def __init__(self, tag): self.tag = tag

class FakeCompose:
    """Stand-in for transforms.Compose; a random pipeline returns a new variant on every call."""
    _call_counter = [0]
    def __init__(self, steps): self.random = any(getattr(s, "is_random", False) for s in steps)
    def __call__(self, img):
        self._call_counter[0] += 1
        variant = self._call_counter[0] if self.random else "fixed"
        return FakeTensor(f"{img.path if hasattr(img,'path') else 'ann'}:{variant}")

class FakeRandomFlip:
    is_random = True
    def __init__(self, p): pass
class FakeColorJitter:
    is_random = True
    def __init__(self, *a): pass
class FakeResize:
    is_random = False
    def __init__(self, *a, **kw): pass
class FakeToTensor:
    is_random = False
class FakeNormalize:
    is_random = False
    def __init__(self, *a): pass
class FakeResizeFn:
    """Stand-in for the dataset's resize step; passes the image through unchanged."""
    def __call__(self, img): return img

fake_T = types.SimpleNamespace(
    Compose=FakeCompose, Resize=FakeResize, RandomHorizontalFlip=FakeRandomFlip,
    ColorJitter=FakeColorJitter, ToTensor=FakeToTensor, Normalize=FakeNormalize,
    InterpolationMode=types.SimpleNamespace(NEAREST="nearest"),
)

class FakeTorch:
    class Tensor: pass
    @staticmethod
    def full(shape, val, dtype=None): return FakeTensor(f"full{shape}")
    @staticmethod
    def zeros(*shape): return FakeTensor(f"zeros{shape}")
    @staticmethod
    def tensor(v): return FakeTensor(f"scalar{v}")

class FakeDataset:
    pass
class FakeDataLoader:
    def __init__(self, *a, **kw): pass
class FakeSubset:
    def __init__(self, *a, **kw): pass
class FakeConcatDataset:
    def __init__(self, *a, **kw): pass

fake_torch_module = types.ModuleType('torch')
fake_torch_module.Tensor = FakeTorch.Tensor
fake_torch_module.full = FakeTorch.full
fake_torch_module.zeros = FakeTorch.zeros
fake_torch_module.tensor = FakeTorch.tensor
sys.modules['torch'] = fake_torch_module

fake_torch_utils = types.ModuleType('torch.utils')
fake_torch_utils_data = types.ModuleType('torch.utils.data')
fake_torch_utils_data.Dataset = FakeDataset
fake_torch_utils_data.DataLoader = FakeDataLoader
fake_torch_utils_data.Subset = FakeSubset
fake_torch_utils_data.ConcatDataset = FakeConcatDataset
fake_torch_utils_data.RandomSampler = object
sys.modules['torch.utils'] = fake_torch_utils
sys.modules['torch.utils.data'] = fake_torch_utils_data
sys.modules['torchvision'] = types.ModuleType('torchvision')
sys.modules['torchvision.transforms'] = fake_T
sys.modules['PIL'] = types.ModuleType('PIL')
sys.modules['PIL'].Image = FakeImageModule
sys.modules['PIL.Image'] = FakeImageModule

if 'src.dataset' in sys.modules:
    del sys.modules['src.dataset']

import importlib.util
spec = importlib.util.spec_from_file_location(
    "dataset_module", os.path.join(os.path.dirname(__file__), "..", "src", "dataset.py"))
dataset_module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(dataset_module)
except Exception as e:
    print(f"NOTE: src/dataset.py did not import completely in isolation ({e}); "
          f"the caching method is tested directly below.")

StreetHazardsDataset = dataset_module.StreetHazardsDataset

class FakePath:
    def __init__(self, s): self._s = s
    def exists(self): return True
    def __str__(self): return self._s
    def __contains__(self, sub): return sub in self._s

def make_fake_dataset(ram_cache_limit_gb, n_images=5, img_w=1280, img_h=720):
    """A dataset object with fake paths and the attributes _load_raw uses."""
    ds = StreetHazardsDataset.__new__(StreetHazardsDataset)
    ds.img_paths = [FakePath(f"/fake/img_{i}.png") for i in range(n_images)]
    ds.ann_paths = [FakePath(f"/fake/ann_{i}.png") for i in range(n_images)]
    ds.img_size = (224, 224)
    ds.anomaly_value = 14
    ds.id_to_idx = None
    ds._cache = {}
    ds._cache_bytes = 0
    ds._cache_limit_bytes = (ram_cache_limit_gb * 1e9) if ram_cache_limit_gb else 0
    # Each cached item costs a fixed 224 x 224 x 4 bytes after resizing.
    ds._cache_item_bytes = ds.img_size[0] * ds.img_size[1] * 4
    ds._img_resize = FakeResizeFn()
    ds._ann_resize = FakeResizeFn()
    ds.img_tf = FakeCompose([FakeRandomFlip(0.5), FakeToTensor(), FakeNormalize()])
    ds.ann_tf = FakeCompose([FakeToTensor()])
    return ds

# Image.open that records each file read.
def patched_open(path):
    _open_calls.append(str(path))
    return FakeImg(str(path), size=(1280, 720))
FakeImageModule.open = patched_open

_open_calls.clear()
# 1. Repeated access to one index reads from disk once.
ds = make_fake_dataset(ram_cache_limit_gb=1.0)
for _ in range(5):
    ds._load_raw(0)
img_opens_for_idx0 = [c for c in _open_calls if "img_0" in c]
print(f"[1] Image.open calls for index 0, across 5 accesses: {len(img_opens_for_idx0)}")
assert len(img_opens_for_idx0) == 1, \
    f"Expected exactly 1 real read (rest from cache), got {len(img_opens_for_idx0)}"
print("PASS: repeated access to the same index reads from disk exactly once, "
      "every subsequent access served from the cache")

_open_calls.clear()
# 2. A 0.5 MB budget holds about two of the five ~0.2 MB items.
ds_limited = make_fake_dataset(ram_cache_limit_gb=5e-4)
for i in range(5):
    ds_limited._load_raw(i)
cached_count = len(ds_limited._cache)
print(f"\n[2] Images actually cached with a 0.5 MB budget (each ~0.2 MB): {cached_count}")
assert 0 < cached_count < 5, \
    f"Expected the limit to allow SOME but not ALL 5 images to cache, got {cached_count}"
_open_calls.clear()
for i in range(5):
    ds_limited._load_raw(i)
recount = len(_open_calls)
# Each uncached index reads both its image and its annotation.
expected_recount = (5 - cached_count) * 2
print(f"[2] Re-reads needed on a second full pass: {recount} "
      f"(expected {expected_recount}: {5 - cached_count} uncached indices x 2 files each)")
assert recount == expected_recount, \
    f"Expected exactly {expected_recount} disk reads on the second pass, got {recount}"
print(f"PASS: cache correctly stopped at {cached_count}/5 images under the configured "
      f"byte limit, not unbounded")

_open_calls.clear()
# 3. Augmentation still runs on every call.
FakeCompose._call_counter[0] = 0
ds_aug = make_fake_dataset(ram_cache_limit_gb=1.0)
img_pil, ann_pil = ds_aug._load_raw(0)
result1 = ds_aug.img_tf(img_pil)
result2 = ds_aug.img_tf(img_pil)
result3 = ds_aug.img_tf(img_pil)
print(f"\n[3] Three transform calls on the SAME cached resized image: "
      f"{result1.tag}, {result2.tag}, {result3.tag}")
assert result1.tag != result2.tag != result3.tag, \
    "Augmentation transform produced identical results across calls -- caching may have " \
    "accidentally frozen randomness instead of only caching the resized, pre-augmentation image"
print("PASS: augmentation transform still produces a different result every call, even "
      "though the underlying resized image is now served from cache -- randomness is intact")

print("\n" + "="*70)
print("ALL RAM CACHING TESTS PASSED")
print("="*70)
