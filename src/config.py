"""Configuration of every phase, seeding, and the fingerprints that decide when
cached results may be reused.

The values below are the starting point of the Phase I search. The search writes
the values it selects (backbone stage, patch configuration, optimiser settings,
PROTOTYPE_SCALE, LAMBDA_REC) into CONFIG in place, per detector (suffix _ML for
the Max Logit system, _PR for the Prototype system); lambda_sem is passed on
separately. The values selected in the reported search are listed in REPRODUCE.md.
"""
import os

# Required by torch.use_deterministic_algorithms; must be set before the first CUDA call.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Project root (this file lives in src/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(BASE_DIR) == "src":
    BASE_DIR = os.path.dirname(BASE_DIR)

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Outputs go to Google Drive when it is mounted (Colab), otherwise to ./outputs.
_DRIVE_ROOT = "/content/drive/MyDrive"
if os.path.isdir(_DRIVE_ROOT):
    _OUTPUT_DIR = os.path.join(_DRIVE_ROOT, "dissertation_outputs")
else:
    _OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# SNR grid: -20 to +20 dB in 2 dB steps, plus the 40 dB clean-channel reference
# point (appended after DEFAULT_SNR_DB and CLS_FIT_SNR are fixed).
_SNR_RANGE = list(range(-20, 22, 2))
_DEFAULT_SNR_DB = 0
_CLS_FIT_SNR = 40.0
_SNR_RANGE = _SNR_RANGE + [40]

def _detect_num_workers(reserve_cores=1, override=None):
    """DataLoader workers: the CPU cores available to this process minus one."""
    if override is not None:
        print(f"[config] NUM_WORKERS: {override} (explicit override, not auto-detected)")
        return override
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        available = os.cpu_count()
    if available is None:
        print("[config] WARNING: could not determine the number of CPU cores -- using NUM_WORKERS=2.")
        return 2
    workers = max(1, available - reserve_cores)
    print(f"[config] NUM_WORKERS: {available} CPU cores available - {reserve_cores} reserved for the main "
          f"process = {workers}")
    return workers

_NUM_WORKERS_OVERRIDE = None   # set an integer to fix the number of workers
_NUM_WORKERS = _detect_num_workers(override=_NUM_WORKERS_OVERRIDE)

CONFIG = {
    # ---- Paths (the notebook sets both explicitly)
    "STREETHAZARDS_ROOT": (os.path.join(DATASETS_DIR, "streethazards")
                           if os.path.isdir(os.path.join(DATASETS_DIR, "streethazards"))
                           or not os.path.isdir("/content/datasets/streethazards")
                           else "/content/datasets/streethazards"),
    "OUTPUT_DIR":         _OUTPUT_DIR,

    # ---- Input size required by the pretrained swin_tiny_patch4_window7_224
    "IMG_H": 224, "IMG_W": 224,

    # ---- Compression ratio: channel symbols per image (Phase III varies it);
    # CR = n_symbols / (3 * 224 * 224)
    "CHANNEL_SYMBOLS": {
        "low":    1024,
        "medium": 2048,
        "high":   4096,
    },
    "DEFAULT_CR": "medium",

    # ---- Channel and evaluation
    "SNR_RANGE_DB":   _SNR_RANGE,
    "DEFAULT_SNR_DB": _DEFAULT_SNR_DB,   # SNR at which the Phase I search scores candidates
    "USE_RAYLEIGH":   True,              # evaluate the Rayleigh channel as well as AWGN

    # Channel noise at each evaluation point is drawn from a seed fixed per
    # (channel, SNR) (src/evaluate.py eval_noise_seed).
    "EVAL_NOISE_SEED": 20261009,
    "SRI_EPSILON":    1e-6,              # avoids division by zero in SRI

    "TARGET_TPR": 0.95,                  # operating point of tau and of FPR95

    # ---- Semantic-collapse analysis (src/collapse.py)
    "COLLAPSE_GRAD_THRESHOLD":  None,    # None: Stage 2 threshold taken from the data
    "COLLAPSE_SENSITIVITY_PERCENTILES": [50, 60, 70, 75, 80, 90],
    "BREAK_TEST_DELTA_AIC":   2.0,       # |delta AIC| threshold of Stage 4 (Burnham and Anderson, 2002)
    "BREAK_TEST_N_BOOTSTRAP": 1000,      # seed resamples for the delta AIC interval

    # ---- Decision Retention levels (src/decision_retention.py)
    "DR_LEVELS":          [0.25, 0.5, 0.75, 0.9],
    "DR_HEADLINE_LEVELS": [0.5, 0.9],

    # ---- Controls run after Phase III
    # Training over a range of SNRs (phases/multisnr_check.py): per-image SNR drawn
    # uniformly from this range, first MULTISNR_N_SEEDS production seeds. The
    # validation pass uses fixed SNR and noise draws (MULTISNR_VAL_SEED).
    "MULTISNR_TRAIN_RANGE_DB": [-20, 20],
    "MULTISNR_N_SEEDS":        5,
    "MULTISNR_VAL_SEED":       20261012,
    # True: fine-tune each seed from its production checkpoints (run_0/multisnr_ft/);
    # False: train from the pretrained backbone (run_0/multisnr/).
    "MULTISNR_WARM_START":     True,
    # A retrained model is flagged as not using the received signal if its AWGN
    # AUROC varies by less than this over the sweep, or if its matched - shuffled
    # AUROC at CLS_FIT_SNR is below COLLAPSE_GUARD_MIN_DEPENDENCE.
    "COLLAPSE_GUARD_AUROC_RANGE":    0.02,
    "COLLAPSE_GUARD_MIN_DEPENDENCE": 0.02,
    # Detector x architecture control (phases/arch2x2_check.py): first N production seeds.
    "ARCH2X2_N_SEEDS":         5,
    # Idealised digital reference (phases/digital_baseline.py): bits per latent value,
    # and the percentiles of the calibration latents that set each quantiser's range.
    "DIGITAL_BITS":              [1, 2, 4, 8],
    "DIGITAL_QUANT_PERCENTILES": [0.1, 99.9],

    # ---- Phase I search: candidates, seeds per candidate (*_VERIFY_N) and starting values
    # Step 0: Swin-Tiny stage, each at its native grid, budget-matched to 1,568 values.
    "BACKBONE_STAGE_ML": 2,
    "BACKBONE_STAGE_PR": 2,
    "STAGE_CANDIDATES": [0, 1, 2, 3],
    "STAGE_VERIFY_N": 6,

    # Step A: (patch_grid, patch_dim) of the transmitted latent at stage 2
    # (14 x 14 x 8 = 1,568 values). Six seeds: the smallest N at which a two-sided
    # exact Wilcoxon test can reach p < 0.05.
    "PATCH_GRID_ML": 14,
    "PATCH_DIM_ML":  8,
    "PATCH_GRID_PR": 14,
    "PATCH_DIM_PR":  8,
    "PATCH_CONFIG_CANDIDATES": [(14, 8), (8, 16)],
    "PATCH_CONFIG_VERIFY_N": 6,

    # Step D: scale of the Prototype detector's cosine logits.
    "PROTOTYPE_SCALE": 32.0,
    "PROTOTYPE_SCALE_GRID": [2.0, 4.0, 8.0, 16.0, 32.0],
    "PROTOTYPE_SCALE_VERIFY_N": 5,

    # Training channel and epoch ceiling (training normally stops earlier).
    "CLS_FIT_SNR": _CLS_FIT_SNR,
    "JOINT_MAX_EPOCHS": 120,

    # Step E: lambda_sem grids (six seeds each); Step F: LAMBDA_REC.
    "LAMBDA_SEM_GRID_MAXLOGIT":  [1.0, 1.5, 2.0, 2.5],
    "LAMBDA_SEM_GRID_PROTOTYPE": [1.5, 2.0, 2.5, 3.0],
    "LAMBDA_REC_ML": 1.0,
    "LAMBDA_REC_PR": 1.0,
    "LAMBDA_REC_SANITY_CANDIDATES": [0.25, 0.5, 1.0, 2.0],

    # Patience of every search run and of production training.
    "GRID_SEARCH_PATIENCE": 15,
    # Search rounds: at most MAX_CONVERGENCE_ROUNDS. The search stops when neither
    # detector's round-end configuration improves on the previous round's (paired
    # tie test at CONVERGENCE_TIE_ALPHA; above 0.0625, the smallest two-sided exact
    # Wilcoxon p-value attainable with five pairs).
    "MAX_CONVERGENCE_ROUNDS": 5,
    "CONVERGENCE_TIE_ALPHA": 0.1,
    # Step C: early-stopping threshold, relative improvement of the validation task
    # loss over the patience window.
    "MIN_REL_IMPROVE_ML": 0.01,
    "MIN_REL_IMPROVE_PR": 0.01,
    "MIN_REL_IMPROVE_CANDIDATES": [0.005, 0.01, 0.02, 0.04],

    # Step B: learning rate; Step C: momentum, polynomial decay power, weight decay.
    "JOINT_LR_ML":     0.003,
    "JOINT_LR_PR":     0.003,
    "JOINT_LR_CANDIDATES": [0.0005, 0.001, 0.003, 0.007, 0.014, 0.02, 0.03, 0.045, 0.07],
    "JOINT_LR_VERIFY_N": 5,
    "JOINT_MOMENTUM_ML":  0.9,
    "JOINT_MOMENTUM_PR":  0.9,
    "JOINT_MOMENTUM_CANDIDATES": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99],
    "JOINT_LR_POWER_ML":  0.9,
    "JOINT_LR_POWER_PR":  0.9,
    "JOINT_LR_POWER_CANDIDATES": [0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 2.5],
    "WEIGHT_DECAY_ML":    1e-4,
    "WEIGHT_DECAY_PR":    1e-4,
    "WEIGHT_DECAY_CANDIDATES": [1e-5, 1e-4, 1e-3, 1e-2],
    "OPTIMIZER_HPARAM_VERIFY_N": 5,      # MIN_REL_IMPROVE and LAMBDA_REC
    "LR_POWER_WD_VERIFY_N": 10,          # JOINT_MOMENTUM, JOINT_LR_POWER, WEIGHT_DECAY
    # lambda_sem used by Steps 0-D of round 0, before the first lambda_sem grid
    # (the middle values of the two grids).
    "PROVISIONAL_LAMBDA_SEM_ML": 2.0,
    "PROVISIONAL_LAMBDA_SEM_PR": 2.5,

    # ---- Seeds. SEED: data loading and the calibration / evaluation split.
    # SEEDS: production repetitions (Phases I-III). SELECTION_SEEDS: Phase I search
    # only, disjoint from SEEDS.
    "SEED":   42,
    "N_RUNS": 15,
    "SEEDS": [42, 123, 456, 789, 999, 111, 222, 333, 444, 555, 666, 777, 888, 1000, 1111],
    "SELECTION_SEEDS": [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010],

    # ---- Execution
    "NUM_WORKERS": _NUM_WORKERS,
    "USE_TORCH_COMPILE": True,           # torch.compile in training (falls back to eager mode)

    # RAM cache of resized images: this fraction of the available RAM, shared by the
    # datasets and worker processes; set RAM_CACHE_TOTAL_GB_OVERRIDE to fix it in GB.
    "RAM_CACHE_TOTAL_GB_OVERRIDE": None,
    "RAM_CACHE_AVAILABLE_FRACTION": 0.65,

    # Batch size: BATCH_SIZE_OVERRIDE fixes it; with None, the largest batch that fits
    # is probed and multiplied by BATCH_SIZE_SAFETY_FACTOR (src/gpu_utils.py).
    "BATCH_SIZE_WARN_ABOVE": 256,
    "BATCH_SIZE_OVERRIDE": 90,
    "BATCH_SIZE_SAFETY_FACTOR": 0.75,
    # SDOP threshold: this percentile of the calibration images' d_sem at CLS_FIT_SNR.
    "SDOP_CALIBRATION_PERCENTILE": 90,
    # Test annotations scanned to identify the anomaly label value.
    "ANOMALY_VALUE_SAMPLE_SIZE": 300,

}


# Deterministic CUDA algorithms are requested unless JSCC_DETERMINISTIC=0.
DETERMINISTIC_ALGOS = os.environ.get("JSCC_DETERMINISTIC", "1") != "0"
LOADER_GENERATORS = []   # the DataLoaders' private generators, re-seeded by set_seed


def set_seed(seed):
    """Seed Python, NumPy, PyTorch (CPU and CUDA) and the DataLoader generators,
    and request deterministic cuDNN and CUDA algorithms. Operations without a
    deterministic implementation give a warning instead of an error."""
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for _g in LOADER_GENERATORS:
        _g.manual_seed(seed)
    if DETERMINISTIC_ALGOS:
        torch.use_deterministic_algorithms(True, warn_only=True)


_FP_TRAIN_KEYS = ("BACKBONE_STAGE", "PATCH_GRID", "PATCH_DIM", "JOINT_LR", "JOINT_MOMENTUM",
                  "JOINT_LR_POWER", "WEIGHT_DECAY", "MIN_REL_IMPROVE", "LAMBDA_REC")


def training_fingerprint(cfg, detector):
    """The settings that determine how one detector's system is trained. Cached
    Phase II/III results are reused only when their stored fingerprint equals the
    current one."""
    sfx = "ML" if detector == "max_logit" else "PR"
    fp = {k: cfg.get(f"{k}_{sfx}", 2 if k == "BACKBONE_STAGE" else None) for k in _FP_TRAIN_KEYS}
    fp.update({"IMG": [cfg["IMG_H"], cfg["IMG_W"]], "JOINT_MAX_EPOCHS": cfg["JOINT_MAX_EPOCHS"],
               "CLS_FIT_SNR": cfg["CLS_FIT_SNR"], "TARGET_TPR": cfg["TARGET_TPR"],
               "SDOP_CALIBRATION_PERCENTILE": cfg["SDOP_CALIBRATION_PERCENTILE"]})
    if detector == "prototype":
        fp["PROTOTYPE_SCALE"] = cfg["PROTOTYPE_SCALE"]
    return fp


def evaluation_fingerprint(cfg):
    """The settings that determine an evaluation sweep."""
    return {"SNR_RANGE_DB": list(cfg["SNR_RANGE_DB"]), "USE_RAYLEIGH": bool(cfg["USE_RAYLEIGH"]),
            "EVAL_NOISE_SEED": cfg.get("EVAL_NOISE_SEED"), "DEFAULT_SNR_DB": cfg["DEFAULT_SNR_DB"],
            "SRI_EPSILON": cfg["SRI_EPSILON"], "eval_version": 2}


def fingerprint_key(obj):
    """Canonical string of a fingerprint, for comparisons after a JSON round trip."""
    import json
    return json.dumps(obj, sort_keys=True)
