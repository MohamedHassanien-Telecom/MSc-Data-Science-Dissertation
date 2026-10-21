# Reproducing the results

How to run the pipeline, what each cell writes and how to resume. Dataset: `README.md`.
All paths below are relative to `OUTPUT_DIR` (set in the paths cell; Google Drive by default).

## 1. Environment

- Colab GPU runtime (the reported run: NVIDIA L4, 24 GB). The clone cell runs `pip install -r requirements.txt`.
- `scipy` and `scikit-learn` are pinned, `numpy` has a floor, and `torch`, `torchvision` and `timm` are not
  pinned (Colab's preinstalled CUDA build is used). The environment cell writes the versions, GPU and git
  commit of every session to `environment/environment_<UTC time>.json`.
- `set_seed` fixes every random number generator and requests deterministic algorithms
  (`JSCC_DETERMINISTIC=0` turns this off). One training operation (CUDA adaptive average pooling, backward
  pass) has no deterministic implementation, so two trainings with the same seed are close but not
  bit-identical. Evaluation is repeatable: channel noise is seeded per (channel, SNR) by
  `src/evaluate.py: eval_noise_seed` (base `EVAL_NOISE_SEED`), and the AUROC pixel subsample has a fixed seed.

## 2. Run order (`run_experiment.ipynb`)

| Cell | Function | Outputs | Dissertation |
|---|---|---|---|
| Setup 1–4 | download StreetHazards, clone, paths, environment record | `environment/` | — |
| Phase I | `phase1_grid_search_and_train` | `run_0/` (search), `run_0_rep{i}/` (production) | 3.1.5, 3.2.1, App. B |
| Phase II | `phase2_aggregated` | `run_0/phase2_aggregated.json`, `run_0/phase2_per_seed.json` | 3.2.2–3.2.4 |
| Phase III | `phase3` | `run_0/phase3.json`, `run_0/phase3_multiseed_state.json` | 3.2.6 |
| Phase III statistics | `tools/compression_stats.py` | printed | 3.2.6 |
| Phase IV | `run_phase4` | `run_0/plots/`, `run_0/tables/`, `run_0/collapse_report.json` | 3.2.2–3.2.6, App. D |
| Scoring-rule check 1–3 | notebook cells, `src/mechanism.py` | `run_0/explainer_state.json`, `run_0/mechanism_retention.json` | F.5 |
| Ideal digital baseline | `run_digital_baseline` | `run_0/digital_baseline/` | F.6 |
| Location audit (about 1 h) | `run_location_prior_check` | `run_0/location_prior/` | F.1 |
| Location-stratified AUROC (about 5–6 h) | `run_stratified_sweep`, `run_digital_stratified` | `run_0/stratified/` | F.2 |
| Region-balanced LS-AUROC and town split | `run_stratified_supplement` | `run_0/stratified_supp/` | F.8 |
| CPU checks | `tools/town_interaction.py`, `tools/verify_retention.py`, `tools/ls_synthetic_validation.py` | `run_0/verify_retention/`, `ls_synthetic_validation/` | F.7, F.8, F.10 |
| Other benchmarks (CPU) | `tools/benchmark_location_map.py` | `benchmarks/` | F.9 |
| Multi-SNR training, cold start | `run_multisnr_check(..., warm_start=False)` | `run_0/multisnr/` | F.4, App. G |
| Multi-SNR training, warm start (about 9 h) | `run_multisnr_check(..., warm_start=True)` | `run_0/multisnr_ft/` | F.4 |
| Detector × architecture (about 11.5 h) | `run_arch2x2_check` | `run_0/arch2x2/` | F.3 |

Every function also writes its table to `run_0/tables/` and its figures to `run_0/plots/`.

**GPU memory.** The Phase I cell keeps the 15 production repetitions on the GPU (about 8.4 GB). Phase III
releases them once its medium-ratio results are stored, and the last three cells release them before
training. A cell that needs them after a release requires a session restart, the setup cells and the
Phase I cell first (it reloads the models without retraining). The two reload cells restore `p2` and `p3`
in a new session.

## 3. Phase I: search and production (`phases/phase1_grid.py`)

Each round runs the steps below for each system. Every candidate is a complete JSCC system and detector
trained jointly at 40 dB and scored by its mean pixel AUROC at 0 dB on the 300-image calibration subset,
over `SELECTION_SEEDS` (disjoint from the production `SEEDS`):

| Step | Hyperparameter | Seeds per candidate |
|---|---|---|
| 0 | Swin stage (`BACKBONE_STAGE`) | 6 |
| A | latent grid (`PATCH_GRID`, `PATCH_DIM`) | 6 |
| B | learning rate (`JOINT_LR`) | 5 |
| C | momentum, learning-rate decay power, weight decay; `MIN_REL_IMPROVE` | 10; 5 |
| D | `PROTOTYPE_SCALE` (Prototype only) | 5 |
| E | λ_sem grid | 6 |
| F | `LAMBDA_REC` | 5 |

Each step keeps the candidate with the highest mean. Rounds repeat until neither system's round-end
configuration improves on the previous round's (paired by seed, Wilcoxon α = 0.1; a tie if p ≥ 0.1 and
|r| is below the null floor of the rank-biserial correlation), or for at most `MAX_CONVERGENCE_ROUNDS` = 5.
Then `N_RUNS` = 15 production repetitions are trained with `SEEDS`, and τ and the SDOP threshold are
calibrated once per model at 40 dB on the calibration subset. The search's own evaluations do not seed
the channel noise; Phases II–IV do.

Values selected in the reported search (dissertation Table 3.1):

| Setting | Max Logit system | Prototype system |
|---|---|---|
| Swin stage; latent grid | 2; 14×14×8 | 3; 7×7×32 |
| Learning rate | 0.02 | 0.003 |
| Momentum | 0.9 | 0.97 |
| Learning-rate decay power | 1.0 | 2.0 |
| Weight decay | 1×10⁻³ | 1×10⁻⁴ |
| `MIN_REL_IMPROVE` | 0.04 | 0.005 |
| λ_rec | 0.5 | 0.5 |
| λ_sem | 2.0 | 2.0 |
| `PROTOTYPE_SCALE` | – | 4.0 |

Files: `run_0/*_verify_state_round{N}.json` (per-seed results, resumable), `run_0/*_verify_summary_round{N}.json`,
`run_0/grid_search_results.json`, `run_0/convergence_summary.json`; production: `run_0_rep{i}/` (checkpoints
and `phase1_meta.json`). In the summary files the keys `kruskal_p` and `epsilon_squared` /
`effect_size_epsilon_sq` hold the Friedman permutation p-value and Kendall's W.

## 4. Phases II–IV

- **Phase II** (`phases/phase2_sweep.py`): every production repetition at every SNR in `SNR_RANGE_DB`
  (−20 to +20 dB in 2 dB steps, and 40 dB), AWGN and Rayleigh, on the 1,200 evaluation images. Per
  repetition `run_0_rep{i}/phase2.json`; mean and SD over repetitions in `run_0/phase2_aggregated.json`;
  every repetition's values in `run_0/phase2_per_seed.json`, which Phase IV and the controls read.
- **Phase III** (`phases/phase3_compression.py`): low, medium and high ratios (`CHANNEL_SYMBOLS` = 1,024,
  2,048 and 4,096 symbols), 15 seeds each. The medium ratio reuses the production models; the low and high
  ratios first re-select λ_sem (5 selection seeds per candidate) and then train both systems per seed.
- **Phase IV** (`phases/phase4_analysis.py`, `src/visualise.py`, `src/collapse.py`,
  `src/decision_retention.py`): figures, tables, Decision Retention (SNR_X per repetition, paired
  comparison) and the four-stage semantic-collapse analysis. The 40 dB point is the reference and is not
  part of the collapse analysis.

## 5. Resuming

Re-run the same cells after a disconnect. Every stored result records the settings it was produced under
(`src/config.py: training_fingerprint`, `evaluation_fingerprint`) and is reused only if they match the
current settings; otherwise it is recomputed.
