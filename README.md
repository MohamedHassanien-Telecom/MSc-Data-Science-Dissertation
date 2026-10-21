# Robust semantic JSCC for wireless anomaly detection

Code for an MSc dissertation on how wireless channel impairments affect anomaly-detection decisions
made on Deep JSCC latents. Two Deep JSCC systems with a Swin-Tiny encoder send a 1,568-value latent over 2,048 complex
channel uses (AWGN and Rayleigh), and a pixel-level anomaly detector works on the received latent:

- **Max Logit system**: Swin stage 2, 14×14×8 latent, Max Logit score (Hendrycks et al., 2022).
- **Prototype system**: Swin stage 3, 7×7×32 latent, cosine similarity to learned class prototypes
  (Fontanel et al., 2021).

Both are evaluated across SNR (−20 to +20 dB, and 40 dB as reference) and three compression ratios with
pixel AUROC, FPR95, semantic distortion (D_sem), the Semantic Robustness Index (SRI) and Decision
Retention. The controls include a location audit and a location-stratified AUROC (LS-AUROC) that removes
the image-independent location prior of the benchmark.

## Quick start

1. Open `run_experiment.ipynb` in Google Colab with a GPU runtime (the reported run used an NVIDIA L4).
2. In the clone cell, set `REPO_URL` to this repository's URL.
3. Run the cells in order. `REPRODUCE.md` gives the run order, the outputs of every cell and how to resume.

## Dataset

StreetHazards only (Hendrycks et al., 2022), official release: https://github.com/hendrycks/anomaly-seg.
The first notebook cell downloads `streethazards_train.tar` and `streethazards_test.tar` and extracts both
under one root (`CONFIG["STREETHAZARDS_ROOT"]`):

| Folder | Content | Use |
|---|---|---|
| `train/images/training/t1-3/` | 5,125 images, towns 1–3 | training |
| `train/images/validation/t4/` | 1,031 images, town 4, no anomalies | validation (early stopping) |
| `test/images/test/{t5,t6}/` | 1,500 images, towns 5–6, with anomalies | 300 for threshold calibration, 1,200 for evaluation (`src/dataset.py`) |

Annotations mirror the image folders under `annotations/`. The location-prior check on other benchmarks
(Appendix F.9) downloads their public label masks in its own cell.

## Repository layout

```
run_experiment.ipynb   Colab notebook; cell order = run order
src/                   configuration, dataset, models, channels, losses, training, evaluation,
                       metrics, Decision Retention, collapse analysis, plots
phases/                Phase I (search and production), Phase II (SNR sweep), Phase III
                       (compression), Phase IV (analysis) and the controls of Appendix F
tools/                 CPU scripts that read saved results: compression statistics, town test,
                       independent recomputation of Decision Retention, synthetic LS-AUROC check,
                       location prior on other benchmarks
REPRODUCE.md           run order, outputs, selected hyperparameters, resuming
requirements.txt       dependencies (the statistical libraries are pinned)
```

The automated tests mentioned in the dissertation are not included in this repository.
