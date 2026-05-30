# Chromatin Accessibility Predictor

Predicts ATAC-seq open chromatin regions from raw DNA sequence using JAX and Flax.  
Dataset: ENCODE REST API, K562 human cell line, hg38 chromosome 22.

---

## Overview

Chromatin accessibility — whether a region of DNA is physically open and available for transcription factor binding — is a fundamental determinant of gene regulation. ATAC-seq measures this experimentally by detecting nucleosome-free regions. This project trains a convolutional neural network to predict accessibility directly from the underlying DNA sequence, without any experimental signal as input.

The core question: does the sequence itself carry enough information to predict whether a region will be open in a given cell type?

---

## Architecture

A Conv1D residual network implemented in Flax.

```
Input:  (B, 4, 2048)   one-hot encoded DNA, channels-first
         |
Stem:   Conv(128, kernel=15) + GELU
         |
Tower:  5 x ResBlock
          LayerNorm -> Conv(128, k=9) -> GELU -> Conv(128, k=9) -> residual add
          average pooling (stride 2) between blocks
         |
Head:   global average pool -> Dense(64) -> GELU -> Dense(1)
         |
Output: (B,)  logit  [sigmoid -> open chromatin probability]
```

Total parameters: approximately 1.2M.

---

## Configuration

| Parameter     | Value  | Notes                                      |
|---------------|--------|--------------------------------------------|
| `SEQ_LEN`     | 2048   | bp window centred on each peak / non-peak  |
| `N_FILTERS`   | 128    | channels in conv tower                     |
| `N_LAYERS`    | 5      | residual blocks                            |
| `BATCH_SIZE`  | 64     |                                            |
| `EPOCHS`      | 100    | best-val-AUROC checkpoint kept             |
| `LR`          | 3e-4   | Adam optimiser                             |
| `SEED`        | 42     |                                            |

Train / val / test split: 70 / 15 / 15, shuffled before splitting.

---

## Data

**Peaks — ENCODE REST API**

IDR-thresholded ATAC-seq narrowPeak file for K562 (Homo sapiens, released experiments). Fetched programmatically: the notebook queries the ENCODE search endpoint, resolves the experiment accession, and downloads the bed.gz peak file. No manual download required.

```
https://www.encodeproject.org/search/
  type=Experiment
  assay_title=ATAC-seq
  biosample_ontology.term_name=K562
  status=released
```

**Reference genome — UCSC**

hg38 chr22 FASTA downloaded from:
```
https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz
```

chr22 is used to keep the download small (~50 MB). The approach generalises to any chromosome.

**Sequence encoding**

Each sample is a 2048 bp window centred on the peak midpoint (positive) or a random non-overlapping genomic window (negative). Windows with more than 10% ambiguous bases (N) are discarded. DNA is one-hot encoded to shape `(4, 2048)` with channel order A / C / G / T.

**Class balance**

1:1 positive to negative ratio. Negative windows are sampled from regions with zero overlap to any annotated peak (checked at 100 bp resolution).

---

## Notebook Structure

Each cell is independently executable. Run them top to bottom on first use; individual cells can be re-run in isolation after the relevant variables are in memory.

| Cell | Content |
|------|---------|
| 1 | Install dependencies |
| 2 | Imports |
| 3 | Paths |
| 4 | Config and hyperparameters |
| 5 | `download_file` helper |
| 6 | ENCODE API query and peak file download |
| 7 | hg38 chr22 FASTA download and decompression |
| 8 | `pyfaidx` import |
| 9 | `one_hot` encoding utility |
| 10 | Load chr22 sequence |
| 11 | Parse narrowPeak file |
| 12 | Build pos/neg dataset and train/val/test split |
| 13 | Print dataset statistics |
| 14 | EDA 1 — peak-length distribution |
| 15 | EDA 2 — GC content open vs closed chromatin |
| 16 | Model definition (`ResBlock`, `ChromatinPredictor`) |
| 17 | Model initialisation, parameter count |
| 18 | Optimiser and train state |
| 19 | JIT-compiled train step and prediction functions |
| 20 | Training loop with tqdm progress bar |
| 21 | Training curves plot |
| 22 | Full test-set evaluation |
| 23 | Input-gradient saliency interpretation |

---

## Exploratory Data Analysis

**EDA 1 — Peak-length distribution**

Histogram of ATAC-seq peak widths across chr22, with median and mean annotated. Most peaks cluster between 150 and 500 bp, consistent with nucleosome-free regions flanked by positioned nucleosomes. Very long peaks typically correspond to broad regulatory regions. This plot serves as a data sanity check: an unexpected distribution (e.g. all peaks the same width) indicates a file format issue.

Output: `plots/eda1_peak_lengths.png`

**EDA 2 — GC content: open vs closed chromatin**

Density histograms of per-window GC content for positive (open) and negative (closed) samples. Open chromatin regions are systematically GC-enriched relative to genomic background, because the transcription factor binding motifs concentrated in accessible regions (SP1, CTCF, EGR1) have GC-rich consensus sequences. The separation between the two distributions confirms that the model has a learnable signal before any training begins.

Output: `plots/eda2_gc_content.png`

---

## Training

Binary cross-entropy loss. Adam optimiser. The best parameters by validation AUROC are checkpointed throughout training and restored at the end, so the final model corresponds to the epoch with highest validation discrimination regardless of overfitting in later epochs.

Output: `plots/training_curves.png`

---

## Evaluation

All metrics computed on the held-out test set (15% of total samples). Threshold-dependent metrics use a decision threshold of 0.5.

**Threshold-independent**

| Metric   | Value  |
|----------|--------|
| AUROC    | 0.8463 |
| PR-AUC   | 0.8929 |
| Log Loss | 0.7070 |

**Threshold-dependent**

| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 0.7630 |
| Balanced Accuracy  | 0.7655 |
| F1 Score           | 0.7726 |
| MCC                | 0.5288 |

**Confusion matrix**

|                    | Predicted Open | Predicted Closed |
|--------------------|---------------|-----------------|
| Actually Open      | TP = 124      | FN = 44         |
| Actually Closed    | FP = 29       | TN = 111        |

**Confusion-matrix derived**

| Metric                   | Value  |
|--------------------------|--------|
| Sensitivity / Recall     | 0.7381 |
| Specificity              | 0.7929 |
| Precision (PPV)          | 0.8105 |
| Negative Predictive Value| 0.7161 |

The model is more precise than it is sensitive: of the 153 windows it calls open, 81% are true peaks (PPV = 0.81), but it misses 26% of real peaks (FN = 44, sensitivity = 0.74). The higher specificity (0.79) relative to sensitivity (0.74) reflects a slight conservative bias — the model is more willing to call a region closed than to make a false positive call. This is consistent with the class imbalance in the difficulty of the task: many non-peak regions share partial sequence similarity with peaks, making false negatives easier to produce than false positives.

Plots: ROC curve, Precision-Recall curve, Confusion matrix.

Output: `plots/eval_curves.png`

---

## Interpretation

Input-gradient saliency via `jax.grad`. The gradient of the model's output score with respect to the one-hot input gives a `(4, 2048)` sensitivity matrix per example — how much a small perturbation to each base at each position would change the prediction. Absolute values are averaged across 50 positive test examples to produce an aggregated importance profile.

The central 200 bp around the peak centre is visualised. A 5-point running mean smooths high-frequency noise. Non-maximum suppression (minimum 15 bp separation) selects the top-5 most salient positions, each annotated with its position relative to peak centre and its saliency value. Clustering of high-saliency positions typically corresponds to known TF footprints within accessible chromatin.

Output: `plots/interpretation_saliency.png`

---

## Installation

```bash
pip install "jax[cuda12]" flax optax requests pyfaidx \
            scikit-learn matplotlib seaborn logomaker tqdm pandas
```

For CPU-only:
```bash
pip install "jax[cpu]" flax optax requests pyfaidx \
            scikit-learn matplotlib seaborn logomaker tqdm pandas
```

---

## Output Files

```
plots/
  eda1_peak_lengths.png
  eda2_gc_content.png
  training_curves.png
  eval_curves.png
  interpretation_saliency.png

data/
  K562_atac_peaks.bed.gz
  hg38_chr22.fa
```

---

## References

- ENCODE Project Consortium. The ENCODE portal. https://www.encodeproject.org
- UCSC Genome Browser. hg38 chromosome downloads. https://hgdownload.soe.ucsc.edu
- Bradbury et al. JAX: composable transformations of Python+NumPy programs. http://github.com/google/jax
- Loshchilov & Hutter. Decoupled Weight Decay Regularization (Adam). ICLR 2019.
- Avsec et al. Effective gene expression prediction from sequence by integrating long-range interactions (Enformer). Nature Methods, 2021.
