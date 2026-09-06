# Uncertainty-Aware Underwater Semantic Segmentation for Robotics

## Overview

A reproducible semantic-segmentation pipeline for the **SUIM (Semantic Segmentation of Underwater Imagery)** benchmark, built around **DeepLabV3 with a MobileNetV3-Large backbone** and extended with **Monte Carlo Dropout uncertainty estimation** and **temperature scaling for probability calibration**.

The project examines not only segmentation accuracy but also whether uncertainty is meaningfully associated with pixel-level prediction errors and whether confidence estimates can be recalibrated before being considered in a conservative underwater-robotics perception workflow.

> **Scientific scope.** This repository documents an experimental research pipeline. It does not claim deployment readiness, exact Bayesian inference, or robustness to arbitrary underwater domain shift.
## Research Questions

The implementation is organized around four questions:

1. How well can a practical DeepLabV3 + MobileNetV3-Large model segment the eight SUIM semantic classes?
2. Does Monte Carlo Dropout produce uncertainty signals that concentrate on difficult or incorrect pixels?
3. Does temperature scaling improve probabilistic calibration without changing the predicted segmentation labels?
4. How can uncertainty be interpreted conservatively in an underwater robotic perception pipeline?

## Core Method

The complete workflow is:

**dataset acquisition and validation → leakage screening → deterministic train/validation split → preprocessing and augmentation → DeepLabV3 training → held-out test evaluation → temperature scaling → Monte Carlo Dropout uncertainty analysis → qualitative visualization → reproducibility record → packaged results**

The main experimental model is the checkpoint selected from the **Cross-Entropy + Dice** training condition. A **Cross-Entropy-only** experiment is retained as an ablation.

## Scientific Background

The segmentation component uses **DeepLabV3**, whose design exploits atrous convolution and multi-scale contextual processing [3][4]. The chosen MobileNetV3-Large backbone provides a comparatively compact feature extractor suitable for a Colab-oriented experiment [5]. The uncertainty component follows the Monte Carlo Dropout view of stochastic neural-network inference [6]: repeated forward passes with dropout active produce a distribution of predictions from which predictive moments and entropy-based quantities can be estimated. The calibration component uses post-hoc **temperature scaling**, a single positive scalar applied to logits before softmax [7].

These components address complementary questions: segmentation metrics quantify *what* the model predicts, calibration evaluates whether confidence tracks empirical correctness, and MC Dropout analysis examines whether stochastic predictive variation is associated with observed pixel errors.

## Dataset: SUIM

The source uses the SUIM benchmark described by the University of Minnesota Interactive Robotics and Vision Laboratory. The official resource reports **1,525 paired train/validation samples** and **110 paired test samples**, with eight RGB-coded semantic classes [1][2].

| Class ID | Symbol | Class | RGB code |
| ---: | :---: | --- | --- |
| 0 | BW | Background / waterbody | `000` |
| 1 | HD | Human divers | `001` |
| 2 | PF | Aquatic plants / sea-grass | `010` |
| 3 | WR | Wrecks / ruins | `011` |
| 4 | RO | Robots / instruments | `100` |
| 5 | RI | Reefs / invertebrates | `101` |
| 6 | FV | Fish / vertebrates | `110` |
| 7 | SR | Sea-floor / rocks | `111` |

The expected official directory structure is:

```text
train_val/
├── images/
└── masks/

TEST/
├── images/
└── masks/
```

The implementation also supports a flatter `suim_merged`-style layout in which class-wise masks are stored under directories such as `BW`, `HD`, `PF`, `WR`, `RO`, `RI`, `FV`, and `SR`.

### Dataset-ingestion behavior in the provided source

The active code expects an existing ZIP archive at `/content/archive.zip` in Google Colab. It then:

- verifies that the archive is a valid ZIP;
- computes its SHA-256 checksum;
- tests ZIP integrity before extraction;
- performs path-traversal-safe extraction;
- detects an official SUIM-style structure or a supported flat mirror layout;
- performs multi-stage image/mask matching;
- reconstructs semantic masks from class-wise binary masks when necessary; and
- materializes a canonical dataset tree under `/content/SUIM`.

The source contains narrative comments describing Google Drive and Kaggle download fallbacks, but the active executable path provided here does **not** invoke `gdown` or perform those downloads. `gdown` is installed in the setup cell, while the dataset cell actually consumes a ZIP already present in the Colab filesystem.

### Robust image/mask matching

For supported archive layouts, the implementation uses progressively more permissive matching strategies:

1. exact filename stem;
2. normalized mask suffix removal;
3. compact alphanumeric normalization;
4. unique numeric signature matching; and
5. high-confidence fuzzy matching.

The fuzzy stage accepts a candidate only when the best sequence-matching score is at least `0.93` and exceeds the second-best candidate by at least `0.03` (unless there is only one candidate).

This design reduces false pairing caused by harmless naming differences while avoiding unconstrained fuzzy assignments.

### Dataset-integrity controls

The source performs several independent integrity checks before model training:

- image and mask readability checks;
- image/mask pairing completeness checks;
- mask color decoding validation;
- automatic repair of image/mask resolution mismatches by resizing the **mask only** with nearest-neighbor interpolation;
- duplicate reporting within each split;
- SHA-256 detection of exact image duplicates across train and test; and
- removal of training-side copies when an exact image also occurs in the held-out test split.

The final cross-split exact-image overlap is asserted to be zero.

## Train, Validation, and Test Protocol

The official `TEST` split is kept separate from training and validation whenever the official SUIM structure is available.

For the `train_val` pool, the source performs a deterministic split using `train_test_split` with:

- validation fraction: `0.15`;
- random state: `20260827`; and
- stratification target: the dominant non-background class in each image, when feasible.

The dominant-class stratification is a practical approximation rather than true multilabel stratification. If scikit-learn cannot construct the stratified split, the code falls back to a deterministic unstratified split using the same random seed.

For a flat mirror containing at least 1,635 validated paired samples, the source constructs a **derived** 1,525/110 train/test partition after deterministic stem sorting. That partition is explicitly **not** the official SUIM benchmark split.

## Preprocessing and Augmentation

All images are resized to **256 × 256** pixels.

Image preprocessing uses ImageNet normalization:

```math
\mu = (0.485,\;0.456,\;0.406),
\qquad
\sigma = (0.229,\;0.224,\;0.225).
```

Geometric transforms are applied identically to the image and mask, while photometric augmentation is applied only to the image.

| Transform | Training behavior |
| --- | --- |
| Resize | `256 × 256` |
| Horizontal flip | Probability `0.50` |
| Rotation | Probability `0.35`, angle sampled uniformly from `[-8°, +8°]` |
| Color jitter | Brightness `0.18`, contrast `0.18`, saturation `0.15`, hue `0.03` |
| Image interpolation | Bilinear |
| Mask interpolation | Nearest neighbor |
| Normalization | ImageNet mean and standard deviation |

Nearest-neighbor interpolation is mandatory for masks because semantic labels are categorical rather than continuous quantities.

## Model Architecture

The primary architecture is **TorchVision DeepLabV3 with a MobileNetV3-Large backbone** and `aux_loss=True`.

TorchVision exposes a dedicated `deeplabv3_mobilenet_v3_large` model builder. The provided source initializes the **MobileNetV3-Large backbone from ImageNet weights** when those weights are available. It deliberately does **not** load the standard DeepLabV3 segmentation weights because those weights correspond to a different semantic label space.

The source also includes a defensive fallback: if the pretrained backbone weights cannot be loaded, the backbone is initialized randomly so that execution can continue.

### Explicit stochastic layer

A `Dropout2d` layer with probability `0.20` is inserted immediately before the final segmentation classifier. The same modification is also applied to the auxiliary classifier when present.

This dropout mechanism is used as an **approximate Bayesian/stochastic inference device** for MC Dropout analysis; it is not exact posterior inference.

### Model output

The main segmentation output is represented by logits with shape:

```text
(batch_size, 8, 256, 256)
```

The auxiliary output is used during training through an auxiliary cross-entropy term and is not used as the primary inference output.

## Objective Function

The baseline training objective consists of a weighted cross-entropy term, a soft multiclass Dice loss, and an auxiliary cross-entropy term.

```math
\mathcal{L}
=
\mathcal{L}_{CE}
+
\lambda_{Dice}\mathcal{L}_{Dice}
+
\lambda_{aux}\mathcal{L}_{aux}.
```

The configured coefficients are:

```math
\lambda_{Dice}=0.6,
\qquad
\lambda_{aux}=0.4.
```

The cross-entropy coefficient is `1.0` in the baseline.

### Soft multiclass Dice loss

The source first converts logits to class probabilities and converts targets to one-hot masks. For class $c$, the implemented Dice term has the form

```math
D_c
=
\frac{2\sum p_c y_c + s}
{\sum p_c + \sum y_c + s},
```

with smoothing constant $s=1$.

The resulting Dice loss is

```math
\mathcal{L}_{Dice}
=
1-
\frac{1}{C}
\sum_{c=1}^{C}D_c,
```

where $C=8$.

### Class-imbalance weighting

The baseline `ce_dice` experiment uses class weights derived from pixel frequencies. If $f_c$ denotes the observed fraction of pixels belonging to class $c$, the source computes a smoothed inverse-frequency weight:

```math
\tilde{w}_c
=
\frac{1}{\sqrt{f_c}+0.05}.
```

The weights are then normalized by their mean:

```math
 w_c
=
\frac{\tilde{w}_c}
{\frac{1}{C}\sum_{j=1}^{C}\tilde{w}_j}.
```

The executable training path supplies these normalized weights to the cross-entropy terms. There is an important implementation subtlety: `segmentation_loss()` automatically falls back to the global `CLASS_WEIGHTS_TENSOR` whenever its `class_weights` argument is `None`. Consequently, the `ce_only` experiment is **still class-weighted in the executed code**, even though `train_experiment()` passes `None` for that condition. The active implementation therefore removes the Dice term but does not remove class weighting.

A second important detail is that the class-frequency statistics are computed from the complete post-cleanup `train_pairs` collection **before** the train/validation split is created. Therefore, the validation subset is not used as a model-update batch, but its class frequencies can influence the class-loss weights. This is a minor form of validation-side information leakage and is documented here explicitly rather than hidden.

## Central Configuration

| Parameter | Value |
| --- | --- |
| Base seed | `20260827` |
| Image size | `256 × 256` |
| CUDA batch size | `8` |
| CPU batch size | `2` |
| Validation fraction | `0.15` |
| Requested epochs | `25` |
| Early-stopping patience | `10` |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Dice weight | `0.6` |
| Auxiliary-loss weight | `0.4` |
| Dropout probability | `0.20` |
| MC Dropout samples | `8` |
| Calibration pixels | `100,000` maximum |
| Smoke-test mode | `False` by default; `True` reduces training to one epoch |

## Optimization and Training Schedule

| Parameter | Value |
| --- | ---: |
| Optimizer | AdamW |
| Initial learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Maximum requested epochs | `25` |
| LR scheduler | CosineAnnealingLR |
| Final scheduler floor | `1e-6` |
| Gradient clipping | `max_norm = 1.0` |
| Early-stopping patience | `10` epochs without improved validation mIoU |
| Batch size | `8` on CUDA, `2` on CPU |
| Mixed precision | Enabled on CUDA |

The scheduler uses `T_max = epochs` and `eta_min = 1e-6`.

A separate deterministic seed is derived for each experiment from the base seed and experiment identifier using CRC32-based hashing. This makes the experiment-specific initialization reproducible while keeping the two training conditions distinct.

## Experiments

Two training conditions are executed:

| Experiment | Executed objective | Purpose |
| --- | --- | --- |
| `experiment_01_baseline_ce_dice` | Weighted CE + Dice + weighted auxiliary CE | Primary model for test, calibration, and uncertainty analysis |
| `experiment_02_ablation_ce_only` | Weighted CE + weighted auxiliary CE; no Dice | Dice-term ablation |

The source keeps the **Cross-Entropy + Dice** checkpoint as the primary model for subsequent calibration and MC Dropout analysis.

For complete fidelity to the implementation, the two total objectives are:

```math
\mathcal{L}_{baseline}
=
\mathcal{L}_{CE,w}
+
0.6\,\mathcal{L}_{Dice}
+
0.4\,\mathcal{L}_{aux,CE,w}.
```

```math
\mathcal{L}_{ablation}
=
\mathcal{L}_{CE,w}
+
0.4\,\mathcal{L}_{aux,CE,w}.
```

The auxiliary cross-entropy term therefore remains active in the `ce_only` experiment. In the executable implementation, the global class weights are also retained; the ablation removes the Dice term but does **not** produce an unweighted cross-entropy baseline.

Each experiment saves its best checkpoint according to validation mIoU rather than final-epoch performance.

## Evaluation Metrics

### Segmentation quality

The code accumulates an $8 \times 8$ confusion matrix over pixels and derives:

**Pixel accuracy**

```math
\mathrm{Accuracy}
=
\frac{\sum_c TP_c}
{\sum_{i,j} CM_{i,j}}.
```

**Per-class Intersection over Union**

```math
\mathrm{IoU}_c
=
\frac{TP_c}
{TP_c+FP_c+FN_c}.
```

**Mean IoU**

```math
\mathrm{mIoU}
=
\frac{1}{|V|}
\sum_{c\in V}\mathrm{IoU}_c,
```

where $V$ contains classes with non-zero union.

**Per-class Dice**

```math
\mathrm{Dice}_c
=
\frac{2TP_c}
{2TP_c+FP_c+FN_c}.
```

**Mean Dice** is the arithmetic mean of valid per-class Dice values.

All of these segmentation metrics are computed at the **pixel level**, not at the whole-image level. The comparison table also records mean confidence, accuracy among pixels above the median confidence, mean inference time per image, and training time.

### Confidence calibration

The source computes pixel-level **Expected Calibration Error (ECE)** using 15 equal-width confidence bins.

For non-empty bin $b$, the implementation compares empirical accuracy with mean confidence and accumulates the weighted absolute gap:

```math
\mathrm{ECE}
=
\sum_{b=1}^{B}
\frac{n_b}{N}
\left|
\mathrm{acc}_b-\mathrm{conf}_b
\right|,
```

with $B=15$.

The source also records mean confidence and an AURC-like summary by sorting pixels by confidence, computing cumulative error risk, and integrating the resulting risk-coverage curve with the trapezoidal rule.

## Results and Evidence Policy

The provided source does not contain precomputed experimental results. All numerical performance, calibration, uncertainty-alignment, timing, and learned-temperature values are generated during execution. Accordingly, this README deliberately reports the **experimental protocol and output schema**, rather than inventing or freezing numerical findings that are not present in the source artifact.

After execution, the principal numerical evidence is stored in `tables/comparison_table.csv`, `tables/calibration_summary.csv`, `tables/mc_dropout_summary.csv`, and `tables/final_experiment_summary.csv`, with per-class and per-image records in the corresponding `metrics/` files.

## Temperature Scaling

The baseline model is calibrated after training using validation pixels only.

Given logits $z$, the source applies a scalar temperature $T$ before the softmax:

```math
p(c\mid x,T)
=
\operatorname{softmax}\left(\frac{z}{T}\right)_c,
\qquad T>0.
```

The parameterization uses `log_temperature`, so positivity is guaranteed by exponentiation. The optimization problem is cross-entropy minimization on a deterministic sample of at most **100,000 validation pixels**.

The source uses **L-BFGS** with:

- learning rate `0.5`;
- at most `50` iterations;
- strong-Wolfe line search; and
- temperature constrained to `[0.05, 20.0]`.

Because temperature scaling multiplies logits by a positive scalar, it does not alter the ordering of class logits for a fixed pixel. Consequently, the argmax segmentation label is expected to remain unchanged while confidence values may become better or worse calibrated.

The learned temperature is then applied to the held-out test set **without fitting on test labels**.

## Monte Carlo Dropout Uncertainty

For a fixed input, the model is evaluated multiple times with dropout active and BatchNorm layers kept in evaluation mode. The source uses **8 stochastic forward passes** by default.

Let $p_t(c\mid x)$ denote the softmax probability for class $c$ during stochastic pass $t$, and let $T$ denote the number of Monte Carlo samples.

### Predictive mean

```math
\bar{p}(c\mid x)
=
\frac{1}{T}
\sum_{t=1}^{T}p_t(c\mid x).
```

### Predictive entropy

```math
H(\bar{p})
=
-\sum_c
\bar{p}_c
\log(\bar{p}_c+\epsilon),
```

with $\epsilon = 10^{-8}$ in the implementation.

### Expected entropy

```math
\mathbb{E}[H(p)]
=
\frac{1}{T}
\sum_{t=1}^{T}
H(p_t).
```

### Mutual-information-style uncertainty

```math
\mathrm{MI}
=
H(\bar{p})
-
\frac{1}{T}
\sum_{t=1}^{T}H(p_t).
```

The source clamps this quantity to be non-negative and uses it as a **mutual-information-style epistemic uncertainty proxy**.

This should not be interpreted as an exact decomposition of aleatoric and epistemic uncertainty. The code itself explicitly treats MC Dropout as a stochastic approximation.

### Predictive variance

The source also computes the class-wise probability variance across stochastic passes using the population-variance convention (`unbiased=False`) and then averages that variance across classes to obtain a per-pixel scalar uncertainty map.

### Predictive confidence

The per-pixel confidence is the largest class probability under the Monte Carlo predictive mean:

```math
\mathrm{conf}(x)
=
\max_c\bar{p}(c\mid x).
```

## Uncertainty Evaluation and Selective Prediction

The central uncertainty test is whether uncertainty is empirically associated with segmentation error.

For each test pixel, the source records:

- MC Dropout mutual-information-style uncertainty;
- predictive confidence; and
- a binary correctness indicator.

The main association statistic is the **Spearman rank correlation** between uncertainty and pixel error.

The code also compares the observed error rates in the highest and lowest uncertainty deciles:

- top 10% uncertainty: pixels with uncertainty at or above the 90th percentile;
- bottom 10% uncertainty: pixels at or below the 10th percentile.

The reported gap is

```math
\Delta_{err}
=
\mathrm{Error}_{high}
-
\mathrm{Error}_{low}.
```

A positive gap indicates that the high-uncertainty group has a higher observed error rate than the low-uncertainty group.

For selective prediction, the code ranks pixels by **low uncertainty first** and evaluates risk as coverage increases. An AURC-like scalar is obtained by trapezoidal integration of cumulative error risk against coverage.

The source also performs a small MC-sample sensitivity analysis using `4`, `8`, and `12` stochastic passes on at most six images.

## Qualitative Analysis

For selected test images, the project stores visualizations containing:

- input RGB image;
- ground-truth segmentation;
- MC mean prediction;
- binary error map;
- predictive confidence;
- mutual-information-style uncertainty; and
- a confidence/uncertainty visualization overlay.

Predictions and masks use the same SUIM color palette to make class-level discrepancies easier to inspect.

## Output Artifacts

All artifacts are written under:

```text
/content/project_outputs/
├── checkpoints/
├── configs/
├── figures/
├── logs/
├── metrics/
├── predictions/
├── tables/
└── uncertainty_maps/
```

The source produces, among others, the following files:

### Checkpoints

```text
checkpoints/experiment_01_baseline_ce_dice_best.pt
checkpoints/experiment_02_ablation_ce_only_best.pt
```

### Training and evaluation tables

```text
metrics/experiment_01_baseline_ce_dice_training_history.csv
metrics/experiment_02_ablation_ce_only_training_history.csv
metrics/experiment_01_baseline_ce_dice_per_class_iou.npy
metrics/experiment_02_ablation_ce_only_per_class_iou.npy
metrics/experiment_01_baseline_ce_dice_per_class_metrics.csv
metrics/experiment_02_ablation_ce_only_per_class_metrics.csv
metrics/raw_reliability_bins.csv
metrics/temperature_scaled_reliability_bins.csv
metrics/mc_dropout_per_image.csv

tables/comparison_table.csv
tables/calibration_summary.csv
tables/mc_dropout_summary.csv
tables/mc_sample_sensitivity.csv
tables/baseline_per_class_metrics.csv
tables/final_experiment_summary.csv
```

### Figures

```text
figures/dataset_class_distribution.png
figures/dataset_samples.png
figures/training_curves.png
figures/reliability_diagram.png
figures/confidence_histograms.png
figures/uncertainty_vs_error.png
figures/uncertainty_histogram.png
figures/per_class_metrics.png
```

### Qualitative predictions and uncertainty maps

Files are written with names such as:

```text
predictions/sample_000.png
predictions/sample_000_prediction.png
predictions/sample_000_error.png

uncertainty_maps/sample_000_mi.png
uncertainty_maps/sample_000_confidence.png
uncertainty_maps/sample_000_overlay.png
```

The qualitative visualization cell writes **up to four** test examples (`min(4, number of test records)`).

### Reproducibility metadata

```text
configs/experiment_01_baseline_ce_dice.json
configs/experiment_02_ablation_ce_only.json
configs/temperature_scaling.json
configs/reproducibility_record.json
REPORT.md
```

The generated `REPORT.md` summarizes the dataset counts, primary model, Monte Carlo sample count, learned temperature, generated metrics, and limitations.

## Running the Project in Google Colab

The provided source is **Colab-oriented exported notebook code**, not a conventional standalone `.py` command-line application. It contains the IPython magic command `!pip -q install gdown`, which is valid in a notebook cell but is invalid syntax in the standard Python interpreter.

### 1. Prepare the Colab runtime

Use a Google Colab runtime with GPU acceleration when available.

The code automatically selects:

```text
cuda → GPU execution + mixed precision
cpu  → CPU fallback + smaller batch size
```

### 2. Place the dataset archive in `/content`

The active dataset preparation path expects:

```text
/content/archive.zip
```

A fallback searches for another `.zip` already present in `/content` if `archive.zip` is absent.

### 3. Execute the notebook sequentially

The intended order is top-to-bottom:

```text
setup/configuration
→ dataset discovery and validation
→ split construction
→ preprocessing
→ model definition
→ loss functions
→ metrics
→ training
→ test evaluation
→ temperature scaling
→ reliability analysis
→ MC Dropout uncertainty
→ qualitative outputs
→ final summaries
→ reproducibility record
→ packaging
```

The final cell creates:

```text
/content/SUIM_uncertainty_segmentation_results.zip
```

and attempts to trigger a browser download through `google.colab.files.download`.

## Dependencies

The source imports or installs the following Python packages and modules:

- NumPy
- pandas
- Matplotlib
- Pillow
- PyTorch
- TorchVision
- scikit-learn
- tqdm
- `gdown` in the setup cell
- Python standard-library modules including `pathlib`, `hashlib`, `zipfile`, `json`, `random`, `time`, and `zlib`

The source does **not** pin exact package versions. The reproducibility record therefore captures the runtime Python version, PyTorch version, device, and GPU name at execution time.

## Reproducibility Design

The implementation records and/or enforces the following controls:

- fixed base seed: `20260827`;
- Python, NumPy, and PyTorch random seeds;
- CUDA seed initialization when available;
- deterministic cuDNN configuration on CUDA;
- experiment-specific deterministic seeds derived from the experiment identifier;
- deterministic train/validation split construction;
- deterministic calibration-pixel subsampling;
- explicit model/checkpoint metadata;
- training history saved as CSV;
- experiment configuration saved as JSON;
- learned temperature saved separately;
- output-tree verification before packaging; and
- final ZIP packaging of generated artifacts.

The project does **not** claim bitwise identity across every possible Colab hardware/software configuration. GPU kernels and dependency versions can still introduce small numerical differences.
