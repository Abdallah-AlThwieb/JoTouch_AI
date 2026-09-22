# JoTouch Model Comparison Results

- **Training budget (identical within each pipeline)**: 12 epochs, batch 64, lr=0.001
- **Alternative-models pipeline**: plain Adam only — no OneCycleLR, no early stopping, no gradient clipping
- **JoTouch pipeline (ours)**: AdamW + OneCycleLR + early stopping (patience 4) + gradient clipping
- **Device**: cpu

## Dataset: Zenodo MDPI FMG (17 subjects, classification + regression)

- **Source**: Zenodo 6632020 (Zakia & Menon, MDPI Data 2022) - real FMG, first 8 of 32 channels, biaxial-stage motions
- **Windows**: 12,500 | **Subjects**: 17 | **Classes**: 5 (1D_X, 1D_Y, 2D_DG, 2D_DM, 2D_SQ)

### Classification

| Model | Params | Size (KB) | Random-split Acc | Cross-Subject Acc (LOSO mean±std) | Generalization Gap | Latency (ms) |
|---|---|---|---|---|---|---|
| Linear (oldest) | 805 | 3.1 | 47.24% | 30.09% ± 3.87% | 17.15 pts | 0.03 |
| Standard CNN (older alternative) | 11,493 | 44.9 | 75.84% | 34.91% ± 3.08% | 40.93 pts | 0.27 |
| LSTM + MobileNetV1 (older alternative) | 42,181 | 164.8 | 92.04% | 34.47% ± 4.32% | 57.57 pts | 1.19 |
| JoTouch (ours) | 65,797 | 257.0 | 96.76% | 35.38% ± 9.64% | 61.38 pts | 1.48 |
| JoTouch-V3 (upgrade candidate) | 70,053 | 273.6 | 96.80% | 35.71% ± 7.31% | 61.09 pts | 1.50 |

#### Per-class accuracy (cross-subject, aggregated over folds)

| Class | Linear (oldest) | Standard CNN (older alternative) | LSTM + MobileNetV1 (older alternative) | JoTouch (ours) | JoTouch-V3 (upgrade candidate) |
|---|---|---|---|---|---|
| 1D_X | 41.1% | 48.8% | 48.6% | 47.4% | 46.3% |
| 1D_Y | 47.2% | 37.4% | 35.9% | 36.4% | 42.5% |
| 2D_DG | 23.6% | 41.2% | 42.7% | 38.5% | 38.0% |
| 2D_DM | 33.2% | 36.8% | 33.6% | 32.8% | 29.7% |
| 2D_SQ | 4.7% | 13.4% | 16.0% | 22.8% | 23.2% |

### Regression (continuous targets)

| Model | Random-split MAE | Random-split R² | Cross-Subject MAE (LOSO mean±std) | Cross-Subject R² |
|---|---|---|---|---|
| Linear (oldest) | 0.0722 | 0.225 | 0.0999 ± 0.0294 | -0.561 ± 0.656 |
| Standard CNN (older alternative) | 0.0590 | 0.447 | 0.0829 ± 0.0110 | -0.134 ± 0.168 |
| LSTM + MobileNetV1 (older alternative) | 0.0349 | 0.808 | 0.0886 ± 0.0008 | -0.303 ± 0.103 |
| JoTouch (ours) | 0.0297 | 0.847 | 0.0781 ± 0.0088 | 0.026 ± 0.119 |
| JoTouch-V3 (upgrade candidate) | 0.0321 | 0.823 | 0.0770 ± 0.0110 | 0.048 ± 0.167 |

## Dataset: Exoskelebox FMG benchmark (20 subjects, 8 FSR, 6 gestures)

- **Source**: Exoskelebox FMG benchmark (arXiv:2007.14918) - 8 forearm FSR, 20 subjects, 6 gestures x 3 wrist orientations, ~100Hz after x5 downsample
- **Windows**: 15,000 | **Subjects**: 20 | **Classes**: 6 (closed, extension, flexion, rest, straight, wide)

### Classification

| Model | Params | Size (KB) | Random-split Acc | Cross-Subject Acc (LOSO mean±std) | Generalization Gap | Latency (ms) |
|---|---|---|---|---|---|---|
| Linear (oldest) | 966 | 3.8 | 29.10% | 19.56% ± 3.52% | 9.54 pts | 0.06 |
| Standard CNN (older alternative) | 11,558 | 45.1 | 46.43% | 25.20% ± 3.21% | 21.23 pts | 0.45 |
| LSTM + MobileNetV1 (older alternative) | 42,246 | 165.0 | 59.03% | 24.79% ± 4.12% | 34.24 pts | 1.97 |
| JoTouch (ours) | 65,862 | 257.3 | 64.47% | 21.23% ± 2.72% | 43.24 pts | 2.44 |
| JoTouch-V3 (upgrade candidate) | 70,118 | 273.9 | 63.27% | 21.76% ± 2.55% | 41.50 pts | 1.78 |

#### Per-class accuracy (cross-subject, aggregated over folds)

| Class | Linear (oldest) | Standard CNN (older alternative) | LSTM + MobileNetV1 (older alternative) | JoTouch (ours) | JoTouch-V3 (upgrade candidate) |
|---|---|---|---|---|---|
| closed | 37.1% | 54.5% | 46.8% | 22.5% | 27.9% |
| extension | 14.7% | 39.2% | 45.1% | 45.2% | 40.4% |
| flexion | 17.3% | 10.6% | 12.0% | 13.0% | 13.1% |
| rest | 11.8% | 16.4% | 12.6% | 6.5% | 6.6% |
| straight | 19.6% | 14.1% | 9.7% | 16.7% | 14.8% |
| wide | 18.9% | 18.0% | 24.8% | 25.0% | 29.2% |

## Dataset: PLoS ONE 2025 FMG (27 participants, 4 gestures + load regression)

- **Source**: PLoS ONE 2025 (Young et al., e0321319) - Zenodo 15420178, 27 participants, 8 FMG ch @100Hz after x20 downsample, 4 gestures x 5 loads; regression target = grasped load (normalized)
- **Windows**: 10,000 | **Subjects**: 27 | **Classes**: 4 (Key, Pinch, Power, Tripod)

### Classification

| Model | Params | Size (KB) | Random-split Acc | Cross-Subject Acc (LOSO mean±std) | Generalization Gap | Latency (ms) |
|---|---|---|---|---|---|---|
| Linear (oldest) | 644 | 2.5 | 32.90% | 27.62% ± 2.12% | 5.28 pts | 0.04 |
| Standard CNN (older alternative) | 11,428 | 44.6 | 39.40% | 28.83% ± 1.08% | 10.57 pts | 0.25 |
| LSTM + MobileNetV1 (older alternative) | 42,116 | 164.5 | 49.40% | 28.46% ± 1.10% | 20.94 pts | 1.23 |
| JoTouch (ours) | 65,732 | 256.8 | 50.65% | 29.36% ± 1.45% | 21.29 pts | 1.37 |
| JoTouch-V3 (upgrade candidate) | 69,988 | 273.4 | 51.10% | 28.55% ± 0.44% | 22.55 pts | 1.73 |

#### Per-class accuracy (cross-subject, aggregated over folds)

| Class | Linear (oldest) | Standard CNN (older alternative) | LSTM + MobileNetV1 (older alternative) | JoTouch (ours) | JoTouch-V3 (upgrade candidate) |
|---|---|---|---|---|---|
| Key | 34.2% | 49.3% | 44.3% | 41.7% | 40.1% |
| Pinch | 24.5% | 26.0% | 23.2% | 19.4% | 18.5% |
| Power | 17.0% | 20.7% | 21.9% | 37.7% | 24.3% |
| Tripod | 34.7% | 19.4% | 24.5% | 18.5% | 31.2% |

### Regression (continuous targets)

| Model | Random-split MAE | Random-split R² | Cross-Subject MAE (LOSO mean±std) | Cross-Subject R² |
|---|---|---|---|---|
| Linear (oldest) | 0.3044 | 0.009 | 0.3272 ± 0.0160 | -0.260 ± 0.171 |
| Standard CNN (older alternative) | 0.2997 | 0.033 | 0.3030 ± 0.0044 | -0.018 ± 0.031 |
| LSTM + MobileNetV1 (older alternative) | 0.2978 | 0.053 | 0.3182 ± 0.0124 | -0.160 ± 0.131 |
| JoTouch (ours) | 0.2985 | 0.049 | 0.3058 ± 0.0001 | -0.024 ± 0.009 |
| JoTouch-V3 (upgrade candidate) | 0.2975 | 0.048 | 0.3051 ± 0.0017 | -0.018 ± 0.019 |

## MobileNetV3 upgrade evaluation (should JoTouch replace MobileNetV2 with MobileNetV3?)

JoTouch (ours, MobileNetV2 + TCN) vs JoTouch-V3 candidate (MobileNetV3 + TCN), trained with the identical JoTouch pipeline on identical splits:

| Dataset | MBV2 Cross-Subject Acc | V3 Cross-Subject Acc | Δ Acc | MBV2 Latency (ms) | V3 Latency (ms) | MBV2 Params | V3 Params |
|---|---|---|---|---|---|---|---|
| Zenodo MDPI FMG (17 subjects, classification + regression) | 35.38% ± 9.64% | 35.71% ± 7.31% | +0.32 pts | 1.48 | 1.50 | 65,797 | 70,053 |
| Exoskelebox FMG benchmark (20 subjects, 8 FSR, 6 gestures) | 21.23% ± 2.72% | 21.76% ± 2.55% | +0.54 pts | 2.44 | 1.78 | 65,862 | 70,118 |
| PLoS ONE 2025 FMG (27 participants, 4 gestures + load regression) | 29.36% ± 1.45% | 28.55% ± 0.44% | -0.81 pts | 1.37 | 1.73 | 65,732 | 69,988 |

**Decision: KEEP MobileNetV2.** MobileNetV3 shows no meaningful gain (mean Δ = +0.02 pts across 3 datasets) while adding SE-block parameters and latency — the theoretical ImageNet advantage does not materialize on 8-channel FMG windows, so the simpler V2 stays.

## Metrics explained

- **Params / Size (KB)**: number of trainable weights and approximate float32 model size — matters for embedding on microcontrollers (flash/RAM limits).
- **Random-split Acc**: accuracy on a random 80/20 window split; windows from the same trial can appear in both sets — the optimistic, in-session number.
- **Cross-Subject Acc (LOSO)**: leave-subjects-out cross-validation — each fold excludes whole subjects from training and tests only on them; mean ± std across folds. The realistic 'new user' number and the key generalization evidence.
- **Generalization Gap**: random-split accuracy minus cross-subject accuracy; smaller = less overfitting.
- **Latency (ms)**: mean of 100 single-window inference passes after a 10-pass warm-up (same method as `test_latency` in JoTouch_AI_module.py); must stay well under the 10 ms real-time budget.
- **MAE / RMSE**: mean absolute / root-mean-square error of the continuous targets (normalized 0–1); lower is better.
- **R²**: coefficient of determination; 1.0 = perfect prediction, 0 = no better than predicting the mean, negative = worse than the mean.
