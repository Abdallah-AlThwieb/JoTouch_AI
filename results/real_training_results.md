# JoTouch Real Training Results (main model)

- **Model**: JoTouchModel (MobileNetV2 + TCN) — MobileNetV2 confirmed over MobileNetV3 by the upgrade evaluation (see model_comparison_results.md)
- **Pipeline (full)**: AdamW + OneCycleLR + gradient clipping + early stopping (patience 8), up to 40 epochs, batch 64, lr=0.001
- **Split per dataset**: 70% train / 15% val / 15% untouched test
- **Device**: cpu

| Dataset | Classes | Epochs trained | Test Accuracy | Latency (ms) | Params | Saved weights |
|---|---|---|---|---|---|---|
| Zenodo MDPI FMG (17 subjects, 5 motions) | 5 (1D_X, 1D_Y, 2D_DG, 2D_DM, 2D_SQ) | 13 | **86.78%** | 3.57 | 65,797 | `jotouch_real_zenodo.pt` |
| Exoskelebox FMG benchmark (20 subjects, 6 gestures) | 6 (closed, extension, flexion, rest, straight, wide) | 40 | **70.92%** | 3.07 | 65,862 | `jotouch_real_exoskelebox.pt` |
| PLoS ONE 2025 FMG (27 participants, 4 gestures) | 4 (Key, Pinch, Power, Tripod) | 12 | **42.61%** | 2.06 | 65,732 | `jotouch_real_plos.pt` |

## Notes

- One model per dataset: the label sets have different meanings across datasets, so merging them into one classifier would be dishonest.
- Input: 8 FMG channels x 20 timesteps (200 ms), min-max normalized per dataset (the loaders set NUM_SENSORS=8 to match the open data; the on-device default of 4 sensors in JoTouch_AI_module.py is unchanged).
- These test accuracies are in-distribution numbers. The cross-subject (new user) estimates remain the LOSO figures in model_comparison_results.md.
