# JoTouch Boot Controller Training Results (generalized FMG -> commands)

- **Purpose**: generalized sensor->actuator relationship for the Teensy boot controller, until per-patient calibration replaces the weights
- **Pipeline**: AdamW + OneCycleLR + gradient clipping + early stopping (patience 8), up to 40 epochs, batch 64
- **Split**: 70% train / 15% val / 15% untouched test

| Boot model | Output (target) | Epochs | Test MAE | Test RMSE | Test R² | Latency (ms) | Weights |
|---|---|---|---|---|---|---|---|
| zenodo | stage position (X, Y) normalized - 2-axis continuous control | 40 | 0.0195 | 0.0307 | 0.917 | 1.43 | `jotouch_boot_zenodo.pt` |
| plos | grasped load [0..1] - grip closure intensity | 17 | 0.3052 | 0.3524 | 0.024 | 1.61 | `jotouch_boot_plos.pt` |

## Control chain on Teensy

1. **Stage 1 (these weights)**: FMG window [8 ch x 20] -> continuous command(s).
2. **Stage 2 (synergy table, hand-designed)**: `motor_cmd[7] = intensity * Posture_g[7]`.
3. **Stage 3 (mechanical)**: 7 motors drive 12-16 finger DOFs via tendons/linkages.

**Input contract**: 8 FMG channels x 20 timesteps @100 Hz, min-max normalized per channel. Normalization stats MUST come from the device calibration step on the patient's own band (open-dataset stats in Content/open_fmg_data/*_norm_stats.npz are reference only).
