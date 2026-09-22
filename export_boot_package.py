"""Export JoTouch boot-controller weights into a Teensy deployment package.

Outputs into teensy_boot/:
  - jotouch_boot_zenodo_weights.h : float32 C arrays (all trained parameters)
  - jotouch_boot_plos_weights.h   : float32 C arrays (grip-intensity model)
  - boot_spec.json                : input/output contract + reference stats
  - synergy_table_template.h      : Stage-2 7-motor synergy table template
"""

import json
import os

import numpy as np
import torch

OUT_DIR = "teensy_boot"
os.makedirs(OUT_DIR, exist_ok=True)


def sanitize(name):
    return name.replace(".", "_")


def export_weights_header(pt_path, header_name, guard):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    meta = ck.get("metadata", {})
    lines = [
        f"// Auto-exported from {pt_path} — JoTouch boot controller weights (float32)",
        f"// Target: {meta.get('target', 'n/a')}",
        f"// Test MAE: {meta.get('test_mae', 0):.4f} | R^2: {meta.get('test_r2', 0):.3f}",
        "// Architecture: Conv1d stem -> MobileNetV2 InvertedResidual x2 -> MaxPool",
        "//               -> TCN TemporalBlock x2 -> shared FC -> boot_head",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
    ]
    total = 0
    for name, tensor in sd.items():
        arr = tensor.detach().numpy().astype(np.float32).ravel()
        total += arr.size
        cname = sanitize(name)
        shape = ",".join(str(s) for s in tensor.shape) or "1"
        lines.append(f"// {name} | shape [{shape}]")
        lines.append(f"static const float w_{cname}[{arr.size}] = {{")
        for i in range(0, arr.size, 8):
            lines.append("  " + ", ".join(f"{v:.8e}f" for v in arr[i:i + 8]) + ",")
        lines.append("};")
        lines.append(f"static const int w_{cname}_shape[] = {{{shape}}};")
        lines.append("")
    lines.append(f"// Total parameters: {total}")
    lines.append(f"#endif  // {guard}")
    path = os.path.join(OUT_DIR, header_name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{path}: {total:,} params")
    return total


def write_synergy_template():
    content = """// JoTouch Stage-2 synergy table — TEMPLATE (fill from your hand kinematics)
// motor_cmd[m] = intensity * SYNERGY_TABLE[gesture][m]
// intensity comes from the Stage-1 boot model (0.0 .. 1.0)
// Each row = the 7-motor target posture (normalized 0..1 of each motor's
// full travel) for one gesture at full closure.

#ifndef JOTOUCH_SYNERGY_TABLE_H
#define JOTOUCH_SYNERGY_TABLE_H

#define NUM_MOTORS   7
#define NUM_GESTURES 4   // adjust to your gesture set

// Gestures (example order — match your classifier's class order):
//   0 = REST, 1 = OPEN/WAVE, 2 = PINCH, 3 = POWER GRIP
static const float SYNERGY_TABLE[NUM_GESTURES][NUM_MOTORS] = {
  // M1    M2    M3    M4    M5    M6    M7   <- your 7 motors
  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },  // REST
  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },  // OPEN  (fill: full extension)
  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },  // PINCH (fill: thumb+index posture)
  { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f },  // GRIP  (example: all finger motors close)
};

// The 12-16 finger DOFs follow mechanically from the 7 motors via the
// tendon/linkage coupling (Stage 3) — no software mapping needed.

#endif  // JOTOUCH_SYNERGY_TABLE_H
"""
    path = os.path.join(OUT_DIR, "synergy_table_template.h")
    with open(path, "w") as f:
        f.write(content)
    print(path)


def write_spec():
    spec = {
        "model": "JoTouchBootModel (MobileNetV2-1D + TCN backbone, regression boot head)",
        "input": {
            "channels": 8,
            "time_steps": 20,
            "sampling_hz": 100,
            "window": "20 x 10ms = 200ms, stride 5 (50ms update)",
            "normalization": "per-channel min-max to [0,1]; stats MUST come from "
                             "on-device calibration on the patient's own band",
            "reference_stats": "Content/open_fmg_data/zenodo_norm_stats.npz (reference only)",
        },
        "outputs": {
            "zenodo": {"dim": 2, "meaning": "2-axis continuous position command [0..1] "
                                            "(e.g. wrist up/down, left/right)",
                       "test_mae": 0.0195, "test_r2": 0.917},
            "plos": {"dim": 1, "meaning": "grip closure intensity [0..1]",
                     "test_mae": 0.3052, "test_r2": 0.024,
                     "warning": "weak on open data — prefer gesture head + stepped "
                                "intensity until own-device data exists"},
        },
        "latency_ms_cpu": {"zenodo": 1.43, "plos": 1.61},
        "deployment": "PyTorch .pt -> this C header -> TFLM int8 quantization per "
                      "Content/JoTouch_AI_Analysis.md (س14)",
        "control_chain": [
            "Stage 1: FMG window -> continuous command (these weights)",
            "Stage 2: motor_cmd[7] = intensity * SYNERGY_TABLE[gesture][7]",
            "Stage 3: 7 motors -> 12-16 finger DOFs mechanically (tendons)",
        ],
    }
    path = os.path.join(OUT_DIR, "boot_spec.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print(path)


if __name__ == "__main__":
    export_weights_header(os.path.join("weights", "jotouch_boot_zenodo.pt"), "jotouch_boot_zenodo_weights.h",
                          "JOTOUCH_BOOT_ZENODO_WEIGHTS_H")
    export_weights_header(os.path.join("weights", "jotouch_boot_plos.pt"), "jotouch_boot_plos_weights.h",
                          "JOTOUCH_BOOT_PLOS_WEIGHTS_H")
    write_synergy_template()
    write_spec()
