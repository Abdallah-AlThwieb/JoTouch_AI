# alternative_models/

Everything in this folder exists **only to justify the main model by comparison**.
The main model — `JoTouch_AI_module.py` (JoTouchModel: MobileNetV2-1D + TCN) — and its
own training, dataset and deployment pipeline stay in the project root.

## Contents

| File | What it is |
|---|---|
| `FMG_Linear_Baseline.py` | Oldest possible baseline: Flatten -> Linear (logistic regression on raw windows). |
| `FMG_CNN_Baseline.py` | Older alternative: plain dense Conv1d stack (no BatchNorm / residual / depthwise). |
| `FMG_RNN_MobileNetV1_Baseline.py` | Older direct alternative: MobileNetV1 blocks + LSTM (the previous generation of both halves of our architecture). |
| `JoTouch_MobileNetV3_candidate.py` | Newer *upgrade candidate*: MobileNetV3 blocks (SE + Hard-Swish) + the same TCN. |
| `model_comparison.py` | The benchmark harness that trains all of the above plus JoTouchModel on identical data/splits and writes the report. |
| `model_comparison_results.md` / `.pdf` | The resulting report (accuracy, LOSO cross-subject, params, latency, MobileNetV3 decision). |
| `benchmark_checkpoint.json` | Resume checkpoint for the harness, so a long benchmark run can be continued. |

All model files follow the same interface contract as `JoTouchModel`:
`forward(x) -> (phase_logits, angles)`, so the shared train / evaluate / latency
helpers work on them unchanged.

## Outcome

**MobileNetV2 stays.** The MobileNetV3 candidate gave a mean change of **+0.02 points**
cross-subject accuracy across three datasets while adding ~4.3K parameters and SE-block
latency, so the simpler V2 backbone was kept in the main model.

## How to run

The files import `JoTouch_AI_module` and the `fmg_*_dataset` loaders from the project
root; each one inserts the project root into `sys.path` at import time, so this works
from anywhere:

```bash
python alternative_models/model_comparison.py
```

The harness `chdir`s to the project root at start (the dataset loaders resolve
`Content/open_fmg_data` relative to the working directory) and writes
`benchmark_checkpoint.json` and `model_comparison_results.md` back into this folder.

To re-render the PDF:

```bash
python report_to_pdf.py alternative_models/model_comparison_results.md
```
