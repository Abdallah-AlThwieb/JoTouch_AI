"""
FMG Standard CNN Baseline Model (older alternative)
===================================================
البديل الأقدم للجزء التلافيفي في معمارية JoTouch:

    JoTouch              -> MobileNetV2 blocks (inverted residual, 2018)
    Standard CNN (هذا)  -> تلافيف قياسية كثيفة (Plain Conv stack) —
                           الجيل الأقدم من الشبكات التلافيفية (قبل كتل
                           الـ Depthwise Separable والـ Residual)

الغرض: يمثل "الحل الأقدم" المقابل لمعماريتنا، لتقف معمارية JoTouch
أمام بدائلها الأقدم بأرقام حقيقية (بدون أي تزييف للنتائج):
نفس البيانات، نفس التقسيمات، نفس بروتوكولات التقييم للجميع.

يتبع نفس عقد الواجهة:
    forward(x) -> (phase_logits, angles_placeholder)
"""

import torch
import torch.nn as nn

# --- path bootstrap: this file now lives in alternative_models/, while the main
# --- JoTouch model and the dataset loaders stay in the project root.
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
import JoTouch_AI_module as cfg


class CNNBaselineModel(nn.Module):
    """
    CNN قياسي أقدم: (Conv1d -> ReLU -> MaxPool) x2 -> FC head.
    بدون BatchNorm، بدون Residual connections، بدون Depthwise Separable.

    المدخلات:  [batch, TIME_STEPS, NUM_SENSORS]
    المخرجات:  (phase_logits [batch, NUM_PHASES], angles placeholder [batch, 1])
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(cfg.NUM_SENSORS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [batch, 32, TIME_STEPS//2]
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # [batch, 64, TIME_STEPS//4]
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, cfg.NUM_PHASES),
        )

    def forward(self, x):
        # x: [batch, TIME_STEPS, NUM_SENSORS] -> [batch, NUM_SENSORS, TIME_STEPS]
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x[:, :, -1]  # آخر خطوة زمنية (نفس أسلوب JoTouchModel)
        phase = self.classifier(x)
        angles = torch.zeros(x.shape[0], 1, device=x.device)  # placeholder
        return phase, angles

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters   : {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        return trainable


if __name__ == "__main__":
    model = CNNBaselineModel()
    print("CNNBaselineModel (standard older CNN) architecture:")
    print(model)
    model.count_parameters()
    dummy = torch.randn(4, cfg.TIME_STEPS, cfg.NUM_SENSORS)
    phase, angles = model(dummy)
    print(f"Input:  {list(dummy.shape)}")
    print(f"Output: phase {list(phase.shape)}, angles {list(angles.shape)}")
