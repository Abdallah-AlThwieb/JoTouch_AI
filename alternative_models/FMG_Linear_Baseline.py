"""
FMG Linear Baseline Model
=========================
النموذج الخطي (الأقدم) للمقارنة مع JoTouchModel.

هذا أبسط نموذج ممكن لتصنيف الإيماءات من إشارات FMG:
يسطّح النافذة الزمنية كلها في متجهة واحدة ويمررها على طبقة خطية واحدة
(Logistic Regression على الميزات الخام) — بدون أي استخلاص ميزات تلافيفي
أو نمذجة زمنية.

الغرض: خط أساس (Baseline) لإثبات تفوق معمارية JoTouch (MobileNetV2 + TCN)
أمام لجنة التحكيم بأرقام فعلية.

يتبع نفس عقد الواجهة المستخدم في JoTouchModel:
    forward(x) -> (phase_logits, angles_placeholder)
حتى تعمل دوال train_model / evaluate_model / test_latency بدون أي تعديل.
"""

import torch
import torch.nn as nn

# --- path bootstrap: this file now lives in alternative_models/, while the main
# --- JoTouch model and the dataset loaders stay in the project root.
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
import JoTouch_AI_module as cfg  # تُقرأ الثوابت عند إنشاء النموذج وليس عند الاستيراد


class LinearBaselineModel(nn.Module):
    """
    نموذج خطي صِرف: Flatten -> Linear.

    المدخلات:  [batch, TIME_STEPS, NUM_SENSORS]
    المخرجات:  (phase_logits [batch, NUM_PHASES], angles placeholder [batch, 1])
    """

    def __init__(self):
        super().__init__()
        input_dim = cfg.TIME_STEPS * cfg.NUM_SENSORS
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, cfg.NUM_PHASES)

    def forward(self, x):
        phase = self.fc(self.flatten(x))
        angles = torch.zeros(x.shape[0], 1, device=x.device)  # placeholder
        return phase, angles

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters   : {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        return trainable


if __name__ == "__main__":
    model = LinearBaselineModel()
    print("LinearBaselineModel architecture:")
    print(model)
    model.count_parameters()
    dummy = torch.randn(4, cfg.TIME_STEPS, cfg.NUM_SENSORS)
    phase, angles = model(dummy)
    print(f"Input:  {list(dummy.shape)}")
    print(f"Output: phase {list(phase.shape)}, angles {list(angles.shape)}")
