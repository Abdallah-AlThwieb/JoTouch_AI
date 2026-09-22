"""
JoTouch MobileNetV3 Upgrade Candidate (experiment)
==================================================
نسخة تجريبية من نموذج JoTouch نستبدل فيها كتل MobileNetV2 بكتل
MobileNetV3 (inverted residual + Squeeze-Excitation + Hard-Swish)،
مع إبقاء الـ TCN كما هو.

الهدف: الإجابة بأرقام حقيقية على سؤال "هل MobileNetV3 أفضل من
MobileNetV2 في مشروعنا؟" — تُدرَّب هذه النسخة على نفس البيانات
وبنفس بروتوكول JoTouch الأصلي تماماً (AdamW + OneCycleLR +
Early Stopping + Gradient Clipping)، ثم تُقارن بالنموذج الأصلي:
  - إن تفوّقت بوضوح في الدقة/التعميم بدون زيادة مؤذية في الزمن
    الاستجابة أو الحجم -> نرقّي النموذج الرئيسي.
  - وإلا -> نبقي MobileNetV2.

خلفية نظرية: MobileNetV3 (Howard et al., 2019) أحدث من V2 وحقق
تفوقاً على ImageNet، لكن مكاسبه الكبرى جاءت من بحث NAS على نماذج
صور كبيرة؛ على نوافذ FMG الصغيرة (8 قنوات × 20 خطوة) قد تكون
الفائدة هامشية بينما تضيف كتل SE معاملات وزمن استجابة إضافيين.
القرار هنا تجريبي وليس نظرياً.

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
from JoTouch_AI_module import TemporalBlock


class SqueezeExcitation1d(nn.Module):
    """Squeeze-and-Excitation لقنوات 1D (مكوّن MobileNetV3 الأساسي)."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        # x: [batch, C, T]
        s = self.pool(x).squeeze(-1)      # [batch, C]
        s = self.fc(s).unsqueeze(-1)      # [batch, C, 1]
        return x * s


class MobileNetV3Block1d(nn.Module):
    """بلوك MobileNetV3 1D: expand -> depthwise -> SE -> project (+ skip)."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 expansion_factor=2, use_se=True):
        super().__init__()
        padding = kernel_size // 2
        hidden = in_channels * expansion_factor
        self.use_res_connect = in_channels == out_channels

        layers = [
            nn.Conv1d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.Hardswish(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size, padding=padding,
                      groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.Hardswish(inplace=True),
        ]
        if use_se:
            layers.append(SqueezeExcitation1d(hidden))
        layers.extend([
            nn.Conv1d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class JoTouchV3CandidateModel(nn.Module):
    """
    نفس JoTouchModel لكن بكتل MobileNetV3 بدل MobileNetV2.

    المدخلات:  [batch, TIME_STEPS, NUM_SENSORS]
    المخرجات:  (phase_logits [batch, NUM_PHASES], angles placeholder [batch, 1])
    """

    def __init__(self):
        super().__init__()

        # Feature Extraction: MobileNetV3 blocks (بدل InvertedResidual1d)
        self.cnn = nn.Sequential(
            nn.Conv1d(cfg.NUM_SENSORS, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.Hardswish(inplace=True),
            MobileNetV3Block1d(32, 32, kernel_size=3, expansion_factor=2),
            MobileNetV3Block1d(32, 64, kernel_size=3, expansion_factor=2),
            nn.MaxPool1d(2),  # [batch, 64, TIME_STEPS//2]
        )

        # Temporal Modeling: نفس TCN الخاص بـ JoTouchModel بالضبط
        self.tcn = nn.Sequential(
            TemporalBlock(64, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            TemporalBlock(64, 64, kernel_size=3, stride=1, dilation=2, padding=2),
        )

        self.shared = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Dropout(0.3))
        self.phase_head = nn.Linear(64, cfg.NUM_PHASES)

    def forward(self, x):
        # x: [batch, TIME_STEPS, NUM_SENSORS] -> [batch, NUM_SENSORS, TIME_STEPS]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.shared(x)
        phase = self.phase_head(x)
        angles = torch.zeros(x.shape[0], 1, device=x.device)  # placeholder
        return phase, angles

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters   : {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        return trainable


if __name__ == "__main__":
    model = JoTouchV3CandidateModel()
    print("JoTouchV3CandidateModel architecture:")
    print(model)
    model.count_parameters()
    dummy = torch.randn(4, cfg.TIME_STEPS, cfg.NUM_SENSORS)
    phase, angles = model(dummy)
    print(f"Input:  {list(dummy.shape)}")
    print(f"Output: phase {list(phase.shape)}, angles {list(angles.shape)}")
