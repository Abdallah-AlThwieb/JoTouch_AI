"""
FMG LSTM + MobileNetV1 Baseline Model (older alternative to JoTouch)
====================================================================
البديل الأقدم المباشر لمعمارية JoTouch — نفس فكرة "مستخلص ميزات تلافيفي
+ نمذجة زمنية" لكن بمكوّنين من الجيل الأقدم:

    JoTouch        -> MobileNetV2 blocks (inverted residual, 2018) + TCN
    هذا النموذج    -> MobileNetV1 blocks (depthwise separable conv,
                      Howard et al. 2017 — الإصدار الأقدم مباشرة من
                      MobileNetV2، بدون inverted residual وبدون linear
                      bottleneck) + LSTM (Hochreiter & Schmidhuber 1997 —
                      الخلية التكرارية الأقدم من عائلة RNN، والبديل الأقدم
                      للـ TCN في النمذجة الزمنية)

ملاحظة تصحيحية: LSTM/GRU ليستا بديلين عن MobileNetV2 — فهما طبقات
زمنية (Temporal) وليستا تلافيفية؛ بدائل MobileNetV2 هي كتل تلافيفية
أخرى (MobileNetV1 الأقدم، MobileNetV3 الأحدث). أما LSTM وGRU فهما
بديلان عن الـ TCN — وكلتاهما من عائلة RNN، وLSTM أقدم من GRU.

الغرض: مقارنة عادلة وصادقة — نفس البيانات ونفس البروتوكولات للجميع،
لتقف معماريتنا أمام بدائلها الأقدم بأرقام حقيقية.

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


class DepthwiseSeparableConv1d(nn.Module):
    """
    بلوك MobileNetV1 الأصلي: Depthwise Conv -> BN -> ReLU ->
    Pointwise Conv (1x1) -> BN -> ReLU.
    (بدون اتصال residual وبدون expansion — هذان جاءا لاحقاً في V2)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class RNNMobileNetV1BaselineModel(nn.Module):
    """
    MobileNetV1-1D (استخلاص ميزات) + LSTM (نمذجة زمنية) -> رأس تصنيف.

    المدخلات:  [batch, TIME_STEPS, NUM_SENSORS]
    المخرجات:  (phase_logits [batch, NUM_PHASES], angles placeholder [batch, 1])
    """

    def __init__(self):
        super().__init__()

        # استخلاص الميزات: كتل MobileNetV1 (الإصدار الأقدم من MobileNetV2)
        self.cnn = nn.Sequential(
            nn.Conv1d(cfg.NUM_SENSORS, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv1d(32, 32),
            DepthwiseSeparableConv1d(32, 64),
            nn.MaxPool1d(2),  # [batch, 64, TIME_STEPS//2]
        )

        # النمذجة الزمنية: LSTM (الخلية التكرارية الأقدم، البديل الأقدم للـ TCN)
        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.shared = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.phase_head = nn.Linear(64, cfg.NUM_PHASES)

    def forward(self, x):
        # x: [batch, TIME_STEPS, NUM_SENSORS] -> [batch, NUM_SENSORS, TIME_STEPS]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # -> [batch, TIME_STEPS//2, 64] للـ RNN
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        x = out[:, -1, :]            # آخر خطوة زمنية (نفس أسلوب JoTouchModel)
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
    model = RNNMobileNetV1BaselineModel()
    print("RNNMobileNetV1BaselineModel architecture:")
    print(model)
    model.count_parameters()
    dummy = torch.randn(4, cfg.TIME_STEPS, cfg.NUM_SENSORS)
    phase, angles = model(dummy)
    print(f"Input:  {list(dummy.shape)}")
    print(f"Output: phase {list(phase.shape)}, angles {list(angles.shape)}")
