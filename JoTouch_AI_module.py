import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

# =========================
# CONFIG
# =========================
TIME_STEPS = 20  # عدد الخطوات الزمنية في كل نافذة (20 × 10ms = 200ms window)
NUM_SENSORS = 8  # 8 FSR sensors (full forearm band)
NUM_DOF = 16     # عدد زوايا المفاصل التي يخرجها رأس الانحدار
                 # 16 = نفس العدد الذي يجمعه data_collection.py عبر MediaPipe،
                 # فهو مصدر التسميات (labels) الحقيقي لهذا الرأس.
NUM_PHASES = 4  # 0=rest, 1=wave, 2=pinch, 3=grip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GESTURE_LABELS = {'rest': 0, 'wave': 1, 'pinch': 2, 'grip': 3}
# تجارب مستبعدة يدوياً من جلسة جمع البيانات القديمة (4 حساسات).
# عند جمع بيانات جديدة بشريط الـ 8 حساسات يجب إعادة تقييم هذه القائمة.
FLAGGED_TRIALS = {'rest': [11, 13, 14], 'wave': [3], 'pinch': [5]}
WINDOW_STRIDE = 5


# =========================
# DATASET
# =========================
class FMGDataset(Dataset):
    """
    Dataset لبيانات حساسات FMG (Force Myography).

    كل عينة تتكون من:
      - X      : [TIME_STEPS, NUM_SENSORS]  - نافذة زمنية من قراءات الحساسات
      - y_phase: int                         - الإيماءة (0=rest, 1=wave, 2=pinch, 3=grip)
      - y_angle: [NUM_DOF]                   - زوايا المفاصل المستهدفة بالراديان

    ملاحظة مهمة حول y_angle:
      ليست كل مجموعة بيانات تحتوي على زوايا مفاصل حقيقية. مجموعات البيانات
      المفتوحة (Zenodo / Exoskelebox / PLoS) تعطي تسميات إيماءات فقط، فتُمرَّر
      زوايا وهمية (placeholder). في تلك الحالة تُضبط self.has_angles = False
      ويتجاهل التدريب خسارة الانحدار تماماً — وإلا لكنّا ندرّب رأس الزوايا على
      إخراج أصفار، وهو أسوأ من عدم تدريبه.

    يمكن تحميل البيانات من ملف .npz أو إنشاؤها اصطناعياً عبر generate_synthetic_data().
    """

    def __init__(self, X: np.ndarray, y_phase: np.ndarray, y_angle: np.ndarray,
                 has_angles: bool = None):
        assert X.shape[1:] == (
            TIME_STEPS,
            NUM_SENSORS,
        ), f"Expected X shape (N, {TIME_STEPS}, {NUM_SENSORS}), got {X.shape}"
        assert y_phase.shape[0] == X.shape[0]

        y_angle = np.asarray(y_angle, dtype=np.float32)
        if y_angle.ndim == 1:
            y_angle = y_angle.reshape(len(X), -1)

        # تسميات زوايا حقيقية فقط إذا كان العرض = NUM_DOF وفيها تباين فعلي
        real = (y_angle.shape == (len(X), NUM_DOF)) and bool(np.any(y_angle))
        self.has_angles = real if has_angles is None else bool(has_angles)

        if y_angle.shape != (len(X), NUM_DOF):
            # placeholder — نوسّعها إلى العرض الصحيح حتى تبقى الأبعاد متسقة
            y_angle = np.zeros((len(X), NUM_DOF), dtype=np.float32)
            self.has_angles = False

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_phase = torch.tensor(y_phase, dtype=torch.long)
        self.y_angle = torch.tensor(y_angle, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_phase[idx], self.y_angle[idx]

    @classmethod
    def from_file(cls, path: str):
        """تحميل dataset محفوظ مسبقاً من ملف .npz"""
        data = np.load(path)
        return cls(data["X"], data["y_phase"], data["y_angle"])

    @staticmethod
    def dataset_has_angles(loader_or_ds) -> bool:
        """يستخرج راية has_angles من DataLoader أو Subset أو Dataset."""
        ds = getattr(loader_or_ds, "dataset", loader_or_ds)
        while hasattr(ds, "dataset"):          # random_split -> Subset
            ds = ds.dataset
        return bool(getattr(ds, "has_angles", False))

    def save(self, path: str):
        """حفظ dataset إلى ملف .npz"""
        np.savez(
            path,
            X=self.X.numpy(),
            y_phase=self.y_phase.numpy(),
            y_angle=self.y_angle.numpy(),
        )
        print(f"Dataset saved to {path}")


# =========================
# SYNTHETIC DATA GENERATOR
# =========================
def generate_synthetic_data(
    num_samples: int = 2000, noise_level: float = 0.05
) -> FMGDataset:
    """
    ينشئ بيانات FMG اصطناعية واقعية لاختبار وتدريب النموذج قبل توفر الحساسات الحقيقية.

    منطق التوليد:
      - Rest  (Phase 0) : إشارات منخفضة ومستقرة        -> زوايا صفرية (يد مفتوحة)
      - Wave  (Phase 1) : إشارات متوسطة بانحدار تدريجي -> زوايا جزئية
      - Pinch (Phase 2) : ذروات في عضلات الانثناء       -> زوايا قبضة دقيقة
      - Grip  (Phase 3) : إشارات مرتفعة على كل القنوات  -> زوايا قبضة كاملة

    تُولَّد هنا زوايا مفاصل حقيقية (وليست أصفاراً)، لذا فهذه البيانات صالحة
    لتدريب رأس الانحدار واختباره قبل توفر بيانات الكاميرا الحقيقية.
    """
    X_list, y_phase_list, y_angle_list = [], [], []

    # توزيع العينات بالتساوي وإضافة الباقي للفئة الأخيرة
    samples_per_class = num_samples // NUM_PHASES
    remainder = num_samples - samples_per_class * NUM_PHASES

    for phase in range(NUM_PHASES):
        count = samples_per_class + (remainder if phase == NUM_PHASES - 1 else 0)
        for _ in range(count):

            if phase == 0:  # Rest
                base_signal = np.random.uniform(0.05, 0.15, (TIME_STEPS, NUM_SENSORS))
                target_angles = np.random.uniform(0.0, 0.1, NUM_DOF)

            elif phase == 1:  # Wave
                base = np.random.uniform(0.3, 0.5, NUM_SENSORS)
                # حركة تدريجية عبر النافذة الزمنية
                ramp = np.linspace(0.1, 1.0, TIME_STEPS).reshape(-1, 1)
                base_signal = ramp * base + np.random.uniform(
                    0.0, 0.1, (TIME_STEPS, NUM_SENSORS)
                )
                target_angles = np.random.uniform(0.5, 1.0, NUM_DOF)

            elif phase == 2:  # Pinch
                # ذروات في حساسات عضلات الانثناء (flexor sensors: 0,1,2,3)
                base_signal = np.random.uniform(0.1, 0.2, (TIME_STEPS, NUM_SENSORS))
                flexor_activation = np.random.uniform(0.7, 1.0, min(4, NUM_SENSORS))
                base_signal[:, :min(4, NUM_SENSORS)] += flexor_activation
                base_signal = np.clip(base_signal, 0, 1)
                target_angles = np.random.uniform(1.2, 1.8, NUM_DOF)

            else:  # Grip (Phase 3)
                base_signal = np.random.uniform(0.5, 0.9, (TIME_STEPS, NUM_SENSORS))
                base_signal = np.clip(base_signal, 0, 1)
                target_angles = np.random.uniform(1.6, 2.2, NUM_DOF)

            noise = np.random.normal(0, noise_level, base_signal.shape)
            base_signal = np.clip(base_signal + noise, 0, 1)

            X_list.append(base_signal)
            y_phase_list.append(phase)
            y_angle_list.append(target_angles)

    X = np.array(X_list, dtype=np.float32)
    y_phase = np.array(y_phase_list, dtype=np.int64)
    y_angle = np.array(y_angle_list, dtype=np.float32)

    # خلط البيانات
    idx = np.random.permutation(len(X))
    return FMGDataset(X[idx], y_phase[idx], y_angle[idx], has_angles=True)


def make_dataloaders(dataset: FMGDataset, batch_size: int = 32, val_split: float = 0.2):
    """تقسيم dataset إلى train/val وإنشاء DataLoaders."""
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(
        f"Train samples: {train_size} | Val samples: {val_size} | Batch size: {batch_size}"
    )
    return train_loader, val_loader


# =========================
# Inverted Residual Block (MobileNetV2 style)
# =========================
class InvertedResidual1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, expansion_factor=2
    ):
        super().__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_res_connect = in_channels == out_channels

        layers = []
        if expansion_factor != 1:
            layers.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                ]
            )

        layers.extend(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# =========================
# Temporal Convolutional Network (TCN) Block
# =========================
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.silu1 = nn.SiLU()
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.silu2 = nn.SiLU()

        self.net = nn.Sequential(self.conv1, self.silu1, self.conv2, self.silu2)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.silu(out + res)


# =========================
# JoTouch Model (MobileNetV2 blocks + TCN + Dual Head)
# =========================
class JoTouchModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature Extraction: Inverted Residuals
        self.cnn = nn.Sequential(
            nn.Conv1d(NUM_SENSORS, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            InvertedResidual1d(32, 32, kernel_size=3, padding=1, expansion_factor=2),
            InvertedResidual1d(32, 64, kernel_size=3, padding=1, expansion_factor=2),
            nn.MaxPool1d(2),  # [batch, 64, TIME_STEPS//2]
        )

        # Temporal Modeling: TCN
        self.tcn = nn.Sequential(
            TemporalBlock(64, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            TemporalBlock(64, 64, kernel_size=3, stride=1, dilation=2, padding=2),
        )

        # Shared representation layer
        self.shared = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Dropout(0.3))

        # Output heads — رأسان فعليان (تصنيف + انحدار)
        self.phase_head = nn.Linear(64, NUM_PHASES)  # الإيماءة المقصودة
        self.angle_head = nn.Linear(64, NUM_DOF)     # زوايا المفاصل المستمرة

    def forward(self, x):
        # x: [batch, TIME_STEPS, NUM_SENSORS] -> [batch, NUM_SENSORS, TIME_STEPS]
        x = x.permute(0, 2, 1)

        x = self.cnn(x)
        x = self.tcn(x)
        x = x[:, :, -1]  # Take last time step: [batch, 64]

        x = self.shared(x)

        phase = self.phase_head(x)
        angles = self.angle_head(x)

        return phase, angles

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters   : {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        return trainable


# =========================
# SAVE & LOAD
# =========================
def save_model(model: JoTouchModel, path: str, metadata: dict = None):
    """
    حفظ أوزان النموذج مع معلومات إضافية (epoch, loss, ...).
    الملف المحفوظ يحتوي على:
      - model_state_dict : أوزان النموذج
      - config           : إعدادات النموذج الثابتة
      - metadata         : معلومات التدريب
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "TIME_STEPS": TIME_STEPS,
            "NUM_SENSORS": NUM_SENSORS,
            "NUM_DOF": NUM_DOF,
            "NUM_PHASES": NUM_PHASES,
        },
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: str = DEVICE) -> JoTouchModel:
    """تحميل نموذج محفوظ مسبقاً وإعادته جاهزاً للاستخدام."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = JoTouchModel().to(device)

    state = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any(k.startswith("angle_head") for k in missing):
        print(
            "\n  ** WARNING ** هذا الملف حُفظ قبل تفعيل رأس الانحدار (angle_head).\n"
            "     أوزان رأس الزوايا عشوائية الآن، لذلك مخرجات الزوايا بلا معنى\n"
            "     حتى يُعاد تدريب النموذج. التصنيف (phase) يعمل بشكل صحيح.\n"
        )
    elif missing or unexpected:
        print(f"  Note: missing={missing} unexpected={unexpected}")
    print(f"Model loaded from {path}")
    if checkpoint["metadata"]:
        print(f"  Metadata: {checkpoint['metadata']}")
    return model


# =========================
# TRAINING
# =========================
def train_model(
    model: JoTouchModel,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 40,
    lr: float = 0.001,
    save_path: str = None,
    patience: int = 8,
    angle_loss_weight: float = 0.7,
    use_angle_loss: bool = None,
) -> dict:
    """
    دالة التدريب الكاملة مع:
      - AdamW optimizer
      - OneCycleLR scheduler (سرعة تقارب عالية)
      - Gradient clipping (استقرار التدريب)
      - حفظ أفضل نموذج تلقائياً إذا تم تحديد save_path
      - Early stopping: يوقف التدريب إذا لم يتحسن النموذج لـ patience epochs
      - خسارة مزدوجة: تصنيف (CrossEntropy) + انحدار الزوايا (MSE)

    use_angle_loss: إذا تُركت None تُستنتج تلقائياً من البيانات — تُفعَّل فقط
    عندما تحمل المجموعة زوايا مفاصل حقيقية (has_angles=True). هذا يمنع تدريب
    رأس الزوايا على أصفار وهمية عند استخدام مجموعات البيانات المفتوحة.
    """
    model.to(DEVICE)

    if use_angle_loss is None:
        use_angle_loss = FMGDataset.dataset_has_angles(train_loader)
    print(f"  Angle-regression loss: {'ON' if use_angle_loss else 'OFF (no angle labels in this dataset)'}")

    loss_phase = nn.CrossEntropyLoss()
    loss_angle = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 5, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0.0

        for X, y_phase, y_angle in train_loader:
            X, y_phase, y_angle = X.to(DEVICE), y_phase.to(DEVICE), y_angle.to(DEVICE)

            pred_phase, pred_angle = model(X)

            loss = loss_phase(pred_phase, y_phase)
            if use_angle_loss:
                loss = loss + angle_loss_weight * loss_angle(pred_angle, y_angle)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # --- Validation ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y_phase, y_angle in val_loader:
                    X, y_phase, y_angle = (
                        X.to(DEVICE),
                        y_phase.to(DEVICE),
                        y_angle.to(DEVICE),
                    )
                    pred_phase, pred_angle = model(X)
                    batch_loss = loss_phase(pred_phase, y_phase)
                    if use_angle_loss:
                        batch_loss = batch_loss + angle_loss_weight * loss_angle(
                            pred_angle, y_angle
                        )
                    val_loss += batch_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            # حفظ أفضل نموذج وتتبع Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                if save_path:
                    save_model(
                        model,
                        save_path,
                        metadata={"epoch": epoch + 1, "val_loss": avg_val_loss},
                    )
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} [best]"
                )
            else:
                epochs_no_improve += 1
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} ({epochs_no_improve}/{patience})"
                )

            # Early Stopping
            if patience and epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)."
                )
                break
        else:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.4f}")

    return history


# =========================
# EVALUATION
# =========================
def evaluate_model(model: JoTouchModel, data_loader: DataLoader,
                   return_angle_mae: bool = False):
    """
    تقييم النموذج: دقة تصنيف الإيماءة + متوسط خطأ الزوايا (MAE).

    القيمة المعادة تبقى phase_acc وحدها افتراضياً (حفاظاً على توافق الكود
    القائم). مرّر return_angle_mae=True للحصول على (phase_acc, angle_mae)؛
    تكون angle_mae مساوية None إذا لم تكن للبيانات زوايا حقيقية.
    """
    model.eval()
    correct = 0
    total = 0
    angle_error = 0.0
    has_angles = FMGDataset.dataset_has_angles(data_loader)

    with torch.no_grad():
        for X, y_phase, y_angle in data_loader:
            X, y_phase, y_angle = X.to(DEVICE), y_phase.to(DEVICE), y_angle.to(DEVICE)

            pred_phase, pred_angle = model(X)

            predicted = torch.argmax(pred_phase, dim=1)
            correct += (predicted == y_phase).sum().item()
            total += y_phase.size(0)

            if has_angles:
                angle_error += torch.sum(torch.abs(pred_angle - y_angle)).item()

    phase_acc = 100.0 * correct / total
    angle_mae = (angle_error / (total * NUM_DOF)) if has_angles else None

    print(f"Phase Accuracy : {phase_acc:.2f}%")
    if angle_mae is not None:
        print(f"Angle MAE      : {angle_mae:.4f} rad ({np.degrees(angle_mae):.2f} deg)")
    else:
        print("Angle MAE      : n/a (لا توجد زوايا مفاصل حقيقية في هذه البيانات)")

    return (phase_acc, angle_mae) if return_angle_mae else phase_acc


# =========================
# LATENCY TEST
# =========================
def test_latency(model: JoTouchModel, n_runs: int = 100):
    """قياس متوسط زمن الاستجابة على DEVICE الحالي."""
    dummy = torch.randn(1, TIME_STEPS, NUM_SENSORS).to(DEVICE)
    model.eval()

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            model(dummy)

    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    elapsed = (time.time() - start) / n_runs * 1000

    print(f"Average inference time ({DEVICE}): {elapsed:.2f} ms")
    if elapsed < 10:
        print("  Status: EXCELLENT - Real-time capable on embedded hardware")
    elif elapsed < 50:
        print("  Status: GOOD - Acceptable for desktop inference")
    else:
        print("  Status: WARNING - May be too slow for real-time use")
    return elapsed


# =========================
# REAL DATA LOADER (FSR CSV files)
# =========================
def load_csv_data(csv_dir: str = ".") -> FMGDataset:
    """
    تحميل بيانات FSR الحقيقية من ملفات CSV.
    يتجاهل التجارب المُشار إليها في FLAGGED_TRIALS.
    ينفذ تطبيع min-max عالمي ثم sliding window segmentation.

    أعمدة الحساسات تُبنى ديناميكياً من NUM_SENSORS (fsr0 .. fsr{NUM_SENSORS-1})،
    فلا تعود مثبتة على 4 قنوات كما كانت. الملفات القديمة ذات الأربع قنوات
    تُرفض برسالة واضحة بدلاً من الفشل الصامت بـ "No windows created".
    """
    import csv as _csv
    from collections import defaultdict

    sensor_cols = [f'fsr{i}' for i in range(NUM_SENSORS)]
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    all_rows = []
    for fpath in csv_files:
        with open(fpath, newline='') as f:
            reader = _csv.DictReader(f)
            missing = [c for c in sensor_cols if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"{os.path.basename(fpath)} لا يحتوي الأعمدة {missing}. "
                    f"النموذج مضبوط على NUM_SENSORS={NUM_SENSORS}، أي يتوقع "
                    f"{sensor_cols}. أعد جمع البيانات بشريط الـ {NUM_SENSORS} "
                    f"حساسات، أو اضبط NUM_SENSORS ليطابق الملف."
                )
            for row in reader:
                gesture = row['gesture']
                trial = int(row['trial'])
                if gesture in FLAGGED_TRIALS and trial in FLAGGED_TRIALS[gesture]:
                    continue
                if gesture not in GESTURE_LABELS:
                    continue
                rec = {'gesture': gesture, 'trial': trial}
                for c in sensor_cols:
                    rec[c] = float(row[c])
                all_rows.append(rec)

    if not all_rows:
        raise ValueError("No valid rows found in CSV files.")

    # Global min-max normalization
    fsr_min = np.array([min(r[c] for r in all_rows) for c in sensor_cols], dtype=np.float32)
    fsr_max = np.array([max(r[c] for r in all_rows) for c in sensor_cols], dtype=np.float32)
    print(f"FSR min: {fsr_min}  max: {fsr_max}")

    # Group by (gesture, trial) preserving time order
    groups = defaultdict(list)
    for row in all_rows:
        key = (row['gesture'], row['trial'])
        values = np.array([row[c] for c in sensor_cols], dtype=np.float32)
        values = (values - fsr_min) / (fsr_max - fsr_min + 1e-8)
        groups[key].append(values)

    X_list, y_list = [], []
    for (gesture, trial), rows_list in groups.items():
        label = GESTURE_LABELS[gesture]
        arr = np.array(rows_list, dtype=np.float32)  # [T, NUM_SENSORS]
        for start in range(0, len(arr) - TIME_STEPS + 1, WINDOW_STRIDE):
            window = arr[start:start + TIME_STEPS]
            if window.shape == (TIME_STEPS, NUM_SENSORS):
                X_list.append(window)
                y_list.append(label)

    if not X_list:
        raise ValueError("No windows created — check TIME_STEPS vs trial length.")

    X = np.array(X_list, dtype=np.float32)
    y_phase = np.array(y_list, dtype=np.int64)
    # ملفات CSV تحمل تسميات إيماءات فقط (بدون زوايا مفاصل من الكاميرا)،
    # لذا نمرّر زوايا وهمية ونُعلم الـ Dataset بعدم تفعيل خسارة الانحدار.
    y_angle = np.zeros((len(X), NUM_DOF), dtype=np.float32)

    print(f"Loaded {len(X)} windows from {len(groups)} (gesture, trial) pairs.")
    label_counts = {k: 0 for k in GESTURE_LABELS.values()}
    for lbl in y_list:
        label_counts[lbl] += 1
    for name, idx in GESTURE_LABELS.items():
        print(f"  {name}: {label_counts[idx]} windows")

    idx_perm = np.random.permutation(len(X))
    return FMGDataset(X[idx_perm], y_phase[idx_perm], y_angle[idx_perm],
                      has_angles=False)


# =========================
# MAIN - تدريب كامل بالبيانات الاصطناعية
# =========================
if __name__ == "__main__":
    print("=" * 55)
    print("  JoTouch AI - Training Pipeline")
    print(f"  Device: {DEVICE}")
    print("=" * 55)

    os.makedirs("weights", exist_ok=True)

    # 1. بناء النموذج
    model = JoTouchModel()
    print("\n[1] Model Architecture:")
    model.count_parameters()

    # 2. تحميل بيانات FSR الحقيقية من CSV
    print("\n[2] Loading real FSR data from CSV files...")
    dataset = load_csv_data(csv_dir=".")
    print(f"    Total windows: {len(dataset)}")

    # [Synthetic data block – disabled, kept for reference]
    # print("\n[2] Generating synthetic FMG data...")
    # dataset = generate_synthetic_data(num_samples=3000, noise_level=0.05)
    # print(f"    Total samples: {len(dataset)}")
    # dataset.save("fmg_synthetic_data.npz")

    # 3. إنشاء DataLoaders
    print("\n[3] Creating DataLoaders...")
    train_loader, val_loader = make_dataloaders(dataset, batch_size=32, val_split=0.2)

    # 4. تدريب النموذج
    print("\n[4] Training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=40,
        lr=0.001,
        save_path=os.path.join("weights", "jotouch_best.pt"),
    )

    # 5. تقييم النموذج
    print("\n[5] Evaluation on validation set:")
    evaluate_model(model, val_loader)

    # 6. اختبار السرعة
    print("\n[6] Latency Test:")
    test_latency(model)

    print("\nDone! Best model saved as 'weights/jotouch_best.pt'")
