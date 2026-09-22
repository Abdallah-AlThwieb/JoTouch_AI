"""
Model Comparison Benchmark
==========================
مقارنة صادقة (بدون أي تزييف) بين نموذجنا وبدائله الأقدم على بيانات FMG
مفتوحة المصدر (8 حساسات) — نفس البيانات والتقسيمات والبروتوكولات للجميع:

  1. LinearBaselineModel           — نموذج خطي صِرف (أقدم/أبسط خط أساس)
  2. CNNBaselineModel              — CNN قياسي من الجيل الأقدم (تلافيف كثيفة
                                     + ReLU، البديل الأقدم لكتل MobileNetV2)
  3. RNNMobileNetV1BaselineModel   — MobileNetV1 (2017، الإصدار الأقدم من
                                     MobileNetV2) + LSTM (1997، الخلية التكرارية
                                     الأقدم من عائلة RNN — البديل الأقدم للـ TCN)
  4. JoTouchModel                  — معماريتنا الكاملة (MobileNetV2 + TCN)
  5. JoTouchV3CandidateModel       — تجربة ترقية: MobileNetV3 + TCN بنفس خط
                                     أنابيب نموذجنا، للإجابة بأرقام حقيقية:
                                     هل نستبدل MobileNetV2 بـ MobileNetV3؟

سياسة التدريب (بناءً على توجيه الفريق — البدائل لا تأخذ ميزاتنا):
  - النماذج البديلة (1-3): محسّن Adam تقليدي فقط، بدون OneCycleLR،
    بدون Early Stopping، بدون Gradient Clipping — عدد Epochs ثابت.
  - نموذجنا ومرشّح الترقية (4-5): خط الأنابيب الكامل من JoTouch_AI_module:
    AdamW + OneCycleLR + Early Stopping + Gradient Clipping.

مهام التقييم:
  - Classification : دقة التصنيف (Random split + Leave-Subjects-Out CV)
  - Regression     : توقع أهداف مستمرة (MAE / RMSE / R²) إن وُجدت في قاعدة البيانات

محاور المقارنة: الدقة، التعميم عبر المشاركين (LOSO mean±std)، فجوة التعميم،
عدد المعاملات، حجم النموذج، زمن الاستجابة (Latency).

تشغيل:
    venv/Scripts/python model_comparison.py
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# --- path bootstrap: this file now lives in alternative_models/, while the main
# --- JoTouch model and the dataset loaders stay in the project root.
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
import JoTouch_AI_module as cfg
from JoTouch_AI_module import JoTouchModel, FMGDataset
from FMG_Linear_Baseline import LinearBaselineModel
from FMG_CNN_Baseline import CNNBaselineModel
from FMG_RNN_MobileNetV1_Baseline import RNNMobileNetV1BaselineModel
from JoTouch_MobileNetV3_candidate import JoTouchV3CandidateModel
from fmg_datasets import fmg_open_dataset

SEED = 42
EPOCHS = 12
PATIENCE = 4          # لنموذجنا فقط (Early Stopping)
LR = 0.001
BATCH_SIZE = 64
VAL_SPLIT = 0.2
LOSO_FOLDS = 3         # folds التصنيف عبر المشاركين
REG_LOSO_FOLDS = 2     # folds الارتداد عبر المشاركين

# (الاسم، الكلاس، هل يستخدم خط الأنابيب المتقدم الخاص بنا؟)
MODELS = [
    ("Linear (oldest)", LinearBaselineModel, False),
    ("Standard CNN (older alternative)", CNNBaselineModel, False),
    ("LSTM + MobileNetV1 (older alternative)", RNNMobileNetV1BaselineModel, False),
    ("JoTouch (ours)", JoTouchModel, True),
    ("JoTouch-V3 (upgrade candidate)", JoTouchV3CandidateModel, True),
]


# ------------------------------------------------------------------
# Training: baseline pipeline (plain Adam — بدون أي ميزات حديثة)
# ------------------------------------------------------------------
def train_baseline(model, train_loader, epochs=EPOCHS, lr=LR):
    """تدريب النماذج القديمة: Adam فقط، بدون Scheduler/Clipping/EarlyStopping."""
    model.to(cfg.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for X_b, y_b, _ in train_loader:
            X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
            pred, _ = model(X_b)
            loss = loss_fn(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train: {total/len(train_loader):.4f}")


def train_baseline_regressor(model, train_loader, epochs=EPOCHS, lr=LR):
    """نفس سياسة النماذج القديمة لمهمة الارتداد: Adam + MSE فقط."""
    model.to(cfg.DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
            loss = loss_fn(model(X_b), y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train MSE: {total/len(train_loader):.4f}")


def train_advanced_regressor(model, train_loader, val_loader,
                             epochs=EPOCHS, lr=LR, patience=PATIENCE):
    """خط الأنابيب المتقدم لنموذجنا في مهمة الارتداد:
    AdamW + OneCycleLR + Gradient Clipping + Early Stopping."""
    model.to(cfg.DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 5, epochs=epochs, steps_per_epoch=len(train_loader))

    best_val, no_improve, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
            loss = loss_fn(model(X_b), y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
                val += loss_fn(model(X_b), y_b).item()
        val /= len(val_loader)

        if val < best_val:
            best_val, no_improve = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1:3d}/{epochs} | Val MSE: {val:.4f} [best]")
        else:
            no_improve += 1
            print(f"  Epoch {epoch+1:3d}/{epochs} | Val MSE: {val:.4f} ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)


# ------------------------------------------------------------------
# Regression model variants (رؤوس ارتداد لكل معمارية)
# ------------------------------------------------------------------
class LinearRegressor(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(cfg.TIME_STEPS * cfg.NUM_SENSORS, out_dim)

    def forward(self, x):
        return self.fc(self.flatten(x))


class CNNRegressor(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.features = CNNBaselineModel().features
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.features(x.permute(0, 2, 1))
        return self.fc(x[:, :, -1])


class RNNMobileNetV1Regressor(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        base = RNNMobileNetV1BaselineModel()
        self.cnn = base.cnn
        self.rnn = base.rnn
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1))
        out, _ = self.rnn(x.permute(0, 2, 1))
        return self.fc(out[:, -1, :])


class JoTouchV3Regressor(JoTouchV3CandidateModel):
    def __init__(self, out_dim=2):
        super().__init__()
        self.reg_head = nn.Linear(64, out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.shared(x)
        return self.reg_head(x)


class JoTouchRegressor(JoTouchModel):
    def __init__(self, out_dim=2):
        super().__init__()
        self.reg_head = nn.Linear(64, out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.shared(x)
        return self.reg_head(x)


REGRESSORS = [
    ("Linear (oldest)", LinearRegressor, False),
    ("Standard CNN (older alternative)", CNNRegressor, False),
    ("LSTM + MobileNetV1 (older alternative)", RNNMobileNetV1Regressor, False),
    ("JoTouch (ours)", JoTouchRegressor, True),
    ("JoTouch-V3 (upgrade candidate)", JoTouchV3Regressor, True),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_dataset(X, y):
    y_angle = np.zeros((len(X), 1), dtype=np.float32)  # placeholder (angle head disabled)
    return FMGDataset(X, y, y_angle)


def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(make_dataset(X, y), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


def make_reg_loader(X, R, batch_size=BATCH_SIZE, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(R))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def split_train_val(X, y, val_split=VAL_SPLIT, R=None):
    idx = np.random.permutation(len(X))
    n_val = int(len(X) * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    out = [(X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])]
    if R is not None:
        out += [R[train_idx], R[val_idx]]
    return out


def per_class_accuracy(model, loader, num_classes):
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for X_b, y_b, _ in loader:
            X_b = X_b.to(cfg.DEVICE)
            pred, _ = model(X_b)
            preds = pred.argmax(dim=1).cpu().numpy()
            for t, p in zip(y_b.numpy(), preds):
                total[t] += 1
                correct[t] += int(t == p)
    acc = np.where(total > 0, 100.0 * correct / np.maximum(total, 1), 0.0)
    return acc, correct, total


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def latency_ms(model, n_runs=100):
    """نفس منهجية test_latency: 10 إحماء + 100 قياس لعينة واحدة."""
    dummy = torch.randn(1, cfg.TIME_STEPS, cfg.NUM_SENSORS).to(cfg.DEVICE)
    model.eval()
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    return (time.time() - start) / n_runs * 1000.0


def regression_metrics(model, loader):
    """MAE و RMSE و R² (على الأهداف المُطبّعة بين 0 و1)."""
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds_all.append(model(X_b.to(cfg.DEVICE)).cpu().numpy())
            targets_all.append(y_b.numpy())
    p = np.concatenate(preds_all)
    t = np.concatenate(targets_all)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t.mean(axis=0)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return mae, rmse, r2


# ------------------------------------------------------------------
# Classification: train + eval لنموذج واحد
# ------------------------------------------------------------------
def train_and_eval(name, cls, advanced, train_loader, val_loader, test_loader):
    set_seed()
    model = cls()
    print(f"\n--- Training: {name} ---")
    if advanced:
        cfg.train_model(model=model, train_loader=train_loader, val_loader=val_loader,
                        epochs=EPOCHS, lr=LR, patience=PATIENCE)
    else:
        train_baseline(model, train_loader)

    acc = cfg.evaluate_model(model, test_loader)
    pc_acc, pc_correct, pc_total = per_class_accuracy(model, test_loader, cfg.NUM_PHASES)
    return {
        "accuracy": acc,
        "per_class": pc_acc,
        "per_class_correct": pc_correct,
        "per_class_total": pc_total,
        "params": count_params(model),
        "size_kb": count_params(model) * 4 / 1024.0,  # float32
        "latency_ms": latency_ms(model),
    }


# ------------------------------------------------------------------
# Regression: train + eval لنموذج واحد
# ------------------------------------------------------------------
def train_and_eval_regressor(name, cls, advanced, train_loader, val_loader, test_loader):
    set_seed()
    model = cls(out_dim=2)
    print(f"\n--- Training regressor: {name} ---")
    if advanced:
        train_advanced_regressor(model, train_loader, val_loader)
    else:
        train_baseline_regressor(model, train_loader)
    mae, rmse, r2 = regression_metrics(model, test_loader)
    print(f"  MAE: {mae:.4f} | RMSE: {rmse:.4f} | R^2: {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ------------------------------------------------------------------
# Checkpointing (استئناف تلقائي بعد أي انقطاع)
# ------------------------------------------------------------------
_HERE = _os.path.dirname(_os.path.abspath(__file__))
CKPT_PATH = _os.path.join(_HERE, "benchmark_checkpoint.json")
REPORT_PATH = _os.path.join(_HERE, "model_comparison_results.md")


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _load_ckpt():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_ckpt(ck):
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(ck), f)


# ------------------------------------------------------------------
# Benchmark لقاعدة بيانات واحدة
# ------------------------------------------------------------------
def benchmark_dataset(label, load_fn, ck):
    if label in ck and ck[label].get("_done"):
        print(f"\n### DATASET '{label}' already completed — loaded from checkpoint")
        return ck[label]["out"]

    set_seed()
    print("\n" + "#" * 66)
    print(f"  DATASET: {label}")
    print("#" * 66)
    X, y, subjects, class_names, source, R = load_fn()
    print(f"Source: {source}")
    print(f"Windows: {len(X)} | Shape: {X.shape} | Classes: {class_names}")
    print(f"Subjects: {len(np.unique(subjects))}")

    has_reg = R is not None and float(np.std(R)) > 1e-6
    saved = ck.get(label, {})
    out = saved.get("out") or {"label": label, "source": source, "n_windows": len(X),
                               "n_subjects": len(np.unique(subjects)), "class_names": class_names,
                               "classification": {n: {} for n, _, _ in MODELS}, "regression": {}}
    fold_data = saved.get("fold_data", {"cls": {}, "reg": {}})

    def checkpoint():
        ck[label] = {"out": out, "fold_data": fold_data, "_done": False}
        _save_ckpt(ck)

    # ---- Classification A: random split ----
    print("\n== Classification | Protocol A: random split ==")
    (Xtr, ytr), (Xva, yva) = split_train_val(X, y, VAL_SPLIT)
    train_loader = make_loader(Xtr, ytr, shuffle=True)
    val_loader = make_loader(Xva, yva)
    for name, cls, adv in MODELS:
        if "random" in out["classification"].get(name, {}):
            print(f"  skip {name} (checkpoint)")
            continue
        out["classification"][name]["random"] = train_and_eval(
            name, cls, adv, train_loader, val_loader, val_loader)
        checkpoint()

    # ---- Classification B: LOSO ----
    print(f"\n== Classification | Protocol B: {LOSO_FOLDS}-fold leave-subjects-out ==")
    uniq = np.unique(subjects)
    full_cov = [s for s in uniq
                if len(set(y[subjects == s].tolist())) == cfg.NUM_PHASES]
    fold_pool = full_cov if len(full_cov) >= LOSO_FOLDS else list(uniq)
    folds = np.array_split(np.array(fold_pool), LOSO_FOLDS)

    for fold_i, held in enumerate(folds):
        if str(fold_i) in fold_data["cls"]:
            print(f"  skip fold {fold_i+1} (checkpoint)")
            continue
        mask = np.isin(subjects, held)
        (Xtr, ytr), (Xva, yva) = split_train_val(X[~mask], y[~mask], 0.1)
        print(f"\n### Fold {fold_i+1}/{len(folds)} — held-out: {[int(s) for s in held]} "
              f"({mask.sum()} test windows)")
        train_loader = make_loader(Xtr, ytr, shuffle=True)
        val_loader = make_loader(Xva, yva)
        test_loader = make_loader(X[mask], y[mask])
        fold_res = {}
        for name, cls, adv in MODELS:
            r = train_and_eval(name, cls, adv, train_loader, val_loader, test_loader)
            fold_res[name] = {"accuracy": r["accuracy"],
                              "per_class_correct": r["per_class_correct"],
                              "per_class_total": r["per_class_total"]}
            if fold_i == 0:
                out["classification"][name]["heldout"] = {
                    "params": r["params"], "size_kb": r["size_kb"],
                    "latency_ms": r["latency_ms"]}
        fold_data["cls"][str(fold_i)] = fold_res
        checkpoint()

    # تجميع نتائج الـ folds
    loso_accs = {n: [] for n, _, _ in MODELS}
    loso_correct = {n: np.zeros(cfg.NUM_PHASES, dtype=np.int64) for n, _, _ in MODELS}
    loso_total = {n: np.zeros(cfg.NUM_PHASES, dtype=np.int64) for n, _, _ in MODELS}
    for fold_i in range(len(folds)):
        for name, _, _ in MODELS:
            fr = fold_data["cls"][str(fold_i)][name]
            loso_accs[name].append(fr["accuracy"])
            loso_correct[name] += np.array(fr["per_class_correct"])
            loso_total[name] += np.array(fr["per_class_total"])

    for name, _, _ in MODELS:
        accs = np.array(loso_accs[name])
        tot = loso_total[name]
        out["classification"][name]["heldout"].update({
            "accuracy": float(accs.mean()), "std": float(accs.std()),
            "fold_accs": loso_accs[name],
            "per_class": np.where(tot > 0, 100.0 * loso_correct[name] / np.maximum(tot, 1), 0.0),
            "per_class_total": tot})
        print(f"{name}: LOSO = {[f'{a:.1f}%' for a in loso_accs[name]]} "
              f"-> {accs.mean():.2f}% +/- {accs.std():.2f}%")
    checkpoint()

    # ---- Regression ----
    if has_reg:
        print("\n== Regression | random split ==")
        (Xtr, ytr), (Xva, yva), Rtr, Rva = split_train_val(X, y, VAL_SPLIT, R)
        reg_train = make_reg_loader(Xtr, Rtr, shuffle=True)
        reg_val = make_reg_loader(Xva, Rva)
        for name, cls, adv in REGRESSORS:
            if "random" in out["regression"].get(name, {}):
                print(f"  skip {name} regressor (checkpoint)")
                continue
            r = train_and_eval_regressor(name, cls, adv, reg_train, reg_val, reg_val)
            out["regression"].setdefault(name, {})["random"] = r
            checkpoint()

        print(f"\n== Regression | {REG_LOSO_FOLDS}-fold leave-subjects-out ==")
        reg_folds = np.array_split(np.unique(subjects), REG_LOSO_FOLDS)
        for fold_i, held in enumerate(reg_folds):
            if str(fold_i) in fold_data["reg"]:
                print(f"  skip reg fold {fold_i+1} (checkpoint)")
                continue
            mask = np.isin(subjects, held)
            (Xtr, ytr), (Xva, yva), Rtr, Rva = split_train_val(
                X[~mask], y[~mask], 0.1, R[~mask])
            print(f"\n### Reg fold {fold_i+1}/{len(reg_folds)} — held-out: "
                  f"{[int(s) for s in held]}")
            reg_train = make_reg_loader(Xtr, Rtr, shuffle=True)
            reg_val = make_reg_loader(Xva, Rva)
            reg_test = make_reg_loader(X[mask], R[mask])
            fold_res = {}
            for name, cls, adv in REGRESSORS:
                fold_res[name] = train_and_eval_regressor(
                    name, cls, adv, reg_train, reg_val, reg_test)
            fold_data["reg"][str(fold_i)] = fold_res
            checkpoint()

        for name, _, _ in REGRESSORS:
            maes = [fold_data["reg"][str(i)][name]["mae"] for i in range(len(reg_folds))]
            r2s = [fold_data["reg"][str(i)][name]["r2"] for i in range(len(reg_folds))]
            out["regression"][name]["heldout"] = {
                "mae": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                "r2": float(np.mean(r2s)), "r2_std": float(np.std(r2s))}
            print(f"{name}: LOSO MAE = {np.mean(maes):.4f} +/- {np.std(maes):.4f} | "
                  f"R^2 = {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")
        checkpoint()
    else:
        print("\n== Regression: skipped (no continuous targets in this dataset) ==")

    ck[label]["_done"] = True
    _save_ckpt(ck)
    return out


# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------
DATASETS = [
    ("Zenodo MDPI FMG (17 subjects, classification + regression)",
     fmg_open_dataset.load_dataset),
    ("Exoskelebox FMG benchmark (20 subjects, 8 FSR, 6 gestures)",
     None),  # يُربط أدناه
    ("PLoS ONE 2025 FMG (27 participants, 4 gestures + load regression)",
     None),  # يُربط أدناه
]

from fmg_datasets import fmg_exo_dataset
from fmg_datasets import fmg_plos_dataset
DATASETS[1] = (DATASETS[1][0], fmg_exo_dataset.load_dataset)
DATASETS[2] = (DATASETS[2][0], fmg_plos_dataset.load_dataset)


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
def report(all_results):
    lines = ["# JoTouch Model Comparison Results", ""]
    lines.append(f"- **Training budget (identical within each pipeline)**: {EPOCHS} epochs, "
                 f"batch {BATCH_SIZE}, lr={LR}")
    lines.append(f"- **Alternative-models pipeline**: plain Adam only — no OneCycleLR, "
                 f"no early stopping, no gradient clipping")
    lines.append(f"- **JoTouch pipeline (ours)**: AdamW + OneCycleLR + early stopping "
                 f"(patience {PATIENCE}) + gradient clipping")
    lines.append(f"- **Device**: {cfg.DEVICE}")
    lines.append("")

    for res in all_results:
        lines.append(f"## Dataset: {res['label']}")
        lines.append("")
        lines.append(f"- **Source**: {res['source']}")
        lines.append(f"- **Windows**: {res['n_windows']:,} | **Subjects**: {res['n_subjects']} | "
                     f"**Classes**: {len(res['class_names'])} ({', '.join(res['class_names'])})")
        lines.append("")
        lines.append("### Classification")
        lines.append("")
        lines.append("| Model | Params | Size (KB) | Random-split Acc | Cross-Subject Acc (LOSO mean±std) | Generalization Gap | Latency (ms) |")
        lines.append("|---|---|---|---|---|---|---|")

        print("\n" + "=" * 82)
        print(f"  RESULTS — {res['label']}")
        print("=" * 82)
        header = f"{'Model':<34} {'Params':>9} {'SizeKB':>8} {'RandAcc':>9} {'XSubjAcc':>16} {'Gap':>7} {'Lat(ms)':>8}"
        print(header)
        print("-" * len(header))

        for name, _, _ in MODELS:
            r = res["classification"][name]["random"]
            h = res["classification"][name]["heldout"]
            gap = r["accuracy"] - h["accuracy"]
            xsubj = f"{h['accuracy']:.2f}+/-{h['std']:.2f}%"
            print(f"{name:<34} {r['params']:>9,} {r['size_kb']:>8.1f} "
                  f"{r['accuracy']:>8.2f}% {xsubj:>16} {gap:>6.2f} {r['latency_ms']:>8.2f}")
            lines.append(f"| {name} | {r['params']:,} | {r['size_kb']:.1f} | "
                         f"{r['accuracy']:.2f}% | {h['accuracy']:.2f}% ± {h['std']:.2f}% | "
                         f"{gap:.2f} pts | {r['latency_ms']:.2f} |")

        # Per-class (مجمّعة عبر الـ folds)
        lines.append("")
        lines.append("#### Per-class accuracy (cross-subject, aggregated over folds)")
        lines.append("")
        lines.append("| Class | " + " | ".join(n for n, _, _ in MODELS) + " |")
        lines.append("|---|" + "---|" * len(MODELS))
        print("\nPer-class accuracy (cross-subject):")
        for ci, cname in enumerate(res["class_names"]):
            row = []
            for n, _, _ in MODELS:
                h = res["classification"][n]["heldout"]
                row.append(f"{h['per_class'][ci]:.1f}%" if h["per_class_total"][ci] > 0 else "n/a")
            print(f"  {cname:<10} " + "  ".join(f"{v:>8}" for v in row))
            lines.append(f"| {cname} | " + " | ".join(row) + " |")

        # Regression
        if res["regression"]:
            lines.append("")
            lines.append("### Regression (continuous targets)")
            lines.append("")
            lines.append("| Model | Random-split MAE | Random-split R² | Cross-Subject MAE (LOSO mean±std) | Cross-Subject R² |")
            lines.append("|---|---|---|---|---|")
            print("\nRegression results:")
            print(f"{'Model':<34} {'RandMAE':>9} {'RandR2':>8} {'XSubjMAE':>16} {'XSubjR2':>9}")
            for name, _, _ in REGRESSORS:
                g = res["regression"][name]
                r, h = g["random"], g["heldout"]
                xm = f"{h['mae']:.4f}+/-{h['mae_std']:.4f}"
                print(f"{name:<34} {r['mae']:>9.4f} {r['r2']:>8.3f} {xm:>16} {h['r2']:>9.3f}")
                lines.append(f"| {name} | {r['mae']:.4f} | {r['r2']:.3f} | "
                             f"{h['mae']:.4f} ± {h['mae_std']:.4f} | {h['r2']:.3f} ± {h['r2_std']:.3f} |")
        lines.append("")

    # ---- MobileNetV3 upgrade evaluation (JoTouch MBV2 vs MBV3) ----
    lines.append("## MobileNetV3 upgrade evaluation (should JoTouch replace MobileNetV2 with MobileNetV3?)")
    lines.append("")
    lines.append("JoTouch (ours, MobileNetV2 + TCN) vs JoTouch-V3 candidate (MobileNetV3 + TCN), "
                 "trained with the identical JoTouch pipeline on identical splits:")
    lines.append("")
    lines.append("| Dataset | MBV2 Cross-Subject Acc | V3 Cross-Subject Acc | Δ Acc | MBV2 Latency (ms) | V3 Latency (ms) | MBV2 Params | V3 Params |")
    lines.append("|---|---|---|---|---|---|---|---|")
    wins, losses, deltas = 0, 0, []
    for res in all_results:
        cls = res["classification"]
        if "JoTouch-V3 (upgrade candidate)" not in cls:
            continue
        a = cls["JoTouch (ours)"]
        b = cls["JoTouch-V3 (upgrade candidate)"]
        d = b["heldout"]["accuracy"] - a["heldout"]["accuracy"]
        deltas.append(d)
        wins += int(d > 0.5)
        losses += int(d < -0.5)
        lines.append(f"| {res['label']} | {a['heldout']['accuracy']:.2f}% ± {a['heldout']['std']:.2f}% | "
                     f"{b['heldout']['accuracy']:.2f}% ± {b['heldout']['std']:.2f}% | {d:+.2f} pts | "
                     f"{a['random']['latency_ms']:.2f} | {b['random']['latency_ms']:.2f} | "
                     f"{a['random']['params']:,} | {b['random']['params']:,} |")
    lines.append("")
    if deltas:
        mean_d = float(np.mean(deltas))
        if wins > losses and mean_d > 0.5:
            verdict = (f"**Decision: UPGRADE.** MobileNetV3 beats MobileNetV2 on {wins} of "
                       f"{len(deltas)} datasets (mean Δ = {mean_d:+.2f} pts cross-subject accuracy) "
                       f"without a harmful latency increase — replace the MobileNetV2 blocks in "
                       f"JoTouch_AI_module with MobileNetV3 blocks.")
        elif losses > wins and mean_d < -0.5:
            verdict = (f"**Decision: KEEP MobileNetV2.** MobileNetV3 loses on {losses} of "
                       f"{len(deltas)} datasets (mean Δ = {mean_d:+.2f} pts) on our small FMG "
                       f"windows — its ImageNet-era gains do not transfer to this task.")
        else:
            verdict = (f"**Decision: KEEP MobileNetV2.** MobileNetV3 shows no meaningful gain "
                       f"(mean Δ = {mean_d:+.2f} pts across {len(deltas)} datasets) while adding "
                       f"SE-block parameters and latency — the theoretical ImageNet advantage does "
                       f"not materialize on 8-channel FMG windows, so the simpler V2 stays.")
        lines.append(verdict)
        lines.append("")
        print("\n" + "=" * 82)
        print("  MOBILENETV3 UPGRADE EVALUATION")
        print("=" * 82)
        print(verdict)

    # شرح المقاييس
    lines.append("## Metrics explained")
    lines.append("")
    lines.append("- **Params / Size (KB)**: number of trainable weights and approximate float32 "
                 "model size — matters for embedding on microcontrollers (flash/RAM limits).")
    lines.append("- **Random-split Acc**: accuracy on a random 80/20 window split; windows from the "
                 "same trial can appear in both sets — the optimistic, in-session number.")
    lines.append("- **Cross-Subject Acc (LOSO)**: leave-subjects-out cross-validation — each fold "
                 "excludes whole subjects from training and tests only on them; mean ± std across "
                 "folds. The realistic 'new user' number and the key generalization evidence.")
    lines.append("- **Generalization Gap**: random-split accuracy minus cross-subject accuracy; "
                 "smaller = less overfitting.")
    lines.append("- **Latency (ms)**: mean of 100 single-window inference passes after a 10-pass "
                 "warm-up (same method as `test_latency` in JoTouch_AI_module.py); must stay well "
                 "under the 10 ms real-time budget.")
    lines.append("- **MAE / RMSE**: mean absolute / root-mean-square error of the continuous "
                 "targets (normalized 0–1); lower is better.")
    lines.append("- **R²**: coefficient of determination; 1.0 = perfect prediction, 0 = no better "
                 "than predicting the mean, negative = worse than the mean.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nReport written to " + REPORT_PATH)


def main():
    ck = _load_ckpt()
    all_results = []
    for label, load_fn in DATASETS:
        all_results.append(benchmark_dataset(label, load_fn, ck))
    report(all_results)
    return all_results


if __name__ == "__main__":
    # the dataset loaders resolve "Content/open_fmg_data" relative to the working
    # directory, so run from the project root regardless of where this was launched.
    _os.chdir(_PROJECT_ROOT)
    main()
