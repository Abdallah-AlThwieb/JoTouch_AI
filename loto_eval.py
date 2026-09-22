"""
Leave-One-Trial-Out (LOTO) Evaluation for JoTouch model.

For each trial number t (1..15):
  - Train  : all windows where trial != t
  - Test   : all windows where trial == t
  - Report : accuracy per fold + overall mean +/- std
"""
import os
import sys
import glob
import csv as _csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from JoTouch_AI_module import (
    JoTouchModel, FMGDataset,
    TIME_STEPS, NUM_SENSORS,
    GESTURE_LABELS, FLAGGED_TRIALS, WINDOW_STRIDE, DEVICE,
)


# ------------------------------------------------------------------
# 1. Load all CSV data and return windows WITH trial numbers
# ------------------------------------------------------------------
def load_raw_windows(csv_dir="."):
    sensor_cols = ["fsr0", "fsr1", "fsr2", "fsr3"]
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    all_rows = []
    for fpath in csv_files:
        with open(fpath, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                gesture = row["gesture"]
                trial = int(row["trial"])
                if gesture in FLAGGED_TRIALS and trial in FLAGGED_TRIALS[gesture]:
                    continue
                if gesture not in GESTURE_LABELS:
                    continue
                all_rows.append({
                    "gesture": gesture,
                    "trial":   trial,
                    "fsr0":    float(row["fsr0"]),
                    "fsr1":    float(row["fsr1"]),
                    "fsr2":    float(row["fsr2"]),
                    "fsr3":    float(row["fsr3"]),
                })

    # Global min-max normalization (using all rows)
    fsr_min = np.array([min(r[c] for r in all_rows) for c in sensor_cols], dtype=np.float32)
    fsr_max = np.array([max(r[c] for r in all_rows) for c in sensor_cols], dtype=np.float32)

    # Group by (gesture, trial) to preserve time order
    groups = defaultdict(list)
    for row in all_rows:
        key = (row["gesture"], row["trial"])
        v = np.array([row["fsr0"], row["fsr1"], row["fsr2"], row["fsr3"]], dtype=np.float32)
        v = (v - fsr_min) / (fsr_max - fsr_min + 1e-8)
        groups[key].append(v)

    X_list, y_list, trial_list = [], [], []
    for (gesture, trial), rows_list in groups.items():
        label = GESTURE_LABELS[gesture]
        arr = np.array(rows_list, dtype=np.float32)
        for start in range(0, len(arr) - TIME_STEPS + 1, WINDOW_STRIDE):
            window = arr[start : start + TIME_STEPS]
            if window.shape == (TIME_STEPS, NUM_SENSORS):
                X_list.append(window)
                y_list.append(label)
                trial_list.append(trial)

    X      = np.array(X_list,     dtype=np.float32)
    y      = np.array(y_list,     dtype=np.int64)
    trials = np.array(trial_list, dtype=np.int64)
    return X, y, trials


# ------------------------------------------------------------------
# 2. Train a fresh model on one fold's training data
# ------------------------------------------------------------------
def train_fold(X_train, y_train, epochs=30, lr=0.001, patience=5):
    y_angle = np.zeros((len(X_train), 1), dtype=np.float32)
    dataset = FMGDataset(X_train, y_train, y_angle)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model     = JoTouchModel().to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 5, epochs=epochs, steps_per_epoch=len(loader)
    )

    best_loss  = float("inf")
    no_improve = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b, _ in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred, _  = model(X_b)
            loss     = loss_fn(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ------------------------------------------------------------------
# 3. Evaluate model on one fold's test data
# ------------------------------------------------------------------
def eval_fold(model, X_test, y_test):
    y_angle = np.zeros((len(X_test), 1), dtype=np.float32)
    dataset = FMGDataset(X_test, y_test, y_angle)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    correct = total = 0
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)

    with torch.no_grad():
        for X_b, y_b, _ in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred, _  = model(X_b)
            preds    = pred.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total   += y_b.size(0)
            for true, p in zip(y_b.cpu().tolist(), preds.cpu().tolist()):
                per_class_total[true]   += 1
                per_class_correct[true] += int(true == p)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, per_class_correct, per_class_total


# ------------------------------------------------------------------
# 4. Full LOTO loop
# ------------------------------------------------------------------
def leave_one_trial_out(csv_dir="."):
    print("=" * 60)
    print("  Leave-One-Trial-Out (LOTO) Evaluation")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    X, y, trials = load_raw_windows(csv_dir)
    unique_trials = sorted(np.unique(trials).tolist())
    label_names   = {v: k for k, v in GESTURE_LABELS.items()}

    print(f"\nTotal windows : {len(X)}")
    print(f"Unique trials : {unique_trials}")
    print(f"Folds         : {len(unique_trials)}\n")
    print(f"{'Trial':>6}  {'Test gestures':<32}  {'Windows':>7}  {'Accuracy':>8}")
    print("-" * 60)

    accs = []
    per_class_correct_all = defaultdict(int)
    per_class_total_all   = defaultdict(int)

    for t in unique_trials:
        test_mask  = trials == t
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        test_classes = sorted(set(y_test.tolist()))
        test_str     = "+".join(label_names[c] for c in test_classes)

        model = train_fold(X_train, y_train, epochs=30, lr=0.001, patience=5)
        acc, pcc, pct = eval_fold(model, X_test, y_test)
        accs.append(acc)

        for cls in pcc:
            per_class_correct_all[cls] += pcc[cls]
            per_class_total_all[cls]   += pct[cls]

        print(f"{t:>6}  {test_str:<32}  {len(X_test):>7}  {acc:>7.1f}%")

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))

    print("-" * 60)
    print(f"\n  LOTO Mean Accuracy : {mean_acc:.2f}%")
    print(f"  LOTO Std           : {std_acc:.2f}%")

    print("\n  Per-gesture accuracy (across all folds):")
    for cls_id, name in sorted(label_names.items(), key=lambda x: x[0]):
        if per_class_total_all[cls_id] > 0:
            g_acc = 100.0 * per_class_correct_all[cls_id] / per_class_total_all[cls_id]
            print(f"    {name:<8}: {g_acc:.1f}%  ({per_class_correct_all[cls_id]}/{per_class_total_all[cls_id]} windows)")

    print("=" * 60)
    return mean_acc, std_acc


if __name__ == "__main__":
    leave_one_trial_out(csv_dir=".")
