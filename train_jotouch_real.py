"""
JoTouch Real Training (main model, full budget)
===============================================
تدريب حقيقي لنموذج JoTouchModel الرئيسي (MobileNetV2 + TCN) على قواعد
البيانات المفتوحة الثلاث (8 حساسات FMG) — ليس تشغيل مقارنة قصير:

  - خط الأنابيب الكامل: AdamW + OneCycleLR + Gradient Clipping +
    Early Stopping (patience أعلى من تشغيل المقارنة)
  - نوافذ أكثر لكل فئة (Caps مرفوعة)
  - حفظ الأوزان المدرّبة على القرص لكل قاعدة بيانات مع Metadata
  - تقييم نهائي على مجموعة اختبار لم تُمس + قياس Latency

يُدرّب نموذجاً منفصلاً لكل قاعدة بيانات (الفئات مختلفة الدلالة بين
القواعد، فلا يمكن دمجها في مهمّة تصنيف واحدة بصدق).

يحفظ Checkpoint بعد كل Epoch ليستأنف تلقائياً بعد أي انقطاع.

تشغيل:
    venv/Scripts/python train_jotouch_real.py
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import JoTouch_AI_module as cfg
from JoTouch_AI_module import JoTouchModel, FMGDataset
from torch.utils.data import DataLoader

from fmg_datasets import fmg_open_dataset
from fmg_datasets import fmg_exo_dataset
from fmg_datasets import fmg_plos_dataset

SEED = 42
EPOCHS = 40
PATIENCE = 8
LR = 0.001
BATCH_SIZE = 64
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

DATASETS = [
    ("zenodo", "Zenodo MDPI FMG (17 subjects, 5 motions)", fmg_open_dataset.load_dataset),
    ("exoskelebox", "Exoskelebox FMG benchmark (20 subjects, 6 gestures)", fmg_exo_dataset.load_dataset),
    ("plos", "PLoS ONE 2025 FMG (27 participants, 4 gestures)", fmg_plos_dataset.load_dataset),
]

RESULTS_PATH = os.path.join("results", "real_training_results.md")

# مجلدات المخرجات (تُنشأ تلقائياً حتى يعمل المشروع بعد نسخة نظيفة)
for _d in ("weights", "results", "runs"):
    os.makedirs(_d, exist_ok=True)



def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_loader(X, y, shuffle=False):
    y_angle = np.zeros((len(X), 1), dtype=np.float32)
    ds = FMGDataset(X, y, y_angle)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def latency_ms(model, n_runs=100):
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


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b, _ in loader:
            pred, _ = model(X_b.to(cfg.DEVICE))
            correct += (pred.argmax(1).cpu() == y_b).sum().item()
            total += len(y_b)
    return 100.0 * correct / max(total, 1)


def train_one(key, label, load_fn):
    ckpt_path = os.path.join("runs", f"real_train_{key}_ckpt.pt")
    model_path = os.path.join("weights", f"jotouch_real_{key}.pt")

    print("\n" + "#" * 66)
    print(f"  REAL TRAINING: {label}")
    print("#" * 66)

    set_seed()
    X, y, subjects, class_names, source, R = load_fn()
    n_classes = len(class_names)
    print(f"Windows: {len(X):,} | Classes: {n_classes} {class_names} | Sensors: {cfg.NUM_SENSORS}")

    idx = np.random.permutation(len(X))
    n_test = int(len(X) * TEST_SPLIT)
    n_val = int(len(X) * VAL_SPLIT)
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    train_loader = make_loader(X[tr], y[tr], shuffle=True)
    val_loader = make_loader(X[va], y[va])
    test_loader = make_loader(X[te], y[te])
    print(f"Split: {len(tr):,} train / {len(va):,} val / {len(te):,} test")

    model = JoTouchModel().to(cfg.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 5, epochs=EPOCHS, steps_per_epoch=len(train_loader))

    start_epoch, best_val, best_state, no_improve = 0, float("inf"), None, 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        if ck.get("n_classes") == n_classes and not ck.get("finished"):
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            start_epoch, best_val, best_state, no_improve = (
                ck["epoch"] + 1, ck["best_val"], ck["best_state"], ck["no_improve"])
            print(f"Resuming from epoch {start_epoch + 1} (best val loss {best_val:.4f})")
        elif ck.get("finished"):
            print("Already finished — evaluating saved model.")
            model.load_state_dict(ck["best_state"])
            return finish(key, label, source, class_names, model, test_loader, ck["best_epoch"])

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        for X_b, y_b, _ in train_loader:
            X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
            pred, _ = model(X_b)
            loss = loss_fn(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b, _ in val_loader:
                X_b, y_b = X_b.to(cfg.DEVICE), y_b.to(cfg.DEVICE)
                pred, _ = model(X_b)
                val_loss += loss_fn(pred, y_b).item()
        val_loss /= len(val_loader)

        marker = ""
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " [best]"
        else:
            no_improve += 1
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | train {total_loss/len(train_loader):.4f} "
              f"| val {val_loss:.4f}{marker}")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_val": best_val, "best_state": best_state,
                    "no_improve": no_improve, "n_classes": n_classes,
                    "best_epoch": epoch + 1, "finished": no_improve >= PATIENCE},
                   ckpt_path)

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no val improvement for {PATIENCE})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return finish(key, label, source, class_names, model, test_loader, epoch + 1)


def finish(key, label, source, class_names, model, test_loader, trained_epochs):
    acc = evaluate(model, test_loader)
    lat = latency_ms(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_path = os.path.join("weights", f"jotouch_real_{key}.pt")
    cfg.save_model(model, model_path, metadata={
        "dataset": label, "source": source, "classes": list(map(str, class_names)),
        "num_sensors": cfg.NUM_SENSORS, "time_steps": cfg.TIME_STEPS,
        "epochs_trained": trained_epochs, "test_acc": acc,
        "pipeline": "AdamW + OneCycleLR + grad clip + early stopping (patience 8)",
    })

    print(f"  ==> TEST ACCURACY: {acc:.2f}% | latency {lat:.2f} ms | "
          f"params {params:,} | saved: {model_path}")
    return {"key": key, "label": label, "source": source,
            "classes": list(map(str, class_names)), "test_acc": acc,
            "latency_ms": lat, "params": params, "epochs": trained_epochs,
            "model_path": model_path}


def write_report(results):
    lines = ["# JoTouch Real Training Results (main model)", ""]
    lines.append("- **Model**: JoTouchModel (MobileNetV2 + TCN) — MobileNetV2 confirmed over "
                 "MobileNetV3 by the upgrade evaluation (see alternative_models/model_comparison_results.md)")
    lines.append(f"- **Pipeline (full)**: AdamW + OneCycleLR + gradient clipping + early stopping "
                 f"(patience {PATIENCE}), up to {EPOCHS} epochs, batch {BATCH_SIZE}, lr={LR}")
    lines.append(f"- **Split per dataset**: {int((1-VAL_SPLIT-TEST_SPLIT)*100)}% train / "
                 f"{int(VAL_SPLIT*100)}% val / {int(TEST_SPLIT*100)}% untouched test")
    lines.append(f"- **Device**: {cfg.DEVICE}")
    lines.append("")
    lines.append("| Dataset | Classes | Epochs trained | Test Accuracy | Latency (ms) | Params | Saved weights |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['label']} | {len(r['classes'])} ({', '.join(r['classes'])}) | "
                     f"{r['epochs']} | **{r['test_acc']:.2f}%** | {r['latency_ms']:.2f} | "
                     f"{r['params']:,} | `{r['model_path']}` |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- One model per dataset: the label sets have different meanings across datasets, "
                 "so merging them into one classifier would be dishonest.")
    lines.append("- Input: 8 FMG channels x 20 timesteps (200 ms), min-max normalized per dataset "
                 "(the loaders set NUM_SENSORS=8 to match the open data; the on-device default "
                 "of 4 sensors in JoTouch_AI_module.py is unchanged).")
    lines.append("- These test accuracies are in-distribution numbers. The cross-subject (new user) "
                 "estimates remain the LOSO figures in alternative_models/model_comparison_results.md.")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {RESULTS_PATH}")


def main():
    results = []
    for key, label, load_fn in DATASETS:
        results.append(train_one(key, label, load_fn))
    write_report(results)


if __name__ == "__main__":
    main()
