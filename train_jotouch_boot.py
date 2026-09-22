"""
JoTouch Boot Controller Training (generalized FMG -> continuous commands)
=========================================================================
يُدرّب "نموذج الإقلاع" (Boot Controller): العلاقة المعمّمة بين حساسات
FMG الثمانية وأوامر التحكم المستمرة التي تُشغّل الطرف الصناعي من اليوم
الأول على Teensy — إلى أن يُعاير نموذج JoTouch على المريض ويرفع الأوزان
الشخصية.

مخرجات النموذج (Stage 1 من سلسلة التحكم):
  - Zenodo  : FMG -> موضع المنصة (X, Y) المُطبّع  => تحكم مستمر بمحورين
              (مثل: معصم أعلى/أسفل، يمين/يسار)
  - PLoS    : FMG -> الحمل المُمسك [0..1]          => شدة القبضة (grip intensity)

ثم على Teensy (Stage 2): motor_cmd[7] = alpha * Posture_g[7] (جدول التآزر)
و (Stage 3): المحركات السبعة تُحرّك درجات الحرية ميكانيكياً عبر الأوتار.

يحفظ Checkpoint بعد كل Epoch ليستأنف تلقائياً بعد أي انقطاع.

تشغيل:
    venv/Scripts/python train_jotouch_boot.py
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import JoTouch_AI_module as cfg
from JoTouch_AI_module import JoTouchModel

from fmg_datasets import fmg_open_dataset
from fmg_datasets import fmg_plos_dataset

SEED = 42
EPOCHS = 40
PATIENCE = 8
LR = 0.001
BATCH_SIZE = 64
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RESULTS_PATH = os.path.join("results", "boot_training_results.md")

# مجلدات المخرجات (تُنشأ تلقائياً حتى يعمل المشروع بعد نسخة نظيفة)
for _d in ("weights", "results", "runs"):
    os.makedirs(_d, exist_ok=True)



class JoTouchBootModel(JoTouchModel):
    """عمود JoTouch الفقري + رأس أوامر مستمرة (بدل رأس التصنيف)."""

    def __init__(self, out_dim):
        super().__init__()
        self.boot_head = nn.Linear(64, out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.shared(x)
        return self.boot_head(x)


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_loader(X, R, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(R))
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
    preds, tgts = [], []
    with torch.no_grad():
        for X_b, R_b in loader:
            preds.append(model(X_b.to(cfg.DEVICE)).cpu().numpy())
            tgts.append(R_b.numpy())
    p, t = np.concatenate(preds), np.concatenate(tgts)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t.mean(axis=0)) ** 2)) + 1e-12
    return mae, rmse, 1.0 - ss_res / ss_tot


def train_boot(key, out_dim, load_fn, target_desc, pick_R):
    ckpt_path = os.path.join("runs", f"boot_train_{key}_ckpt.pt")
    model_path = os.path.join("weights", f"jotouch_boot_{key}.pt")

    print("\n" + "#" * 66)
    print(f"  BOOT CONTROLLER: {key} -> {target_desc}")
    print("#" * 66)

    set_seed()
    X, y, subjects, class_names, source, R = load_fn()
    R = pick_R(R)
    print(f"Windows: {len(X):,} | Sensors: {cfg.NUM_SENSORS} | Targets: {R.shape[1]} ({target_desc})")

    idx = np.random.permutation(len(X))
    n_test = int(len(X) * TEST_SPLIT)
    n_val = int(len(X) * VAL_SPLIT)
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    train_loader = make_loader(X[tr], R[tr], shuffle=True)
    val_loader = make_loader(X[va], R[va])
    test_loader = make_loader(X[te], R[te])
    print(f"Split: {len(tr):,} train / {len(va):,} val / {len(te):,} test")

    model = JoTouchBootModel(out_dim).to(cfg.DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 5, epochs=EPOCHS, steps_per_epoch=len(train_loader))

    start_epoch, best_val, best_state, no_improve = 0, float("inf"), None, 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        if ck.get("finished"):
            model.load_state_dict(ck["best_state"])
            return finish(key, target_desc, source, model, test_loader, ck["best_epoch"], model_path)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_val, best_state, no_improve = (
            ck["epoch"] + 1, ck["best_val"], ck["best_state"], ck["no_improve"])
        print(f"Resuming from epoch {start_epoch + 1} (best val {best_val:.4f})")

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, EPOCHS):
        last_epoch = epoch
        model.train()
        total_loss = 0.0
        for X_b, R_b in train_loader:
            X_b, R_b = X_b.to(cfg.DEVICE), R_b.to(cfg.DEVICE)
            loss = loss_fn(model(X_b), R_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, R_b in val_loader:
                X_b, R_b = X_b.to(cfg.DEVICE), R_b.to(cfg.DEVICE)
                val_loss += loss_fn(model(X_b), R_b).item()
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
                    "no_improve": no_improve, "best_epoch": epoch + 1,
                    "finished": no_improve >= PATIENCE}, ckpt_path)

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return finish(key, target_desc, source, model, test_loader, last_epoch + 1, model_path)


def finish(key, target_desc, source, model, test_loader, trained_epochs, model_path):
    mae, rmse, r2 = evaluate(model, test_loader)
    lat = latency_ms(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    torch.save({"state_dict": model.state_dict(),
                "metadata": {"target": target_desc, "source": source,
                             "num_sensors": cfg.NUM_SENSORS,
                             "time_steps": cfg.TIME_STEPS,
                             "epochs_trained": trained_epochs,
                             "test_mae": mae, "test_r2": r2}}, model_path)

    print(f"  ==> TEST MAE: {mae:.4f} | RMSE: {rmse:.4f} | R^2: {r2:.3f} | "
          f"latency {lat:.2f} ms | saved: {model_path}")
    return {"key": key, "target": target_desc, "source": source, "mae": mae,
            "rmse": rmse, "r2": r2, "latency_ms": lat, "params": params,
            "epochs": trained_epochs, "model_path": model_path}


def write_report(results):
    lines = ["# JoTouch Boot Controller Training Results (generalized FMG -> commands)", ""]
    lines.append("- **Purpose**: generalized sensor->actuator relationship for the Teensy boot "
                 "controller, until per-patient calibration replaces the weights")
    lines.append(f"- **Pipeline**: AdamW + OneCycleLR + gradient clipping + early stopping "
                 f"(patience {PATIENCE}), up to {EPOCHS} epochs, batch {BATCH_SIZE}")
    lines.append(f"- **Split**: {int((1-VAL_SPLIT-TEST_SPLIT)*100)}% train / "
                 f"{int(VAL_SPLIT*100)}% val / {int(TEST_SPLIT*100)}% untouched test")
    lines.append("")
    lines.append("| Boot model | Output (target) | Epochs | Test MAE | Test RMSE | Test R² | Latency (ms) | Weights |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['key']} | {r['target']} | {r['epochs']} | {r['mae']:.4f} | "
                     f"{r['rmse']:.4f} | {r['r2']:.3f} | {r['latency_ms']:.2f} | `{r['model_path']}` |")
    lines.append("")
    lines.append("## Control chain on Teensy")
    lines.append("")
    lines.append("1. **Stage 1 (these weights)**: FMG window [8 ch x 20] -> continuous command(s).")
    lines.append("2. **Stage 2 (synergy table, hand-designed)**: `motor_cmd[7] = intensity * Posture_g[7]`.")
    lines.append("3. **Stage 3 (mechanical)**: 7 motors drive 12-16 finger DOFs via tendons/linkages.")
    lines.append("")
    lines.append("**Input contract**: 8 FMG channels x 20 timesteps @100 Hz, min-max normalized "
                 "per channel. Normalization stats MUST come from the device calibration step "
                 "on the patient's own band (open-dataset stats in "
                 "Content/open_fmg_data/*_norm_stats.npz are reference only).")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {RESULTS_PATH}")


def main():
    results = []
    results.append(train_boot(
        "zenodo", 2, fmg_open_dataset.load_dataset,
        "stage position (X, Y) normalized - 2-axis continuous control",
        lambda R: R))
    results.append(train_boot(
        "plos", 1, fmg_plos_dataset.load_dataset,
        "grasped load [0..1] - grip closure intensity",
        lambda R: R[:, :1]))
    write_report(results)


if __name__ == "__main__":
    main()
