"""
PLoS ONE 2025 FMG/EMG Dataset Loader
====================================
قاعدة بيانات PLoS ONE (27 مشاركاً) — Zenodo record 15420178:

  Young PR, et al. (2025) "The effects of limb position and grasped load on
  hand gesture classification using electromyography, force myography, and
  their combination." PLoS ONE 20(4): e0321319.

  البنية: Data/Par <1..27>/<Gesture>/<Load>/<Gesture Load Rep>.csv
    - 4 إيماءات: Key, Pinch, Power, Tripod (فئات التصنيف)
    - 5 أحمال: 0, 250, 500, 750, 1000 غرام (تُستخدم كهدف ارتداد مستمر)
    - كل CSV: 36001 صف × 16 قناة @ 2000Hz (18 ثانية)
      القنوات 1-8 = FMG (نستخدمها)، القنوات 9-16 = EMG (نتجاهلها)
  - نخفض العينات x20 إلى 100Hz لمطابقة المشروع (نافذة 20 خطوة = 200ms).
  - الأحمال المختلفة داخل فئة الإيماءة تعمل كتنويع طبيعي (روبستness).

المهام:
  - Classification : الإيماءة (4 فئات)
  - Regression     : الحمل المُمسك بالغرام مُطبّعاً [0..1] (هدف مستمر حقيقي)

الواجهة نفس fmg_open_dataset:
    load_dataset() -> (X, y, subjects, class_names, source, R)
"""

import os
import re
import glob

import numpy as np

# --- path bootstrap: هذا الملف داخل fmg_datasets/ بينما JoTouch_AI_module في الجذر
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import JoTouch_AI_module as cfg

DATA_DIR = os.path.join("Content", "open_fmg_data", "plos_extracted", "Data")
CACHE_PATH = os.path.join("Content", "open_fmg_data", "plos_windows.npz")

FMG_COLS = list(range(8))   # القنوات 1-8 = FMG
DOWNSAMPLE = 20             # 2000Hz -> 100Hz
WINDOW_STRIDE = 40          # خطوة أكبر للنوافذ (تجارب طويلة 18s)
MAX_PER_CLASS = 3000        # real training budget


def _iter_trials():
    """يعيد مولّد (subject_str, gesture, load_grams, csv_path)."""
    pattern = os.path.join(DATA_DIR, "Par *", "*", "*", "*.csv")
    for path in sorted(glob.glob(pattern)):
        m = re.match(r"Par (\d+)[\\/]+(\w+)[\\/]+(\d+)[\\/]", os.path.relpath(path, DATA_DIR))
        if m:
            yield m.group(1), m.group(2), float(m.group(3)), path


def build_windows():
    import pandas as pd

    files = list(_iter_trials())
    print(f"Found {len(files)} trial CSVs")
    if not files:
        raise FileNotFoundError(f"No PLoS CSVs under {DATA_DIR}")

    X_list, y_list, s_list, r_list = [], [], [], []
    class_names = sorted({g for _, g, _, _ in files})
    gest_to_id = {g: i for i, g in enumerate(class_names)}
    subj_ids = sorted({int(s) for s, _, _, _ in files})
    subj_to_id = {s: i for i, s in enumerate(subj_ids)}

    all_min = np.full(8, np.inf, dtype=np.float32)
    all_max = np.full(8, -np.inf, dtype=np.float32)
    trials = []
    for i, (subject, gesture, load, path) in enumerate(files):
        try:
            fmg = pd.read_csv(path).to_numpy(dtype=np.float32)[:, FMG_COLS]
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        if fmg.shape[0] < cfg.TIME_STEPS * DOWNSAMPLE:
            continue
        all_min = np.minimum(all_min, fmg.min(axis=0))
        all_max = np.maximum(all_max, fmg.max(axis=0))
        trials.append((subj_to_id[int(subject)], gest_to_id[gesture],
                       load / 1000.0, fmg))
        if (i + 1) % 500 == 0:
            print(f"  read {i+1}/{len(files)} files ...")

    # حفظ إحصاءات التطبيع للنشر على Teensy
    np.savez(os.path.join("Content", "open_fmg_data", "plos_norm_stats.npz"),
             fmin=all_min, fmax=all_max)

    for subject, label, load_norm, fmg in trials:
        fmg = (fmg - all_min) / (all_max - all_min + 1e-8)
        seg = fmg[::DOWNSAMPLE]
        for start in range(0, len(seg) - cfg.TIME_STEPS + 1, WINDOW_STRIDE):
            X_list.append(seg[start:start + cfg.TIME_STEPS])
            y_list.append(label)
            s_list.append(subject)
            r_list.append([load_norm, load_norm])  # هدف الارتداد: الحمل (مكرر على بُعدين)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(s_list, dtype=np.int64)
    R = np.array(r_list, dtype=np.float32)

    print(f"Built {len(X):,} windows | {len(class_names)} gestures x "
          f"{len(subj_ids)} subjects")
    for g, i in gest_to_id.items():
        print(f"  {g}: {int((y == i).sum()):,} windows")

    np.savez_compressed(CACHE_PATH, X=X, y=y, subjects=subjects, R=R,
                        class_names=np.array(class_names))
    return X, y, subjects, class_names, R


def load_dataset():
    """يعيد (X, y, subjects, class_names, source, R)."""
    cfg.NUM_SENSORS = 8
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached windows from {CACHE_PATH}")
        d = np.load(CACHE_PATH, allow_pickle=False)
        X, y, subjects, R = d["X"], d["y"], d["subjects"], d["R"]
        class_names = [str(g) for g in d["class_names"]]
    else:
        X, y, subjects, class_names, R = build_windows()

    cfg.NUM_PHASES = len(class_names)
    source = ("PLoS ONE 2025 (Young et al., e0321319) - Zenodo 15420178, 27 participants, "
              "8 FMG ch @100Hz after x20 downsample, 4 gestures x 5 loads; "
              "regression target = grasped load (normalized)")

    # سقف طبقي لكل فئة لضبط زمن التدريب
    rng = np.random.default_rng(42)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) > MAX_PER_CLASS:
            idx = rng.choice(idx, size=MAX_PER_CLASS, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    if len(keep) < len(X):
        print(f"Capped to {len(keep):,} windows ({MAX_PER_CLASS}/class)")
        X, y, subjects, R = X[keep], y[keep], subjects[keep], R[keep]

    perm = np.random.permutation(len(X))
    return X[perm], y[perm], subjects[perm], class_names, source, R[perm]


if __name__ == "__main__":
    X, y, subjects, class_names, source, R = load_dataset()
    print(f"\nSource : {source}")
    print(f"X      : {X.shape}  y: {y.shape}  subjects: {len(np.unique(subjects))}")
    print(f"Classes: {class_names}  R range: [{R.min():.2f}, {R.max():.2f}]")
