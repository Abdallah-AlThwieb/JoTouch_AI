"""
Exoskelebox 8-FSR FMG Dataset Loader
====================================
قاعدة بيانات Exoskelebox المفتوحة لتصنيف إيماءات اليد بتقنية FMG:

  المصدر: github.com/exoskelebox/force-myography-hand-gesture-recognition-benchmark-data
          (الورقة المرافقة: arXiv:2007.14918) — 20 مشاركاً، سوار BioX
          بـ 8 حساسات FSR على الساعد (نستخدم قنوات الساعد الثمانية فقط).

  - المهمة: تصنيف 6 إيماءات (closed, extension, flexion, rest, straight, wide).
    كل إيماءة مسجلة بثلاث وضعيات للمعصم (neutral/supine/prone) — ندمج الوضعيات
    كتنويع طبيعي داخل الفئة الواحدة لزيادة صعوبة الاختبار وواقعيته.
  - معدل العينات ~500Hz — نخفضه x5 إلى ~100Hz لمطابقة مشروعنا
    (نافذة 20 خطوة = 200ms).
  - لا توجد أهداف ارتداد مستمرة -> تُتخطى مهمة الارتداد لهذه القاعدة.

الواجهة نفس fmg_open_dataset:
    load_dataset() -> (X, y, subjects, class_names, source, R)
"""

import os

import numpy as np

# --- path bootstrap: هذا الملف داخل fmg_datasets/ بينما JoTouch_AI_module في الجذر
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import JoTouch_AI_module as cfg

DATA_DIR = os.path.join("Content", "open_fmg_data", "exoskelebox", "data")
CSV_PATH = os.path.join(DATA_DIR, "sensor readings.csv")
CACHE_PATH = os.path.join("Content", "open_fmg_data", "exo_windows.npz")

ARM_COLS = [f"arm_sensor_{i}" for i in range(1, 9)]  # 8 قنوات FSR على الساعد
META_COLS = ["subject_id", "repetition", "gesture"]
DOWNSAMPLE = 5  # ~500Hz -> ~100Hz (نافذة 20 خطوة = 200ms كما في المشروع)
MAX_PER_CLASS = 4000  # real training budget


def _cap_per_class(X, y, subjects, cap, R=None):
    """عينة عشوائية طبقية بحد أقصى cap نافذة لكل فئة."""
    rng = np.random.default_rng(42)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    print(f"Capped to {len(keep):,} windows ({cap}/class)")
    if R is not None:
        return X[keep], y[keep], subjects[keep], R[keep]
    return X[keep], y[keep], subjects[keep]


def build_windows():
    """يقرأ CSV الضخم (684MB) ويبني النوافذ — يُخزّن مؤقتاً في npz."""
    import pandas as pd

    print(f"Reading {CSV_PATH} (this takes a minute or two) ...")
    df = pd.read_csv(CSV_PATH, usecols=ARM_COLS + META_COLS)
    print(f"  {len(df):,} rows")

    # الفئة = الإيماءة فقط بدون وضعية المعصم ("neutral closed" -> "closed")
    df["gesture"] = df["gesture"].str.split().str[-1]
    class_names = sorted(df["gesture"].unique())
    gid = df["gesture"].map({g: i for i, g in enumerate(class_names)}).to_numpy()

    subj = df["subject_id"].to_numpy()
    rep = df["repetition"].to_numpy()
    fmg = df[ARM_COLS].to_numpy(dtype=np.float32)

    # مفتاح التجربة: (subject, repetition, gesture-class) — الترتيب داخل الملف زمني
    key = ((subj.astype(np.int64) * 1000 + rep.astype(np.int64)) * 100 + gid)
    order = np.argsort(key, kind="stable")
    key, fmg, gid, subj = key[order], fmg[order], gid[order], subj[order]

    # حدود التجارب المتتالية
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]

    # تطبيع Min-Max عام (نفس أسلوب المشروع)
    fmin = fmg.min(axis=0)
    fmax = fmg.max(axis=0)
    fmg = (fmg - fmin) / (fmax - fmin + 1e-8)

    X_list, y_list, s_list = [], [], []
    for st, en in zip(starts, ends):
        seg = fmg[st:en:DOWNSAMPLE]
        if len(seg) < cfg.TIME_STEPS:
            continue
        label = gid[st]
        subject = subj[st]
        for start in range(0, len(seg) - cfg.TIME_STEPS + 1, cfg.WINDOW_STRIDE):
            X_list.append(seg[start:start + cfg.TIME_STEPS])
            y_list.append(label)
            s_list.append(subject)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(s_list, dtype=np.int64)

    print(f"Built {len(X):,} windows | {len(class_names)} gestures x "
          f"{len(np.unique(subjects))} subjects")
    for g, i in sorted(((g, i) for i, g in enumerate(class_names))):
        print(f"  {g}: {int((y == i).sum()):,} windows")

    np.savez_compressed(CACHE_PATH, X=X, y=y, subjects=subjects,
                        class_names=np.array(class_names))
    return X, y, subjects, class_names


def load_dataset():
    """
    يعيد (X, y, subjects, class_names, source, R).
      R: أصفار — لا أهداف ارتداد في هذه القاعدة (مهمة تصنيف فقط).
    """
    cfg.NUM_SENSORS = 8
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached windows from {CACHE_PATH}")
        d = np.load(CACHE_PATH, allow_pickle=False)
        X, y, subjects = d["X"], d["y"], d["subjects"]
        class_names = [str(g) for g in d["class_names"]]
    else:
        X, y, subjects, class_names = build_windows()

    cfg.NUM_PHASES = len(class_names)
    R = np.zeros((len(X), 2), dtype=np.float32)  # لا ارتداد
    source = ("Exoskelebox FMG benchmark (arXiv:2007.14918) - 8 forearm FSR, "
              "20 subjects, 6 gestures x 3 wrist orientations, ~100Hz after x5 downsample")

    X, y, subjects, R = _cap_per_class(X, y, subjects, MAX_PER_CLASS, R)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm], subjects[perm], class_names, source, R[perm]


if __name__ == "__main__":
    X, y, subjects, class_names, source, R = load_dataset()
    print(f"\nSource : {source}")
    print(f"X      : {X.shape}  y: {y.shape}  subjects: {len(np.unique(subjects))}")
    print(f"Classes: {class_names}")
