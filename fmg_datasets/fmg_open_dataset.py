"""
Open-Source FMG Dataset Loader (8 channels)
===========================================
يحمّل قاعدة بيانات FMG مفتوحة المصدر ويجهّزها بنفس صيغة مشروع JoTouch:

  المصدر: "Dataset on Force Myography for Human Robot Interactions"
          U. Zakia & C. Menon, MDPI Data 2022 — Zenodo record 6632020
          https://zenodo.org/records/6632020

  نستخدم ملفات الـ Biaxial Stage:
    - كل ملف CSV: تجربة واحدة لمشارك واحد يؤدي حركة واحدة (بدون Header).
    - العمودان الأول والثاني: إحداثيات المنصة (X, Y).
    - الأعمدة 3..34 : قنوات FMG عدد 32 — نأخذ أول 8 قنوات فقط
      لمطابقة مواصفة مشروعنا (8 حساسات FMG).
    - الفئة (Class): نوع الحركة من اسم الملف
      (1D_X, 1D_Y, 2D_DG, 2D_SQ, 2D_DM) → 5 فئات.
    - المشارك (Subject): S1..S15 — يُستخدم في اختبار التعميم
      (Cross-Subject / Held-Out-Subject) لقياس مقاومة الـ Overfitting.

إذا فشل التحميل (شبكة/صيغة)، نرجع تلقائياً إلى بيانات اصطناعية
بـ 8 حساسات مع توضيح ذلك في النتائج.

الواجهة:
    load_dataset() -> (X, y, subjects, class_names, source)
      X         : float32 [N, TIME_STEPS, 8]
      y         : int64   [N]
      subjects  : int64   [N]  رقم المشارك لكل نافذة
      class_names: list[str]
      source    : str   وصف مصدر البيانات المستخدم فعلياً
"""

import os
import re
import json
import urllib.request

import numpy as np

# --- path bootstrap: هذا الملف داخل fmg_datasets/ بينما JoTouch_AI_module في الجذر
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import JoTouch_AI_module as cfg

ZENODO_RECORD = "6632020"
DATA_DIR = os.path.join("Content", "open_fmg_data")
FILE_PREFIX = "pHRI_BiaxialStage_"
NUM_CHANNELS = 8          # نأخذ 8 قنوات فقط من أصل 32 (مطابقة مواصفة المشروع)
SKIP_COLUMNS = 2          # أول عمودين = إحداثيات المنصة X,Y وليست FMG
MAX_PER_CLASS = 5000      # real training: effectively the full Zenodo dataset


# ------------------------------------------------------------------
# 1. Download
# ------------------------------------------------------------------
def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "jotouch-benchmark"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_dataset(data_dir: str = DATA_DIR) -> list:
    """يحمّل ملفات Biaxial Stage من Zenodo ويعيد قائمة المسارات المحلية."""
    os.makedirs(data_dir, exist_ok=True)

    record = _fetch_json(f"https://zenodo.org/api/records/{ZENODO_RECORD}")
    wanted = [f for f in record["files"] if f["key"].startswith(FILE_PREFIX)]
    print(f"Zenodo record {ZENODO_RECORD}: {len(wanted)} biaxial-stage files")

    local_paths = []
    for i, f in enumerate(wanted):
        dest = os.path.join(data_dir, f["key"])
        if not os.path.exists(dest) or os.path.getsize(dest) != f["size"]:
            url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{f['key']}/content"
            print(f"  [{i+1}/{len(wanted)}] downloading {f['key']} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "jotouch-benchmark"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
                out.write(resp.read())
        local_paths.append(dest)

    return local_paths


# ------------------------------------------------------------------
# 2. Parse & window
# ------------------------------------------------------------------
def _parse_filename(path: str):
    """pHRI_BiaxialStage_S15_2D_DG_Rep1.csv -> ('S15', '2D_DG')"""
    name = os.path.basename(path)
    m = re.match(r"pHRI_BiaxialStage_(S\d+)_((?:1D|2D)_[A-Z]+)(?:_Session\d+|_diffsize)?_Rep\d+\.csv", name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _load_csv_tolerant(path: str):
    """يقرأ CSV بدون Header ويعيد (fmg[T,8], target[T,2]).

    - الأعمدة 0,1 : إحداثيات المنصة X,Y — تُستخدم كهدف لمهمة الارتداد (Regression).
    - الأعمدة 2..9: قنوات FMG الثمانية.
    يتجاهل الصفوف الفارغة/المعطوبة.
    """
    fmg_rows, tgt_rows = [], []
    with open(path, "r") as fh:
        for line in fh:
            parts = [v for v in line.strip().split(",") if v.strip() != ""]
            if len(parts) >= SKIP_COLUMNS + NUM_CHANNELS:
                try:
                    vals = [float(v) for v in parts[:SKIP_COLUMNS + NUM_CHANNELS]]
                except ValueError:
                    continue
                tgt_rows.append(vals[:SKIP_COLUMNS])
                fmg_rows.append(vals[SKIP_COLUMNS:SKIP_COLUMNS + NUM_CHANNELS])
    if not fmg_rows:
        return (np.empty((0, 0), dtype=np.float32),) * 2
    return (np.array(fmg_rows, dtype=np.float32),
            np.array(tgt_rows, dtype=np.float32))


def build_windows(paths: list):
    """يبني نوافذ [TIME_STEPS, 8] من ملفات CSV مع تطبيع Min-Max عام + أهداف الارتداد."""
    trials = []  # (subject_str, motion_str, fmg[T,8], tgt[T,2])
    for p in paths:
        subject, motion = _parse_filename(p)
        if subject is None:
            continue
        fmg, tgt = _load_csv_tolerant(p)
        if fmg.ndim != 2 or fmg.shape[1] != NUM_CHANNELS or tgt.shape[0] != fmg.shape[0]:
            continue
        if fmg.shape[0] >= cfg.TIME_STEPS:
            trials.append((subject, motion, fmg, tgt))

    if not trials:
        raise ValueError("No usable trials parsed from downloaded files.")

    # Global min-max normalization (نفس أسلوب load_csv_data في المشروع)
    all_vals = np.concatenate([t[2] for t in trials], axis=0)
    fmin = all_vals.min(axis=0)
    fmax = all_vals.max(axis=0)
    # تطبيع أهداف الارتداد أيضاً
    all_tgt = np.concatenate([t[3] for t in trials], axis=0)
    tmin = all_tgt.min(axis=0)
    tmax = all_tgt.max(axis=0)

    # حفظ إحصاءات التطبيع — ضرورية لتطبيع بيانات الحساسات على Teensy بنفس الطريقة
    np.savez(os.path.join(DATA_DIR, "zenodo_norm_stats.npz"),
             fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)

    subjects_str = sorted({t[0] for t in trials}, key=lambda s: int(s[1:]))
    motions = sorted({t[1] for t in trials})
    subj_to_id = {s: i for i, s in enumerate(subjects_str)}
    mot_to_id = {m: i for i, m in enumerate(motions)}

    X_list, y_list, s_list, r_list = [], [], [], []
    for subject, motion, fmg, tgt in trials:
        fmg = (fmg - fmin) / (fmax - fmin + 1e-8)
        tgt = (tgt - tmin) / (tmax - tmin + 1e-8)
        for start in range(0, len(fmg) - cfg.TIME_STEPS + 1, cfg.WINDOW_STRIDE):
            window = fmg[start:start + cfg.TIME_STEPS]
            if window.shape == (cfg.TIME_STEPS, NUM_CHANNELS):
                X_list.append(window)
                y_list.append(mot_to_id[motion])
                s_list.append(subj_to_id[subject])
                # هدف الارتداد: متوسط موضع المنصة خلال النافذة
                r_list.append(tgt[start:start + cfg.TIME_STEPS].mean(axis=0))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(s_list, dtype=np.int64)
    R = np.array(r_list, dtype=np.float32)

    print(f"Built {len(X)} windows from {len(trials)} trials "
          f"({len(subjects_str)} subjects x {len(motions)} motions)")
    for m, mid in mot_to_id.items():
        print(f"  {m}: {int((y == mid).sum())} windows")

    return X, y, subjects, motions, R


# ------------------------------------------------------------------
# 3. Synthetic fallback
# ------------------------------------------------------------------
def _synthetic_fallback():
    """بيانات اصطناعية بـ 8 حساسات و 5 فئات عند فشل تحميل البيانات الحقيقية."""
    print("Falling back to SYNTHETIC 8-sensor data (results marked accordingly).")
    cfg.NUM_PHASES = 5
    dataset = cfg.generate_synthetic_data(num_samples=5000, noise_level=0.05)
    X = dataset.X.numpy()
    y = dataset.y_phase.numpy()
    # جلسات وهمية (5 مجموعات متتالية) لإتاحة اختبار التعميم
    subjects = (np.arange(len(X)) * 5 // len(X)).astype(np.int64)
    class_names = [f"class_{i}" for i in range(cfg.NUM_PHASES)]
    R = np.zeros((len(X), 2), dtype=np.float32)  # لا أهداف ارتداد في الوضع الاصطناعي
    return X, y, subjects, class_names, R


# ------------------------------------------------------------------
# 4. Public API
# ------------------------------------------------------------------
def load_dataset():
    """
    يعيد (X, y, subjects, class_names, source, R).
      R: float32 [N, 2] أهداف الارتداد (موضع المنصة X,Y مُطبّع) لكل نافذة.
    يضبط cfg.NUM_SENSORS = 8 دائماً لأن البيانات المفتوحة 8 قنوات.
    """
    cfg.NUM_SENSORS = NUM_CHANNELS
    try:
        paths = download_dataset()
        X, y, subjects, class_names, R = build_windows(paths)
        source = ("Zenodo 6632020 (Zakia & Menon, MDPI Data 2022) - "
                  "real FMG, first 8 of 32 channels, biaxial-stage motions")
    except Exception as e:
        print(f"Open dataset unavailable ({e}).")
        X, y, subjects, class_names, R = _synthetic_fallback()
        source = "SYNTHETIC fallback (generate_synthetic_data, 8 sensors)"

    cfg.NUM_PHASES = len(class_names)

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
    print(f"Classes: {class_names}  R: {R.shape}")
