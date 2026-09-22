# JoTouch — AI & Control

مشروع تخرج: يد صناعية علوية مُشغَّلة بإشارات **FMG** (Force Myography) من ثمانية
حساسات على الساعد. هذا المستودع يضم نموذج الذكاء الاصطناعي وأنبوب البيانات
والتدريب والتصدير إلى العتاد.

## بنية المشروع

```
JoTouch_AI_module.py      ★ النموذج الرئيسي: JoTouchModel = MobileNetV2-1D + TCN
                            برأسين: تصنيف الإيماءة + انحدار زوايا المفاصل.
                            يحتوي أيضاً Dataset والتدريب والتقييم وقياس الزمن.

data_collection.py          جمع البيانات: قراءة الحساسات + استخراج 16 زاوية
                            مفصل بالكاميرا عبر MediaPipe -> ملف .npz
fmg_datasets/               مُحمِّلات مجموعات البيانات المفتوحة الثلاث
                              fmg_open_dataset.py   Zenodo MDPI FMG
                              fmg_exo_dataset.py    Exoskelebox
                              fmg_plos_dataset.py   PLoS ONE 2025

train_jotouch_real.py       تدريب النموذج الرئيسي على كل مجموعة بيانات
train_jotouch_boot.py       تدريب متحكّم الإقلاع (FMG -> أمر مستمر) للـ Teensy
loto_eval.py                تقييم Leave-One-Trial-Out
export_boot_package.py      تصدير الأوزان إلى ترويسات C لـ Teensy -> teensy_boot/
report_to_pdf.py            تحويل تقرير .md إلى PDF منسّق

weights/                    الأوزان النهائية (.pt) — هذه هي الملفات المهمة
results/                    تقارير النتائج (.md / .pdf)
runs/                       ملفات استئناف التدريب المؤقتة (يمكن حذفها)
teensy_boot/                مخرجات العتاد: ترويسات .h + boot_spec.json
Content/                    المستندات، العروض، وبيانات open_fmg_data
JoTouch_CAD/                كود CadQuery للتصميم الميكانيكي
alternative_models/         نماذج بديلة للمقارنة فقط (انظر README الخاص بها)
_to_delete/                 ملفات ميتة جاهزة للحذف اليدوي
```

## التشغيل

شغّل كل شيء من **جذر المشروع** — مُحمِّلات البيانات تحل مسار
`Content/open_fmg_data` نسبةً إلى مجلد العمل.

```bash
python train_jotouch_real.py          # تدريب النموذج الرئيسي
python train_jotouch_boot.py          # تدريب متحكّم الإقلاع
python export_boot_package.py         # تصدير أوزان الـ Teensy
python alternative_models/model_comparison.py   # مقارنة بالبدائل
```

## ثوابت النموذج

| الثابت | القيمة | المعنى |
|---|---|---|
| `TIME_STEPS` | 20 | نافذة 20 × 10ms = 200 ms |
| `WINDOW_STRIDE` | 5 | تحديث كل 50 ms |
| `NUM_SENSORS` | 8 | ثمانية حساسات FMG حول الساعد |
| `NUM_PHASES` | 4 | rest / wave / pinch / grip |
| `NUM_DOF` | 16 | زوايا المفاصل (نفس عدد ما يجمعه data_collection.py) |

**رأس الانحدار (`angle_head`) مُفعَّل.** خسارة الانحدار تُفعَّل تلقائياً فقط عندما
تحمل البيانات زوايا مفاصل حقيقية (`has_angles=True`)؛ مجموعات البيانات المفتوحة
تحمل تسميات إيماءات فقط، فتُدرَّب بالتصنيف وحده. الأوزان المحفوظة في `weights/`
حُفظت قبل تفعيل هذا الرأس، لذا سيطبع `load_model` تحذيراً واضحاً: التصنيف يعمل،
أما مخرجات الزوايا فبلا معنى حتى إعادة التدريب.
