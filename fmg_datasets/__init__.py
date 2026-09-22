"""
مُحمِّلات مجموعات بيانات FMG المفتوحة.

كل وحدة هنا تقرأ مجموعة بيانات منشورة وتحوّلها إلى نوافذ
[N, TIME_STEPS, NUM_SENSORS] جاهزة لنموذج JoTouch:

    fmg_open_dataset  -> Zenodo MDPI FMG   (Zakia & Menon, 2022)
    fmg_exo_dataset   -> Exoskelebox       (arXiv:2007.14918)
    fmg_plos_dataset  -> PLoS ONE 2025     (Young et al., e0321319)

الاستخدام:
    from fmg_datasets import fmg_open_dataset

ملاحظة: هذه الوحدات تحل مسارات البيانات (Content/open_fmg_data) نسبةً إلى
مجلد العمل، لذا شغّل السكربتات من جذر المشروع.
"""
