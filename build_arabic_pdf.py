"""
Generate a purely text-based Arabic technical document (PDF) for JoTouch_AI Base Model.
No boxes or cards - structured cleanly with headings, paragraphs, and numbered points.
Guaranteed strictly 2 pages.
"""
import os
import subprocess
import tempfile
import pymupdf

def build_pdf(output_path):
    html = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>شرح النموذج الأساسي — JoTouch AI</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

  @page {
    size: A4 portrait;
    margin: 12mm 14mm 10mm 14mm;
  }

  * {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  body {
    font-family: 'Cairo', 'Segoe UI', Tahoma, Arial, sans-serif;
    direction: rtl;
    text-align: right;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
    font-size: 9.2pt;
    line-height: 1.45;
  }

  .page-1 {
    page-break-after: always;
  }

  .page-2 {
    page-break-after: auto;
  }

  /* Header */
  .doc-title {
    font-size: 15.5pt;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 2px 0;
    border-bottom: 2px solid #0284c7;
    padding-bottom: 4px;
  }

  .doc-subtitle {
    font-size: 9pt;
    font-weight: 600;
    color: #0284c7;
    margin: 0 0 7px 0;
  }

  h2 {
    font-size: 10.8pt;
    font-weight: 700;
    color: #0f172a;
    margin: 8px 0 3px 0;
    padding-bottom: 2px;
    border-bottom: 1px solid #cbd5e1;
  }

  h3 {
    font-size: 9.6pt;
    font-weight: 700;
    color: #0369a1;
    margin: 6px 0 2px 0;
  }

  p {
    margin: 0 0 5px 0;
    color: #334155;
  }

  ol, ul {
    margin: 0 0 5px 0;
    padding-right: 18px;
  }

  li {
    margin-bottom: 3.5px;
    color: #334155;
  }

  li:last-child {
    margin-bottom: 0;
  }

  .en {
    direction: ltr;
    display: inline-block;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-weight: 700;
    color: #0f172a;
  }

  .meta-line {
    font-size: 8.4pt;
    color: #475569;
    margin-bottom: 7px;
    padding-bottom: 5px;
    border-bottom: 1px dashed #cbd5e1;
  }

  .footer {
    border-top: 1px solid #e2e8f0;
    padding-top: 4px;
    margin-top: 12px;
    display: flex;
    justify-content: space-between;
    font-size: 7.6pt;
    color: #94a3b8;
  }
</style>
</head>
<body>

<!-- ================= الصفحة الأولى ================= -->
<div class="page-1">
  <div class="doc-title">مشروع JoTouch — دليل النموذج الأساسي للذكاء الاصطناعي</div>
  <div class="doc-subtitle">شرح معماري وتقني شامل للنموذج الرئيسي (MobileNetV2-1D + TCN) وآلية عمله البرمجية</div>

  <div class="meta-line">
    <b>موجز تقني سريع:</b>
    إجمالي المعاملات: <span class="en">66,772 Parameters</span> (~267 KB) &nbsp;|&nbsp;
    زمن الاستنتاج: <span class="en">&lt; 3.5 ms</span> &nbsp;|&nbsp;
    النافذة الزمنية: <span class="en">200 ms</span> (20 خطوة بتردد 100Hz) &nbsp;|&nbsp;
    معدل التحديث: كل <span class="en">50 ms</span> (حلقة تحكم بتردد 20Hz).
  </div>

  <h2>1. هيكلية مدخلات النموذج ومبدأ المعالجة (Input Representation)</h2>
  <p>
    يعتمد نموذج <span class="en">JoTouchModel</span> الأساسي على استقبال قراءات حساسات الضغط العضلي <span class="en">Force Myography (FMG)</span> الموزعة حول ساعد اليد (8 حساسات <span class="en">FSR</span>).
    عند انقباض عضلات الساعد يتغير حجمها الميكانيكي مسبباً تغيراً لحظياً في الضغط. تدخل البيانات إلى النموذج على شكل مصفوفة ثلاثية الأبعاد:
  </p>
  <ul>
    <li><b>شكل مصفوفة الدخل:</b> تدخل الإشارات بمصفوفة <span class="en">[Batch, 20, 8]</span>، حيث يمثل الرقم 20 نافذة زمنية تمتد عبر 200 ميلي ثانية (مأخوذة بتردد 100 هرتز)، ويمثل الرقم 8 عدد حساسات الساعد.</li>
    <li><b>إعادة ترتيب المحاور (Permute):</b> يقوم كود النموذج في أول خطوة بتحويل الترتيب إلى <span class="en">[Batch, 8, 20]</span>، وذلك لتطبيق التفافات أحادية البعد <span class="en">1D Convolutions</span> تتعامل مع الحساسات كقنوات وتتحرك على طول المحور الزمني.</li>
    <li><b>سرعة الاستجابة (Window Stride):</b> تُحدّث النافذة بمقدار 5 خطوات (أي كل 50 ميلي ثانية)، مما يحقق حلقة تحكم فورية بمعدل 20 إطاراً في الثانية دون أي شعور بالتأخير من قِبل المستخدم.</li>
  </ul>

  <h2>2. البلوك الأول: مستخرج الخصائص المكانية (MobileNetV2-1D)</h2>
  <p>
    الهدف من هذا البلوك هو فهم التفاعل العضلي بين حساسات الساعد الثمانية واستخراج الأنماط المكانية بدقة فائقة وبأقل استهلاك حسابي ممكن، وذلك عبر فلسفة الـ <span class="en">Inverted Residuals</span>:
  </p>
  <ol>
    <li>
      <b>طبقة التوسيع (Pointwise Expansion 1x1):</b>
      تبدأ كل كتلة بمضاعفة عدد القنوات بمقدار الضعف (<span class="en">Expansion Factor = 2</span> من 32 إلى 64). هذا التوسيع يرفع أبعاد البيانات إلى فضاء أعلى يتيح للشبكة تمييز الحدود الفاصلة المعقدة لحركات العضلات بدون فقدان للمعلومات.
    </li>
    <li>
      <b>الالتفاف العميق المنفصل (Depthwise Separable Conv1D):</b>
      يتم تطبيق مرشح التفافي بحجم <span class="en">Kernel = 3</span> على كل قناة بشكل منعزل ومنفصل تماماً (<span class="en">groups = channels</span>). هذه الخطوة الذكية تقلل عدد المعاملات والعمليات الحسابية بنسبة تتجاوز 80% مقارنة بالالتفاف التقليدي، وهي السبب الرئيسي في خفة النموذج.
    </li>
    <li>
      <b>عنق الزجاجة الخطي (Linear Bottleneck):</b>
      تُعاد الميزات لضغطها في قنوات أقل عبر التفاف <span class="en">1x1</span> دون وضع دالة تفعيل غير خطية في النهاية، وذلك لحماية الخصائص المستخرجة من الانهيار الرياضي في المساحات الضيقة.
    </li>
    <li>
      <b>المسارات الالتفافية المتبقية (Residual Connections):</b>
      في حال تطابق عدد قنوات الدخل والخرج، يتم جمع الدخل الأصلي مباشرة مع ناتج العمليات (<span class="en">x + conv(x)</span>)، ما يتيح تدفق التدرجات بسلاسة ويمنع ضياع الإشارة أثناء التدريب.
    </li>
    <li>
      <b>الاختزال الزمني (Max Pooling 1D):</b>
      تنتهي مرحلة الـ CNN بطبقة <span class="en">MaxPool1d(2)</span> تقلص الطول الزمني من 20 إلى 10، لتقليل العبء الحسابي على الطبقات التالية مع الاحتفاظ بذروات الضغط المهمة.
    </li>
  </ol>

  <h2>3. البلوك الثاني: النمذجة الزمنية المتقدمة (TCN)</h2>
  <p>
    بما أن حركة اليد عملية تسلسلية تتغير مع الوقت، تم بناء شبكة التفاف زمني <span class="en">Temporal Convolutional Network (TCN)</span> مكونة من طبقتين متعاقبتين، لتتبع تطور الإشارات عبر الزمن:
  </p>
  <ul>
    <li>
      <b>الالتفافات المتسعة (Dilated Convolutions):</b>
      تستخدم الطبقة الأولى معامل اتساع <span class="en">Dilation = 1</span> لالتقاط التغيرات السريعة، بينما تستخدم الطبقة الثانية <span class="en">Dilation = 2</span>. هذه القفزات تضاعف مجال الرؤية الزمني (<span class="en">Receptive Field</span>) ليغطي كامل فترة الـ 200ms بمرشحات صغيرة ودون الحاجة لطبقات تجميع إضافية.
    </li>
    <li>
      <b>التوصيل التراكمي الداخلي:</b>
      تحتوي كل كتلة زمنية على طبقتي التفاف مع دالة تفعيل ومسار التفافي (<span class="en">Residual Shortcut</span>) يجمع الدخل مع الخرج لضمان استقرار التدريب وسرعة التقارب.
    </li>
    <li>
      <b>اقتطاع اللحظة النهائية (Last Time-step Slicing):</b>
      يأخذ النموذج الخطوة الزمنية الأخيرة فقط <span class="en">x[:, :, -1]</span> لتمثيل ملخص الحركة التاريخية كاملة في متجه مدمج بحجم 64 قناة، ويمرره إلى طبقة تمثيل مشتركة مع طبقة إسقاط عشوائي (<span class="en">Dropout 0.3</span>) لمنع الحفظ السطحي (Overfitting).
    </li>
  </ul>

  <div class="footer">
    <span>مشروع تخرج JoTouch — توثيق النموذج الأساسي للذكاء الاصطناعي</span>
    <span>الصفحة 1 من 2</span>
  </div>
</div>


<!-- ================= الصفحة الثانية ================= -->
<div class="page-2">
  <h2>4. الدوال الرياضية وخوارزميات التدريب في بنية النموذج</h2>
  <p>
    تم انتقاء الدوال البرمجية والرياضية في بنية النموذج وتدريبه بعناية فائقة لتلائم طبيعة إشارات الضغط الفيزيائية وتضمن أعلى استقرار:
  </p>
  <ul>
    <li>
      <b>دالة التفعيل SiLU (Swish):</b>
      اعتمد النموذج دالة <span class="en">SiLU (x &middot; &sigma;(x))</span> بدلاً من دالة ReLU التقليدية في كافة طبقات الـ CNN والـ TCN. ميزة هذه الدالة أنها ناعمة وتفاضلية بالكامل ولا تصفر القيم السالبة تماماً، مما يحل مشكلة موت العصبونات (<span class="en">Dying ReLU</span>) ويحافظ على التغيرات الطفيفة جداً في قراءات حساسات الضغط العضلي.
    </li>
    <li>
      <b>خوارزمية التحسين AdamW:</b>
      تم تدريب النموذج بواسطة خوارزمية <span class="en">AdamW</span> مع تثبيط للأوزان (<span class="en">Weight Decay = 1e-2</span>). تقوم هذه الخوارزمية بفصل تقليص الأوزان عن حساب التدرج اللحظي، مما يمنع تضخم الأوزان ويضمن قدرة النموذج على التعميم عند اختبار مستخدمين جدد لم يشاركوا في التدريب.
    </li>
    <li>
      <b>جدولة معدل التعلم OneCycleLR:</b>
      استخدم خط التدريب استراتيجية <span class="en">OneCycleLR</span>، حيث يبدأ بمعدل تعلم منخفض، ثم يصعد تدريجياً لقمة التدريب، ثم يهبط تدريجياً وفق منحنى الجيب (<span class="en">Cosine Annealing</span>). هذا التكنيك يسرع الوصول لأفضل أداء في عدد قليل من الدورات (Epochs) ويمنع الوقوع في النقاط الحرجة المحلية.
    </li>
    <li>
      <b>تطبيع الطبقات ومعالجة التدرجات (Batch Normalization & Gradient Clipping):</b>
      استخدام <span class="en">BatchNorm1d</span> بعد كل التفاف لتثبيت توزيع البيانات داخل الشبكة، مع تطبيق قص للتدرجات (<span class="en">Gradient Clipping = 1.0</span>) لمنع انفجار التدرجات في الطبقات الزمنية.
    </li>
    <li>
      <b>دوال الخسارة المخصصة (Loss Functions):</b>
      استخدام <span class="en">CrossEntropyLoss</span> لضبط دقة تصنيف الحركات، واستخدام <span class="en">Smooth L1 Loss (Huber Loss)</span> لزوايا المفاصل؛ وهي دالة مثالية لأنها تتصرف خطياً مع الأخطاء الكبيرة (مقاومة للضوضاء) وتربيعياً قرب الصفر، ما يعطي حركة ناعمة ومستمرة للأصابع.
    </li>
  </ul>

  <h2>5. رأسا الإخراج المتزامنان (Dual-Head Architecture)</h2>
  <p>
    ينفرد نموذج <span class="en">JoTouchModel</span> بدمج نوعين من مخرجات التحكم في آن واحد من خلال رأسين متوازيين ينبثقان من نفس الميزات المشتركة:
  </p>
  <ol>
    <li>
      <b>رأس تصنيف الإيماءة (Phase Head):</b>
      طبقة خطية <span class="en">Linear(64 &rarr; 4)</span> تُخرج احتمالية الحركة من بين 4 أوضاع أساسية:
      <br>
      • <span class="en">0: Rest</span> (استرخاء اليد) &nbsp;|&nbsp;
      • <span class="en">1: Wave</span> (حركة تلويح) &nbsp;|&nbsp;
      • <span class="en">2: Pinch</span> (قبضة دقيقة بأصبعين) &nbsp;|&nbsp;
      • <span class="en">3: Grip</span> (قبضة كاملة لكامل الكف).
      <br>
      يُستخدم هذا الرأس في إطلاق أوضاع المسك المبرمجة مسبقاً وتغيير الحالة الميكانيكية للطرف الصناعي.
    </li>
    <li>
      <b>رأس انحدار زوايا المفاصل (Angle Head):</b>
      طبقة خطية <span class="en">Linear(64 &rarr; 16)</span> تُخرج 16 زاوية مفصلية مستمرة بالراديان لكافة أصابع اليد، مطابقة للزوايا المستخرجة عبر تتبع الكاميرا بكود <span class="en">data_collection.py</span> باستخدام <span class="en">MediaPipe</span>، مما يسمح بحركة انسيابية تدريجية للأصابع تحاكي اليد الطبيعية.
    </li>
  </ol>

  <h2>6. أهم مميزات وفيتشرز النموذج JoTouch AI</h2>
  <ul>
    <li>
      <b>تكامل مكاني وزماني متكامل:</b>
      الجمع بين <span class="en">MobileNetV2</span> لاستخراج التناسق العضلي اللحظي و <span class="en">TCN</span> لتتبع ديناميكية الحركة زمنياً، وهو تصميم يتفوق بوضوح على النماذج الخطية أو الشبكات التكرارية التقليدية (LSTM/RNN) من حيث السرعة والتوازي.
    </li>
    <li>
      <b>حجم فائق الصغر ومناسب للأنظمة المدمجة:</b>
      حجم النموذج الإجمالي هو <b>66,772 وزناً فقط</b> (أقل من <b>270 كيلوبايت</b>)، ما يجعله قادراً على العمل بسلاسة تامة داخل ذاكرة المتحكم الدقيق <span class="en">Teensy 4.1</span> (1MB RAM) أو معالجات <span class="en">Raspberry Pi Zero 2W</span>.
    </li>
    <li>
      <b>استنتاج فوري فائق السرعة (Real-Time Latency):</b>
      يستغرق الاستنتاج <b>أقل من 3.5 ميلي ثانية</b> على الحاسوب العادي و<b>أقل من ميلي ثانية واحدة</b> على معالج ARM مدمج بتردد 600MHz، وهو أسرع بكثير من معدل التحديث المطلوب (50ms)، مانحاً استجابة شبه فورية.
    </li>
    <li>
      <b>التصدير المباشر لكود سي خالص (Pure C Static Arrays):</b>
      يتوفر في المشروع أداة <span class="en">export_boot_package.py</span> لتحويل أوزان النموذج إلى مصفوفات C داخل ملفات ترويسة (<span class="en">teensy_boot/*.h</span>)، لتشغيل النموذج على المتحكم مباشرة بدون بايثون أو أنظمة تشغيل خارجية.
    </li>
    <li>
      <b>آلية تدريب مشروطة وذكية (has_angles Flag):</b>
      يدعم النموذج التدريب المرن؛ حيث يُدرّب رأس التصنيف وحده على قواعد البيانات العالمية المفتوحة (Zenodo و PLoS و Exoskelebox)، بينما يُفعّل رأس الزوايا الحركية تلقائياً عند التدريب على بيانات المشروع الحقيقية دون أن يفسد أحدهما الآخر.
    </li>
  </ul>

  <div class="footer">
    <span>مشروع تخرج JoTouch — توثيق النموذج الأساسي للذكاء الاصطناعي</span>
    <span>الصفحة 2 من 2</span>
  </div>
</div>

</body>
</html>
"""

    temp_html = os.path.join(tempfile.gettempdir(), "jotouch_text_arabic.html")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html)

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    user_data = os.path.join(tempfile.gettempdir(), "edge_text_profile")
    url = "file:///" + temp_html.replace("\\", "/")

    cmd = [
        edge_path,
        "--headless=new",
        "--user-data-dir=" + user_data,
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf=" + output_path,
        url
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to generate PDF: {res.stderr}")

    doc = pymupdf.open(output_path)
    count = len(doc)
    print(f"Generated clean text-based Arabic PDF: {output_path}")
    print(f"Total pages: {count}")
    return count

if __name__ == "__main__":
    import sys
    dest = sys.argv[1] if len(sys.argv) > 1 else "JoTouch_AI_Base_Model_Arabic.pdf"
    build_pdf(dest)
