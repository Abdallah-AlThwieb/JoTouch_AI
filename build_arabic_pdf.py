"""
Build Arabic PDF for JoTouch_AI Base Model Overview.
Uses clean HTML/CSS and Microsoft Edge headless renderer for perfect RTL typography and layout.
"""
import os
import subprocess
import tempfile
import pymupdf

def generate_arabic_pdf(output_pdf_path):
    html_content = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>شرح النموذج الأساسي — JoTouch AI</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

  @page {
    size: A4 portrait;
    margin: 10mm 12mm 10mm 12mm;
  }

  * {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  body {
    font-family: 'Cairo', 'Segoe UI', Tahoma, Arial, sans-serif;
    direction: rtl;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
    font-size: 9.6pt;
    line-height: 1.45;
  }

  .page {
    width: 100%;
    height: 275mm;
    max-height: 275mm;
    position: relative;
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .page:last-child {
    page-break-after: avoid;
  }

  /* Header Bar */
  .header {
    border-bottom: 2px solid #0284c7;
    padding-bottom: 6px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header-titles h1 {
    font-size: 16pt;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 2px 0;
    letter-spacing: -0.3px;
  }

  .header-titles p {
    font-size: 9pt;
    color: #0284c7;
    font-weight: 600;
    margin: 0;
  }

  .header-badge {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    color: #0369a1;
    font-weight: 700;
    font-size: 8pt;
    padding: 4px 10px;
    border-radius: 6px;
    text-align: center;
  }

  /* Stats Quick Bar */
  .stats-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 9px;
  }

  .stat-box {
    flex: 1;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 6px 8px;
    text-align: center;
  }

  .stat-label {
    font-size: 7.5pt;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 1px;
  }

  .stat-value {
    font-size: 10pt;
    font-weight: 800;
    color: #0f172a;
    direction: ltr;
    display: inline-block;
  }

  /* Section Titles */
  .section-title {
    font-size: 11pt;
    font-weight: 700;
    color: #0f172a;
    margin: 6px 0 5px 0;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .section-title::before {
    content: '';
    display: inline-block;
    width: 4px;
    height: 14px;
    background: #0284c7;
    border-radius: 2px;
  }

  /* Cards */
  .card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 7px;
    padding: 8px 11px;
    margin-bottom: 7px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }

  .card-highlight {
    background: #f8fafc;
    border-right: 3.5px solid #0284c7;
  }

  .card-title {
    font-size: 9.8pt;
    font-weight: 700;
    color: #0369a1;
    margin: 0 0 4px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .card p {
    margin: 0 0 4px 0;
    color: #334155;
  }

  .card p:last-child {
    margin-bottom: 0;
  }

  /* English Term Badge */
  .badge {
    background: #f1f5f9;
    color: #0f172a;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 8.5pt;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    font-weight: 600;
    display: inline-block;
    direction: ltr;
    unicode-bidi: embed;
    border: 1px solid #cbd5e1;
  }

  .badge-blue {
    background: #e0f2fe;
    color: #0369a1;
    border-color: #bae6fd;
  }

  .badge-green {
    background: #dcfce7;
    color: #15803d;
    border-color: #bbf7d0;
  }

  .badge-purple {
    background: #f3e8ff;
    color: #7e22ce;
    border-color: #e9d5ff;
  }

  /* List Items */
  .feature-list {
    margin: 3px 0 0 0;
    padding-right: 14px;
  }

  .feature-list li {
    margin-bottom: 3.5px;
    color: #334155;
    font-size: 9.2pt;
  }

  .feature-list li:last-child {
    margin-bottom: 0;
  }

  .feature-list strong {
    color: #0f172a;
  }

  /* Two Column Layout */
  .grid-2 {
    display: flex;
    gap: 8px;
    margin-bottom: 6px;
  }

  .grid-2 > div {
    flex: 1;
  }

  /* Footer */
  .footer {
    border-top: 1px solid #e2e8f0;
    padding-top: 4px;
    display: flex;
    justify-content: space-between;
    font-size: 7.5pt;
    color: #94a3b8;
    margin-top: auto;
  }
</style>
</head>
<body>

<!-- ==================== الصفحة الأولى ==================== -->
<div class="page">
  <div>
    <!-- الترويسة -->
    <div class="header">
      <div class="header-titles">
        <h1>مشروع JoTouch — النموذج الأساسي للذكاء الاصطناعي</h1>
        <p>المعمارية التقنية للنموذج الرئيسي: MobileNetV2-1D + TCN مع رأسي إخراج متزامنين</p>
      </div>
      <div class="header-badge">
        وثيقة تقنية مبسطة<br>موجّهة لفريق العمل
      </div>
    </div>

    <!-- شريط الأرقام السريعة -->
    <div class="stats-bar">
      <div class="stat-box">
        <div class="stat-label">إجمالي المعاملات (Parameters)</div>
        <div class="stat-value">66,772</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">حجم الأوزان في الذاكرة</div>
        <div class="stat-value">~267 KB (FP32)</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">زمن الاستنتاج (Latency)</div>
        <div class="stat-value">&lt; 3.5 ms</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">أبعاد نافذة الدخل</div>
        <div class="stat-value">20 Steps &times; 8 FSR</div>
      </div>
    </div>

    <!-- طبيعة الدخل والإشارة -->
    <div class="card card-highlight">
      <div class="card-title">
        <span>1. مدخلات النموذج ومبدأ القياس (Input Representation)</span>
        <span class="badge badge-blue">100 Hz &rarr; 20 Hz Control</span>
      </div>
      <p>
        يعتمد النموذج على إشارات <b>Force Myography (FMG)</b> القادمة من <b>8 حساسات ضغط (FSR)</b> موزعة حول ساعد اليد.
        عند انقباض العضلات يتغير حجمها الميكانيكي مسبباً تغيراً لحظياً في الضغط.
      </p>
      <ul class="feature-list">
        <li><b>أبعاد مصفوفة الدخل:</b> تدخل الإشارات بمصفوفة أبعادها <span class="badge">[Batch, 20, 8]</span>، حيث يمثل الرقم 20 نافذة زمنية مدتها <b>200 ميلي ثانية</b> (مأخوذة بتردد 100 هرتز)، ويمثل الرقم 8 عدد حساسات الساعد.</li>
        <li><b>تحويل الأبعاد الداخلي:</b> يقوم النموذج فوراً بقلب المحاور لتصبح <span class="badge">[Batch, 8, 20]</span> لتطبيق التفافات أحادية البعد <span class="badge">1D Convolutions</span> عالية الكفاءة عبر قنوات الحساسات.</li>
        <li><b>معدل التحديث (Stride):</b> تُزاح النافذة بمقدار 5 خطوات (أي كل <b>50 ميلي ثانية</b>)، مما يمنح اليد الصناعية حلقة تحكم فورية بتردد <b>20 هرتز</b> تمنع أي شعور بالتأخير.</li>
      </ul>
    </div>

    <!-- البلوك الأول: MobileNetV2-1D -->
    <div class="section-title">2. البلوك الأول: استخراج الخصائص العضلية (MobileNetV2-1D)</div>
    <div class="card">
      <p style="margin-bottom: 6px;">
        تم استلهام هذا البلوك من معمارية <b>MobileNetV2</b> وتكييفها لتعمل مع الإشارات أحادية البعد <span class="badge">1D Signals</span>.
        وظيفته الأساسية هي فهم وتفكيك التفاعل المكاني بين عضلات الساعد الثمانية بأقل عدد ممكن من العمليات الحسابية:
      </p>
      
      <div class="grid-2">
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 7px;">
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 2px; font-size: 9pt;">
            أ. طبقة التوسيع (Pointwise Expansion)
          </div>
          <p style="font-size: 8.8pt; color: #475569; margin: 0;">
            تستخدم التفاف <span class="badge">1x1 Conv1D</span> مع معامل توسيع <span class="badge">Expansion = 2</span> لمضاعفة عدد القنوات (من 32 إلى 64). الهدف هو رفع أبعاد البيانات لتمكين النموذج من تعلم فواصل لاخطية معقدة دون فقدان الخصائص.
          </p>
        </div>

        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 7px;">
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 2px; font-size: 9pt;">
            ب. الالتفاف العميق المنفصل (Depthwise Conv)
          </div>
          <p style="font-size: 8.8pt; color: #475569; margin: 0;">
            تطبق فلترة التفافية <span class="badge">Kernel = 3</span> على كل قناة بمفردها بشكل منفصل تماماً (<span class="badge">groups = hidden_dim</span>)، مما يخفض العمليات الحسابية والمعاملات بنسبة هائلة مقارنة بالالتفاف العادي.
          </p>
        </div>
      </div>

      <div class="grid-2">
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 7px;">
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 2px; font-size: 9pt;">
            ج. طبقة الإسقاط الخطي (Linear Bottleneck)
          </div>
          <p style="font-size: 8.8pt; color: #475569; margin: 0;">
            تُعيد ضغط القنوات إلى الحجم المطلوب عبر <span class="badge">1x1 Conv1D</span> مع حذف دالة التفعيل في النهاية، لمنع انهيار وتموت الخصائص الرياضية في الطبقات الضيقة.
          </p>
        </div>

        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 7px;">
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 2px; font-size: 9pt;">
            د. التوصيلات الالتفافية (Residual Connection)
          </div>
          <p style="font-size: 8.8pt; color: #475569; margin: 0;">
            عند تطابق أبعاد الدخل والخرج، يتم جمع الدخل مباشرة مع الخرج (<span class="badge">x + conv(x)</span>)، مما يضمن تدفق التدرجات بسلاسة أثناء التدريب دون تلاشيها.
          </p>
        </div>
      </div>

      <p style="margin-top: 5px; font-size: 8.8pt; color: #64748b;">
        <b>طبقة الاختزال الزمني:</b> تنتهي مرحلة الـ CNN بطبقة <span class="badge">MaxPool1d(2)</span> تختزل الطول الزمني من 20 إلى 10، لتقليل التعقيد الحسابي للمراحل التالية مع الحفاظ على ذروات الضغط المهمة.
      </p>
    </div>
  </div>

  <!-- التذييل -->
  <div class="footer">
    <span>مشروع تخرج JoTouch — طرف صناعي علوي ذكي</span>
    <span>الصفحة 1 من 2</span>
  </div>
</div>


<!-- ==================== الصفحة الثانية ==================== -->
<div class="page">
  <div>
    <!-- ترويسة مصغرة -->
    <div class="header" style="margin-bottom: 7px;">
      <div class="header-titles">
        <h1 style="font-size: 13pt;">تابع: النمذجة الزمنية (TCN)، رأسا الإخراج، والمميزات التنافسية</h1>
      </div>
      <div class="header-badge" style="padding: 2px 8px;">JoTouch_AI</div>
    </div>

    <!-- البلوك الثاني: TCN -->
    <div class="section-title">3. البلوك الثاني: النمذجة الزمنية (Temporal Convolutional Network - TCN)</div>
    <div class="card">
      <p style="margin-bottom: 5px;">
        حركة اليد ليست مجرد ضغطة ثابتة بل هي تسلسل زمني ديناميكي مستمر. بدلاً من استخدام الشبكات التكرارية التقليدية مثل <span class="badge">LSTM</span> أو <span class="badge">GRU</span>، تم بناء طبقتين من <b>TCN</b>:
      </p>
      <ul class="feature-list">
        <li>
          <b>الالتفاف المتسع (Dilated Convolutions):</b>
          تستخدم الطبقة الأولى معامل اتساع <span class="badge">Dilation = 1</span> لالتقاط تغيرات السرعة اللحظية، بينما تستخدم الطبقة الثانية <span class="badge">Dilation = 2</span> لمضاعفة مجال الرؤية الزمني (Receptive Field). يتيح ذلك للنموذج استيعاب الحركة عبر كامل نافذة الـ 200ms دون زيادة المعاملات ودون فقدان التزامن.
        </li>
        <li>
          <b>التوصيل التراكمي (Residual Shortcut):</b>
          كل كتلة TCN تحتوي على التفافين متتاليين مع دالة تفعيل <span class="badge">SiLU</span>، ومسار التفافي يجمع الدخل الأصلي مع الناتج لتسهيل التدريب واستقرار الأوزان.
        </li>
        <li>
          <b>استخلاص الحالة النهائية (Bottleneck Slicing):</b>
          بعد انتهاء الـ TCN، يتم أخذ الخطوة الزمنية الأخيرة فقط <span class="badge">x[:, :, -1]</span> لتمثيل ملخص كامل الحركة في متجه مكثف بحجم <b>64 خريطة ميزات</b>، وتمريره لطبقة تمثيل مشتركة مع إسقاط عشوائي <span class="badge">Dropout(0.3)</span> لمنع فرط التخصيص.
        </li>
      </ul>
      <div style="background: #f1f5f9; border-radius: 5px; padding: 5px 8px; margin-top: 5px; font-size: 8.6pt; color: #0f172a;">
        <b>لماذا تفوق الـ TCN على الـ RNN والـ LSTM في هذا المشروع؟</b>
        (1) معالجة متوازية بالكامل تخفض وقت التدريب بأكثر من 70%. (2) استهلاك ذاكرة منخفض ووقت استنتاج ثابت ومضمون (Deterministic) يناسب المعالجات المدمجة. (3) انعدام مشكلة تلاشي التدرج الحسابي.
      </div>
    </div>

    <!-- رأسا الإخراج -->
    <div class="section-title">4. رأسا الإخراج المتزامنان (Dual-Head Multi-Task Output)</div>
    <div class="card card-highlight">
      <p style="margin-bottom: 6px;">
        اليد البشرية تقوم بنوعين من الأفعال: اختيار نوع القبضة (قرار تصنيفي) وتحريك الأصابع بزوايا دقيقة (حركة مستمرة). لذا تم تزويد النموذج برأسين متوازيين ينبثقان من نفس الميزات المشتركة:
      </p>

      <div class="grid-2">
        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px;">
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 2px;">
            أ. رأس تصنيف الإيماءة (Phase Head)
          </div>
          <p style="font-size: 8.7pt; color: #475569; margin-bottom: 3px;">
            عبارة عن طبقة خطية <span class="badge">Linear(64 &rarr; 4)</span> تدرب بدالة خسارة <span class="badge">CrossEntropyLoss</span> لتحديد إحدى 4 إيماءات أساسية:
          </p>
          <div style="display: flex; gap: 4px; flex-wrap: wrap;">
            <span class="badge badge-blue">0: Rest (استرخاء)</span>
            <span class="badge badge-blue">1: Wave (تلويح)</span>
            <span class="badge badge-blue">2: Pinch (قبضة دقيقة)</span>
            <span class="badge badge-blue">3: Grip (قبضة كاملة)</span>
          </div>
        </div>

        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px;">
          <div style="font-weight: 700; color: #0f172a; margin-bottom: 2px;">
            ب. رأس انحدار زوايا المفاصل (Angle Head)
          </div>
          <p style="font-size: 8.7pt; color: #475569; margin-bottom: 3px;">
            عبارة عن طبقة خطية <span class="badge">Linear(64 &rarr; 16)</span> تدرب بدالة <span class="badge">Smooth L1 Loss</span> لإخراج زوايا المفاصل بالراديان:
          </p>
          <p style="font-size: 8.5pt; color: #334155; margin: 0;">
            تخرج <b>16 زاوية مفصلية مستمرة</b> مطابقة تماماً للمفاصل التي يسجلها كود جمع البيانات <span class="badge">data_collection.py</span> عبر كاميرا MediaPipe، لتحريك أصابع اليد بنعومة وواقعية.
          </p>
        </div>
      </div>
    </div>

    <!-- أهم مميزات النموذج -->
    <div class="section-title">5. أهم مميزات النموذج المبتكرة (Key Project Advantages)</div>
    <div class="card" style="padding-bottom: 5px;">
      <ul class="feature-list">
        <li>
          <b>حجم فائق الصغر (Ultra-Lightweight):</b>
          يحتوي النموذج على <b>66,772 وزناً فقط</b>، وبحجم ملف لا يتجاوز <b>267 كيلوبايت</b>، مما يتيح تحميله وتشغيله بسهولة تامة داخل ذاكرة المتحكم الدقيق <b>Teensy 4.1</b> (التي تحوي 1MB RAM فقط) أو معالج <b>Raspberry Pi Zero 2W</b>.
        </li>
        <li>
          <b>زمن استجابة فائق السرعة (Real-Time Latency):</b>
          يحتاج النموذج إلى <b>أقل من 3.5 ميلي ثانية</b> للاستنتاج على معالج عادي، و<b>أقل من ميلي ثانية واحدة</b> على معالج ARM مدمج، وهو زمن أقل بكثير من حد الإدراك الحسي للإنسان (50ms)، مانحاً المستخدم تحكماً طبيعياً خالياً من البطء.
        </li>
        <li>
          <b>تصدير مباشر إلى لغة السي الخام (Pure C Headers):</b>
          يتضمن المشروع أداة <span class="badge">export_boot_package.py</span> تقوم بتحويل أوزان بايتورش إلى مصفوفات C ثابتة في ملفات <span class="badge">teensy_boot/*.h</span>، ليتم حرقها وتشغيلها على المتحكم مباشرة بدون أي بيئة برمجية خارجية أو نظام تشغيل.
        </li>
        <li>
          <b>بروتوكول تدريب ذكي ومرن (Conditional Training):</b>
          يمتلك النموذج راية ذكية <span class="badge">has_angles</span>؛ عند تدريبه على قواعد البيانات العالمية المفتوحة (Zenodo, PLoS, Exoskelebox) التي لا تحتوي على زوايا مفاصل يتم تدريب رأس التصنيف فقط، بينما يُفعل رأس الزوايا تلقائياً بمجرد توفر تسجيلات الكاميرا الخاصة بالمشروع.
        </li>
      </ul>
    </div>
  </div>

  <!-- التذييل -->
  <div class="footer">
    <span>مشروع تخرج JoTouch — طرف صناعي علوي ذكي</span>
    <span>الصفحة 2 من 2</span>
  </div>
</div>

</body>
</html>
"""

    temp_html = os.path.join(tempfile.gettempdir(), "jotouch_arabic_overview.html")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    user_data = os.path.join(tempfile.gettempdir(), "edge_ar_profile")
    url = "file:///" + temp_html.replace("\\", "/")

    cmd = [
        edge_path,
        "--headless=new",
        "--user-data-dir=" + user_data,
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf=" + output_pdf_path,
        url
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Edge PDF generation failed: {res.stderr}")

    # Validate with PyMuPDF
    doc = pymupdf.open(output_pdf_path)
    page_count = len(doc)
    print(f"Arabic PDF successfully generated at: {output_pdf_path}")
    print(f"Total pages: {page_count}")
    return page_count

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "JoTouch_AI_Base_Model_Arabic.pdf"
    generate_arabic_pdf(out)
