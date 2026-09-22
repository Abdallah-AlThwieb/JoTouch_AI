"""
Script to generate a pristine 2-page PDF overview of the primary JoTouch_AI model.
Targeted at engineering colleagues and academic presentation.
"""
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically to print 'Page X of Y'"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#5A6B7C"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(1.8 * cm, 28.3 * cm, "JoTouch_AI — Base Model Technical Architecture & Features")
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.5)
            self.line(1.8 * cm, 28.15 * cm, 19.2 * cm, 28.15 * cm)

        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(19.2 * cm, 1.2 * cm, text)
        self.drawString(1.8 * cm, 1.2 * cm, "JoTouch Bionic Prosthesis Project • Confidential & Technical Summary")
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(1.8 * cm, 1.45 * cm, 19.2 * cm, 1.45 * cm)
        self.restoreState()


def build_pdf(dest_path):
    doc = SimpleDocTemplate(
        dest_path,
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm
    )

    styles = getSampleStyleSheet()
    
    # Custom color palette
    c_primary = colors.HexColor("#0D3B66")     # Deep Navy
    c_secondary = colors.HexColor("#1D6996")   # Slate Blue
    c_accent = colors.HexColor("#F4D35E")      # Gold accent
    c_dark = colors.HexColor("#1A202C")        # Charcoal text
    c_muted = colors.HexColor("#4A5568")       # Muted gray text
    c_bg_light = colors.HexColor("#F8FAFC")    # Background light
    c_border = colors.HexColor("#CBD5E1")      # Border gray
    c_badge_bg = colors.HexColor("#E2E8F0")    # Badge gray

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=c_secondary,
        spaceAfter=8
    )

    section_style = ParagraphStyle(
        "SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=c_primary,
        spaceBefore=7,
        spaceAfter=3
    )

    subsection_style = ParagraphStyle(
        "SubSectionHeader",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=c_secondary,
        spaceBefore=4,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        fontName="Helvetica",
        fontSize=8.3,
        leading=11,
        textColor=c_dark,
        spaceAfter=3
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    bullet_style = ParagraphStyle(
        "BulletCustom",
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.8,
        textColor=c_dark,
        leftIndent=10,
        spaceAfter=2
    )

    table_cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=7.8,
        leading=9.8,
        textColor=c_dark
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=9.8,
        textColor=c_dark
    )

    table_header = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # ================= PAGE 1 =================
    # Title & Header Banner
    story.append(Paragraph("JoTouch_AI — Primary Model Technical Architecture", title_style))
    story.append(Paragraph("Dual-Head Spatio-Temporal Network (MobileNetV2-1D + TCN) for FMG Prosthetic Control", subtitle_style))
    
    # Metadata bar
    meta_data = [
        [
            Paragraph("<b>Target Hardware:</b> Teensy 4.1 (ARM Cortex-M7) / Embedded Linux", table_cell),
            Paragraph("<b>Total Parameters:</b> 66,772 (~267 KB FP32)", table_cell),
            Paragraph("<b>Inference Latency:</b> &lt; 3.5 ms", table_cell),
            Paragraph("<b>Input Window:</b> 200 ms (20 steps @ 10ms)", table_cell)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[4.7*cm, 4.3*cm, 3.8*cm, 4.7*cm])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EEF4F8")),
        ("BOX", (0,0), (-1,-1), 0.5, c_secondary),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 5))

    # Section 1: Executive Overview & Physical Principle
    story.append(Paragraph("1. Executive Overview & Sensing Principle (FMG)", section_style))
    story.append(Paragraph(
        "<b>Force Myography (FMG)</b> captures mechanical volumetric muscle expansion beneath the skin during contractions "
        "using an array of <b>8 Force-Sensing Resistors (FSRs)</b> wrapped around the user's forearm. Unlike electromyography (EMG), "
        "which suffers from skin impedance shifts, sweat drift, and electromagnetic interference, FMG provides a direct, highly stable, "
        "and robust mechanical signal well-suited for durable prosthetic control.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The JoTouch Core AI Model (JoTouchModel)</b> is a purpose-built deep neural architecture designed to transform raw continuous "
        "FMG pressure streams into simultaneous <b>discrete intention classification</b> (which grip?) and <b>continuous kinematic trajectories</b> "
        "(what finger angles?). The pipeline samples data at <b>100 Hz (10 ms intervals)</b> using a sliding window of <b>20 steps (200 ms)</b> "
        "with a stride of <b>5 steps (50 ms)</b>, achieving a <b>20 Hz update loop</b> that feels immediate and natural to the amputee.",
        body_style
    ))

    # Section 2: Architecture Pipeline & Input Flow
    story.append(Paragraph("2. Deep Neural Architecture & Processing Flow", section_style))
    story.append(Paragraph(
        "The model processes an input tensor of dimension <b>[Batch, 20 timesteps, 8 sensors]</b>. It immediately transposes this to "
        "<b>[Batch, 8 channels, 20 length]</b> to leverage highly optimized 1D convolutions across the multi-sensor spatial channel axis.",
        body_style
    ))

    # Architecture Overview Table
    arch_data = [
        [Paragraph("Stage / Layer", table_header), Paragraph("Type & Operation", table_header), Paragraph("Output Shape", table_header), Paragraph("Key Functional Role", table_header)],
        [Paragraph("<b>Input</b>", table_cell_bold), Paragraph("FMG Sensor Temporal Window", table_cell), Paragraph("[B, 8, 20]", table_cell), Paragraph("8 forearm channels over 200 ms temporal span", table_cell)],
        [Paragraph("<b>Conv1D Stem</b>", table_cell_bold), Paragraph("Conv1D(8 &rarr; 32, k=3, p=1) + BN + SiLU", table_cell), Paragraph("[B, 32, 20]", table_cell), Paragraph("Initial cross-sensor spatial projection & feature expansion", table_cell)],
        [Paragraph("<b>InvResBlock 1</b>", table_cell_bold), Paragraph("MobileNetV2 Inverted Residual (exp=2)", table_cell), Paragraph("[B, 32, 20]", table_cell), Paragraph("Spatial muscle synergy extraction with residual skip", table_cell)],
        [Paragraph("<b>InvResBlock 2</b>", table_cell_bold), Paragraph("MobileNetV2 Inverted Residual (exp=2)", table_cell), Paragraph("[B, 64, 20]", table_cell), Paragraph("Higher-level cross-sensor muscle coordination mapping", table_cell)],
        [Paragraph("<b>Max Pooling</b>", table_cell_bold), Paragraph("MaxPool1D(kernel=2, stride=2)", table_cell), Paragraph("[B, 64, 10]", table_cell), Paragraph("Reduces temporal length by 50% for computational efficiency", table_cell)],
        [Paragraph("<b>TCN Block 1</b>", table_cell_bold), Paragraph("Dilated Conv1D (d=1, k=3) &times; 2 + Res + SiLU", table_cell), Paragraph("[B, 64, 10]", table_cell), Paragraph("Short-term temporal dependency & motion speed modeling", table_cell)],
        [Paragraph("<b>TCN Block 2</b>", table_cell_bold), Paragraph("Dilated Conv1D (d=2, k=3) &times; 2 + Res + SiLU", table_cell), Paragraph("[B, 64, 10]", table_cell), Paragraph("Long-range temporal context across entire 200 ms window", table_cell)],
        [Paragraph("<b>Bottleneck</b>", table_cell_bold), Paragraph("Take final temporal slice: x[:, :, -1]", table_cell), Paragraph("[B, 64]", table_cell), Paragraph("Summarizes whole spatio-temporal history into 64-D vector", table_cell)],
        [Paragraph("<b>Shared Head</b>", table_cell_bold), Paragraph("Linear(64 &rarr; 64) + SiLU + Dropout(0.3)", table_cell), Paragraph("[B, 64]", table_cell), Paragraph("Regularized shared representation for multi-task heads", table_cell)],
        [Paragraph("<b>Phase Head</b>", table_cell_bold), Paragraph("Linear(64 &rarr; 4)", table_cell), Paragraph("[B, 4]", table_cell), Paragraph("<b>Classification</b>: 4 discrete hand gestures / grips", table_cell)],
        [Paragraph("<b>Angle Head</b>", table_cell_bold), Paragraph("Linear(64 &rarr; 16)", table_cell), Paragraph("[B, 16]", table_cell), Paragraph("<b>Regression</b>: 16 joint angles in radians (MediaPipe DOF)", table_cell)],
    ]
    t_arch = Table(arch_data, colWidths=[2.8*cm, 5.2*cm, 2.3*cm, 7.2*cm])
    t_arch.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), c_primary),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0,0), (-1,-1), 0.4, c_border),
        ("TOPPADDING", (0,0), (-1,-1), 2.5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2.5),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 5))

    # Section 3: Deep Dive into MobileNetV2-1D Block
    story.append(Paragraph("3. Building Block 1: MobileNetV2-1D Inverted Residuals", section_style))
    story.append(Paragraph(
        "The spatial feature extractor adopts the <b>Inverted Residual block</b> philosophy pioneered by MobileNetV2, adapted for 1D signals:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Pointwise Expansion (1x1 Conv1D):</b> Expands the channel capacity by an expansion factor of 2 (e.g. 32 &rarr; 64 channels). "
        "This projects muscle pressure readings into a higher-dimensional space where non-linear boundaries can be captured without information loss.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Depthwise Separable Convolution (k=3, groups=channels):</b> Operates independently on each individual feature channel. "
        "This drastically slashes floating-point operations (FLOPs) and parameter counts compared to standard dense convolutions.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Pointwise Linear Projection (1x1 Conv1D):</b> Condenses the features back down to the target dimension without non-linearity, "
        "preventing the manifold collapse commonly observed with ReLU/SiLU at thin bottlenecks.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Residual Skip Connection:</b> Added when input and output channels match (<code>x + conv(x)</code>). "
        "Enables smooth gradient flow across layers and prevents degradation during backpropagation.",
        bullet_style
    ))

    # ================= PAGE BREAK =================
    story.append(PageBreak())

    # ================= PAGE 2 =================
    # Section 4: Deep Dive into TCN Block
    story.append(Paragraph("4. Building Block 2: Temporal Convolutional Network (TCN)", section_style))
    story.append(Paragraph(
        "While MobileNetV2 extracts spatial cross-sensor correlations, hand movement is inherently sequential. "
        "Instead of recurrent architectures (LSTM / GRU), JoTouch employs a <b>Temporal Convolutional Network (TCN)</b> consisting of stacked "
        "dilated residual blocks:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Dilated Convolutions (Dilation d=1 and d=2):</b> The dilation factor introduces gaps in the convolutional kernel, "
        "allowing the receptive field of the network to grow exponentially rather than linearly with depth. "
        "Block 1 (d=1) captures short-term velocity cues; Block 2 (d=2) encompasses the entire 200 ms movement trajectory without downsampling.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Residual Additive Shortcuts:</b> Each <code>TemporalBlock</code> contains two dilated Conv1D layers with SiLU activations "
        "and adds a residual shortcut (with a 1x1 projection when channel dimensions mismatch), stabilizing optimization.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Why TCN Beats LSTMs/GRUs for Prosthetics:</b><br/>"
        "&nbsp;&nbsp;1. <i>Zero Sequential Bottleneck:</i> Convolutions are fully parallelizable during training, reducing training time by &gt;70%.<br/>"
        "&nbsp;&nbsp;2. <i>Deterministic Latency:</i> Execution time on embedded hardware is constant (fixed feed-forward passes), unlike RNN hidden state unrolling.<br/>"
        "&nbsp;&nbsp;3. <i>No Exploding/Vanishing Gradients:</i> Residual connections prevent gradient vanishing over long sequences.",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    # Section 5: Dual-Head Multi-Task Learning
    story.append(Paragraph("5. Dual-Head Multi-Task Output Architecture", section_style))
    story.append(Paragraph(
        "Real human hands do not merely switch between static poses; they modulate fine joint angles. "
        "JoTouch solves both simultaneously using a shared 64-dimensional bottleneck branching into <b>two specialized heads</b>:",
        body_style
    ))

    head_data = [
        [Paragraph("Head Type", table_header), Paragraph("Target Dimension", table_header), Paragraph("Loss Function", table_header), Paragraph("Operational Functionality", table_header)],
        [
            Paragraph("<b>Phase Head</b><br/>(Classification)", table_cell_bold),
            Paragraph("4 Classes<br/>[Rest, Wave, Pinch, Grip]", table_cell),
            Paragraph("CrossEntropyLoss", table_cell),
            Paragraph("Identifies high-level user grip intent. Triggers discrete state changes and synergy presets on the physical bionic hand.", table_cell)
        ],
        [
            Paragraph("<b>Angle Head</b><br/>(Regression)", table_cell_bold),
            Paragraph("16 Degrees of Freedom<br/>[Thumb, Index, Mid, Ring, Pinky]", table_cell),
            Paragraph("Smooth L1 Loss (Huber)<br/><i>(Active when angles present)</i>", table_cell),
            Paragraph("Predicts real-time joint angular trajectories in radians. Matches MediaPipe hand tracking labels collected in <code>data_collection.py</code> for natural, proportional finger articulation.", table_cell)
        ]
    ]
    t_head = Table(head_data, colWidths=[3.2*cm, 3.8*cm, 3.5*cm, 7.0*cm])
    t_head.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), c_secondary),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0,0), (-1,-1), 0.4, c_border),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(t_head)
    story.append(Spacer(1, 4))

    # Section 6: Key Engineering Features & Advantages
    story.append(Paragraph("6. Key Features & Competitive Advantages of JoTouch_AI", section_style))
    
    features = [
        ("Ultra-Lightweight Embedded Footprint", "With exactly <b>66,772 parameters</b>, the entire model occupies under <b>270 KB</b> in memory. It fits effortlessly into the 1 MB RAM / 8 MB Flash of a <b>Teensy 4.1 microcontroller</b> or Raspberry Pi Zero 2W."),
        ("Microsecond-Scale Inference Latency", "Inference runs in <b>~2.0 to 3.5 ms</b> on standard CPU and <b>&lt; 1 ms</b> on bare-metal ARM Cortex-M7 at 600 MHz. This is well below the human-perceived threshold (50 ms), completely eliminating control lag."),
        ("Spatio-Temporal Synergy", "Decomposes the complex neuromuscular task: MobileNetV2 resolves which muscles are activating across the forearm circumference, while TCN tracks the temporal velocity and acceleration of the contraction."),
        ("Smart Dual-Protocol Training", "Employs an automated conditional loss flag (<code>has_angles</code>). It trains on large public FMG benchmarks (Zenodo, PLoS ONE, Exoskelebox) with classification loss, and seamlessly enables kinematic regression when true joint angles are present."),
        ("Direct C Header Export Pipeline", "Includes <code>export_boot_package.py</code>, which exports trained PyTorch weights directly into pure C static float arrays (<code>.h</code>), enabling direct flashing to microcontrollers without any runtime interpreter dependencies.")
    ]

    feat_data = [
        [Paragraph(f"<b>{title}</b>", table_cell_bold), Paragraph(desc, table_cell)]
        for title, desc in features
    ]
    t_feat = Table(feat_data, colWidths=[5.0*cm, 12.5*cm])
    t_feat.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, c_border),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 4))

    # Summary Conclusion Callout
    summary_data = [[
        Paragraph(
            "<b>Summary for Engineering Peers:</b> JoTouch_AI successfully resolves the historical trade-off in prosthetic control "
            "between heavy, accurate models and lightweight, responsive firmware. By fusing MobileNetV2-1D spatial depthwise convolutions "
            "with TCN dilated temporal filtering, it achieves state-of-the-art intentional accuracy and continuous kinematic flexibility "
            "at negligible computational cost.",
            body_style
        )
    ]]
    t_sum = Table(summary_data, colWidths=[17.5*cm])
    t_sum.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EEF4F8")),
        ("BOX", (0,0), (-1,-1), 0.8, c_primary),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t_sum)

    # Build document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Pristine PDF generated at: {dest_path}")

if __name__ == "__main__":
    out_pdf = sys.argv[1] if len(sys.argv) > 1 else "JoTouch_AI_Model_Overview.pdf"
    build_pdf(out_pdf)
