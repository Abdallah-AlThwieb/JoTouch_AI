"""Render model_comparison_results.md to a styled PDF using reportlab."""
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "model_comparison_results.md"
DST = sys.argv[2] if len(sys.argv) > 2 else SRC.replace(".md", ".pdf")

h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, spaceAfter=10, textColor=colors.HexColor("#1a3a5c"))
h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13.5, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a3a5c"))
h3 = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#2d5a80"))
h4 = ParagraphStyle("h4", fontName="Helvetica-BoldOblique", fontSize=10.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2d5a80"))
body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=4)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4)

def inline(t):
    t = t.replace("**", "")          # drop bold markers (styles carry emphasis)
    t = t.replace("`", "")
    return t.strip()

def flush_table(rows, story):
    ncols = len(rows[0])
    data = [[Paragraph(inline(c), ParagraphStyle("tc", fontName="Helvetica-Bold" if i == 0 else "Helvetica",
             fontSize=8, leading=10, textColor=colors.white if i == 0 else colors.black)) for c in r]
            for i, r in enumerate(rows)]
    avail = 17.5 * cm
    tbl = Table(data, colWidths=[avail / ncols] * ncols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d5a80")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c6d4")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))

story = []
table_rows = []
for raw in open(SRC, encoding="utf-8"):
    line = raw.rstrip()
    if line.startswith("|"):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue  # separator row
        table_rows.append(cells)
        continue
    if table_rows:
        flush_table(table_rows, story)
        table_rows = []
    if not line:
        continue
    if line.startswith("#### "):
        story.append(Paragraph(inline(line[5:]), h4))
    elif line.startswith("### "):
        story.append(Paragraph(inline(line[4:]), h3))
    elif line.startswith("## "):
        story.append(Paragraph(inline(line[3:]), h2))
    elif line.startswith("# "):
        story.append(Paragraph(inline(line[2:]), h1))
    elif line.startswith("- "):
        story.append(Paragraph("• " + inline(line[2:]), bullet))
    else:
        story.append(Paragraph(inline(line), body))
if table_rows:
    flush_table(table_rows, story)

doc = SimpleDocTemplate(DST, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        title="JoTouch Model Comparison Results")
doc.build(story)
print("PDF written:", DST)
