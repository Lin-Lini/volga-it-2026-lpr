#!/usr/bin/env python3
"""Render a Markdown file (headings, paragraphs, bullet lists, pipe tables, code blocks) to PDF with a
Cyrillic-capable font (PT Sans Narrow shipped with the generator).

    python tools/md2pdf.py docs/note.md docs/note.pdf
"""
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "..", "generator", "fonts", "PTSansNarrow-Bold.ttf")
FONT_R = os.path.join(HERE, "..", "generator", "fonts", "RobotoCondensed-wght.ttf")


def esc(t):
    t = t.replace("→", "->").replace("⇒", "=>").replace("←", "<-").replace("≈", "~").replace("×", "x").replace("−", "-")
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<font face='Head'>\1</font>", t)
    t = re.sub(r"_\{(.+?)\}_", r"<i>\1</i>", t)
    return t


def main(src, dst):
    pdfmetrics.registerFont(TTFont("Body", FONT_R))
    pdfmetrics.registerFont(TTFont("Head", FONT))
    body = ParagraphStyle("body", fontName="Body", fontSize=9.2, leading=11.6, alignment=TA_JUSTIFY, spaceAfter=3)
    small = ParagraphStyle("small", fontName="Body", fontSize=8, leading=9.6)
    h1 = ParagraphStyle("h1", fontName="Head", fontSize=15, leading=18, spaceBefore=4, spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName="Head", fontSize=11.5, leading=14, spaceBefore=7, spaceAfter=3)
    h3 = ParagraphStyle("h3", fontName="Head", fontSize=10, leading=12.5, spaceBefore=5, spaceAfter=2)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=2)
    code = ParagraphStyle("code", fontName="Body", fontSize=7.8, leading=9.4, backColor=colors.whitesmoke, leftIndent=4)
    doc = SimpleDocTemplate(dst, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    story = []
    lines = open(src, encoding="utf-8").read().splitlines()
    i = 0
    para = []

    def flush():
        nonlocal para
        if para:
            story.append(Paragraph(esc(" ".join(para)), body))
            para = []

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            flush()
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            story.append(Preformatted("\n".join(buf), code))
            i = j + 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*-", lines[i + 1]):
            flush()
            rows = []
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                if not re.match(r"^\|\s*-", lines[j]):
                    rows.append([Paragraph(esc(c.strip()), small) for c in lines[j].strip().strip("|").split("|")])
                j += 1
            t = Table(rows, repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
            story.append(t); story.append(Spacer(1, 4))
            i = j
            continue
        if ln.startswith("# "):
            flush(); story.append(Paragraph(esc(ln[2:]), h1))
        elif ln.startswith("### "):
            flush(); story.append(Paragraph(esc(ln[4:]), h3))
        elif ln.startswith("## "):
            flush(); story.append(Paragraph(esc(ln[3:]), h2))
        elif re.match(r"^\s*[-*] ", ln) or re.match(r"^\s*\d+\. ", ln):
            flush()
            is_bullet = bool(re.match(r"^\s*[-*] ", ln))
            item = [re.sub(r"^\s*[-*] ", "", ln).strip() if is_bullet else ln.strip()]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not re.match(r"^\s*([-*]|\d+\.) ", lines[j]) and not lines[j].startswith(("#", "|", "```")):
                item.append(lines[j].strip()); j += 1
            story.append(Paragraph(esc(" ".join(item)), bullet, bulletText="•" if is_bullet else None))
            i = j
            continue
        elif not ln.strip():
            flush()
        else:
            para.append(ln.strip())
        i += 1
    flush()
    doc.build(story)
    print("->", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
