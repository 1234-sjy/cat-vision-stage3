"""Render the completed Chinese project report for sharing."""
from pathlib import Path
import sys
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catvision.common import ROOT


def main():
    from matplotlib import font_manager
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

    try:
        font = font_manager.findfont(font_manager.FontProperties(family="Microsoft YaHei"), fallback_to_default=False)
        pdfmetrics.registerFont(TTFont("ReportChinese", font, subfontIndex=0))
    except (ValueError, OSError):
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font = None
    face = "ReportChinese" if font else "STSong-Light"
    body = ParagraphStyle("Body", fontName=face, fontSize=10, leading=17,
                          spaceAfter=9, wordWrap="CJK", alignment=TA_LEFT, textColor=colors.HexColor("#26342f"))
    heading = ParagraphStyle("Heading", parent=body, fontSize=13, leading=20,
                             spaceBefore=14, spaceAfter=8, keepWithNext=True, textColor=colors.HexColor("#287764"))
    title = ParagraphStyle("Title", parent=heading, fontSize=19, leading=29, spaceAfter=18)
    paragraphs = (ROOT / "docs/项目说明与实现感悟.txt").read_text(encoding="utf-8").strip().split("\n\n")
    story = []
    for index, text in enumerate(paragraphs):
        if text == "实际案例":
            chart = ROOT / "artifacts/comparison/comparison.png"
            story.append(Image(str(chart), width=480, height=480 * 4.5 / 11))
            story.append(Spacer(1, 10))
        style = title if index == 0 else heading if len(text) < 24 and "。" not in text else body
        story.append(Paragraph(escape(text).replace("\n", "<br/>"), style))
    output = ROOT / "docs/项目说明与实现感悟.pdf"

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(face, 8)
        canvas.setFillColor(colors.HexColor("#788078"))
        canvas.drawString(43, 27, "看图识猫 · 第二阶段")
        canvas.drawRightString(A4[0] - 43, 27, str(doc.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=43, rightMargin=43,
                                 topMargin=39, bottomMargin=43,
                                 title="看图识猫：检测分割与视觉大模型的同图比较", author="项目作者")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print("PDF SAVED", output, flush=True)


if __name__ == "__main__":
    main()
