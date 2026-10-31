"""One-page ReportLab export with measured wrapping and a strict page boundary."""

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from .commentary import DISCLAIMER, SOURCE_URL, Brief, parameter_text

NAVY = colors.HexColor("#132A44")
TEAL = colors.HexColor("#087E8B")
INK = colors.HexColor("#263A4F")
MUTED = colors.HexColor("#536579")


def _paragraph(text: str, size: float, *, bold: bool = False, color=INK) -> Paragraph:
    return Paragraph(
        escape(text),
        ParagraphStyle(
            "report",
            fontName="Helvetica-Bold" if bold else "Helvetica",
            fontSize=size,
            leading=size * 1.35,
            textColor=color,
            alignment=TA_LEFT,
            splitLongWords=True,
        ),
    )


def _content(brief: Brief, size: float, width: float):
    blocks = []
    for number, (title, text) in enumerate(brief.sections):
        blocks.append((_paragraph(title.upper(), size + 0.8, bold=True, color=TEAL), 4))
        if number == 0:
            headings = ("TENOR", "YIELD (%)", "PRIOR CHANGE (bps)", "5-OBS CHANGE (bps)")
            data = [[_paragraph(v, size - 0.5, bold=True, color=colors.white) for v in headings]]
            data += [[_paragraph(v, size - 0.3) for v in row] for row in brief.snapshot]
            table = Table(data, colWidths=[width * x for x in (0.13, 0.19, 0.34, 0.34)])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.HexColor("#F0F5F8"), colors.white],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            blocks.append((table, 5))
        blocks.append((_paragraph(text, size), 8))
        if number == 3:
            blocks.append(
                (_paragraph("Parameters: " + parameter_text(brief), size - 0.5, color=MUTED), 8)
            )
    blocks.append(
        (
            _paragraph(
                f"Source: Bank of Canada Valet API | {SOURCE_URL} | "
                f"Retrieval: {brief.source} | Fetched: {brief.fetched_at}",
                size - 0.5,
                color=MUTED,
            ),
            6,
        )
    )
    blocks.append((_paragraph(DISCLAIMER, size - 0.5, color=MUTED), 0))
    return blocks


def render_pdf(brief: Brief) -> bytes:
    """Fit measured paragraphs on one readable A4 page, refusing silent overflow."""
    output = BytesIO()
    page_width, page_height = A4
    margin = 36
    width = page_width - 2 * margin
    content_top = page_height - 116
    available = content_top - 38
    selected = None
    for size in (9.5, 9.25, 9.0, 8.75, 8.5):
        blocks = _content(brief, size, width)
        measured = [(flowable, flowable.wrap(width, available)[1], gap) for flowable, gap in blocks]
        if sum(height + gap for _, height, gap in measured) <= available:
            selected = measured
            break
    if selected is None:
        raise ValueError("Research brief exceeds one readable page; shorten custom parameter text")
    pdf = canvas.Canvas(output, pagesize=A4, invariant=1)
    pdf.setTitle("Canada Rates Strategy Lab - Canadian Rates Monitor")
    pdf.setAuthor("Canada Rates Strategy Lab")
    pdf.setFillColor(NAVY)
    pdf.rect(0, page_height - 74, page_width, 74, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(margin, page_height - 31, "CANADA RATES STRATEGY LAB")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(margin, page_height - 51, "Canadian Rates Monitor")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawString(
        margin, page_height - 90, f"REPORT {brief.report_date}   |   DATA AS OF {brief.as_of}"
    )
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(margin, page_height - 103, f"Analysis observations: {brief.period}")
    y = content_top
    for flowable, height, gap in selected:
        y -= height
        flowable.drawOn(pdf, margin, y)
        y -= gap
    pdf.setStrokeColor(TEAL)
    pdf.line(margin, 28, page_width - margin, 28)
    pdf.setFont("Helvetica", 7)
    pdf.setFillColor(MUTED)
    pdf.drawString(margin, 17, "PUBLIC DATA / REPRODUCIBLE RESEARCH")
    pdf.drawRightString(page_width - margin, 17, "1 / 1")
    pdf.showPage()
    pdf.save()
    return output.getvalue()
