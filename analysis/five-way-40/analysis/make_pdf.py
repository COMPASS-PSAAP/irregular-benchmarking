"""Build results/five-way-campaign-report.pdf: results/SUMMARY.md with the key figures
placed under the section they support, then every figure in an appendix.

Handles the Markdown SUMMARY.md uses: # and ## headings, paragraphs, "- " bullets
(nested by two spaces), | tables |, `code`, **bold**, *italic*.

Usage (from this folder): python make_pdf.py
"""
import re
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from common import FIGS, RESULTS

OUT = RESULTS / "five-way-campaign-report.pdf"
TEXT_WIDTH = letter[0] - 1.5 * inch

# Figures shown inside each numbered section of SUMMARY.md; all figures go in the appendix.
SECTION_FIGURES = {
    "1": ["stall_quantum_excess_fine_crankseg_2_STANDARD_16nodes.png", "stall_quantum_coherence_by_cell.png",
          "stall_steps_pm_crankseg_2_4nodes.png"],
    "2": ["ladder_worst_regret_vs_cg.png", "worst_case_regret.png", "regret_pm_vs_abcgspmv_4nodes.png", "regret_coarse_vs_abcgspmv_4nodes.png",
          "pairwise_agreement_pm_vs_abcgspmv.png"],
    "3": ["regret_coarse_vs_fine_4nodes.png", "coarse_structure_ratio_4nodes.png"],
    "4": ["h0_best_arm_abcgspmv_2nodes.png", "h0_best_arm_abcgspmv_16nodes.png", "h0_selection_gain_abcgspmv.png"],
    "5": ["ratio_fine_vs_pm_4nodes.png", "ratio_pm_vs_abcgspmv_4nodes.png"],
    "6": ["h13_median_vs_spread_abcgspmv_2nodes.png"],
    "7": ["h8_nstar_abcgspmv_LOCALITY_vs_MPI.png"],
    "8": ["position_effect_by_nodes.png", "launch_stall_score_vs_start.png", "variance_share_4nodes.png"],
    "9": ["estimand_median_vs_rom.png", "calibration_size_minority.png", "calibration_power_10pct.png",
          "sample_size_normal_vs_median.png", "replication_chronological.png"],
    "10": ["tracking_summary.png", "tracking_pm_vs_abcgspmv.png"],
}
APPENDIX_GROUPS = [
    ("Stalls", r"^stall_"),
    ("H17: regret of each stand-in's choice", r"^regret_(pm|fine|coarse)_vs_abcg"),
    ("H17: pairwise ordering agreement", r"^pairwise_agreement_"),
    ("H17/H18: worst-case regret", r"^worst_case_regret"),
    ("H18: coarse against fine, and what the coarse capture changes", r"^(regret_coarse_vs_fine|coarse_structure)"),
    ("Paired time ratios", r"^ratio_"),
    ("H0: best arm and the value of choosing per workload", r"^h0_"),
    ("H13: fastest arm against steadiest arm", r"^h13_"),
    ("H8: break-even iterations", r"^h8_"),
    ("Order effects, stall timing, and variance", r"^(position_|launch_stall|variance_)"),
    ("The stand-in ladder scored against ABCG's CG", r"^ladder_"),
    ("Estimands: median, mean, and ratio of means", r"^estimand_"),
    ("Test calibration: false positives, power, coverage", r"^calibration_"),
    ("Launches needed for a +/-10% interval", r"^sample_size_"),
    ("Replication of verdicts across halves of the launches", r"^replication_"),
    ("Whether each stand-in tracks the run above it", r"^tracking_"),
]

base = getSampleStyleSheet()
STYLES = {
    "title": ParagraphStyle("title", parent=base["Title"], fontSize=18, leading=22, alignment=TA_LEFT,
                            spaceAfter=4),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, leading=16, spaceBefore=14, spaceAfter=6,
                         keepWithNext=1),
    "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=4,
                         keepWithNext=1),
    "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.8, leading=13.2, spaceAfter=6),
    "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8.4, leading=10.2),
    "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=8, leading=10,
                            textColor=colors.HexColor("#555555")),
}


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<i>\1</i>", text)
    return text


def table_flowable(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines
            if not re.match(r"^\|\s*-", ln.strip())]
    longest = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    weights = [min(max(n, 6), 45) for n in longest]
    widths = [TEXT_WIDTH * w / sum(weights) for w in weights]
    data = [[Paragraph(("<b>%s</b>" if i == 0 else "%s") % inline(c), STYLES["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 8)]


def bullet_flowables(lines):
    items = []
    for ln in lines:
        m = re.match(r"^(\s*)- (.*)$", ln)
        if m:
            items.append([len(m.group(1)) // 2, m.group(2)])
        else:
            items[-1][1] += " " + ln.strip()
    out = []
    for level, text in items:
        style = ParagraphStyle(f"b{level}", parent=STYLES["body"], leftIndent=14 + 16 * level,
                               bulletIndent=4 + 16 * level, spaceAfter=3)
        out.append(Paragraph(inline(text), style, bulletText="•" if level == 0 else "–"))
    out.append(Spacer(1, 4))
    return out


def figure(name, width=5.6 * inch):
    path = FIGS / name
    w, h = ImageReader(str(path)).getSize()
    if w / h > 1.8:
        width = TEXT_WIDTH
    height = width * h / w
    max_height = 4.0 * inch
    if height > max_height:
        width, height = width * max_height / height, max_height
    return KeepTogether([Image(str(path), width=width, height=height, hAlign="CENTER"),
                         Paragraph(f"figures/{name}", STYLES["small"]), Spacer(1, 10)])


def markdown_flowables(text):
    story, block, kind = [], [], None

    def flush():
        nonlocal block, kind
        if block:
            if kind == "table":
                story.extend(table_flowable(block))
            elif kind == "list":
                story.extend(bullet_flowables(block))
            else:
                story.append(Paragraph(inline(" ".join(s.strip() for s in block)), STYLES["body"]))
        block, kind = [], None

    section = None
    for ln in text.splitlines():
        if ln.startswith("# "):
            flush()
            story.append(Paragraph(inline(ln[2:]), STYLES["title"]))
            story.append(Paragraph(f"Generated {date.today().isoformat()} from results/ by analysis/make_pdf.py",
                                   STYLES["small"]))
            story.append(Spacer(1, 10))
        elif ln.startswith("## "):
            flush()
            if section in SECTION_FIGURES:
                story.extend(figure(f) for f in SECTION_FIGURES[section])
            m = re.match(r"(\d+)\.", ln[3:])
            section = m.group(1) if m else None
            story.append(Paragraph(inline(ln[3:]), STYLES["h2"]))
        elif not ln.strip():
            flush()
        elif ln.lstrip().startswith("|"):
            if kind != "table":
                flush()
                kind = "table"
            block.append(ln)
        elif re.match(r"^\s*- ", ln):
            if kind != "list":
                flush()
                kind = "list"
            block.append(ln)
        else:
            if kind is None:
                kind = "para"
            block.append(ln)
    flush()
    if section in SECTION_FIGURES:
        story.extend(figure(f) for f in SECTION_FIGURES[section])
    return story


def natural(name):
    return re.sub(r"\d+", lambda m: m.group(0).zfill(4), name)


def appendix():
    story = [PageBreak(), Paragraph("Appendix: every figure", STYLES["title"]),
             Paragraph("All figures in results/figures/, grouped by analysis. Each caption says what the plot shows; "
                       "the numbers behind each are in results/tables/.", STYLES["body"])]
    names = sorted((p.name for p in FIGS.glob("*.png")), key=natural)
    placed = set()
    for heading, pattern in APPENDIX_GROUPS:
        group = [n for n in names if re.search(pattern, n) and n not in placed]
        if not group:
            continue
        story.append(Paragraph(heading, STYLES["h3"]))
        for n in group:
            story.append(figure(n))
            placed.add(n)
    missing = [n for n in names if n not in placed]
    assert not missing, f"figures with no appendix group: {missing}"
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(0.75 * inch, 0.5 * inch, "Five-way campaign, 2026-10-13: statistical analysis")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"page {doc.page}")
    canvas.restoreState()


def main():
    story = markdown_flowables((RESULTS / "SUMMARY.md").read_text(encoding="utf-8")) + appendix()
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                            title="Five-way campaign, 2026-10-13: what the data shows",
                            author="Nicholas Bacon", subject="Phase 1C-shaped analysis on Dane")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT} ({doc.page} pages)")


if __name__ == "__main__":
    main()
