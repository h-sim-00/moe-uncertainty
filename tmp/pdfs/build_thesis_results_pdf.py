from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "thesis-experiment-results-compendium.pdf"
ASSETS = ROOT / "tmp" / "pdfs" / "thesis-results-assets"
ASSETS.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)


NAVY = "#14213D"
BLUE = "#2F5D8A"
TEAL = "#1F8A70"
ORANGE = "#E07A3F"
RED = "#B84A4A"
PALE = colors.HexColor("#F2F5F8")
INK = colors.HexColor("#18212B")
MID = colors.HexColor("#5B6570")
RULE = colors.HexColor("#D6DDE5")
TEAL_C = colors.HexColor(TEAL)
ORANGE_C = colors.HexColor(ORANGE)


def register_fonts():
    regular = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    italic = Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ThesisSans", str(regular)))
        pdfmetrics.registerFont(TTFont("ThesisSans-Bold", str(bold)))
        if italic.exists():
            pdfmetrics.registerFont(TTFont("ThesisSans-Italic", str(italic)))
        return "ThesisSans", "ThesisSans-Bold", "ThesisSans-Italic" if italic.exists() else "ThesisSans"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self._on_page))

    def _on_page(self, canvas, doc):
        page = canvas.getPageNumber()
        if page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 15 * mm, A4[0] - doc.rightMargin, 15 * mm)
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MID)
        canvas.drawString(doc.leftMargin, 10.5 * mm, "Thesis experiment results compendium - verified 31 August 2026")
        canvas.drawRightString(A4[0] - doc.rightMargin, 10.5 * mm, f"Page {page}")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "CoverTitle",
        fontName=FONT_BOLD,
        fontSize=27,
        leading=31,
        textColor=colors.HexColor(NAVY),
        alignment=TA_LEFT,
        spaceAfter=8 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "CoverSub",
        fontName=FONT,
        fontSize=13,
        leading=18,
        textColor=MID,
        spaceAfter=5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "Section",
        fontName=FONT_BOLD,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor(NAVY),
        spaceAfter=3 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "Kicker",
        fontName=FONT_BOLD,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor(TEAL),
        uppercase=True,
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "BodyThesis",
        fontName=FONT,
        fontSize=9.4,
        leading=13.4,
        textColor=INK,
        spaceAfter=2.5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "Small",
        fontName=FONT,
        fontSize=7.5,
        leading=10,
        textColor=MID,
    )
)
styles.add(
    ParagraphStyle(
        "Box",
        fontName=FONT,
        fontSize=9.2,
        leading=13,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        "Callout",
        fontName=FONT_BOLD,
        fontSize=12,
        leading=16,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        "TableCell",
        fontName=FONT,
        fontSize=7.2,
        leading=8.8,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        "TableHead",
        fontName=FONT_BOLD,
        fontSize=7.1,
        leading=8.6,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
)


def P(text, style="BodyThesis"):
    return Paragraph(text, styles[style])


def clean(text):
    return text.replace("–", "-").replace("—", "-").replace("−", "-").replace("‑", "-")


def result_header(number, title, status, metric):
    return [
        P(f"RESULT {number}", "Kicker"),
        P(clean(title), "Section"),
        Table(
            [[P(f"<b>Evidence:</b> {clean(status)}", "Small"), P(f"<b>Metric orientation:</b> {clean(metric)}", "Small")]],
            colWidths=[83 * mm, 83 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE),
                    ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            ),
        ),
        Spacer(1, 4 * mm),
    ]


def data_table(rows, widths=None, font_size=7.2, repeat_rows=1, highlights=None):
    cooked = []
    for ridx, row in enumerate(rows):
        style = "TableHead" if ridx == 0 else "TableCell"
        cooked.append([P(clean(str(cell)), style) for cell in row])
    table = Table(cooked, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
    ]
    if highlights:
        for row_idx, colour in highlights:
            commands.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor(colour)))
    table.setStyle(TableStyle(commands))
    return table


def inference_box(text):
    box = Table(
        [[P("<b>Inference.</b> " + clean(text), "Box")]],
        colWidths=[166 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF5F1")),
                ("BOX", (0, 0), (-1, -1), 1, TEAL_C),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        ),
    )
    return KeepTogether([Spacer(1, 4 * mm), box])


def source_line(text):
    return KeepTogether([Spacer(1, 3 * mm), P("<b>Primary source:</b> " + clean(text), "Small")])


def save_grouped_bar(filename, labels, series, ylabel="AUROC", ylim=(0, 1), chance=True, value_labels=False):
    drawing = Drawing(470, 195)
    chart = VerticalBarChart()
    chart.x = 48
    chart.y = 36
    chart.height = 125
    chart.width = 405
    chart.data = [tuple(values) for values in series.values()]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = FONT
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.angle = 0
    chart.categoryAxis.labels.dy = -8
    chart.valueAxis.valueMin = ylim[0]
    chart.valueAxis.valueMax = ylim[1]
    step = (ylim[1] - ylim[0]) / 5
    chart.valueAxis.valueStep = step
    chart.valueAxis.labels.fontName = FONT
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.gridStrokeColor = colors.HexColor("#DDE3EA")
    chart.valueAxis.gridStrokeWidth = 0.4
    chart.groupSpacing = 6
    chart.barSpacing = 1
    palette = [BLUE, ORANGE, TEAL, RED, "#7566A8"]
    for idx in range(len(series)):
        chart.bars[idx].fillColor = colors.HexColor(palette[idx])
        chart.bars[idx].strokeColor = colors.white
        chart.bars[idx].strokeWidth = 0.4
    drawing.add(chart)
    if chance:
        line_y = chart.y + chart.height * (0.5 - ylim[0]) / (ylim[1] - ylim[0])
        drawing.add(Line(chart.x, line_y, chart.x + chart.width, line_y, strokeColor=colors.HexColor("#555555"), strokeWidth=0.8, strokeDashArray=[3, 2]))
    legend_x = 55
    legend_y = 181
    for idx, name in enumerate(series):
        drawing.add(Rect(legend_x, legend_y - 5, 8, 8, fillColor=colors.HexColor(palette[idx]), strokeColor=None))
        drawing.add(String(legend_x + 11, legend_y - 4, name, fontName=FONT, fontSize=7.2, fillColor=INK))
        legend_x += 120
    if chance:
        drawing.add(Line(legend_x, legend_y - 1, legend_x + 13, legend_y - 1, strokeColor=colors.HexColor("#555555"), strokeWidth=0.8, strokeDashArray=[3, 2]))
        drawing.add(String(legend_x + 16, legend_y - 4, "Chance = 0.5", fontName=FONT, fontSize=7.2, fillColor=INK))
    drawing.add(String(8, 95, ylabel, fontName=FONT, fontSize=7.5, fillColor=INK, angle=90))
    return drawing


def save_horizontal_bar(filename, labels, values, colours, xlabel, xlim):
    drawing = Drawing(470, 165)
    chart = HorizontalBarChart()
    chart.x = 120
    chart.y = 45
    chart.height = 80
    chart.width = 320
    chart.data = [tuple(values)]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = FONT
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = xlim[0]
    chart.valueAxis.valueMax = xlim[1]
    chart.valueAxis.valueStep = 0.1
    chart.valueAxis.labels.fontName = FONT
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.gridStrokeColor = colors.HexColor("#DDE3EA")
    chart.bars[0].fillColor = colors.HexColor(colours[0])
    chart.bars[0].strokeColor = colors.white
    drawing.add(chart)
    zero_x = chart.x + chart.width * (0 - xlim[0]) / (xlim[1] - xlim[0])
    drawing.add(Line(zero_x, chart.y, zero_x, chart.y + chart.height, strokeColor=colors.HexColor("#555555"), strokeWidth=0.8))
    drawing.add(String(160, 20, xlabel, fontName=FONT, fontSize=8, fillColor=INK))
    for idx, value in enumerate(values):
        y = chart.y + chart.height * (idx + 0.5) / len(values)
        drawing.add(String(445, y - 2, f"{value:.3f}", fontName=FONT_BOLD, fontSize=7, fillColor=INK))
    return drawing


def save_heatmap(filename, rows, cols, data, title):
    table_rows = [[P(title, "Kicker")] + [P(c, "TableHead") for c in cols]]
    for row_name, values in zip(rows, data):
        table_rows.append([P(row_name, "TableCell")] + [P(f"{v:.3f}", "TableCell") for v in values])
    table = Table(table_rows, colWidths=[48 * mm] + [29.5 * mm] * len(cols), hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for ridx, values in enumerate(data, start=1):
        for cidx, value in enumerate(values, start=1):
            if value < 0.5:
                t = min(1.0, (0.5 - value) / 0.5)
                colour = colors.Color(0.97 - 0.25 * t, 0.97 - 0.68 * t, 0.97 - 0.68 * t)
            else:
                t = min(1.0, (value - 0.5) / 0.5)
                colour = colors.Color(0.97 - 0.78 * t, 0.97 - 0.43 * t, 0.97 - 0.55 * t)
            commands.append(("BACKGROUND", (cidx, ridx), (cidx, ridx), colour))
            if value < 0.16 or value > 0.88:
                commands.append(("TEXTCOLOR", (cidx, ridx), (cidx, ridx), colors.white))
        commands.append(("BACKGROUND", (0, ridx), (0, ridx), PALE))
    table.setStyle(TableStyle(commands))
    return table


def fig_image(path, height_mm=65):
    return path


story = []


# Cover
story += [
    Spacer(1, 22 * mm),
    P("Thesis experiment results compendium", "CoverTitle"),
    P("Variational mixture-of-experts router uncertainty across all completed experiments", "CoverSub"),
    Spacer(1, 5 * mm),
    Table(
        [[P("<b>Verified evidence base</b><br/>31 August 2026", "Callout"), P("<b>Primary score</b><br/>Inf-Logit-Var (ILV)", "Callout")]],
        colWidths=[80 * mm, 80 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(BLUE)),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(TEAL)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.white),
                ("INNERGRID", (0, 0), (-1, -1), 2, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 16),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
            ]
        ),
    ),
    Spacer(1, 14 * mm),
    P(
        "This document lists the quantitative results one by one. Each result is paired with a thesis-safe interpretation that distinguishes held-out test evidence from validation, historical, and diagnostic findings. Numerical values were reconciled against the comprehensive synthesis, the independent source audit, and the cited raw reports.",
        "BodyThesis",
    ),
    Spacer(1, 10 * mm),
    P("Core conclusion", "Kicker"),
    P(
        "Router posterior variance can be informative for answer-only MCQA, but its empirical orientation is learned, target-sensitive, position-dependent, and conditioning-dependent. Explanation supervision can preserve predictive quality while reversing the final-position ILV ranking.",
        "Section",
    ),
    Spacer(1, 14 * mm),
    P("Prepared as a thesis-drafting reference, not as a substitute for the raw result files.", "Small"),
    PageBreak(),
]


# Reading guide
story += [P("How to read the results", "Section")]
guide_rows = [
    ["Quantity", "Positive class / direction", "What a high value means"],
    ["Wrong-answer AUROC", "wrong answer = 1; larger score intended for wrong", "The score ranks mistakes above correct answers"],
    ["OOD AUROC / AUPRC", "OOD = 1; larger score intended for OOD", "The score ranks shifted examples above ID examples"],
    ["Token-level transfer", "Spearman(ILV, predictive entropy) within each sequence", "Positive means both token trajectories rise together"],
    ["ECE / NLL", "lower is better", "Answer probabilities are better calibrated / assign more probability to truth"],
]
story += [data_table(guide_rows, [39 * mm, 63 * mm, 64 * mm]), Spacer(1, 5 * mm)]
story += [
    inference_box(
        "AUROC below 0.5 is an inversion under the predeclared direction, not a successful detector after flipping. Token-level transfer and explanation-level OOD AUROC are separate analyses: the first correlates two token trajectories; the second reduces a whole explanation to one scalar score and measures dataset separation. Bootstrap intervals quantify example resampling, not independently trained model or inference-seed uncertainty."
    ),
    Spacer(1, 6 * mm),
    P("Common experimental backbone", "Kicker"),
    P(
        "The main model was IBM Granite 3.1 3B A800M Instruct. Full-covariance variational routers replaced ten MoE routers at layers 5, 6, 7, 8, 19, 20, 28, 29, 30, and 31. The posterior covariance is LL^T; ILV is tr(LL^T), averaged across the selected layers. Stochastic evaluation used 35 router-logit samples, averaged in probability space before Top-8 expert selection. ILV is read analytically from L; it is not the empirical variance of those 35 samples.",
        "BodyThesis",
    ),
    PageBreak(),
]


# Result 1
chart = save_grouped_bar(
    "r01_obqa_foundation.png",
    ["ARC-C", "ARC-E", "MedMCQA", "MMLU law"],
    {"Answer entropy": [0.617, 0.512, 0.736, 0.764], "ILV beta 0.01": [0.687, 0.593, 0.792, 0.942]},
    value_labels=True,
)
story += result_header(1, "Corrected answer-only OBQA reproduced useful ILV OOD detection", "Historical held-out test; n=500 ID; beta 0.01", "OOD = 1; higher ILV intended for OOD")
story += [fig_image(chart, 63)]
story += [
    data_table(
        [
            ["FCVR", "ACC", "NLL", "ECE", "MCE", "Macro ILV OOD AUROC"],
            ["beta 0.01", "0.772", "0.6369", "0.0996", "0.2631", "0.754"],
            ["beta 0.1", "0.798", "0.5934", "0.0794", "0.3220", "0.510"],
        ],
        [33 * mm, 23 * mm, 25 * mm, 25 * mm, 25 * mm, 35 * mm],
    ),
    inference_box(
        "Correcting the answer-mask/collator bug restored the intended ILV direction. At beta 0.01, ILV exceeded answer entropy on all four shifts and its macro AUROC of 0.754 closely matched the paper-scale reference of about 0.749. Raising beta to 0.1 improved ACC, NLL, and ECE but reduced macro ILV OOD AUROC to 0.510, showing that ordinary predictive calibration and router-variance OOD ranking are not interchangeable. This result remains qualified by the old padding-inclusive KL reduction."
    ),
    source_line("results/fcvr/*ansmask* and logs/overnight-granite-obqa-ansmask-20260815-190519.log"),
    PageBreak(),
]


# Result 2
story += result_header(2, "Historical FCVR configurations were frequently inverted", "Historical held-out test summaries; legacy files omit CIs", "OOD = 1; below 0.5 is inverted")
hist_rows = [
    ["Configuration", "ACC", "ECE", "Entropy macro", "ILV macro", "Reading"],
    ["Correct answer-mask, beta 0.01", "0.772", "0.100", "0.657", "0.754", "intended"],
    ["Correct answer-mask, beta 0.1", "0.798", "0.079", "0.675", "0.510", "chance"],
    ["Experts, beta 0.01", "0.784", "0.133", "0.713", "0.292", "inverted"],
    ["Experts, beta 0.1", "0.778", "0.152", "0.715", "0.286", "inverted"],
    ["MAP prior", "0.702", "0.084", "0.705", "0.233", "inverted"],
    ["Pretrained prior, beta 0.01", "0.756", "0.059", "0.705", "0.232", "inverted"],
    ["Pretrained prior, beta 0.1", "0.756", "0.055", "0.695", "0.263", "inverted"],
    ["Pretrained prior, unsuffixed", "0.682", "0.091", "0.696", "0.165", "strong inversion"],
    ["Layers 27-31 only", "0.730", "0.137", "0.699", "0.326", "inverted"],
]
story += [data_table(hist_rows, [50 * mm, 19 * mm, 19 * mm, 27 * mm, 24 * mm, 27 * mm])]
story += [
    inference_box(
        "The successful answer-mask run was not the default behaviour of FCVR. Several plausible configurations assigned lower ILV to OOD examples, sometimes with AUROC as low as 0.165, even while answer entropy remained above chance. These are failures under the intended score direction and establish why preprocessing, target construction, and explicit orientation checks are central to the thesis rather than implementation footnotes."
    ),
    source_line("Historical results/fcvr families summarised in thesis-results-comprehensive.md Section 6.1"),
    PageBreak(),
]


# Result 3
chart = save_grouped_bar(
    "r03_token_debug.png",
    ["pre .01", "pre .1", "untrained", "2 trunks", "beta .001", "beta 0", "layer norm", "longer", "prior sd 2"],
    {
        "ILV": [0.233, 0.265, 0.541, 0.239, 0.288, 0.648, 0.253, 0.208, 0.467],
        "Off-diagonal": [0.760, 0.776, 0.670, 0.770, 0.765, 0.535, 0.788, 0.756, 0.728],
    },
)
story += result_header(3, "Inversion ablations did not yield a principled architectural fix", "Historical diagnostic ablations", "Macro OOD AUROC; OOD = 1")
story += [fig_image(chart, 62)]
story += [
    P("Beta 0 produced ILV macro AUROC 0.648, but removed the KL term. Other changes - two trunks, beta 0.001, layer normalisation, longer training, or prior standard deviation 2 - yielded ILV macro AUROC 0.208-0.467. Off-diagonal statistics remained 0.728-0.788 in many inverted runs.", "BodyThesis"),
    inference_box(
        "No ablation both restored positive trace-ILV separation and retained the intended variational constraint. The positive beta-zero result is therefore not evidence that Bayesian FCVR worked; it is evidence that the training objective controls score orientation. The covariance off-diagonal structure often still carried domain information when the trace was inverted, motivating future component-wise analyses rather than treating total variance as the only possible readout."
    ),
    source_line("results-debug/phase0 and results-debug/arms"),
    PageBreak(),
]


# Result 4
chart = save_grouped_bar(
    "r04_medexqa_transfer.png",
    ["Teacher forced", "Generated s42", "Generated s43", "Generated s44"],
    {"beta 0.01": [0.373, 0.319, 0.287, 0.297], "beta 0.1": [0.191, 0.065, 0.084, 0.047]},
    ylabel="Mean per-example Spearman",
    ylim=(0, 0.45),
    chance=False,
    value_labels=True,
)
story += result_header(4, "MedExQA beta 0.01 showed positive token-level transfer", "Exploratory validation; n=50; no held-out test", "Spearman(ILV, entropy); positive expected")
story += [fig_image(chart, 63)]
story += [
    P("Teacher-forced beta-0.01 Spearman was 0.373 [0.346, 0.398]. Generated values were 0.319 [0.287, 0.351], 0.287 [0.250, 0.325], and 0.297 [0.262, 0.332] across seeds 42-44. Beta 0.1 was much weaker. Online/post-hoc ILV agreement was about 0.984-0.994.", "BodyThesis"),
    inference_box(
        "The beta-0.01 router signal co-moved with predictive entropy along MedExQA explanation trajectories, and the finding was stable across three stochastic generation seeds. This validates a local token-level relationship on the validation set, not sequence correctness or OOD detection. The larger beta again suppressed variation, consistent with posterior contraction."
    ),
    source_line("results/token_analysis/*_val_medexqa* and logs/iter1-granite-medexqa-20260817-163435.log"),
    PageBreak(),
]


# Result 5
story += result_header(5, "MedExQA transfer did not become reliable wrong-answer detection", "Exploratory validation; n=50; correctness label partly probe-derived", "Wrong answer = 1; higher score intended for wrong")
story += [
    data_table(
        [
            ["beta", "Labelled n", "Wrong", "Last-10 ILV AUROC [95% CI]", "Entropy-max", "Length", "Primary correctness"],
            ["0.01", "50", "21", "0.435 [0.276, 0.586]", "0.567", "0.525", "0.580"],
            ["0.1", "50", "22", "0.516 [0.357, 0.679]", "0.657", "0.643", "0.560"],
        ],
        [16 * mm, 20 * mm, 17 * mm, 43 * mm, 22 * mm, 19 * mm, 29 * mm],
    ),
    Spacer(1, 6 * mm),
    data_table(
        [
            ["Quality / label statistic (beta 0.01, seed 42)", "Value"],
            ["Option-judgeable outputs", "32 / 50"],
            ["Option accuracy among judgeable", "0.625"],
            ["Probe correctness", "0.640"],
            ["NLI entailment / contradiction", "0.520 / 0.380"],
            ["Token F1 / ROUGE-L / BERTScore-F1", "0.481 / 0.311 / 0.706"],
        ],
        [112 * mm, 54 * mm],
    ),
    inference_box(
        "Positive within-sequence ILV-entropy correlation did not imply that incorrect sequences received higher aggregate ILV. The beta-0.01 point estimate was inverted and both confidence intervals were wide. Because only 32 outputs were directly option-judgeable and the remaining labels used a probe fallback, this pilot is best used to motivate stricter correctness labelling and larger samples, not as a confirmed abstention result."
    ),
    source_line("results/abstention and results/labels MedExQA validation families"),
    PageBreak(),
]


# Result 6
story += result_header(6, "MedExQA input OOD detection depended strongly on readout position", "Historical held-out test; n=175 per class", "OOD = 1; higher score intended for OOD")
story += [
    data_table(
        [
            ["beta", "OOD", "Final ILV AUROC / AUPRC", "Prompt-mean ILV", "Answer entropy"],
            ["0.01", "OBQA", "0.621 / 0.580", "0.577 / 0.573", "0.704 / 0.727"],
            ["0.01", "MMLU law", "0.700 / 0.661", "0.003 / 0.308", "0.562 / 0.505"],
            ["0.1", "OBQA", "0.648 / 0.646", "0.140 / 0.330", "0.777 / 0.801"],
            ["0.1", "MMLU law", "0.334 / 0.392", "0.053 / 0.314", "0.694 / 0.632"],
        ],
        [18 * mm, 28 * mm, 43 * mm, 39 * mm, 38 * mm],
    ),
    inference_box(
        "The same beta-0.01 model detected law OOD at 0.700 AUROC from the final prompt position but produced an almost perfectly reversed 0.003 AUROC after averaging ILV across the prompt. This early result anticipated the later OBQA positional diagnosis: ILV cannot be treated as position-invariant. Answer entropy was more consistently oriented across these four comparisons."
    ),
    source_line("results/input_level_ood"),
    PageBreak(),
]


# Result 7
chart = save_horizontal_bar(
    "r07_medmcqa_transfer.png",
    ["Generated validation", "Teacher-forced gold"],
    [-0.404, -0.306],
    [RED, ORANGE],
    "Mean per-example Spearman(ILV, entropy)",
    (-0.5, 0.1),
)
story += result_header(7, "Original MedMCQA explanation generation produced strong inverse transfer", "Full validation; n=1,000; test not run", "Spearman(ILV, entropy); positive expected")
story += [fig_image(chart, 56)]
story += [
    data_table(
        [
            ["Condition", "Tokens", "Mean Spearman [95% CI]", "Pooled z-scored"],
            ["Generated", "95,080", "-0.404 [-0.417, -0.392]", "-0.437"],
            ["Teacher forced", "126,566", "-0.306 [-0.315, -0.295]", "-0.299"],
        ],
        [42 * mm, 31 * mm, 58 * mm, 35 * mm],
    ),
    inference_box(
        "This is a precise negative result rather than a noisy near-zero result. Only 1.8% of sequences had positive correlation, and high-entropy tokens aligned with ILV troughs. Online/post-hoc agreement of 0.993 +/- 0.005 rules out a capture mismatch. The positive MedExQA result therefore did not generalise to this larger MedMCQA explanation model; token-level transfer is dataset- and training-dependent."
    ),
    source_line("results/token_analysis/step1_medmcqa_gen_val_*"),
    PageBreak(),
]


# Result 8
chart = save_grouped_bar(
    "r08_medmcqa_abstention.png",
    ["Frozen last-10 ILV", "Best listed raw ILV", "Maximum entropy"],
    {"Wrong-answer AUROC": [0.500, 0.530, 0.570]},
    value_labels=True,
)
story += result_header(8, "MedMCQA aggregate ILV did not support selective abstention", "Validation; 840 option-labelled of 1,000 generated", "Wrong answer = 1; higher score intended for wrong")
story += [fig_image(chart, 59)]
story += [
    P("Frozen last-ten ILV AUROC was 0.500 [0.460, 0.539] with AURC 0.562. Maximum entropy reached 0.570 [0.533, 0.608] with AURC 0.501. Of 1,000 generations, 840 were option-judgeable, 160 unlabelled, 166 hit the 256-token cap, and option accuracy among labelled responses was 0.448.", "BodyThesis"),
    inference_box(
        "The model usually responded, but response is not correctness. The intended policy was to generate, score uncertainty, and withhold answers likely to be wrong. Frozen ILV was chance for that task, while a simple entropy baseline was modestly useful. Because the held-out test was never evaluated, even the entropy result remains validation evidence rather than final deployment performance."
    ),
    source_line("results/abstention/medmcqa_gen/frozen_beta0.01-S35.json and results/labels/audit_medmcqa*"),
    PageBreak(),
]


# Result 9
story += result_header(9, "Controlled MedMCQA: explanations improved calibration but degraded ILV error ranking", "Held-out test; n=1,000; primary seed 42", "Wrong answer = 1; higher score intended for wrong")
story += [
    data_table(
        [
            ["FCVR arm", "ACC", "NLL", "ECE", "Entropy AUROC", "ILV AUROC"],
            ["A: answer-only", "0.467", "1.298", "0.173", "0.652", "0.610"],
            ["B: answer + explanation", "0.444", "1.220", "0.052", "0.672", "0.439"],
            ["Paired B - A [95% CI]", "-0.023 [-0.057, 0.009]", "-0.078 [-0.130, -0.024]", "-0.121 [-0.152, -0.078]", "+0.020 [-0.020, 0.062]", "-0.170 [-0.224, -0.120]"],
        ],
        [37 * mm, 29 * mm, 30 * mm, 30 * mm, 21 * mm, 21 * mm],
    ),
    Spacer(1, 6 * mm),
    P("Mean maximum answer probability was about 0.638 for Arm A against accuracy 0.467, versus 0.485 for Arm B against accuracy 0.444. Thus Arm B was substantially less overconfident even though its accuracy was not significantly different.", "BodyThesis"),
    inference_box(
        "This cleaner A/B comparison separates two meanings of uncertainty. Explanation supervision improved NLL and ECE by reducing overconfidence, but significantly reversed the ranking of router covariance with mistakes. Good output calibration therefore does not imply that ILV is correctly oriented. Unequal early-stopping points are outcomes of the shared rule but remain a training-exposure qualification."
    ),
    source_line("results/reports/letter_arms_test.md; seed 42 is primary because seeds 43/44 changed data rows"),
    PageBreak(),
]


# Result 10
chart = save_grouped_bar(
    "r10_medmcqa_ood.png",
    ["MedExQA", "OBQA", "ARC-E", "ARC-C", "SciQ", "MMLU law"],
    {"Arm A": [0.398, 0.439, 0.383, 0.546, 0.211, 0.871], "Arm B": [0.390, 0.587, 0.441, 0.327, 0.578, 0.029]},
    value_labels=True,
)
story += result_header(10, "Controlled MedMCQA OOD behaviour was heterogeneous and worse on average with explanations", "Held-out test; seed 42; 175 or 500 per class", "OOD = 1; higher ILV intended for OOD")
story += [fig_image(chart, 62)]
story += [
    P("Paired B - A changes ranged from +0.367 [0.321, 0.415] on SciQ to -0.842 [-0.866, -0.816] on MMLU law. Macro paired change was -0.083 [-0.105, -0.058].", "BodyThesis"),
    inference_box(
        "Explanation supervision did not uniformly damage or improve domain separation. Arm B improved on OBQA and SciQ but became severely inverted on ARC-Challenge and especially law. The negative macro effect is supported, but the domain heterogeneity argues against a single global interpretation of ILV scale. Medical subject composition, target style, prompt length, and learned covariance orientation may interact differently with each shift."
    ),
    source_line("results/ilv_ood_arms/medmcqa-arms_test_data-s42_mc-s42.md"),
    PageBreak(),
]


# Result 11
story += result_header(11, "OBQA pipeline comparison preserved accuracy while reversing final-position ILV", "Held-out test; n=500; frozen split; seed 42", "Wrong answer = 1; higher score intended for wrong")
story += [
    data_table(
        [
            ["FCVR pipeline", "ACC", "NLL", "ECE", "Entropy AUROC", "Gate AUROC", "ILV AUROC"],
            ["Arm A answer-only", "0.806", "0.647", "0.088", "0.793", "0.288", "0.766"],
            ["Arm B answer + explanation", "0.810", "0.664", "0.104", "0.792", "0.419", "0.377"],
            ["Paired B - A [95% CI]", "+0.004 [-0.030, 0.036]", "+0.016 [-0.078, 0.116]", "+0.016 [-0.016, 0.045]", "-0.002 [-0.054, 0.052]", "+0.131 [0.054, 0.205]", "-0.390 [-0.478, -0.296]"],
        ],
        [37 * mm, 24 * mm, 24 * mm, 24 * mm, 20 * mm, 18 * mm, 20 * mm],
    ),
    Spacer(1, 5 * mm),
    P("Across stochastic readout seeds 42-44, Arm A ILV AUROC was 0.769 +/- 0.002 and Arm B was 0.390 +/- 0.015. The same frozen 500 examples were retained across OBQA seeds.", "BodyThesis"),
    inference_box(
        "The answer-only and explanation-trained pipelines performed similarly on accuracy and output entropy, yet the ILV relationship with errors reversed by about 0.39 AUROC. This is the clearest evidence that explanation-target training changed the learned mapping from hidden states to covariance rather than merely adding information to a fixed uncertainty representation. Because the OBQA arms also differ in training prompt, row construction, Stage-1 provenance, and optimisation exposure, this is a pipeline comparison rather than a clean causal target-only ablation."
    ),
    source_line("OBQA-comparison/results/reports/obqa_gen/letter_arms_test.md"),
    PageBreak(),
]


# Result 12
chart = save_grouped_bar(
    "r12_obqa_ood.png",
    ["MedExQA", "MedMCQA", "ARC-E", "ARC-C", "SciQ", "MMLU law"],
    {"Arm A": [0.702, 0.812, 0.574, 0.680, 0.515, 0.943], "Arm B": [0.188, 0.159, 0.329, 0.269, 0.401, 0.032]},
    value_labels=True,
)
story += result_header(12, "OBQA final-position ILV OOD detection inverted in Arm B on all six shifts", "Held-out test; n=175 or 500 per class; 2,000 paired bootstraps", "OOD = 1; no sign flipping")
story += [fig_image(chart, 63)]
story += [
    P("Paired B - A changes were -0.515, -0.654, -0.244, -0.411, -0.114, and -0.911 by domain. The macro change was -0.475 [-0.498, -0.451]. Arm A detected five shifts and was inconclusive on SciQ; Arm B was significantly inverted on every shift.", "BodyThesis"),
    inference_box(
        "The result is an actual score reversal: Arm A mean ILV rose on OOD inputs, whereas Arm B mean ILV fell. The effect is far larger than the example-level confidence intervals and cannot be explained as random sampling error within this evaluation. It still cannot be attributed solely to explanation supervision because the two pipelines were not controlled in every training detail."
    ),
    source_line("OBQA-comparison/results/ilv_ood_arms/obqa-arms_test_data-s42_mc-s42.md"),
    PageBreak(),
]


# Result 13
story += result_header(13, "Prompt crossover showed that inversion followed the trained pipeline", "Diagnostic on the same OBQA ID/OOD examples", "OOD = 1; final-prompt ILV")
story += [
    data_table(
        [
            ["OOD", "Arm A + MCQ", "Arm A + comparison", "Arm B + MCQ", "Arm B + comparison"],
            ["MedExQA", "0.702", "0.718", "0.168", "0.188"],
            ["MedMCQA", "0.812", "0.810", "0.161", "0.159"],
            ["ARC-Easy", "0.574", "0.618", "0.327", "0.329"],
            ["ARC-Challenge", "0.680", "0.699", "0.264", "0.269"],
            ["SciQ", "0.515", "0.558", "0.402", "0.401"],
            ["MMLU law", "0.943", "0.959", "0.033", "0.032"],
        ],
        [36 * mm, 31 * mm, 34 * mm, 31 * mm, 34 * mm],
    ),
    Spacer(1, 5 * mm),
    P("Macro prompt effect (comparison minus MCQ): Arm A +0.022 [0.015, 0.030]; Arm B +0.004 [-0.003, 0.011]. Model effect (Arm B minus Arm A): -0.479 [-0.502, -0.456] under MCQ and -0.497 [-0.519, -0.473] under the comparison prompt.", "BodyThesis"),
    inference_box(
        "Changing only the inference prompt produced small effects, while holding the prompt fixed left an approximately 0.48-0.50 AUROC pipeline gap. The simplest claim that Arm B was inverted only because it saw a different test prompt is therefore rejected. The diagnostic does not remove the training-prompt and row-construction confounds because each model was still trained under its original pipeline."
    ),
    source_line("OBQA-comparison/results/analyze_obqa_gen/obqa-gen-analysis_test_data-s42_mc-s42.md"),
    PageBreak(),
]


# Result 14
story += result_header(14, "Teacher-forced traces revealed a position-dependent reorganisation", "Diagnostic; OBQA ID versus MedExQA / MedMCQA OOD", "OOD = 1; different token positions and aggregates")
story += [
    data_table(
        [
            ["Score", "Arm A MedExQA / MedMCQA", "Arm B MedExQA / MedMCQA"],
            ["Final prompt ILV", "0.697 / 0.811", "0.183 / 0.167"],
            ["Explanation mean ILV", "0.861 / 0.992", "0.735 / 0.662"],
            ["Explanation max ILV", "0.961 / 0.991", "0.998 / 0.985"],
            ["Explanation second max", "0.983 / 0.998", "0.997 / 0.981"],
            ["Explanation last-10 ILV", "0.726 / 0.969", "0.823 / 0.702"],
        ],
        [55 * mm, 56 * mm, 55 * mm],
    ),
    Spacer(1, 5 * mm),
    P("Arm B mean ILV by position on ID: prompt mean 38.540; prompt-final 38.054; answer letter 38.519; explanation marker 35.350; explanation region 32.408; EOS 38.308. Wrong-answer AUROC within teacher-forced explanations stayed near chance. Independent evaluator agreement on ID ranks was Spearman 0.928 for Arm A and 0.908 for Arm B.", "BodyThesis"),
    inference_box(
        "Arm B's final-prompt score was inverted, yet large explanation-region aggregates separated the same domains. This supports a learned positional reorganisation rather than a broken extractor. The 0.928/0.908 correlations are sanity checks comparing two evaluation paths; they are not token-level transfer or correctness results. Strong OOD separation inside gold explanations does not imply advance warning that the answer will be wrong."
    ),
    source_line("OBQA prompt-crossover and teacher-forced trace reports"),
    PageBreak(),
]


# Result 15
rows = ["A mean ILV", "A max ILV", "A last-10 ILV", "B mean ILV", "B max ILV", "B last-10 ILV", "Length"]
cols = ["MedExQA", "ScienceQA", "ECQA", "AQuA-RAT"]
tf_data = [
    [0.861, 0.563, 0.520, 0.949],
    [0.961, 0.873, 0.654, 0.970],
    [0.726, 0.431, 0.472, 0.903],
    [0.735, 0.966, 0.957, 0.989],
    [0.998, 0.999, 0.992, 0.989],
    [0.823, 0.935, 0.963, 0.987],
    [1.000, 0.998, 0.937, 0.995],
]
chart = save_heatmap("r15_teacher_forced.png", rows, cols, tf_data, "Teacher-forced gold explanations")
story += result_header(15, "Teacher-forced explanation readouts strongly distinguished dataset-specific gold targets", "Diagnostic; OBQA gold explanations as ID", "OOD = 1; explanation-level aggregates")
story += [fig_image(chart, 72)]
story += [
    inference_box(
        "Ignoring length, these tables support strong operational domain discrimination inside the explanation trajectory, especially for Arm B, and suggest that explanation supervision relocated domain-sensitive information away from the final prompt position. They do not isolate epistemic uncertainty: gold targets expose dataset-specific style, vocabulary, topic, and token composition, and maxima have more opportunities to rise in longer sequences. Length alone achieved AUROC 0.937-1.000, so the spectacular teacher-forced values must not be presented as deployment evidence."
    ),
    source_line("results/ood_expl_readout/obqa-ood-expl*_tf*"),
    PageBreak(),
]


# Result 16
gen_data = [
    [0.870, 0.462, 0.402, 0.861],
    [0.819, 0.476, 0.435, 0.911],
    [0.727, 0.412, 0.470, 0.547],
    [0.264, 0.449, 0.618, 0.588],
    [0.318, 0.386, 0.565, 0.507],
    [0.270, 0.399, 0.593, 0.606],
    [0.651, 0.411, 0.465, 0.697],
]
chart = save_heatmap("r16_generated.png", rows, cols, gen_data, "Self-generated explanations")
story += result_header(16, "Arm B's teacher-forced ILV advantage largely collapsed under self-generation", "Deployment-like diagnostic; greedy generation up to 256 tokens", "OOD = 1; explanation-level aggregates")
story += [fig_image(chart, 72)]
story += [
    P("Arm B teacher forcing to generation macro changes: mean ILV -0.432; max -0.550; second max -0.545; last-10 -0.460. Generated macro B - A changes were -0.169, -0.216, -0.222, and -0.072 for those ILV aggregates. Arm B output entropy and surprisal instead improved over Arm A by +0.463 and +0.443.", "BodyThesis"),
    inference_box(
        "The model did not retain the gold-target separation when conditioned on its own generated trajectory. This is the decisive reason not to equate teacher-forced AUROC with deployment-time epistemic uncertainty. For Arm B, conventional output entropy and surprisal were the more consistent generated-condition signals. Any uncertainty-guided decoder should therefore be validated during actual generation rather than only on gold continuations."
    ),
    source_line("results/ood_expl_readout/obqa-ood-expl*_gen*"),
    PageBreak(),
]


# Result 17
chart = save_grouped_bar(
    "r17_generation_lengths.png",
    ["OBQA", "MedExQA", "ScienceQA", "ECQA", "AQuA-RAT"],
    {"Arm A mean length": [53.1, 82.4, 59.0, 44.5, 189.5], "Arm B mean length": [12.4, 15.2, 11.7, 11.6, 16.5]},
    ylabel="Generated tokens",
    ylim=(0, 220),
    chance=False,
    value_labels=True,
)
story += result_header(17, "The two OBQA pipelines generated qualitatively different continuation distributions", "Descriptive behaviour; full datasets plus 20-example qualitative viewer", "Length and accuracy are behavioural, not uncertainty metrics")
story += [fig_image(chart, 63)]
story += [
    data_table(
        [
            ["Dataset", "Arm A letter ACC", "Arm B letter ACC", "Arm A truncation", "Arm B truncation"],
            ["OBQA", "0.792", "0.816", "0", "0"],
            ["MedExQA", "0.577", "0.566", "not reported", "not reported"],
            ["ScienceQA", "0.565", "0.542", "not reported", "not reported"],
            ["ECQA", "0.586", "0.648", "not reported", "not reported"],
            ["AQuA-RAT", "0.230", "0.258", "0.402", "0"],
        ],
        [35 * mm, 31 * mm, 31 * mm, 35 * mm, 34 * mm],
    ),
    inference_box(
        "Arm B learned short, templated fact-style explanations, whereas answer-only Arm A often continued much longer and sometimes hallucinated or contradicted its selected option. This distribution shift helps explain why maximum and second-maximum ILV are unsafe cross-arm comparisons and why teacher-forced and generated trajectories diverge. Shorter output is not evidence of better explanation quality; the viewer is qualitative context, not a statistical quality study."
    ),
    source_line("OBQA-comparison/results/generation_examples and full generation reports"),
    PageBreak(),
]


# Result 18
story += result_header(18, "Early transfer probes already showed that token-level transfer was not universal", "Historical pilots; point estimates", "Spearman(ILV, entropy); positive expected")
story += [
    data_table(
        [
            ["Dataset / condition", "n", "Tokens", "Mean per-example Spearman", "Reading"],
            ["MedExQA beta 0.01 teacher forced", "175", "21,509", "0.357", "positive"],
            ["MedExQA beta 0.01 generated, 128", "175", "19,686", "0.314", "positive"],
            ["MedExQA beta 0.01 generated, 256", "175", "21,678", "0.308", "positive"],
            ["MedExQA beta 0.1 teacher forced", "175", "not recorded", "0.163", "weak"],
            ["MedExQA beta 0.1 generated, 128", "175", "not recorded", "0.059", "negligible"],
            ["OBQA deterministic pilot", "50", "4,867", "-0.0089", "near zero"],
            ["Built-in prompts", "15", "476", "0.174", "too small"],
        ],
        [57 * mm, 16 * mm, 29 * mm, 38 * mm, 26 * mm],
    ),
    inference_box(
        "The larger MedExQA test pilot consistently supported positive transfer at beta 0.01, whereas early OBQA was essentially zero and the later MedMCQA model was strongly negative. The cross-dataset pattern rules out a universal claim that router variance automatically tracks token entropy during generation. It instead motivates studying target, token category, position, and regularisation as moderators of the relationship."
    ),
    source_line("Non-_val_ results/token_analysis MedExQA, OBQA, and built-in families"),
    PageBreak(),
]


# Subject appendix
story += [P("Appendix A", "Kicker"), P("MedMCQA subject-level FCVR accuracy / ECE", "Section")]
subjects = [
    ("Dental", 102, "0.412 / 0.129", "0.402 / 0.116"),
    ("Pharmacology", 90, "0.511 / 0.215", "0.500 / 0.099"),
    ("Gynaecology & Obstetrics", 87, "0.506 / 0.114", "0.540 / 0.158"),
    ("Medicine", 81, "0.457 / 0.229", "0.358 / 0.185"),
    ("Pediatrics", 75, "0.493 / 0.221", "0.467 / 0.111"),
    ("Surgery", 74, "0.514 / 0.121", "0.459 / 0.067"),
    ("Pathology", 66, "0.515 / 0.210", "0.515 / 0.136"),
    ("Biochemistry", 64, "0.594 / 0.230", "0.453 / 0.112"),
    ("Anatomy", 61, "0.475 / 0.165", "0.443 / 0.080"),
    ("Social & Preventive Medicine", 57, "0.386 / 0.219", "0.404 / 0.122"),
    ("Physiology", 53, "0.415 / 0.264", "0.377 / 0.165"),
    ("Microbiology", 40, "0.400 / 0.289", "0.375 / 0.201"),
    ("Ophthalmology", 32, "0.531 / 0.262", "0.469 / 0.137"),
    ("Forensic Medicine", 32, "0.469 / 0.263", "0.438 / 0.158"),
    ("ENT", 25, "0.360 / 0.296", "0.520 / 0.133"),
    ("Anaesthesia", 18, "0.389 / 0.240", "0.222 / 0.242"),
    ("Radiology", 16, "0.312 / 0.362", "0.438 / 0.231"),
    ("Orthopaedics", 10, "0.200 / 0.436", "0.300 / 0.251"),
    ("Skin", 8, "0.500 / 0.289", "0.375 / 0.285"),
    ("Psychiatry", 7, "0.429 / 0.341", "0.714 / 0.388"),
]
subject_rows = [["Subject", "n", "Arm A ACC / ECE", "Arm B ACC / ECE"]] + [list(x) for x in subjects]
story += [data_table(subject_rows, [67 * mm, 14 * mm, 43 * mm, 43 * mm])]
story += [
    inference_box(
        "No uniform subject-wise advantage is visible. Arm B was better in some small specialties and worse in several larger ones, while its ECE was often lower even when accuracy declined. Because subject counts range from 7 to 102, this table is descriptive and should not be used for inferential claims without subject-specific uncertainty intervals."
    ),
    source_line("results/reports/letter_arms_test.md; two Unknown rows excluded"),
    PageBreak(),
]


# Synthesis
story += [P("Thesis-ready synthesis", "Section")]
claim_rows = [
    ["Supported claim", "Boundary / qualification"],
    ["Answer-only OBQA FCVR can provide useful final-position ILV.", "The original positive run used padding-inclusive KL; the later corrected Arm A confirms strong error/OOD ranking."],
    ["Explanation-trained pipelines can preserve answer performance while reversing ILV.", "OBQA is a pipeline comparison, not a fully controlled target-only causal ablation."],
    ["Conventional calibration and ILV orientation are distinct.", "ECE evaluates answer probabilities, not the semantic validity of router covariance."],
    ["ILV is position- and conditioning-dependent.", "Final prompt, answer letter, explanation region, EOS, teacher forcing, and generation can disagree."],
    ["Teacher-forced explanation scores can separate datasets.", "Length, style, vocabulary, topic, and gold-token exposure remain major confounds."],
    ["Output entropy / surprisal are essential baselines.", "They were often more stable than ILV, especially under self-generation."],
]
story += [data_table(claim_rows, [78 * mm, 88 * mm]), Spacer(1, 5 * mm)]
story += [
    inference_box(
        "The complete evidence supports a boundary result, not a universal positive or negative verdict. FCVR ILV can be a strong uncertainty signal in the corrected answer-only OBQA setting. It is not invariant to supervised target, dataset, readout position, or conditioning regime. Explanation supervision changed the learned mapping from token representations to posterior covariance: the formula tr(LL^T) stayed fixed, but high final-position ILV predicted OOD/errors for Arm A and not for Arm B. Therefore, an uncertainty estimator must be validated at the exact position and under the exact generation process in which it will be deployed."
    ),
    Spacer(1, 6 * mm),
    P("Highest-priority follow-up experiments", "Kicker"),
    P(
        "1) rerun MedMCQA with a fixed data seed and separate MC seeds; 2) perform a fully controlled OBQA target ablation with identical rows, prompt, initialisation, and update budget; 3) evaluate frozen generation scores on held-out test; 4) use length/style-matched or residualised OOD controls; 5) repeat full training across seeds; 6) compare covariance components and pre-register any score orientation on validation; 7) prioritise generated-condition evaluation; and 8) assess explanation faithfulness and correctness beyond surface-form similarity.",
        "BodyThesis",
    ),
    PageBreak(),
]


# Provenance
story += [P("Evidence and provenance", "Section")]
story += [
    P(
        "The numerical authority for this compendium is thesis-results-comprehensive.md, independently checked against THESIS_RESULTS_SOURCE_AUDIT.md. The prior task's follow-up discussion supplied interpretation corrections that are reflected here: token-level transfer versus nine-score OOD analysis; the strict AUROC orientations; the meaning of selective prediction; the difference between MAP and zero-shot; the evaluator-agreement sanity correlation; and the teacher-forcing length/style caveats.",
        "BodyThesis",
    ),
    data_table(
        [
            ["Evidence family", "Primary local source"],
            ["Foundational OBQA", "results/fcvr and exp4 training log"],
            ["Historical / ablations", "results-exp1, results/vtsr, results-debug"],
            ["MedExQA generation", "results/token_analysis, results/abstention, results/labels"],
            ["Original MedMCQA generation", "results/token_analysis, results/abstention/medmcqa_gen, results/data"],
            ["Controlled MedMCQA", "results/reports and results/ilv_ood_arms"],
            ["OBQA pipeline comparison", "OBQA-comparison/results/reports and results/ilv_ood_arms"],
            ["OBQA inversion diagnostics", "OBQA-comparison/results/analyze_obqa_gen and results/ood_expl_readout"],
            ["Qualitative generation", "OBQA-comparison/results/generation_examples"],
        ],
        [58 * mm, 108 * mm],
    ),
    Spacer(1, 6 * mm),
    P(
        "Reporting cautions: legacy JSONs may omit sample sizes and CIs; MedMCQA seeds 43/44 changed data rows; the original MedMCQA and MedExQA frozen abstention studies have no held-out test; correctness policies differ across branches; OBQA arms are not fully controlled; bootstrap CIs cover examples rather than model-training or MC-seed variability; and stale generated-report prose must be subordinate to JSON configuration objects and logs.",
        "BodyThesis",
    ),
    Spacer(1, 10 * mm),
    P("End of compendium", "Kicker"),
]


doc = NumberedDocTemplate(
    str(OUT),
    pagesize=A4,
    rightMargin=22 * mm,
    leftMargin=22 * mm,
    topMargin=20 * mm,
    bottomMargin=22 * mm,
    title="Thesis experiment results compendium",
    author="Compiled from the moe-uncertainty project evidence base",
    subject="Verified experimental results and thesis-safe interpretations",
)
doc.build(story)
print(OUT)
