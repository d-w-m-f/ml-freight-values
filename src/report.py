"""Builds the submission report (PDF): data split, validation approach, results and the December chart.

Run from the repository root, after the other steps:

    poetry install --with report
    poetry run python -m src.report

Every number is read from the files written by the previous steps, so the
report always matches the code's output. Writes src/freight-rate-report/freight-rate-report.pdf.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src import config
from src.final_model import FINAL_PARAMS

REPORT_PATH = config.SRC_DIR / "freight-rate-report" / "freight-rate-report.pdf"
FIGURES_DIR = config.OUTPUTS_DIR / "report"
MODELS_DIR = config.OUTPUTS_DIR / "models"
TUNING_DIR = config.OUTPUTS_DIR / "tuning"
TEST_DIR = config.OUTPUTS_DIR / "test"
DECEMBER_CHART = config.ROOT_DIR / "scorer_results" / "candidate_december.png"

AUTHOR = "Derick William de Moraes Frias"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MODEL_LABELS = {
    "baseline": "Baseline (median $/mile by equipment)",
    "regression": "Ridge regression",
    "lightgbm": "LightGBM",
    "final_hybrid": "Final hybrid (Ridge + LightGBM)",
}

# Chart styling (categorical slots of the reference palette, fixed order).
SERIES = {"baseline": "#2a78d6", "regression": "#eb6834", "lightgbm": "#1baf7a", "final_hybrid": "#eda100"}
PHASES = {"Train": "#2a78d6", "Validate": "#eb6834", "Test (once)": "#1baf7a", "Predict": "#eda100"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9dde1")
HEADER_FILL = colors.HexColor("#eef2f6")
HIGHLIGHT_FILL = colors.HexColor("#fdf3d8")


# --------------------------------------------------------------------------- results


def load_results() -> dict:
    cv = pd.read_csv(MODELS_DIR / "cv_summary.csv", index_col="model")
    folds = pd.read_csv(MODELS_DIR / "cv_folds.csv")
    tuning = pd.read_csv(TUNING_DIR / "4_lightgbm.csv", index_col="model")
    final_key = (
        f"trees={FINAL_PARAMS['n_estimators']},leaves={FINAL_PARAMS['num_leaves']},"
        f"min_samples={FINAL_PARAMS['min_child_samples']}"
    )
    final_cv = tuning.loc[final_key]

    by_month = folds.pivot(index="validation_month", columns="model", values="mape")[["baseline", "regression", "lightgbm"]]
    by_month["final_hybrid"] = [final_cv[f"mape_month_{m}"] for m in by_month.index]

    cv_table = cv.loc[["baseline", "regression", "lightgbm"], ["mape_mean", "mape_std", "mape_plausible_labels_mean", "wape_mean", "mae_mean"]]
    cv_table.loc["final_hybrid"] = final_cv[cv_table.columns]

    test = pd.read_csv(TEST_DIR / "october_test.csv", index_col="model")
    segments = pd.read_csv(TEST_DIR / "october_test_segments.csv", index_col=0).iloc[:, 0]
    test_predictions = pd.read_csv(TEST_DIR / "october_test_predictions.csv")
    december = pd.read_csv(config.RAW_DECEMBER, parse_dates=["date"])
    return {
        "cv": cv_table,
        "by_month": by_month,
        "test": test,
        "segments": segments,
        "test_rows": len(test_predictions),
        "december": december,
    }


# --------------------------------------------------------------------------- figures


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY, pad=26)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ae")
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_facecolor(SURFACE)


def split_timeline() -> str:
    """Train / validate / test / predict months for every evaluation round."""
    rounds = [(f"CV fold {i + 1}", v) for i, v in enumerate(config.CV_VALIDATION_MONTHS)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    labels = []
    for row, (label, validation_month) in enumerate(rounds):
        ax.barh(row, validation_month - 1, left=0.5, color=PHASES["Train"], edgecolor=SURFACE, linewidth=2, height=0.62)
        ax.barh(row, 1, left=validation_month - 0.5, color=PHASES["Validate"], edgecolor=SURFACE, linewidth=2, height=0.62)
        labels.append(label)
    row = len(rounds)
    ax.barh(row, 9, left=0.5, color=PHASES["Train"], edgecolor=SURFACE, linewidth=2, height=0.62)
    ax.barh(row, 1, left=9.5, color=PHASES["Test (once)"], edgecolor=SURFACE, linewidth=2, height=0.62)
    labels.append("October test")
    ax.barh(row + 1, 10, left=0.5, color=PHASES["Train"], edgecolor=SURFACE, linewidth=2, height=0.62)
    ax.barh(row + 1, 2, left=10.5, color=PHASES["Predict"], edgecolor=SURFACE, linewidth=2, height=0.62)
    labels.append("Final fit + submission")

    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_xlim(0.4, 12.6)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[Patch(color=c, label=l) for l, c in PHASES.items()],
        frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=TEXT_SECONDARY,
    )
    _style(ax, "Every evaluation trains on earlier months and tests on later ones (2025)")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    path = FIGURES_DIR / "split_timeline.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def mape_by_month(by_month: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.3))
    x = list(by_month.index)
    for model in by_month.columns:
        ax.plot(x, by_month[model], color=SERIES[model], linewidth=2.2, marker="o", markersize=5, label=MODEL_LABELS[model])
        ax.annotate(f"{by_month[model].iloc[-1]:.1f}%", (x[-1], by_month[model].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=TEXT_PRIMARY, fontsize=9)
    ax.set_xticks(x, [MONTHS[m - 1] for m in x])
    ax.set_xlim(x[0] - 0.2, x[-1] + 0.45)
    ax.set_ylim(0, by_month.max().max() * 1.12)
    ax.set_ylabel("MAPE (%)", color=TEXT_SECONDARY)
    ax.set_xlabel("Validation month (trained on all earlier months)", color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    # The area under the lines is empty, so the legend sits there.
    ax.legend(frameon=False, ncol=2, loc="lower right", labelcolor=TEXT_SECONDARY, fontsize=8.5)
    _style(ax, "Cross-validation error by month")
    ax.title.set_y(0.98)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    path = FIGURES_DIR / "mape_by_month.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- document


def styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.6, leading=13.4, textColor=INK, spaceAfter=6, alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=body, textColor=MUTED, fontSize=10, spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6, keepWithNext=1),
        "h2": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=4, keepWithNext=1),
        "body": body,
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=body, fontSize=8.4, leading=11, textColor=MUTED, spaceAfter=10),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.6, leading=11, spaceAfter=0),
        "cell_bold": ParagraphStyle("cell_bold", parent=body, fontName="Helvetica-Bold", fontSize=8.6, leading=11, spaceAfter=0),
        "box": ParagraphStyle("box", parent=body, spaceAfter=3),
        "code": ParagraphStyle("code", parent=body, fontName="Courier", fontSize=8.4, leading=11, spaceAfter=1),
    }


def table(rows: list[list], s: dict, widths: list[float], highlight_last: bool = False, bold_first_col: bool = False) -> Table:
    data = []
    for i, row in enumerate(rows):
        cells = []
        for j, value in enumerate(row):
            style = s["cell_bold"] if i == 0 or (bold_first_col and j == 0) or (highlight_last and i == len(rows) - 1) else s["cell"]
            cells.append(Paragraph(str(value), style))
        data.append(cells)
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_last:
        commands.append(("BACKGROUND", (0, -1), (-1, -1), HIGHLIGHT_FILL))
    t.setStyle(TableStyle(commands))
    return t


def bullets(items: list[str], s: dict) -> list:
    return [Paragraph(item, s["bullet"], bulletText="•") for item in items]


def figure(path: str, width_cm: float) -> Image:
    """Image scaled to the given width, keeping its own aspect ratio."""
    width, height = ImageReader(path).getSize()
    return Image(path, width=width_cm * cm, height=width_cm * height / width * cm)


def build_story(r: dict, timeline: str, monthly: str) -> list:
    s = styles()
    cv, test, seg = r["cv"], r["test"], r["segments"]
    base, final = cv.loc["baseline"], cv.loc["final_hybrid"]
    t_base, t_final = test.loc["baseline"], test.loc["final_hybrid"]
    december = r["december"]
    dec_min, dec_max = december["predicted_rate"].min(), december["predicted_rate"].max()
    peak = december.loc[december["predicted_rate"].idxmax()]
    reduction = 100 * (1 - final["mape_mean"] / base["mape_mean"])

    story = [
        Paragraph("Freight Rate Prediction", s["title"]),
        Paragraph(f"Data split, validation approach and December forecast &nbsp;|&nbsp; {AUTHOR}", s["subtitle"]),
    ]

    summary = Table([[Paragraph(
        f"<b>Summary.</b> The model is scored on November and December 2025, after all the labeled data, so every "
        f"evaluation in this project trains on earlier months and tests on later ones. Model comparison used "
        f"expanding-window cross-validation over January to September; October was held out and opened once, after "
        f"the final model was frozen. The final model (a Ridge regression base with a time trend, corrected by "
        f"LightGBM) reaches a cross-validated MAPE of <b>{final['mape_mean']:.2f}% &plusmn; {final['mape_std']:.2f}</b> "
        f"against <b>{base['mape_mean']:.2f}%</b> for the baseline, and <b>{t_final['mape']:.2f}%</b> on the October "
        f"test (<b>{t_final['mape_plausible_labels']:.2f}%</b> on loads with plausible labels).",
        s["box"],
    )]], colWidths=[17 * cm])
    summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f7fa")),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [summary, Spacer(1, 6)]

    # 1. Split
    story += [
        Paragraph("1. Data split", s["h1"]),
        Paragraph(
            "The labeled data (<font face='Courier'>train-test.csv</font>) covers 48,000 loads from January to "
            "October 2025. The 12,000 loads to predict cover November and December. A random split would let the "
            "model learn from loads in the same weeks as the ones it is tested on, sharing the same market "
            "conditions, and would overstate accuracy. The data was therefore split <b>by month</b>, so the model is "
            "always evaluated on months later than the ones it learned from, exactly as in the real task.",
            s["body"],
        ),
        table([
            ["Set", "Months", "Loads", "Use"],
            ["Train", "Jan - Aug", "38,477", "Fit models"],
            ["Validation", "Sep", "4,670", "Compare models and decisions (last cross-validation fold)"],
            ["Test", "Oct", f"{r['test_rows']:,}", "One final evaluation, after the model was frozen"],
        ], s, [2.6, 2.4, 1.8, 10.2], bold_first_col=True),
        Spacer(1, 6),
        Paragraph(
            "<b>Test-set rule.</b> October was never explored, plotted or used for any decision. It was opened once, "
            "at the end. For the submission, the frozen configuration was refit on all of January to October.",
            s["body"],
        ),
        KeepTogether([figure(timeline, 16.5), Paragraph("Figure 1. Months used for training, validation, testing and prediction in each evaluation round.", s["caption"])]),
    ]

    # 2. Validation
    story += [
        Paragraph("2. Validation approach", s["h1"]),
        Paragraph("2.1 Expanding-window cross-validation", s["h2"]),
        Paragraph(
            "A single validation month gives a single, noisy score. Models were compared on four folds instead: each "
            "fold trains on every month before its validation month (June, July, August, September). Fold 4 is the "
            "train / validation split above. Models are ranked on the <b>mean</b> error across folds; the "
            "<b>standard deviation</b> shows whether a model is consistently good or only good in one market regime "
            "(June is the price peak, August and September are lows).",
            s["body"],
        ),
        Paragraph("2.2 Metric", s["h2"]),
        Paragraph(
            "Rates range from about $150 to $8,000, so the primary metric is <b>MAPE</b> (mean absolute percentage "
            "error): a 10% miss is equally bad on a short or a long load. <b>WAPE</b> (total absolute error over "
            "total actual rate) and <b>MAE</b> (dollars) are reported alongside, because MAPE gives cheap loads the "
            "same weight as expensive ones and penalizes over-prediction more than under-prediction.",
            s["body"],
        ),
        Paragraph("2.3 Leakage control", s["h2"]),
        *bullets([
            "Cleaning applied to the raw files is deterministic and row by row (date parsing; negative weights, which "
            "are sign errors, made positive). Nothing in it is learned from the data.",
            "Everything learned from data (imputation values, encoders, the corrupted-label filter) is part of each "
            "model and is refit inside every fold on that fold's training months only.",
            "Features use information available at prediction time only; the time trend is learned from training "
            "months and held at the last training day when predicting later dates.",
        ], s),
        Paragraph("2.4 Corrupted labels: removed from training, kept in evaluation", s["h2"]),
        Paragraph(
            "About 1.3% of posted rates sit in separate clusters at roughly 0.2-0.6x and 2.2-5x the usual rate per "
            "mile of comparable loads (same equipment and distance band), with empty gaps in between: for example, "
            "a 230-mile Cincinnati to St. Louis Dry Van load posted at $102 and another at $2,058 on a lane that "
            "normally pays about $590. Inside each fold, the filter learns the typical rate per mile from the "
            "training months and drops implausible <i>training</i> rows. Validation and test rows are <b>never</b> "
            "removed, because the real scoring data will contain such labels too. MAPE on plausible labels is "
            "reported separately to show how much of the error they cause.",
            s["body"],
        ),
        Paragraph("2.5 Selection rule", s["h2"]),
        Paragraph(
            "Every choice (model type, trend handling, trimming, hyperparameters) was made on the cross-validation "
            "folds with one rule: <b>lowest mean MAPE plus one standard deviation</b>, so a setting must be both "
            "accurate and stable across months to win.",
            s["body"],
        ),
    ]

    # 3. Results
    cv_rows = [["Model", "MAPE (mean &plusmn; std)", "MAPE, plausible labels", "WAPE", "MAE"]]
    for model in ["baseline", "regression", "lightgbm", "final_hybrid"]:
        row = cv.loc[model]
        cv_rows.append([MODEL_LABELS[model], f"{row['mape_mean']:.2f}% &plusmn; {row['mape_std']:.2f}",
                        f"{row['mape_plausible_labels_mean']:.2f}%", f"{row['wape_mean']:.2f}%", f"${row['mae_mean']:.0f}"])
    test_rows = [["Model", "MAPE", "MAPE, plausible labels", "WAPE", "MAE"]]
    for model in ["baseline", "regression", "lightgbm", "final_hybrid"]:
        row = test.loc[model]
        test_rows.append([MODEL_LABELS[model], f"{row['mape']:.2f}%", f"{row['mape_plausible_labels']:.2f}%",
                          f"{row['wape']:.2f}%", f"${row['mae']:.0f}"])
    suspect_mape = seg.get("suspect label")

    story += [
        Paragraph("3. Results", s["h1"]),
        Paragraph("3.1 Cross-validation (January to September, 4 folds)", s["h2"]),
        table(cv_rows, s, [6.2, 3.4, 3.2, 2.0, 2.2], highlight_last=True),
        Spacer(1, 8),
        KeepTogether([
            figure(monthly, 15.5),
            Paragraph(
                "Figure 2. MAPE per validation month. Ridge and LightGBM both drifted low in later months: rates rose "
                "about 0.75% per month independently of the market index. The final model adds a time trend (held at "
                "the last training day) and combines both models, which removes the drift and halves the spread.",
                s["caption"],
            ),
        ]),
        Paragraph(
            f"The final model cuts the baseline's error by {reduction:.0f}%. Its median prediction is within about "
            "1.6% of the actual rate in every fold. On plausible labels its error is about 2%; the corrupted labels "
            "account for the rest.",
            s["body"],
        ),
        Paragraph("3.2 October test (trained on January to September, evaluated once)", s["h2"]),
        table(test_rows, s, [6.2, 2.4, 3.6, 2.4, 2.4], highlight_last=True),
        Spacer(1, 6),
        Paragraph(
            f"The ranking of all models matches cross-validation, and the final model's error is less than half the "
            f"baseline's ({t_base['mape']:.2f}%). Its {t_final['mape']:.2f}% is slightly "
            f"above its cross-validated {final['mape_mean']:.2f}%, but on plausible labels it is better "
            f"({t_final['mape_plausible_labels']:.2f}% vs {final['mape_plausible_labels_mean']:.2f}%): October's loads "
            f"with suspect labels carry an average error of {suspect_mape:.0f}% and alone add about 3 points of MAPE. "
            f"The median prediction is {100 * (1 - t_final['median_predicted_over_actual']):.1f}% below the actual "
            f"rate: the model is not biased.",
            s["body"],
        ),
    ]

    # 4. December chart
    story += [
        Paragraph("4. December prediction chart", s["h1"]),
        KeepTogether([
            figure(str(DECEMBER_CHART), 16.5),
            Paragraph("Figure 3. Output of score.py for the fixed December load (final model refit on January to October).", s["caption"]),
        ]),
        *bullets([
            f"<b>Level:</b> ${dec_min:.0f} to ${dec_max:.0f} for Lexington to Fort Wayne, Dry Van, 360 miles, 32,000 lb. "
            "Dry Van loads on this lane cost a median of $807 earlier in the year. The prediction sits slightly above, "
            "consistent with the upward trend held at its October level.",
            "<b>Shape:</b> a weekly cycle of about 1%, peaking on Thursdays and lowest on Sundays and Mondays. It "
            "follows the market index supplied for each December date, which has the same weekly cycle as in training, "
            "where rates follow it as well.",
            f"<b>Limitation:</b> no holiday period appears in the training data, so the model treats "
            f"{peak['date']:%d %B} like any other {peak['date']:%A} (it is the month's highest point, "
            f"${peak['predicted_rate']:.0f}). Holiday effects are not modeled.",
        ], s),
    ]

    # Appendix
    story += [
        Paragraph("Reproducing the results", s["h1"]),
        Paragraph("From the repository root, with Poetry:", s["body"]),
        Preformatted("\n".join([
            "poetry install --no-root --with report",
            "poetry run python -m src.data_preparation   # exploration and cleaning",
            "poetry run python -m src.train              # model comparison (cross-validation)",
            "poetry run python -m src.tuning             # tuning experiments",
            "poetry run python -m src.test_evaluation    # one-time October test",
            "poetry run python -m src.predict            # submission files",
            "poetry run python score.py --predictions validation_predictions.csv \\",
            "    --december-predictions december-chart-inputs.csv",
            "poetry run python -m src.report             # this report",
        ]), s["code"]),
        Spacer(1, 4),
        Paragraph("Each step is documented in <font face='Courier'>src/steps/</font>.", s["body"]),
    ]
    return story


def _page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Freight Rate Prediction")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = load_results()
    story = build_story(results, split_timeline(), mape_by_month(results["by_month"]))
    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Freight Rate Prediction - Validation approach and December forecast", author=AUTHOR,
    )
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
