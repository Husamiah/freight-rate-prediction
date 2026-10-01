"""Builds reports/Freight_Rate_Report.pdf from reports/*.json and reports/figures/*. Usage: python -m src.make_report"""
import json
import numpy as np, pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether

R = json.load(open("reports/results_summary.json")); M = json.load(open("reports/metrics.json"))
W = R["windows"]; names = list(W)
TEAL = colors.HexColor("#064A56")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=13, textColor=TEAL, spaceBefore=10, spaceAfter=4)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=19, textColor=TEAL, alignment=0, spaceAfter=2)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.2, leading=12.4, spaceAfter=4)
SM = ParagraphStyle("S", parent=B, fontSize=7.8, leading=10, textColor=colors.HexColor("#455A60"))
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)
CELL = ParagraphStyle("C", parent=B, fontSize=8, leading=10, spaceAfter=0)

def tbl(data, widths, header=True, zebra=True):
    data = [[Paragraph(str(c), CELL) if isinstance(c, str) else c for c in r] for r in data]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#C9D5D8")),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header: st += [("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    if zebra: st += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F6F7")])]
    t.setStyle(TableStyle(st)); return t
def hdr(*xs): return [Paragraph(f'<font color="white"><b>{x}</b></font>', CELL) for x in xs]
def bullets(items): return [Paragraph(i, BUL, bulletText="-") for i in items]
def img(path, w): 
    from PIL import Image as PI; a, b = PI.open(path).size; return Image(path, width=w, height=w * b / a)

mean = lambda k, m: np.mean([W[n][k][m] for n in names])
cv = M["cv_variant_scores"]["True"]
S = []
S += [Paragraph("Freight Rate Prediction - Methodology & Results", TITLE),
      Paragraph("Musab Alhusamiah", ParagraphStyle("A", parent=B, fontSize=11, textColor=TEAL, spaceAfter=2)),
      Paragraph("Machine Learning Engineer assessment | Target: posted_rate | 48,000 labeled loads (Jan-Oct 2025), 12,000 loads to predict (Nov-Dec 2025)", SM), Spacer(1, 6)]

S += [Paragraph("1. Summary", H1),
      Paragraph(f"The final model predicts <b>log(rate per mile)</b> and multiplies back by distance. It is an equal-weight blend of a shallow LightGBM (Huber loss) and a spline + Ridge model that carries a linear time-trend term. "
                f"In expanding-window validation over three future windows (May-Jun, Jul-Aug, Sep-Oct) it reaches a mean <b>MAE of ${mean('blend','MAE'):.0f}</b> (MAPE {mean('blend','MAPE_%'):.1f}%, median APE {mean('blend','medAPE_%'):.1f}%), "
                f"versus ${mean('baseline','MAE'):.0f} MAE (MAPE {mean('baseline','MAPE_%'):.1f}%) for a strong rule-based baseline (median rate/mile by equipment x distance band), i.e. about {(1-mean('blend','MAE')/mean('baseline','MAE'))*100:.0f}% lower MAE. "
                "The most important finding was a <b>temporal drift</b>: rates rise over time in a way the features do not explain, and tree models under-predict later months. A linear trend in the linear component fixes most of it.", B)]

S += [Paragraph("2. Data exploration & data-quality issues", H1), img("reports/figures/eda.png", 6.9 * inch),
      Paragraph("Rate per mile (RPM) is far more regular than the raw rate, so modelling log(RPM) removes the dominant distance effect. Short hauls cost more per mile (about $2.75 under 250 mi vs about $1.91 over 2,000 mi); Reefer &gt; Flatbed &gt; Dry Van. "
                "RPM follows a seasonal pattern that mirrors the monthly mean of market_index, so market_index carries the seasonality and we avoid raw month/day-of-year features that trees cannot extrapolate. "
                "quote_signal has a non-linear (U-shaped) relation to RPM, which motivates tree models and interaction features.", B), Spacer(1, 3)]
dq = [hdr("Issue found", "Size", "Treatment"),
      ["Negative weights", "292 train / 145 validation", "Sign errors (abs. values look like the normal weight distribution) -> take absolute value, add flag"],
      ["Weight capped at 47,500", "1,191 train / 301 validation", "Censored value -> flag weight_capped"],
      ["Missing weight", "300 train / 165 validation", "Median of the equipment type (fit on train only) + missing flag"],
      ["Missing market_index", "374 train / 249 validation", "Left as NaN for LightGBM; median fill + flag in the linear model"],
      ["Price spikes (RPM &gt; $5, up to $14)", "about 0.7% of loads, uniform over months/equipment", "Noise, not seasonality. Kept (cannot be identified in validation); handled with a robust Huber loss on log scale. Metrics also reported excluding them"],
      ["Distance vs. coordinates", "Haversine ratio median 1.18; a few extreme values", "Kept; distance_ratio feature added"],
      ["Unseen lanes in validation", "1,461 loads (12%)", "Lane/city target encoding tested and dropped (no gain); model uses geography/distance instead"]]
S += [tbl(dq, [1.55 * inch, 1.65 * inch, 3.8 * inch])]

S += [KeepTogether([Paragraph("3. Train / validation split and leakage control", H1), img("reports/figures/split_timeline.png", 6.6 * inch),
      Paragraph("The test period (Nov-Dec) lies strictly after the labeled data, so a random split would be optimistic. I used <b>expanding-window, rolling-origin validation</b> with three folds (validate on May-Jun, Jul-Aug, Sep-Oct; always train on everything earlier) and scored on the dollar scale (MAE, MAPE, median APE; RMSE is dominated by the 0.7% unpredictable spikes and is reported only for completeness).", B)])]
S += bullets(["All imputation values and city codes are fitted on the training fold only (<font face='Courier'>FeatureBuilder.fit</font>); the same object transforms validation and December rows.",
              "No labels are used from validation.csv. The 'same-day market mean' features (mean market_index / quote_signal of all loads on a date) use features only, within each file, and were chosen by CV.",
              "Calendar/holiday features were tested and <b>removed</b>: removing them improved CV MAE from about 121 to 112 and cannot be verified for Nov-Dec.",
              "Seeds are fixed (42); one command (<font face='Courier'>python -m src.run_all</font>) reproduces every output."])
S += [Paragraph("<b>Honest note on the last window.</b> I first planned Sep-Oct as an untouched hold-out. Its result exposed the drift problem below, so I used it for diagnosis and for choosing the fix. Sep-Oct figures are therefore the last fold of the rolling validation, not a fully independent test. Final model selection used the mean over all three folds.", B)]

S += [Paragraph("4. Models, features and results", H1),
      Paragraph("Features (about 35): log distance, haversine distance, distance ratio, bearing, coordinates, equipment, cleaned weight and flags, market_index, quote_signal, interactions (quote_signal deviation, quote_signal x log distance, weight x equipment) and same-day market means. "
                "Target: log(RPM). LightGBM: Huber (alpha 0.1), 7 leaves, min 150 samples per leaf; the grid showed that simpler trees are marginally better (108.9 vs 111.9 MAE), meaning most remaining variance is noise.", B)]
rows = [hdr("Model", *[f"MAE {n}" for n in names], "Mean MAE", "Mean MAPE", "Mean medAPE")]
for k, lab in [("baseline", "Baseline (median $/mile)"), ("lgbm", "LightGBM (log RPM, Huber)"), ("spline_ridge", "Spline + Ridge (with trend)"), ("blend", "<b>Final blend</b>")]:
    rows.append([lab] + [f"{W[n][k]['MAE']:.1f}" for n in names] + [f"{mean(k,'MAE'):.1f}", f"{mean(k,'MAPE_%'):.2f}%", f"{mean(k,'medAPE_%'):.2f}%"])
S += [tbl(rows, [1.85 * inch, .75 * inch, .75 * inch, .75 * inch, .8 * inch, .85 * inch, .85 * inch]), Spacer(1, 3),
      Paragraph("RMSE stays near 630 for every model because of the unpredictable spikes. Excluding loads with RPM &gt; $5, the final blend on Sep-Oct has MAE $%.0f and MAPE %.1f%%." % (W["Sep-Oct"]["spikes_excluded_blend"]["MAE"], W["Sep-Oct"]["spikes_excluded_blend"]["MAPE_%"]), SM)]

S += [KeepTogether([Paragraph("4.1 Temporal drift and the fix", H1), img("reports/figures/models_drift.png", 6.9 * inch),
      Paragraph("All tree-based models under-predict more and more in later windows (right panel): rates drift upward beyond what market_index and the other features capture, and trees cannot extrapolate. "
                "A spline + Ridge model without a trend drifts even more (-3.5% in Sep-Oct). Adding a <b>linear time term only to the Ridge component</b> and blending it with LightGBM brings the bias to about -0.5%. "
                "Trade-off: the trend slightly over-predicts early windows (+1.2% in May-Jun), and a linear trend is an assumption when extrapolating to December. Recency weighting and short training windows were also tested but were unstable across folds.", B)])]

hb = pd.read_csv("reports/holdout_breakdown.csv")
S += [Paragraph("4.2 Error breakdown (Sep-Oct window, final blend)", H1)]
rows = [hdr("Slice", "Group", "n", "MAE ($)", "MAPE %", "medAPE %")]
ORDER = ["(0, 250]", "(250, 500]", "(500, 1000]", "(1000, 1500]", "(1500, 2000]", "(2000, 4000]", "Dry Van", "Flatbed", "Reefer"]
hb = hb.assign(o=hb.iloc[:, 1].astype(str).map({k: i for i, k in enumerate(ORDER)})).sort_values(["slice", "o"], ascending=[True, True]).sort_values("slice", key=lambda c: c.map({"equipment": 0, "distance_band": 1}), kind="stable")
for _, r in hb.iterrows():
    if r["slice"] in ("equipment", "distance_band"):
        rows.append([r["slice"].replace("_", " "), str(r.iloc[1]).replace("(", "").replace("]", "").replace(", ", "-"), f"{int(r['n']):,}", f"{r['MAE']:.1f}", f"{r['MAPE']:.2f}", f"{r['medAPE']:.2f}"])
S += [tbl(rows, [1.2 * inch, 1.5 * inch, .7 * inch, .9 * inch, .9 * inch, .9 * inch]),
      Paragraph("Percentage errors are consistent across equipment and distance bands (about 4-5% MAPE); absolute dollar errors scale with distance, as expected.", SM)]

S += [KeepTogether([Paragraph("5. Fixed December 2025 prediction (Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb)", H1), img("reports/figures/candidate_december.png", 6.4 * inch),
      Paragraph("This is the exact chart produced by the provided score.py. The December input file has no market_index or quote_signal, so for each date I used the <b>average of those two features over that date's loads in validation.csv</b> (features only, no labels). "
                "The day-to-day movement therefore comes from daily market conditions, not from weekday or holiday effects. Predictions range from $%.0f to $%.0f (mean $%.0f); the historical median for this lane is $849, based on only 32 loads, and December market_index is below the yearly average." % tuple(
                    [pd.read_csv("outputs/december_chart_inputs.csv").predicted_rate.min(), pd.read_csv("outputs/december_chart_inputs.csv").predicted_rate.max(), pd.read_csv("outputs/december_chart_inputs.csv").predicted_rate.mean()]), B)])]
S += [Paragraph("6. Limitations & next steps", H1)] + bullets([
    "No December (or Nov) labels exist; the level of the prediction relies on the linear trend and on market_index. Real holiday-season effects are not modelled.",
    "The curve is flat-ish (about +/-2%) because market_index has a modest effect on the rate; holiday demand shocks would not appear.",
    "The 0.7% price spikes cap achievable RMSE; if the production metric is RMSE-based, the ranking of models may differ from MAE/MAPE.",
    "Next: quantile/interval predictions, lane-level effects with more data, and a periodic re-fit to track the drift."])

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor("#6C7A7E")); c.drawRightString(letter[0] - .7 * inch, .45 * inch, f"Page {d.page}"); c.restoreState()
SimpleDocTemplate("reports/Freight_Rate_Report.pdf", pagesize=letter, leftMargin=.7 * inch, rightMargin=.7 * inch, topMargin=.65 * inch, bottomMargin=.7 * inch,
                  title="Freight Rate Prediction - Methodology & Results", author="Musab Alhusamiah").build(S, onFirstPage=footer, onLaterPages=footer)
print("built")
