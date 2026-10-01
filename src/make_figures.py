"""Figures + results summary for the report. Usage: python -m src.make_figures"""
import json, shutil, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.features import FeatureBuilder, load
from src.models import build_X, fit_predict, metrics

warnings.filterwarnings("ignore")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
TEAL, ORANGE, GREY, RED = "#064A56", "#E08A1E", "#9DAFB3", "#B5483A"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E3EAEC", "grid.linewidth": .7})
df = load("data/train_test.csv"); df["rpm"] = df.posted_rate / df.distance; df["lrpm"] = np.log(df.rpm); df["m"] = df.date.dt.month
WIN = {"May-Jun": ("2025-05-01", "2025-07-01"), "Jul-Aug": ("2025-07-01", "2025-09-01"), "Sep-Oct": ("2025-09-01", "2025-11-01")}

# ---------- 1. split timeline ----------
fig, ax = plt.subplots(figsize=(7.2, 2.3))
rows = [("Fold 1", 0, 4, 4, 6), ("Fold 2", 0, 6, 6, 8), ("Fold 3", 0, 8, 8, 10), ("Final model", 0, 10, 10, 12)]
for i, (n, a, b, c, d) in enumerate(rows[::-1]):
    ax.barh(i, b - a, left=a, color=TEAL, height=.55); ax.barh(i, d - c, left=c, color=(ORANGE if n != "Final model" else RED), height=.55)
    ax.text(a + .15, i, "train", color="white", va="center", fontsize=8); ax.text(c + .1, i, "validate" if n != "Final model" else "predict", color="white", va="center", fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xticks(range(0, 13, 1))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", ""]); ax.set_xlim(0, 12); ax.grid(axis="y", visible=False)
ax.set_title("Rolling-origin validation: train on the past, validate on the next 2 months", loc="left", fontsize=9, fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "split_timeline.png", dpi=200); plt.close(fig)

# ---------- 2. EDA ----------
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
band = pd.cut(df.distance, [0, 250, 500, 1000, 1500, 2000, 4000]); g = df.groupby(band, observed=True).rpm.median()
ax[0].bar(range(len(g)), g.values, color=TEAL); ax[0].set_xticks(range(len(g))); ax[0].set_xticklabels(["<250", "250-500", "500-1k", "1-1.5k", "1.5-2k", ">2k"], rotation=40, fontsize=7); ax[0].set_title("Rate/mile by distance (mi)", fontsize=8.5, loc="left"); ax[0].set_ylim(1.5, 3)
mm = df.groupby("m").agg(rpm=("rpm", "median"), mi=("market_index", "mean")); ax[1].plot(mm.index, mm.rpm, color=TEAL, marker="o", ms=3); ax[1].set_ylabel("median rate/mile", color=TEAL, fontsize=7.5)
a2 = ax[1].twinx(); a2.plot(mm.index, mm.mi, color=ORANGE, marker="s", ms=3); a2.set_ylabel("mean market_index", color=ORANGE, fontsize=7.5); a2.grid(False); a2.spines["right"].set_visible(True)
ax[1].set_title("Seasonality tracks market_index", fontsize=8.5, loc="left"); ax[1].set_xlabel("month 2025", fontsize=7.5)
ax[2].hist(np.log10(df.rpm.clip(upper=15)), bins=80, color=TEAL); ax[2].axvline(np.log10(5), color=RED, ls="--", lw=1); ax[2].text(np.log10(5) + .02, ax[2].get_ylim()[1] * .7, "0.7% of loads\nrate/mile > $5", color=RED, fontsize=7)
ax[2].set_title("Price spikes are noise", fontsize=8.5, loc="left"); ax[2].set_xlabel("log10 rate/mile", fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "eda.png", dpi=200); plt.close(fig)

# ---------- 3. per-window metrics: baseline vs models, and bias / drift ----------
def baseline_pred(tr, va):
    bins = [0, 250, 500, 1000, 1500, 2000, 5000]; t = tr.assign(b=pd.cut(tr.distance, bins)); med = t.groupby(["equipment", "b"], observed=True).rpm.median()
    v = va.assign(b=pd.cut(va.distance, bins)); return np.array([med[(e, b)] for e, b in zip(v.equipment, v.b)]) * va.distance.values

res, bias = {}, {}
for name, (s, e) in WIN.items():
    tr, va = df[df.date < s], df[(df.date >= s) & (df.date < e)]; fb = FeatureBuilder().fit(tr)
    Xtr, Xva = build_X(fb, tr, True), build_X(fb, va, True); P = fit_predict(Xtr, tr.lrpm, Xva)
    X0, V0 = Xtr.copy(), Xva.copy(); X0["t"] = 0.0; V0["t"] = 0.0
    from src.models import _spline_ridge; P["ridge_no_trend"] = _spline_ridge(X0, tr.lrpm, V0)
    y, d, ok = va.posted_rate.values, va.distance.values, (va.rpm.values < 5)
    res[name] = {"baseline": metrics(baseline_pred(tr, va), y), **{k: metrics(np.exp(P[k]) * d, y) for k in ["lgbm", "spline_ridge", "blend"]}}
    res[name]["spikes_excluded_blend"] = metrics((np.exp(P["blend"]) * d)[ok], y[ok])
    bias[name] = {k: float((P[k] - va.lrpm.values)[ok].mean() * 100) for k in ["lgbm", "ridge_no_trend", "blend"]}
json.dump({"windows": res, "bias_pct": bias}, open("reports/results_summary.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6)); w = list(WIN); x = np.arange(3)
for i, (k, lab, c) in enumerate([("baseline", "Baseline (median $/mile)", GREY), ("lgbm", "LightGBM", TEAL), ("spline_ridge", "Spline+Ridge (trend)", ORANGE), ("blend", "Final blend", RED)]):
    ax[0].bar(x + (i - 1.5) * .2, [res[n][k]["MAE"] for n in w], .2, label=lab, color=c)
ax[0].set_xticks(x); ax[0].set_xticklabels(w); ax[0].set_ylabel("MAE ($)"); ax[0].set_title("Error per validation window", fontsize=8.5, loc="left"); ax[0].legend(fontsize=6.5, frameon=False)
for k, lab, c, ls in [("lgbm", "LightGBM (no trend)", TEAL, "-"), ("ridge_no_trend", "Spline+Ridge without trend", GREY, "--"), ("blend", "Final blend (with trend term)", RED, "-")]:
    ax[1].plot(w, [bias[n][k] for n in w], marker="o", color=c, ls=ls, label=lab)
ax[1].axhline(0, color="black", lw=.8); ax[1].set_ylabel("mean log error, % (neg = under-predict)", fontsize=7); ax[1].set_title("Temporal drift: models under-predict later months", fontsize=8.5, loc="left"); ax[1].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(FIG / "models_drift.png", dpi=200); plt.close(fig)

# ---------- 4. December chart (the exact image produced by score.py) ----------
shutil.copy("scorer_results/candidate_december.png", FIG / "candidate_december.png")
print(json.dumps({"bias_pct": {n: {k: round(v, 2) for k, v in b.items()} for n, b in bias.items()}, "MAE": {n: {k: round(res[n][k]["MAE"], 1) for k in ["baseline", "lgbm", "spline_ridge", "blend"]} for n in w}}, indent=1))
