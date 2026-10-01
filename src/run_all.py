"""End-to-end: rolling CV -> clean hold-out -> final fit -> predictions -> December chart inputs.
Usage: python -m src.run_all
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from src.features import FeatureBuilder, load
from src.models import build_X, fit_predict, metrics

warnings.filterwarnings("ignore")
D, OUT, REP = Path("data"), Path("outputs"), Path("reports")
for p in (OUT, REP): p.mkdir(exist_ok=True)
df = load(D / "train_test.csv"); df["rpm"] = df.posted_rate / df.distance; df["lrpm"] = np.log(df.rpm)
FOLDS = [("2025-05-01", "2025-07-01"), ("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]  # rolling-origin (expanding window)
HOLD = ("2025-09-01", "2025-11-01")                                         # untouched until variant is chosen


def evaluate(start, end, use_day):
    tr, va = df[df.date < start], df[(df.date >= start) & (df.date < end)]
    fb = FeatureBuilder().fit(tr)
    P = fit_predict(build_X(fb, tr, use_day), tr.lrpm, build_X(fb, va, use_day))
    return {k: metrics(np.exp(v) * va.distance.values, va.posted_rate.values) for k, v in P.items()}, va, P

# ---- 1. choose feature variant by rolling CV only -------------------------------------------
cv = {}
for use_day in (False, True):
    rs = [evaluate(s, e, use_day)[0]["blend"] for s, e in FOLDS]
    cv[use_day] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    print(f"[CV] day-level features={use_day!s:5}: " + " | ".join(f"{k} {v:.2f}" for k, v in cv[use_day].items()))
USE_DAY = cv[True]["MAE"] < cv[False]["MAE"]
print("-> chosen by CV: use_day =", USE_DAY)

# ---- 2. clean hold-out (Sep-Oct), models trained on Jan-Aug -----------------------------------
res, ho, P = evaluate(*HOLD, USE_DAY)
print("\n[HOLD-OUT Sep-Oct]")
for k, m in res.items(): print(f"  {k:13s}", {a: round(b, 2) for a, b in m.items()})
base_pred = None
pred = np.exp(P["blend"]) * ho.distance.values
bd = ho.assign(pred=pred, ae=np.abs(pred - ho.posted_rate), ape=np.abs(pred - ho.posted_rate) / ho.posted_rate * 100)
bd["distance_band"] = pd.cut(bd.distance, [0, 250, 500, 1000, 1500, 2000, 4000]).astype(str); bd["month"] = bd.date.dt.month
tables = {k: bd.groupby(k).agg(n=("ae", "size"), MAE=("ae", "mean"), MAPE=("ape", "mean"), medAPE=("ape", "median")).round(2)
          for k in ["equipment", "distance_band", "month"]}
for k, t in tables.items(): print(f"\n  by {k}:\n{t.to_string()}")
clean = ho.rpm < 5
print("\n  last-window (Sep-Oct) excluding the 0.7% price spikes (rpm>5):", {a: round(b, 2) for a, b in metrics(pred[clean], ho.posted_rate.values[clean]).items()})
pd.concat(tables, names=["slice"]).to_csv(REP / "holdout_breakdown.csv")
json.dump({"cv_variant_scores": {str(k): v for k, v in cv.items()}, "use_day_features": bool(USE_DAY), "holdout": res}, open(REP / "metrics.json", "w"), indent=2)

# ---- 3. final fit on ALL labeled data, predict validation.csv -------------------------------
val = load(D / "validation.csv"); fb = FeatureBuilder().fit(df)
Pv = fit_predict(build_X(fb, df, USE_DAY), df.lrpm, build_X(fb, val, USE_DAY))
val["predicted_rate"] = np.exp(Pv["blend"]) * val.distance.values
tmpl = pd.read_csv(D / "validation_predictions_template.csv")[["load_id"]]
sub = tmpl.merge(val[["load_id", "predicted_rate"]], on="load_id", how="left").round({"predicted_rate": 2})
assert len(sub) == 12000 and sub.predicted_rate.notna().all() and (sub.predicted_rate > 0).all()
sub.to_csv("validation_predictions.csv", index=False)
print("\nvalidation_predictions.csv:", sub.predicted_rate.describe().round(1).to_dict())

# ---- 4. December fixed-input chart ----------------------------------------------------------
dec = pd.read_csv(D / "december_chart_inputs.csv"); dec["date"] = pd.to_datetime(dec["date"])
allr = pd.concat([df, val]); coords = {}
for side in ("pickup", "delivery"):
    for c, la, lo in allr[[side, f"{side}_lat", f"{side}_lon"]].drop_duplicates().itertuples(index=False): coords[c] = (la, lo)
raw = dec.drop(columns="predicted_rate").copy()
raw["pickup_lat"], raw["pickup_lon"] = zip(*raw.pickup.map(coords)); raw["delivery_lat"], raw["delivery_lon"] = zip(*raw.delivery.map(coords))
# market_index / quote_signal are not in the December file: use the daily mean of that date's loads in validation.csv (features only)
daily = val.groupby("date")[["market_index", "quote_signal"]].mean()
raw["market_index"] = raw.date.map(daily.market_index).values; raw["quote_signal"] = raw.date.map(daily.quote_signal).values
assert raw[["market_index", "quote_signal"]].notna().all().all()
Pd = fit_predict(build_X(fb, df, USE_DAY), df.lrpm, build_X(fb, raw, USE_DAY))
dec["predicted_rate"] = (np.exp(Pd["blend"]) * 360).round(2)
dec["date"] = dec["date"].dt.strftime("%Y-%m-%d"); dec.to_csv(OUT / "december_chart_inputs.csv", index=False)
hist = df[(df.pickup == "Lexington") & (df.delivery == "Fort Wayne")]
print("\nDecember predicted rate: min %.0f | mean %.0f | max %.0f ; historical Lexington->Fort Wayne (n=%d) median rate %.0f"
      % (dec.predicted_rate.min(), dec.predicted_rate.mean(), dec.predicted_rate.max(), len(hist), hist.posted_rate.median()))
