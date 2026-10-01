"""Exploratory data analysis. Prints every number quoted in the report's EDA / data-quality sections.
Usage: python -m src.eda   (output saved to reports/eda_log.txt by `make eda`)
"""
import numpy as np, pandas as pd

t = pd.read_csv("data/train_test.csv", parse_dates=["date"]); v = pd.read_csv("data/validation.csv", parse_dates=["date"])
t["rpm"] = t.posted_rate / t.distance
section = lambda s: print(f"\n===== {s} =====")

section("1. Shape, ranges, missing values")
print("train", t.shape, "| validation", v.shape)
print("train dates", t.date.min().date(), "->", t.date.max().date(), "| validation dates", v.date.min().date(), "->", v.date.max().date())
print("missing train:", t.isna().sum()[lambda s: s > 0].to_dict(), "| missing validation:", v.isna().sum()[lambda s: s > 0].to_dict())
print("duplicate rows (excluding id):", t.duplicated(subset=[c for c in t.columns if c != "load_id"]).sum())

section("2. Target distribution")
print(t.posted_rate.describe().round(1).to_dict())
print("skew raw", round(t.posted_rate.skew(), 2), "| skew log", round(np.log1p(t.posted_rate).skew(), 2))
print("quantiles:", t.posted_rate.quantile([.5, .9, .99, .999]).round(0).to_dict())
print("corr(distance, rate) =", round(t.distance.corr(t.posted_rate), 3))

section("3. Rate per mile by equipment, distance band, month, weekday")
print(t.groupby("equipment").rpm.describe().round(3))
print(t.groupby(pd.cut(t.distance, [0, 250, 500, 1000, 1500, 2000, 4000]), observed=True).rpm.median().round(3).to_dict())
m = t.date.dt.month
print(t.groupby(m).agg(n=("rpm", "size"), rpm=("rpm", "median"), rate=("posted_rate", "median"), market_index=("market_index", "mean"), quote_signal=("quote_signal", "mean")).round(3))
print("weekday median rpm:", t.groupby(t.date.dt.dayofweek).rpm.median().round(3).to_dict())
print("validation monthly feature means:\n", v.groupby(v.date.dt.month).agg(n=("distance", "size"), market_index=("market_index", "mean"), quote_signal=("quote_signal", "mean"), distance=("distance", "mean")).round(3))

section("4. Price spikes")
print("share rpm>4/5/8:", {k: round((t.rpm > k).mean(), 4) for k in [4, 5, 8]})
hi = t[t.rpm > 5]
print("spikes: mean distance", round(hi.distance.mean()), "| share <500mi", round((hi.distance < 500).mean(), 2), "| by equipment", hi.equipment.value_counts(normalize=True).round(2).to_dict())
print("spikes by month:", hi.date.dt.month.value_counts().sort_index().to_dict())
print(t.sort_values("rpm", ascending=False).head(5)[["pickup", "delivery", "distance", "equipment", "weight", "date", "posted_rate", "rpm"]].to_string())

section("5. Signals: market_index, quote_signal, weight (rpm < 5)")
c = t[t.rpm < 5]
print("corr with rpm:", c[["rpm", "market_index", "quote_signal", "weight"]].corr().round(3)["rpm"].to_dict())
print("rpm by market_index quintile:", c.groupby(pd.qcut(c.market_index, 5), observed=True).rpm.median().round(3).to_dict())
print("rpm by quote_signal quintile (U-shape):", c.groupby(pd.qcut(c.quote_signal, 5), observed=True).rpm.median().round(3).to_dict())
print("rpm by |weight| band:", t.groupby(pd.cut(t.weight.abs(), [0, 1e4, 2e4, 3e4, 4e4, 48e3]), observed=True).rpm.median().round(3).to_dict())
print("market_index within-day sd:", round(t.groupby("date").market_index.std().mean(), 3), "| across-day sd of daily mean:", round(t.groupby("date").market_index.mean().std(), 3))

section("6. Weight quality")
w = t.weight
print("negative", (w < 0).sum(), "| zero", (w == 0).sum(), "| ==47,500", (w == 47500).sum(), "| ==-47,500", (w == -47500).sum(), "| missing", w.isna().sum())
print("abs(negative weights) summary:", w[w < 0].abs().describe().round(0).to_dict())
print("validation: negative", (v.weight < 0).sum(), "| |w|>=47,500", (v.weight.abs() >= 47500).sum(), "| missing", v.weight.isna().sum())

section("7. Geography, lanes, distance consistency")
def hav(a, b, c_, d):
    a, b, c_, d = map(np.radians, [a, b, c_, d]); x = np.sin((c_ - a) / 2) ** 2 + np.cos(a) * np.cos(c_) * np.sin((d - b) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))
ratio = t.distance / hav(t.pickup_lat, t.pickup_lon, t.delivery_lat, t.delivery_lon)
print("distance / haversine:", ratio.describe().round(3).to_dict())
print("one coordinate pair per city:", t.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique().max().to_dict(), "| pickup==delivery rows:", (t.pickup == t.delivery).sum())
lanes = t.set_index(["pickup", "delivery"]).index
print("distinct lanes in train:", lanes.nunique(), "| validation loads on unseen lanes:", int((~v.set_index(["pickup", "delivery"]).index.isin(lanes)).sum()))

section("8. December inputs (what the scorer's fixed shipment needs)")
d = v[v.date >= "2025-12-01"].groupby("date").agg(n=("distance", "size"), market_index=("market_index", "mean"), quote_signal=("quote_signal", "mean"), mi_sd=("market_index", "std"))
print(d.round(3).iloc[::3].to_string())
print("Lexington->Fort Wayne loads in train:", int(((t.pickup == "Lexington") & (t.delivery == "Fort Wayne")).sum()), "| median rate:", t[(t.pickup == "Lexington") & (t.delivery == "Fort Wayne")].posted_rate.median())
