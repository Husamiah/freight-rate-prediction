"""Temporal-drift diagnosis: do models under-predict later windows, and which fix works?
Compares: base LightGBM, recency weighting, short training windows, LightGBM + spline/Ridge with linear trend.
Usage: python -m experiments.cv_round5_drift
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.features import FeatureBuilder, load
from src.models import build_X, _lgbm, _spline_ridge

df = load("data/train_test.csv"); df["rpm"] = df.posted_rate / df.distance; df["lrpm"] = np.log(df.rpm)
WINDOWS = {"F1": ("2025-05-01", "2025-07-01"), "F2": ("2025-07-01", "2025-09-01"), "F3": ("2025-09-01", "2025-11-01")}

def run(tau=None, trend_ridge=False, last_only=None):
    out = {}
    for n, (s, e) in WINDOWS.items():
        tr, va = df[df.date < s], df[(df.date >= s) & (df.date < e)]
        if last_only: tr = tr[tr.date >= pd.Timestamp(s) - pd.Timedelta(days=last_only)]
        fb = FeatureBuilder().fit(tr); Xtr, Xva = build_X(fb, tr, True), build_X(fb, va, True)
        w = np.exp(-(pd.Timestamp(s) - tr.date).dt.days.values / tau) if tau else None
        p = _lgbm().fit(Xtr.drop(columns="t"), tr.lrpm, sample_weight=w).predict(Xva.drop(columns="t"))
        if trend_ridge: p = (p + _spline_ridge(Xtr, tr.lrpm, Xva)) / 2
        pr = np.exp(p) * va.distance.values; y = va.posted_rate.values
        out[n] = (round(float(np.abs(pr - y).mean()), 1), round(float((p - va.lrpm.values)[va.rpm.values < 5].mean()) * 100, 2))
    return out   # window -> (MAE, mean log error % ; negative = under-prediction)

if __name__ == "__main__":
    print("(MAE $, bias %) per window F1 May-Jun | F2 Jul-Aug | F3 Sep-Oct")
    print("base LightGBM            ", run())
    for tau in (60, 120, 240): print(f"recency weights tau={tau:<4}", run(tau=tau))
    for lo in (120, 180): print(f"last {lo} days only       ", run(last_only=lo))
    print("LightGBM + Ridge w/ trend", run(trend_ridge=True))
