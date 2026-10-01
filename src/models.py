"""Feature assembly + the three-model blend used for the final submission.

Target is log(rate per mile); prediction = exp(blend) * distance.
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from src.features import FeatureBuilder

SEED = 42
CALENDAR = ["dow", "is_weekend", "dom", "days_to_month_end", "days_since_holiday", "days_to_holiday", "near_holiday"]


def build_X(fb: FeatureBuilder, raw: pd.DataFrame, use_day: bool) -> pd.DataFrame:
    """Cleaned features + interactions (+ optional same-day market means). Calendar features are dropped:
    in rolling CV they hurt (MAE 121 -> 112 when removed)."""
    X = fb.transform(raw).drop(columns=CALENDAR)
    X["qs_dev"] = (X.quote_signal - 2.06).abs()
    X["qs_x_logd"] = X.quote_signal * X.log_distance
    X["w_x_equip"] = X.weight * (X.equipment + 1)
    X["t"] = (pd.to_datetime(raw["date"]) - pd.Timestamp("2025-01-01")).dt.days.values / 100.0  # linear trend, ridge only
    if use_day:  # label-free: mean of the features of all loads on the same date, within the same file
        g = raw.groupby("date")
        X["mi_day"] = g["market_index"].transform("mean").values
        X["qs_day"] = g["quote_signal"].transform("mean").values
        X["mi_dev"] = X.market_index - X.mi_day
        X["qs_dev_day"] = X.quote_signal - X.qs_day
    return X


def _lgbm(linear_tree=False):
    return lgb.LGBMRegressor(objective="huber", alpha=0.1, learning_rate=0.03 if linear_tree else 0.05,
                             num_leaves=7, min_child_samples=150, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1,
                             n_estimators=600 if linear_tree else 1500, linear_tree=linear_tree)


def _spline_ridge(Xtr, ytr, Xte):
    med = Xtr["market_index"].median()
    def prep(X):
        X = X.drop(columns=["pickup_city", "delivery_city"]).copy()
        X["market_index"] = X["market_index"].fillna(med)
        if "mi_dev" in X: X["mi_dev"] = X["mi_dev"].fillna(0)
        for e in range(3):
            X[f"eq{e}"] = (X.equipment == e).astype(float); X[f"eq{e}_ld"] = X[f"eq{e}"] * X.log_distance
        return X
    A, B = prep(Xtr), prep(Xte)
    num = [c for c in ["log_distance", "distance", "haversine", "quote_signal", "market_index", "weight", "distance_ratio", "qs_dev"] if c in A]
    pipe = make_pipeline(ColumnTransformer([("s", SplineTransformer(n_knots=8, degree=3), num)], remainder="passthrough"),
                         StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100]))
    return pipe.fit(A, ytr).predict(B)


def fit_predict(Xtr, ytr, Xte) -> dict[str, np.ndarray]:
    """Return log-RPM predictions of each component and of the equal-weight blend."""
    Ttr, Tte = Xtr.drop(columns="t"), Xte.drop(columns="t")
    p = {"lgbm": _lgbm().fit(Ttr, ytr).predict(Tte),
         "lgbm_linear": _lgbm(True).fit(Ttr, ytr).predict(Tte),
         "spline_ridge": _spline_ridge(Xtr, ytr, Xte)}   # keeps the linear trend term t
    # lgbm_linear is reported for reference only; the blend is tree model + trend-aware spline/ridge
    p["blend"] = (p["lgbm"] + p["spline_ridge"]) / 2
    return p


def metrics(pred_rate, y):
    ape = np.abs(pred_rate - y) / y
    return {"MAE": float(np.abs(pred_rate - y).mean()), "MAPE_%": float(ape.mean() * 100),
            "medAPE_%": float(np.median(ape) * 100), "RMSE": float(np.sqrt(((pred_rate - y) ** 2).mean()))}
