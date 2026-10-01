"""Cleaning and feature engineering for the freight-rate challenge.

One FeatureBuilder is used for train, hold-out, validation.csv and the December
chart inputs, so every file goes through identical logic.
Anything learned from data (medians, ...) is learned in `fit()` on TRAIN ONLY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

WEIGHT_CAP = 47_500.0
EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
TARGET = "posted_rate"


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


class FeatureBuilder:
    def __init__(self):
        self.weight_median_by_equip_: dict[str, float] = {}
        self.weight_median_: float = np.nan
        self.city_codes_: dict[str, int] = {}
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        w = df["weight"].abs()
        self.weight_median_ = float(w.median())
        self.weight_median_by_equip_ = w.groupby(df["equipment"]).median().to_dict()
        cities = sorted(set(df["pickup"]) | set(df["delivery"]))
        self.city_codes_ = {c: i for i, c in enumerate(cities)}
        self.fitted_ = True
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted_, "call fit() first"
        d = pd.DataFrame(index=df.index)
        date = pd.to_datetime(df["date"])

        # --- data-quality handling -------------------------------------------------
        w_raw = df["weight"]
        d["weight_missing"] = w_raw.isna().astype(int)
        d["weight_was_negative"] = (w_raw < 0).astype(int)      # sign errors -> abs()
        w = w_raw.abs()
        d["weight_capped"] = (w >= WEIGHT_CAP).astype(int)       # censored at 47,500
        fill = df["equipment"].map(self.weight_median_by_equip_).fillna(self.weight_median_)
        d["weight"] = w.fillna(fill)
        d["market_index_missing"] = df["market_index"].isna().astype(int)
        d["market_index"] = df["market_index"]                   # LightGBM handles NaN natively
        d["quote_signal"] = df["quote_signal"]

        # --- distance / geography --------------------------------------------------
        d["distance"] = df["distance"]
        d["log_distance"] = np.log(df["distance"])
        hav = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
        d["haversine"] = hav
        d["distance_ratio"] = df["distance"] / hav.clip(lower=1)
        d["bearing"] = bearing_deg(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
        for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
            d[c] = df[c]
        d["dlat"] = df["delivery_lat"] - df["pickup_lat"]
        d["dlon"] = df["delivery_lon"] - df["pickup_lon"]
        d["pickup_city"] = df["pickup"].map(self.city_codes_).fillna(-1).astype(int)
        d["delivery_city"] = df["delivery"].map(self.city_codes_).fillna(-1).astype(int)
        d["equipment"] = df["equipment"].map(EQUIP).astype(int)

        # --- calendar (no raw date/month/day-of-year: they cannot extrapolate to Nov-Dec)
        d["dow"] = date.dt.dayofweek
        d["is_weekend"] = (d["dow"] >= 5).astype(int)
        d["dom"] = date.dt.day
        d["days_to_month_end"] = date.dt.days_in_month - d["dom"]
        years = list(range(int(date.dt.year.min()) - 1, int(date.dt.year.max()) + 2))
        hol = holidays.US(years=years)
        hol_dates = pd.to_datetime(sorted(hol.keys()))
        idx = np.searchsorted(hol_dates.values, date.values)
        prev = hol_dates[np.clip(idx - 1, 0, len(hol_dates) - 1)]
        nxt = hol_dates[np.clip(idx, 0, len(hol_dates) - 1)]
        d["days_since_holiday"] = (date.values - prev.values).astype("timedelta64[D]").astype(int)
        d["days_to_holiday"] = (nxt.values - date.values).astype("timedelta64[D]").astype(int)
        d["near_holiday"] = ((d["days_since_holiday"] <= 2) | (d["days_to_holiday"] <= 2)).astype(int)
        return d


def load(path: str, with_target: bool = False):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df
