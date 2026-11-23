"""Transparent small-sample baseline models; NOT a validated shelf-life predictor."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

TARGETS = {
    "firmness_source_units": "Твёрдость мякоти",
    "puncture_source_units": "Сопротивление проколу",
    "respiration_source_units": "Показатель дыхания",
    "fructose_source_units": "Фруктоза",
    "glucose_source_units": "Глюкоза",
    "sucrose_source_units": "Сахароза",
    "oxalic_acid_source_units": "Щавелевая кислота",
    "malic_acid_source_units": "Яблочная кислота",
    "brix": "°Brix",
}
TRAIN_VARIANTS = ("V1", "V2", "V3")
DAYS = (3, 5, 7, 10)
FEATURES = ["day", "baseline"]


def validate_dataset(df: pd.DataFrame) -> None:
    need = {"variant", "storage_day", *TARGETS}
    missing = need - set(df.columns)
    if missing:
        raise ValueError("Отсутствуют столбцы: " + ", ".join(sorted(missing)))
    if df.duplicated(["variant", "storage_day"]).any():
        raise ValueError("Одинаковые (variant, storage_day) повторяются. Сначала определите правила агрегации.")


def pairs(df: pd.DataFrame, target: str, variants=TRAIN_VARIANTS, days=DAYS) -> pd.DataFrame:
    """One row for each future day; predictors are D0 measurement and forecast horizon ONLY."""
    validate_dataset(df)
    if target not in TARGETS:
        raise ValueError("Неизвестный показатель")
    rows = []
    for variant in variants:
        one = df[df["variant"] == variant].copy()
        d0 = one[one["storage_day"] == 0]
        if len(d0) != 1:
            continue
        baseline = pd.to_numeric(d0[target], errors="coerce").iloc[0]
        if pd.isna(baseline):
            continue
        for day in days:
            row = one[one["storage_day"] == day]
            if len(row) != 1:
                continue
            observed = pd.to_numeric(row[target], errors="coerce").iloc[0]
            if pd.isna(observed):
                continue
            rows.append({"variant": variant, "day": int(day), "baseline": float(baseline), "observed": float(observed)})
    return pd.DataFrame(rows, columns=["variant", "day", "baseline", "observed"])


def new_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def group_validation(frame: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-variant-out. Train and test never share a variant."""
    output = []
    for held_out in sorted(frame["variant"].unique()):
        train = frame[frame["variant"] != held_out]
        test = frame[frame["variant"] == held_out]
        if train["variant"].nunique() < 2 or len(test) == 0:
            continue
        model = new_model()
        model.fit(train[FEATURES], train["observed"])
        pred = model.predict(test[FEATURES])
        for (_, item), value in zip(test.iterrows(), pred):
            output.append({
                "held_out": held_out, "day": int(item["day"]),
                "actual": float(item["observed"]),
                "predicted": float(value),
                "D0_constant": float(item["baseline"]),
                "abs_error_ML": float(abs(item["observed"] - value)),
                "abs_error_D0": float(abs(item["observed"] - item["baseline"])),
            })
    return pd.DataFrame(output)


def trained_model(frame: pd.DataFrame):
    if frame.empty or frame["variant"].nunique() < 2:
        raise ValueError("Для обучения необходимо не менее двух вариантов с измерениями D0 и последующими сроками")
    model = new_model()
    model.fit(frame[FEATURES], frame["observed"])
    return model


def prediction_table(model, baseline: float, days=DAYS) -> pd.DataFrame:
    X = pd.DataFrame({"day": list(days), "baseline": [float(baseline)] * len(days)})
    X["prediction"] = model.predict(X[FEATURES])
    return X
