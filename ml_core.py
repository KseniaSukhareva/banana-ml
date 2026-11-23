"""Banana storage: exploratory MULTIVARIATE D0 -> D3/D5/D7/D10 forecasts.
All baseline predictors come from D0. Only V1..V3 train; V4 visualisation only.
No validated shelf-life, chilling or physiological state classifier.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {
    'respiration_source_units': 'Дыхание',
    'puncture_source_units': 'Сопротивление проколу',
    'firmness_source_units': 'Твёрдость мякоти',
    'oxalic_acid_source_units': 'Щавелевая кислота',
    'malic_acid_source_units': 'Яблочная кислота',
    'fructose_source_units': 'Фруктоза',
    'glucose_source_units': 'Глюкоза',
    'sucrose_source_units': 'Сахароза',
}
BASELINE_COLUMNS = list(TARGETS)
FEATURES = ['forecast_day'] + ['D0_' + c for c in BASELINE_COLUMNS]
DAYS = [3, 5, 7, 10]
TRAIN_VARIANTS = ['V1', 'V2', 'V3']

def clean_data(df):
    df = df.copy()
    required = {'variant', 'storage_day'} | set(BASELINE_COLUMNS)
    absent = required - set(df.columns)
    if absent:
        raise ValueError('Отсутствуют поля: ' + ', '.join(sorted(absent)))
    df['storage_day'] = pd.to_numeric(df['storage_day'], errors='coerce')
    for c in BASELINE_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['variant'] = df['variant'].astype(str).str.strip()
    if df.duplicated(['variant', 'storage_day']).any():
        raise ValueError('Обнаружены повторные сочетания вариант/сутки — нужна агрегация по методике.')
    return df

def d0_profiles(df):
    df = clean_data(df)
    baseline = df[(df.storage_day == 0) & df.variant.isin(TRAIN_VARIANTS)].set_index('variant')
    if set(TRAIN_VARIANTS) - set(baseline.index):
        raise ValueError('Нет исходных D0 для всех V1–V3')
    if baseline[BASELINE_COLUMNS].isna().any().any():
        raise ValueError('В D0 у V1–V3 есть пропуски среди восьми исходных показателей')
    return baseline[BASELINE_COLUMNS].copy()

def make_learning_table(df, target):
    if target not in TARGETS:
        raise ValueError('Неизвестный целевой показатель')
    df = clean_data(df)
    profiles = d0_profiles(df)
    entries = []
    for variant in TRAIN_VARIANTS:
        s = df[(df.variant == variant) & df.storage_day.isin(DAYS)]
        d0 = profiles.loc[variant]
        for _, obs in s.iterrows():
            if pd.isna(obs[target]):
                continue
            record = {'variant': variant, 'forecast_day': int(obs.storage_day), 'actual': float(obs[target]),
                      'constant_D0': float(d0[target])}
            record.update({'D0_' + c: float(d0[c]) for c in BASELINE_COLUMNS})
            entries.append(record)
    return pd.DataFrame(entries)

def new_model():
    # fixed regularisation, no tuning on the tiny dataset
    return make_pipeline(StandardScaler(), Ridge(alpha=30.0))

def leave_one_variant_out(frame):
    records=[]
    for held in TRAIN_VARIANTS:
        train = frame[frame.variant != held]
        test = frame[frame.variant == held]
        if train.variant.nunique() < 2 or test.empty:
            continue
        model = new_model()
        model.fit(train[FEATURES], train['actual'])
        forecast = model.predict(test[FEATURES])
        for (_, row), predicted in zip(test.iterrows(), forecast):
            records.append({'excluded_variant': held, 'day': int(row.forecast_day),
                            'actual': float(row.actual), 'predicted': float(predicted),
                            'constant_D0': float(row.constant_D0),
                            'abs_error_ML': abs(float(row.actual)-float(predicted)),
                            'abs_error_D0': abs(float(row.actual)-float(row.constant_D0))})
    return pd.DataFrame(records)

def fit_predict(frame, d0_input, days=DAYS):
    if frame.empty or frame.variant.nunique() < 3:
        raise ValueError('Для расчёта необходимы V1–V3 с последующими сроками')
    model = new_model()
    model.fit(frame[FEATURES], frame['actual'])
    rows=[]
    for day in days:
        row = {'forecast_day': day}
        row.update({'D0_' + c: float(d0_input[c]) for c in BASELINE_COLUMNS})
        rows.append(row)
    X=pd.DataFrame(rows)
    return np.asarray(model.predict(X[FEATURES]),dtype=float)

def validate_all(df):
    entries=[]
    details={}
    for target in TARGETS:
        frame=make_learning_table(df,target)
        val=leave_one_variant_out(frame)
        if val.empty:
            continue
        details[target]=val
        entries.append({'target':target,'label':TARGETS[target], 'n_variants':frame.variant.nunique(),
                        'n_test_points':len(val), 'ML_MAE':val.abs_error_ML.mean(),
                        'D0_MAE':val.abs_error_D0.mean(),
                        'beats_constant_D0':val.abs_error_ML.mean()<val.abs_error_D0.mean()})
    return pd.DataFrame(entries),details
