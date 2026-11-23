"""Run: python tests_smoke.py"""
from pathlib import Path
import pandas as pd
from ml_core import TARGETS, pairs, group_validation, trained_model, prediction_table

df = pd.read_csv(Path(__file__).parent / 'data' / 'banana_ml_dataset_v1.csv', encoding='utf-8-sig')
for target in TARGETS:
    frame = pairs(df, target)
    if frame['variant'].nunique() >= 3:
        validation = group_validation(frame)
        assert len(validation) > 0 and validation['abs_error_ML'].notna().all(), target
        model = trained_model(frame)
        forecast = prediction_table(model, float(frame.iloc[0]['baseline']))
        assert len(forecast) == 4 and forecast['prediction'].notna().all(), target
print('OK: dataset, leave-one-variant-out, forecast')
