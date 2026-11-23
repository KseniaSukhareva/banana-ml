"""Streamlit UI: research prototype for banana storage observations."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from ml_core import TARGETS, TRAIN_VARIANTS, DAYS, pairs, group_validation, trained_model, prediction_table, validate_dataset

st.set_page_config(page_title="Banana ML | Исследовательский прототип", page_icon="🍌", layout="wide")
st.title("Прогноз показателей бананов при хранении")
st.warning("Исследовательский прототип. Данных мало (V1–V3); результаты нельзя использовать для решения о сроке годности, безопасности или коммерческой приёмке партий без независимой валидации.")
st.caption("Прогноз только для исследованного режима ≈12,8 °C, без газации и без моделирования V4з. Значения показателей сохраняют единицы исходной таблицы; единицы следует сверить с лабораторными протоколами.")

with st.sidebar:
    st.header("Данные")
    uploaded = st.file_uploader("Загрузить объединённую CSV-базу (не исходную книгу с 21 листом)", type=["csv"])
    st.caption("Файл по умолчанию уже включён в проект. Для подготовки данных из исходной книги потребуется отдельный этап импорта.")

try:
    df = pd.read_csv(uploaded if uploaded else Path(__file__).parent / "banana_ml_dataset_v1.csv", encoding="utf-8-sig")
    df["storage_day"] = pd.to_numeric(df["storage_day"], errors="coerce")
    for target in TARGETS:
        df[target] = pd.to_numeric(df[target], errors="coerce")
    validate_dataset(df)
except (ValueError, OSError, KeyError, UnicodeDecodeError) as exc:
    st.error(f"Ошибка загрузки набора данных: {exc}")
    st.stop()

st.subheader("Наблюдения")
c1,c2,c3 = st.columns(3)
c1.metric("Записей", len(df))
c2.metric("Вариантов", df["variant"].nunique())
c3.metric("Сроков хранения", df["storage_day"].nunique())
choice = st.selectbox("Измеряемый показатель", list(TARGETS), format_func=lambda k: TARGETS[k])
plot_data = df.dropna(subset=[choice]).copy()
fig = px.line(plot_data, x="storage_day", y=choice, color="variant", markers=True, title=f"Экспериментальная динамика: {TARGETS[choice]}", labels={"storage_day":"Сутки хранения", choice:"Единицы исходной таблицы", "variant":"Вариант"})
st.plotly_chart(fig, use_container_width=True)
with st.expander("Посмотреть исходную сводную таблицу"):
    st.dataframe(df, use_container_width=True)
    st.download_button("Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"), "banana_data.csv", "text/csv")

st.subheader("Обучение и проверка на исключённом варианте")
st.markdown("Для прогноза на D3, D5, D7, D10 используется **только исходное измерение выбранного показателя на D0** и горизонт прогноза. Не применяются данные из будущего. Валидация: V1, V2 и V3 по очереди исключаются полностью.")
frame = pairs(df, choice)
if frame.empty or frame["variant"].nunique() < 3:
    st.info("Для этого показателя недостаточно полных данных D0–D10 по V1–V3, чтобы провести трёхгрупповую проверку. Прогноз не выполняется.")
    st.stop()
val = group_validation(frame)
if val.empty:
    st.info("Недостаточно вариантов для групповой проверки.")
    st.stop()
a = val["abs_error_ML"].mean()
b = val["abs_error_D0"].mean()
k1,k2,k3=st.columns(3)
k1.metric("MAE модели", f"{a:.3f}")
k2.metric("MAE постоянного D0", f"{b:.3f}")
k3.metric("Проверочных точек", len(val))
if a >= b:
    st.error("ML в данном тесте не превосходит простой прогноз от D0. Использовать эту модель для практических решений не рекомендуется.")
else:
    st.info("ML лучше простого D0 в этом ограниченном тесте; независимая внешняя валидация всё равно необходима.")
fig2 = px.scatter(val, x="actual", y="predicted", color="held_out", hover_data=["day"], title="Проверка: фактические и прогнозные значения", labels={"actual":"Факт", "predicted":"Прогноз ML", "held_out":"Исключённый вариант"})
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(val.round(3), use_container_width=True)
st.download_button("Скачать результаты проверки", val.to_csv(index=False).encode("utf-8-sig"), "validation.csv", "text/csv")

st.subheader("Исследовательский прогноз для исходного значения D0")
st.caption("Вводить можно только значения из диапазона исходных измерений, на которых модель обучалась. Прогнозируем только D3, D5, D7 и D10.")
base = frame.drop_duplicates("variant")["baseline"]
low, high = float(base.min()),float(base.max())
input_value = st.number_input(f"{TARGETS[choice]} в день приёмки (D0)", min_value=low, max_value=high, value=float(base.median()), format="%.4f")
model = trained_model(frame)
forecast = prediction_table(model, input_value)
forecast = forecast.rename(columns={"day":"Сутки", "baseline":"Значение D0", "prediction":"Модельный прогноз"})
st.dataframe(forecast.round(3), use_container_width=True, hide_index=True)
st.line_chart(forecast.set_index("Сутки")[["Модельный прогноз"]])
st.download_button("Скачать прогноз", forecast.to_csv(index=False).encode("utf-8-sig"), "banana_forecast.csv", "text/csv")
st.caption("Модель обучена на 3 вариантах. Область применения крайне узкая; оценка неопределённости и порогов товарной пригодности ещё не разработана.")
