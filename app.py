"""Streamlit application: transparent multivariate banana research model."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from ml_core import (BASELINE_COLUMNS, TARGETS, DAYS, FEATURES, TRAIN_VARIANTS,
                     clean_data, d0_profiles, make_learning_table, leave_one_variant_out,
                     fit_predict, validate_all)

st.set_page_config(page_title='Banana ML | многопараметрический прогноз',layout='wide')
st.title('Banana ML — многопараметрический прогноз при хранении')
st.warning('Исследовательский прототип. Всего три независимых исходных профиля V1–V3. Прогноз не валидирован на новых партиях и НЕ определяет срок годности или безопасность.')
st.caption('Модель использует одновременно 8 показателей на D0 и горизонт прогноза. Только условия опыта № 1 (~12,8 °C, без газации). V4 и V4з не входят в обучение.')

with st.sidebar:
    st.header('Источники')
    upload=st.file_uploader('Необязательно: загрузить сводный CSV той же структуры',type=['csv'])
    st.caption('Файл по умолчанию уже включён в проект. Исходный Excel с 21 листом сюда напрямую загружать не нужно.')
try:
    if upload is not None:
        df=pd.read_csv(upload,encoding='utf-8-sig')
    else:
        root=Path(__file__).resolve().parent
        choices=[root/'data'/'banana_ml_dataset_v1.csv',root/'banana_ml_dataset_v1.csv']
        source=next((p for p in choices if p.is_file()),None)
        if source is None:
            raise FileNotFoundError('Не найден banana_ml_dataset_v1.csv — поместите его в корень репозитория или data/')
        df=pd.read_csv(source,encoding='utf-8-sig')
    df=clean_data(df)
    profiles=d0_profiles(df)
except Exception as exc:
    st.error(f'Ошибка чтения данных: {exc}')
    st.stop()

st.subheader('Экспериментальная база')
a,b,c=st.columns(3)
a.metric('Записей',len(df)); b.metric('Вариантов в базе',df.variant.nunique()); c.metric('Профилей для обучения',len(profiles))
with st.expander('Исходные данные и D0 профили'):
    st.dataframe(df,use_container_width=True)
    st.markdown('**Восемь исходных показателей D0, используемых одновременно:**')
    st.dataframe(profiles.rename(columns=TARGETS),use_container_width=True)

st.subheader('Проверка моделей без пересечения вариантов')
st.caption('На каждом шаге один вариант полностью исключается из обучения. Показываем ошибку многопараметрической модели и наивного прогноза: значение D0 остаётся неизменным. Проверка крайне нестабильна при трёх вариантах.')
summary,details=validate_all(df)
if summary.empty:
    st.error('Недостаточно наблюдений для проверки модели.')
    st.stop()
view=summary[['label','n_test_points','ML_MAE','D0_MAE','beats_constant_D0']].rename(columns={
    'label':'Показатель','n_test_points':'Точек в проверке','ML_MAE':'MAE модели',
    'D0_MAE':'MAE постоянного D0','beats_constant_D0':'ML лучше D0?'})
st.dataframe(view.style.format({'MAE модели':'{:.3f}','MAE постоянного D0':'{:.3f}'}),use_container_width=True,hide_index=True)
st.download_button('Скачать таблицу проверки',summary.to_csv(index=False).encode('utf-8-sig'),'banana_multi_validation.csv','text/csv')

st.subheader('Ввод исходного профиля партии (D0)')
st.caption('Измените все или некоторые восемь входных признаков. Использование вне диапазонов V1–V3 — экстраполяция, а не подтверждённый прогноз.')
col1,col2=st.columns(2)
values={}
for i,key in enumerate(BASELINE_COLUMNS):
    series=profiles[key].astype(float)
    with (col1 if i%2==0 else col2):
        values[key]=st.number_input(TARGETS[key]+' — D0',value=float(series.median()),format='%.4f',key='d0_'+key)
outside=[TARGETS[key] for key,val in values.items() if not (profiles[key].min()<=val<=profiles[key].max())]
if outside:
    st.error('За пределами наблюдавшегося диапазона D0: '+', '.join(outside)+'. Результаты являются экстраполяцией.')

st.subheader('Многопараметрический прогноз на D3 / D5 / D7 / D10')
chosen=st.multiselect('Показатели для графика',options=list(TARGETS),default=['firmness_source_units','respiration_source_units','fructose_source_units'],format_func=lambda x:TARGETS[x])
if st.button('Рассчитать прогноз по всем 8 исходным показателям',type='primary'):
    allrows=[]
    for target in TARGETS:
        frame=make_learning_table(df,target)
        predicted=fit_predict(frame,values)
        metric=summary.loc[summary.target==target].iloc[0]
        for day,pred in zip(DAYS,predicted):
            allrows.append({'День':day,'Показатель':TARGETS[target], 'Код показателя':target,
                            'Прогноз':float(pred),'ML лучше D0 в проверке':bool(metric.beats_constant_D0)})
    forecast=pd.DataFrame(allrows)
    st.dataframe(forecast[['День','Показатель','Прогноз','ML лучше D0 в проверке']].style.format({'Прогноз':'{:.3f}'}),use_container_width=True,hide_index=True)
    st.download_button('Скачать прогноз CSV',forecast.to_csv(index=False).encode('utf-8-sig'),'banana_multivariate_forecast.csv','text/csv')
    for target in chosen:
        data=forecast[forecast['Код показателя']==target]
        fig=px.line(data,x='День',y='Прогноз',markers=True,title=TARGETS[target])
        st.plotly_chart(fig,use_container_width=True)
    st.info('Важное ограничение: это модель зависимости от D0, а не валидированный прогноз комплексной физиологической стадии, газации, холодового повреждения или срока годности. Доверительные интервалы не рассчитаны.')

st.subheader('Экспериментальные кривые (наблюдения, не прогноз)')
target=st.selectbox('Показатель для просмотра',list(TARGETS),format_func=lambda key:TARGETS[key])
observed=df.dropna(subset=[target]).copy()
st.plotly_chart(px.line(observed,x='storage_day',y=target,color='variant',markers=True,
                         labels={'storage_day':'Сутки','variant':'Вариант',target:'Единицы исходной таблицы'},
                         title=TARGETS[target]),use_container_width=True)
st.caption('Входные показатели и их единицы необходимо подтвердить по первичным протоколам. Точки разных сроков одной партии не являются независимыми биологическими повторностями.')
