import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_selection import VarianceThreshold


df = pd.read_csv('data/DatasetExam_upd.csv')
y = df['y_yes']
X = df.drop(['y_yes'], axis=1)

with open("models/BaggingClassifier.pkl", "rb") as f:
    model1 = pickle.load(f)
with open("models/GradientBoostingClassifier.pkl", "rb") as f:
    model2 = pickle.load(f)
with open("models/KNeighborsClassifier.pkl", "rb") as f:
    model3 = pickle.load(f)

models = {
    'BaggingClassifier': model1,
    'GradientBoostingClassifier': model2,
    'KNeighborsClassifier': model3,
}

st.set_page_config(page_title='Predict', page_icon='🎰')

st.write('# Предсказание моделей машинного обучения')

st.write('**Введите данные для предсказания 👇**')

labels = (
    'возраст',
    'должность',
    'семейное положение',
    'образование',
    'имеет ли кредит',
    'есть ли жилищный кредит',
    'есть ли потребительский кредит',
    'тип контактной связи',
    'месяц',
    'день недели',
    'кампания',
    'последний контакт (кол-во дней)',
    'кол-во контактов до',
    'предыдущий результат',
    'уровень занятости',
    'индекс цен',
    'доверие потребителей',
    'ставка еврибора',
    'кол-во сотрудников'
)

inp = {}
for ix, label in zip(X.columns, labels):
    inp[ix] = st.slider(f'**{label} =**', min(X[ix]), max(X[ix]))

X_inp = pd.DataFrame([inp])
st.write(
    '## Для следующего набора данных:', 
    X_inp, 
    '## Получаем следующие предсказания:')
    
vt = VarianceThreshold(1.0)
vt.fit(X)
X_inp = vt.transform(X_inp)

for name, model in models.items():
    st.write(f'### `{name}`: клиент', '' if model.predict(X_inp) else 'не', 'подписался на срочный депозит')
             
