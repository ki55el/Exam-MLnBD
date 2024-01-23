import pickle

import numpy as np
import pandas as pd
import streamlit as st


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

inp = {}
for ix in X.columns:
    inp[ix] = st.slider(f'**{ix} =**', min(X[ix]), max(X[ix]))

X_inp = pd.DataFrame([inp])
st.write(
    '## Для следующего набора данных:', 
    X_inp, 
    '## Получаем следующие предсказания:')
    
for name, model in models.items():
    st.write(f'### `{name}`: клиент', '' if model.predict(X_inp) else 'не', 'подписался на срочный депозит')
             
