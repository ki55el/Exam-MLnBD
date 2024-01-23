import streamlit as st
import pandas as pd


df = pd.read_csv('data/DatasetExam.csv')
df_upd = pd.read_csv('data/DatasetExam_upd.csv')

st.set_page_config(page_title='Dataset', page_icon='💾')

st.markdown('''
# Информация о наборе данных

## Тематика
Этот датасет содержит различные признаки, относящиеся к кампании маркетинга банковского учреждения. 
Цель классификации - предсказать, подпишется ли клиент на срочный депозит.

**Датасет до предобработки 👇**
''')
st.dataframe(df)

st.markdown('''
## Особенности предобработки данных
            
* Выделены поля, конкатетенирующие в себе несколько значений, либо указывающие диапазон, и разбиты на несколько других
```
dummies = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y'], drop_first=True, dtype=int)
df = pd.DataFrame(data=dummies)
```
* с помощью `df.isna().sum().sum()` узнаем, что количество пропущенных переменных равно нулю. Следовательно, их обработка не требуется

В результате получаем, что
* Числовыми признаками являются:
    * 1 - `age`: int64
    * 11 - `campaign`: int64
    * 15 -`emp.var.rate `: float64
    * 16 - `cons.price.idx`: float64
    * 17 - `cons.conf.idx`: float64
    * 18 - `euribor3m `: float64
    * 19 -`nr.employed`: float64

* Все остальные признаки - категориальные (содержат 0 и 1)

* Столбцы `age`, `emp.var.rate`, `cons.conf.idx` содержали выбросы, но были удалены путем определения нижнего/верхнего предела нормального диапазона значений

**Датасет после предобработки 👇**
''')
st.dataframe(df_upd)
