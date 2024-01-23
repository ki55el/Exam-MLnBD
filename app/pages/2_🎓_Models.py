import streamlit as st


st.markdown('''
# Информация о построенных моделях

## `KNeighborsClassifier`:
```
{'Accuracy': 0.8480848153214774,
'F1-score': 0.8412777817480168,
'Precision': 0.8707100591715976,
'ROC_AUC': 0.8477271296773502,
'Recall': 0.8137702198257984}
```          

## `BaggingClassifier`:
```            
{'Accuracy': 0.901436388508892,
'F1-score': 0.8986282096377066,
'Precision': 0.9147808650816385,
'ROC_AUC': 0.9012445889420792,
'Recall': 0.8830360846121941}            
```
            
## `GradientBoostingClassifier`:
```
{'Accuracy': 0.8480848153214774,
'F1-score': 0.8412777817480168,
'Precision': 0.8707100591715976,
'ROC_AUC': 0.8477271296773502,
'Recall': 0.8137702198257984}
``` 

*Очевидно, что лучшей моделью будет `BaggingClassifier`*
''')
