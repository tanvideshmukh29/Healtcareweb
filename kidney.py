



import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

df = pd.read_csv("kidney_disease.csv")

df[['htn', 'dm', 'cad', 'pe', 'ane']] = df[['htn', 'dm', 'cad', 'pe', 'ane']].replace(to_replace={'yes': 1, 'no': 0})
df[['rbc', 'pc']] = df[['rbc', 'pc']].replace(to_replace={'abnormal': 1, 'normal': 0})
df[['pcc', 'ba']] = df[['pcc', 'ba']].replace(to_replace={'present': 1, 'notpresent': 0})

df['classification'] = df['classification'].replace(to_replace={'ckd': 1.0, 'ckd\t': 1.0, 'notckd': 0.0, 'no': 0.0})
df.rename(columns={'classification': 'class'}, inplace=True)


df.drop('id', axis=1, inplace=True)
df = df.dropna(axis=0)

cols = ['bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc']
X = df[cols]
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, stratify=y)
print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))


pickle.dump(model,open('modelkidney.pkl','wb'))
model=pickle.load(open('modelkidney.pkl','rb'))
