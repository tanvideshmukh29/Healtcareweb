import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('heart.csv')



X=df.drop(['target','ca','slope','thal','oldpeak'], axis = 1)
print(X)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))



log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predlog = log_reg.predict(X_test)

print('Accuracy : {}'.format(accuracy_score(y_test, y_predlog)))

pickle.dump(log_reg,open('modelheart.pkl','wb'))
model=pickle.load(open('modelheart.pkl','rb'))