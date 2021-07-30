
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle
df = pd.read_csv('diabetes.csv')
print(df)


df['Pregnancies'].replace(0, np.nan, inplace=True)


df['Glucose'].replace(0, np.nan, inplace=True)
df['BloodPressure'].replace(0, np.nan, inplace=True)
df['SkinThickness'].replace(0, np.nan, inplace=True)
df['Insulin'].replace(0, np.nan, inplace=True)
df['BMI'].replace(0, np.nan, inplace=True)
df.drop(['SkinThickness' ,'Insulin'], axis=1, inplace=True)
df = df.dropna(axis=0)







y = df['Outcome']
print(y)
X = df.drop('Outcome', axis=1)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2
                                                    , random_state=1)

print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

#model = ensemble.RandomForestClassifier()
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))




log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predlog = log_reg.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_predlog)))
#same both



pickle.dump(log_reg,open('modeldiabetes.pkl','wb'))
model=pickle.load(open('modeldiabetes.pkl','rb'))
