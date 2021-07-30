import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

patients=pd.read_csv('indian_liver_patient.csv')

patients[['Gender']] = patients[['Gender']].replace(to_replace={'Male': 1, 'Female': 0})
print(patients[['Gender']])

patients=patients.dropna(axis=0)

X=patients[['Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']]
y=patients['Dataset']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

model = LogisticRegression()
model.fit(X_train, y_train)
y_predlog = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_predlog)))
pickle.dump(model,open('modelliver.pkl','wb'))
model=pickle.load(open('modelliver.pkl','rb'))
