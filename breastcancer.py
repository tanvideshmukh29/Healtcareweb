
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import pickle
df = pd.read_csv('data.csv')
print(df)
new_df = df.drop('Unnamed: 32',axis='columns')
print(new_df)

new_df.drop(new_df.columns[[0,-1]], axis=1, inplace=True)
print(new_df)
xdata = new_df.drop(['diagnosis'], axis=1)
print(xdata)
ydata = new_df['diagnosis']
print(ydata)


ydata = np.asarray([1 if c == 'M' else 0 for c in ydata])
print(ydata)
print(xdata)
cols = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean','smoothness_se']


xdata = df[cols]
print(xdata.columns)
print(xdata)
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata,
                                                    test_size=0.3,
                                                    random_state=43)
print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))




model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

#log_reg = LogisticRegression()
#log_reg.fit(X_train, y_train)
#y_predlog = model.predict(X_test)
#print('Accuracy : {}'.format(accuracy_score(y_test, y_predlog)))
#same both

pickle.dump(model,open('modelcancer.pkl','wb'))
model=pickle.load(open('modelcancer.pkl','rb'))




