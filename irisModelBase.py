# 2025.3.10.
# 프로젝트2 붓꽃분류기 만들기
from operator import irshift

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
iris_df = pd.read('iris.csv')
print(iris_df)
y=iris_df['species']
X=iris_df.drop('species', axis=1)

kn= KNeighborsClassifier()
rfc= RandomForestClassifier()
model_kn = kn.fit(X,y)
model_rfc = rfc.fit(X,y)

joblib.dump(model_rfc,'model_rfc.pkl')

# X_new = np.array(([3,3,3,3]))
# prediction = model_kn.predict(X_new)
# print(prediction)
# probability= model_kn.predict_proba(X_new)
# print(probability)

model_rfc=joblib.load('model_rfc.pkl')
X_new = np.array(([1, 4.2,1.4, 7]))
# prediction = model_kn.predict(X_new)
prediction = model_rfc.predict(X_new)
print(prediction)
probability= model_kn.predict_proba(X_new)
print(probability)
