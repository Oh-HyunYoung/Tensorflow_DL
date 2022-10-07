# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

## 1. data load

df = pd.read_csv('../detect_sort.csv',header=None)
target = pd.read_csv('../target.csv',header=None)

X = df.loc[:]
y = target.loc[1:,1]


## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)

# from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# X = le.fit_transform(X)
y = le.fit_transform(y)


## 3. train / validation / test data split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = \
            train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            


## 4. data preprocessing
# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# ss.fit(X_train)
# X_train_std = ss.transform(X_train)
# X_test_std = ss.transform(X_test)




## 5. algorithm training
from xgboost import XGBClassifier


xgb = XGBClassifier(tree_method='hist', random_state=42,use_label_encoder=False,eval_metric='logloss')
xgb.fit(X_train, y_train)

xgb_train_score = xgb.score(X_train, y_train)
xgb_test_score = xgb.score(X_test, y_test)

print('XGBoost 훈련 정확도/테스트 정확도 %.3f/%.3f' % (xgb_train_score, xgb_test_score))

## 6. performance measure
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
y_train_xgb_pred = xgb.predict(X_train)
y_test_xgb_pred = xgb.predict(X_test)
y_test_xgb_prob = xgb.predict_proba(X_test)

print("test data accuracy: %.3f" %accuracy_score(y_test,y_test_xgb_pred))
print("test data recall: %.3f" %recall_score(y_test,y_test_xgb_pred, average='weighted'))
print("test data precison: %.3f" %precision_score(y_test,y_test_xgb_pred, average='weighted'))
print("test data f1 score: %.3f" %f1_score(y_test,y_test_xgb_pred, average='weighted'))
print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_xgb_prob, multi_class='ovr'))
print("test data Confusion matrix:")
print(confusion_matrix(y_test,y_test_xgb_pred))