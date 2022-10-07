# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

## 1. data load
'''
df_all_1 = pd.read_csv('../ECG_data/feature/All.csv',header=None)
rhythmAll = ['AFIB','GSVT','SB','SR']

df1 = pd.read_csv('../ECG_data/feature/AFIB.csv',header=None) 
df2 = pd.read_csv('../ECG_data/feature/GSVT.csv',header=None) 
df3 = pd.read_csv('../ECG_data/feature/SB.csv',header=None) 
df4 = pd.read_csv('../ECG_data/feature/SR.csv',header=None) 

AFIB = np.zeros(len(df1))
GSVT = np.ones(len(df2))
SB = np.full((len(df3)),2)
SR = np.full((len(df4)),3) 
arr = np.concatenate([AFIB,GSVT,SB,SR],axis=0)

X = df_all_1.loc[:]
y = arr
'''
df = pd.read_csv('../PPGdata_new_features.csv',header=None)

X = df.loc[:,1:19]
y = df.loc[:,20]


## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)


## 3. train / validation / test data split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = \
            train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            


## 4. data preprocessing
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)




## 5. algorithm training
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

clf1 = SVC(kernel='linear', random_state=42, probability=True)
clf2 = LogisticRegression(random_state=42)
clf3 = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')


from sklearn.ensemble import StackingClassifier

stack = StackingClassifier(estimators=[
                                ('svm',clf1),
                                ('lr', clf2),
                                ('knn',clf3)],
                            final_estimator=LogisticRegression(random_state=42))
        
stack.fit(X_train_std, y_train)
        


## 6. performance measure
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

y_test_pred = stack.predict(X_test_std)
y_test_prob = stack.predict_proba(X_test_std)

print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
print("test data recall: %.3f" %recall_score(y_test, y_test_pred, average='macro'))
print("test data precison: %.3f" %precision_score(y_test, y_test_pred, average='macro'))
print("test data f1 score: %.3f" %f1_score(y_test, y_test_pred, average='macro'))
print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
print("test data Confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

