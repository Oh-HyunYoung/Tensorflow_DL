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
'''
df = pd.read_csv("../value/value/value_5.csv")
print(df)
X = df.iloc[:,:12]
y = df.iloc[:,-1]
'''
'''
df = pd.read_csv('../PPGdata_new_features.csv',header=None)

X = df.loc[:,1:19]
y = df.loc[:,20]
'''
df = pd.read_csv('../detect_ds_sm.csv',header=None)
target = pd.read_csv('../target_100.csv',header=None)

X = df.loc[:,1:19]
y = target.loc[:,1]
## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', max_depth=1, random_state=42)

ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500, 
                         random_state=42)



tree = tree.fit(X_train, y_train)

ada = ada.fit(X_train, y_train)



## 6. performance measure
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

y_train_tree_pred = tree.predict(X_train)
y_test_tree_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_tree_pred)
tree_test = accuracy_score(y_test, y_test_tree_pred)
print('결정 트리의 훈련 정확도/테스트 정확도 %.3f/%.3f' % (tree_train, tree_test))


y_train_ada_pred = ada.predict(X_train)
y_test_ada_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_ada_pred) 
ada_test = accuracy_score(y_test, y_test_ada_pred) 
print('에이다부스트의 훈련 정확도/테스트 정확도 %.3f/%.3f' % (ada_train, ada_test))

