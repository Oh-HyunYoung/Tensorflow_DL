# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

## 1. data load
df = pd.read_csv('...csv',header=None) # 해당하는 csv파일 불러오기 또는 데이터 로드
target = pd.read_csv('...csv',header=None)

X = df.loc[:,:]
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

