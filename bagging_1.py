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
df = pd.read_csv('../PPGdata_features.csv',header=None)

X = df.loc[:,1:12]
y = df.loc[:,13]
'''
'''
df = pd.read_csv('../PPG_features.csv',header=None)

X = df.loc[:,1:16]
y = df.loc[:,17]
'''
df = pd.read_csv('../detect_sort.csv',header=None)
target = pd.read_csv('../target.csv',header=None)

X = df.loc[:]
y = target.loc[1:,1]

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
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)

bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500, 
                        max_samples=1.0, 
                        max_features=1.0, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=-1, 
                        random_state=42)



tree = tree.fit(X_train, y_train)

bag = bag.fit(X_train, y_train)



## 6. performance measure
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

y_train_tree_pred = tree.predict(X_train)
y_test_tree_pred = tree.predict(X_test)
y_test_tree_prob = tree.predict_proba(X_test)

tree_train = accuracy_score(y_train, y_train_tree_pred)
tree_test = accuracy_score(y_test, y_test_tree_pred)
print('결정 트리의 훈련 정확도/테스트 정확도 %.3f/%.3f' % (tree_train, tree_test))

print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_tree_pred))
print("test data recall: %.3f" %recall_score(y_test, y_test_tree_pred, average='weighted'))
print("test data precison: %.3f" %precision_score(y_test, y_test_tree_pred, average='weighted'))
print("test data f1 score: %.3f" %f1_score(y_test, y_test_tree_pred, average='weighted'))
print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_tree_prob, multi_class='ovr'))
print("test data Confusion matrix:")
print(confusion_matrix(y_test, y_test_tree_pred))

y_train_bag_pred = bag.predict(X_train)
y_test_bag_pred = bag.predict(X_test)
y_test_bag_prob = bag.predict_proba(X_test)

bag_train = accuracy_score(y_train, y_train_bag_pred) 
bag_test = accuracy_score(y_test, y_test_bag_pred) 
print('배깅의 훈련 정확도/테스트 정확도 %.3f/%.3f' % (bag_train, bag_test))

print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_bag_pred))
print("test data recall: %.3f" %recall_score(y_test, y_test_bag_pred, average='weighted'))
print("test data precison: %.3f" %precision_score(y_test, y_test_bag_pred, average='weighted'))
print("test data f1 score: %.3f" %f1_score(y_test, y_test_bag_pred, average='weighted'))
print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_bag_prob, multi_class='ovr'))
print("test data Confusion matrix:")
print(confusion_matrix(y_test, y_test_bag_pred))