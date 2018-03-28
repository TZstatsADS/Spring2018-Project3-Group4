# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
# load train data
siftpath=os.path.expanduser("..\data\\train\SIFT_train.csv")
sift_feature=pd.read_csv(siftpath,header=None)
label_path=os.path.expanduser("..\data\\train\label_train.csv")
label=pd.read_csv(label_path)
sift_feature.insert(loc=0, column='label', value=label.label)

data=sift_feature 
X_train=data.iloc[:,2:]
y_train=data.iloc[:,0]
clf = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,subsample=0.1)
clf.fit(X_train,y_train)

X_test_path="..//data//test//SIFT_test.csv"
X_test=pd.read_csv(X_test_path)

test_pred=clf.predict(X_test.iloc[:,1:])
pd.DataFrame(test_pred).to_csv("..//output//base_pred.csv")