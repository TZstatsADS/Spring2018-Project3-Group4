# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import argparse

def Main():
    parser=argparse.ArgumentParser()
    parser.add_argument("train",help="whether it is train data 1 is true 0 is false",type=int)
    args=parser.parse_args()
    if args.train:
        # load train data
        siftpath=os.path.expanduser("..\\data\\train\\SIFT_train.csv")
        print("start loading the data")
        sift_feature=pd.read_csv(siftpath,header=None)
        label_path=os.path.expanduser("..\\data\\train\\label_train.csv")
        label=pd.read_csv(label_path)
        sift_feature.insert(loc=0, column='label', value=label.label)

        data=sift_feature 
        X_train=data.iloc[:,2:]
        y_train=data.iloc[:,0]
        clf = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,subsample=0.1)
        print("start train the model")
        clf.fit(X_train,y_train)
        print("we finished training now we are storing the model")
        with open("..\\output\\basemodel.pickle","bw") as f:
            pickle.dump(clf,f)
    else:
        print("we load the model")
        with open("..\\output\\basemodel.pickle","br") as f:
            clf=pickle.load(f)
        print("we load the data")
        X_test_path="..//data//test//SIFT_test.csv"
        X_test=pd.read_csv(X_test_path)
        test_pred=clf.predict(X_test.iloc[:,1:])
        print("we output the test")
        pd.DataFrame(test_pred).to_csv("..//output//base_pred.csv")

if __name__=="__main__":
  
    Main()

