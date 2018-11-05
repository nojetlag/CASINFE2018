# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:26:12 2018

@author: stuck
"""

import numpy as np
import pandas as pd

test = pd.DataFrame

def clean_data(data,testfile,trainfile):

    combine = pd.concat([pd.read_csv(testfile, sep=";"), pd.read_csv(trainfile, sep=";")])

    data["fare"] = data["fare"].fillna(combine["fare"].dropna().median())
    data["age"] = data["age"].fillna(combine["age"].dropna().median())

    data.loc[data["sex"] == "male", "sex"] = 0
    data.loc[data["sex"] == "female", "sex"] = 1

    data["body"] = data["body"].fillna(0)
    data.loc[data["body"] > 0, "body"] = 1
    

    data["embarked"] = data["embarked"].fillna("S")
    data.loc[data["embarked"] == "S", "embarked"] = 0
    data.loc[data["embarked"] == "C", "embarked"] = 1
    data.loc[data["embarked"] == "Q", "embarked"] = 2
    
#neu
    data["boat"] = data["boat"].fillna(0)
    data.loc[data["boat"] == "A", "boat"] = 17
    data.loc[data["boat"] == "B", "boat"] = 18
    data.loc[data["boat"] == "C", "boat"] = 19
    data.loc[data["boat"] == "D", "boat"] = 20
    data.loc[data["boat"] == "C D", "boat"] = 21
    data.loc[data["boat"] == "5 9", "boat"] = 21
    data.loc[data["boat"] == "5 7", "boat"] = 21
    data.loc[data["boat"] == "8 10", "boat"] = 21
    data.loc[data["boat"] == "13 15 B", "boat"] = 21
    data.loc[data["boat"] == "13 15", "boat"] = 21
    data.loc[data["boat"] == "15 16", "boat"] = 21
    

def write_prediction(prediction, testfile, predictionfile):
    print("write prediction")
    test = pd.read_csv(testfile, sep=";")
    PassengerId = np.array(test["id"]).astype(int)
    solution = pd.DataFrame(prediction, PassengerId, columns = ["survived"])
    solution.to_csv(predictionfile, index_label = ["id"], sep=";")
