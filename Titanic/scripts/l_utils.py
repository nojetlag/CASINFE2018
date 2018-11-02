# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:26:12 2018

@author: stuck
"""

import numpy as np
import pandas as pd

test = pd.read_csv('../data/titanic3_test.csv', sep=';')
train = pd.read_csv('../data/titanic3_train.csv', sep=';')

print(test.mean())
print(train.mean())




def clean_data(data):
    test = pd.read_csv('../data/titanic3_test.csv', sep=';')
    train = pd.read_csv('../data/titanic3_train.csv', sep=';')

    combine = pd.concat([train, test])

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
    data["boat"] = data["boat"].fillna(999)
    data.loc[data["boat"] == "A", "boat"] = 995
    data.loc[data["boat"] == "B", "boat"] = 996
    data.loc[data["boat"] == "C", "boat"] = 997
    data.loc[data["boat"] == "D", "boat"] = 998
    data.loc[data["boat"] == "C D", "boat"] = 881
    data.loc[data["boat"] == "5 9", "boat"] = 990
    data.loc[data["boat"] == "5 7", "boat"] = 991
    data.loc[data["boat"] == "8 10", "boat"] = 992
    data.loc[data["boat"] == "13 15 B", "boat"] = 993
    data.loc[data["boat"] == "13 15", "boat"] = 880
    data.loc[data["boat"] == "15 16", "boat"] = 994
    
    
    

def write_prediction(prediction, name):
    PassengerId = np.array(test["id"]).astype(int)
    solution = pd.DataFrame(prediction, PassengerId, columns = ["survived"])
    solution.to_csv(name, index_label = ["id"])
