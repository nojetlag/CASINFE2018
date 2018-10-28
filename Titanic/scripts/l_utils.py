# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:26:12 2018

@author: stuck
"""

import numpy as np
import pandas as pd

test = pd.read_csv('titanic/data/titanic3_test.csv', sep=';')

def clean_data(data):
    data["fare"] = data["fare"].fillna(data["fare"].dropna().median())
    data["age"] = data["age"].fillna(data["age"].dropna().median())

    data.loc[data["sex"] == "male", "sex"] = 0
    data.loc[data["sex"] == "female", "sex"] = 1

    data["embarked"] = data["embarked"].fillna("S")
    data.loc[data["embarked"] == "S", "embarked"] = 0
    data.loc[data["embarked"] == "C", "embarked"] = 1
    data.loc[data["embarked"] == "Q", "embarked"] = 2

    data["boat"] = data["boat"].fillna(999)
    data.loc[data["boat"] == "A", "boat"] = 995
    data.loc[data["boat"] == "B", "boat"] = 996
    data.loc[data["boat"] == "C", "boat"] = 997
    data.loc[data["boat"] == "D", "boat"] = 998
    data.loc[data["boat"] == "C D", "boat"] = 998
    data.loc[data["boat"] == "5 9", "boat"] = 990
    data.loc[data["boat"] == "5 7", "boat"] = 991
    data.loc[data["boat"] == "8 10", "boat"] = 992
    data.loc[data["boat"] == "13 15 B", "boat"] = 993
    data.loc[data["boat"] == "13 15", "boat"] = 994
    data.loc[data["boat"] == "15 16", "boat"] = 994



def write_prediction(prediction, name):
    PassengerId = np.array(test["id"]).astype(int)
    solution = pd.DataFrame(prediction, PassengerId, columns = ["survived"])
    solution.to_csv(name, index_label = ["id"], sep=';')
