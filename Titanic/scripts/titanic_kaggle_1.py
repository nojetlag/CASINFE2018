from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import json
import sys
import csv
import os
import l_utils as utils

print('Modul overview:')
print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

test = pd.read_csv('titanic/data/titanic3_test.csv',sep=';')
train = pd.read_csv('titanic/data/titanic3_train.csv',sep=';')

print ("\nCleaning up some data")

utils.clean_data(train)
utils.clean_data(test)

print(train.shape)
target = train["survived"].values
features = train[["pclass", "age", "sex", "fare", "sibsp", "parch", "embarked"]].values

gbm = GradientBoostingClassifier(
    learning_rate = 0.005,
    min_samples_split=40,
    min_samples_leaf=1,
    max_features=2,
    max_depth=12,
    n_estimators=1500,
    subsample=0.75,
    random_state=1)
gbm = gbm.fit(features, target)

print(gbm.feature_importances_)
print(gbm.score(features, target))

test_features = test[["pclass", "age", "sex", "fare", "sibsp", "parch", "embarked"]].values
prediction_gbm = gbm.predict(test_features)
utils.write_prediction(prediction_gbm, "titanic/submissions/gmb.csv")