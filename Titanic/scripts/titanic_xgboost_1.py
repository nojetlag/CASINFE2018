from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import warnings
import sklearn
import sys
import os
import l_utils as utils
from xgboost import XGBClassifier

#Prediction Score: 0.850574

print('Modul overview:')
print('sklearn: {}'.format(sklearn.__version__))
print('pandas: {}'.format(pd.__version__))
print('Python: {}'.format(sys.version))

test = pd.read_csv('titanic/data/titanic3_test.csv',sep=';')
train = pd.read_csv('titanic/data/titanic3_train.csv',sep=';')

print ("\nCleaning up some data")

utils.clean_data(train)
utils.clean_data(test)

print(train.shape)
target = train["survived"].values
features = train[["pclass", "age", "sex", "fare", "sibsp", "parch", "embarked","boat"]].values

# XGBoost 

xgbc = XGBClassifier(
        learning_rate=0.005,
        max_depth=7,
        n_estimators=1500,
        subsample=0.75,
        random_state=1
)
xgbc.fit(features,target)

print("importances", xgbc.feature_importances_)
print("score", xgbc.score(features, target))

test_features = test[["pclass", "age", "sex", "fare", "sibsp", "parch", "embarked","boat"]].values
prediction_xgbc = xgbc.predict(test_features)
utils.write_prediction(prediction_xgbc, "titanic/submissions/submission_xgbc_2.csv")