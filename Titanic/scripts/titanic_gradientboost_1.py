from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import warnings
import sklearn
import sys
import l_utils as utils

#Prediction Score: 0.83908

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
features = train[["pclass", "age", "sex", "fare", "sibsp", "parch", "embarked"]].values

gbc = GradientBoostingClassifier(
    learning_rate = 0.005,
    min_samples_split=40,
    min_samples_leaf=1,
    max_features=7,
    max_depth=12,
    n_estimators=1500,
    subsample=0.75,
    random_state=1)
gbc = gbc.fit(features, target)

print(gbc.feature_importances_)
print(gbc.score(features, target))

test_features = test[["pclass", "age", "sex", "fare", "sibsp", "parch", "embarked"]].values
prediction_gbc = gbc.predict(test_features)
utils.write_prediction(prediction_gbc, "titanic/submissions/submission_gradientboost_1.csv")