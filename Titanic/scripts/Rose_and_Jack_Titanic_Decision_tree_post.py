# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 09:45 2018
Team Rose & Jack 
"""

import l_utils as utils
import l_plot as plot
import pandas as pd
from sklearn import tree, model_selection
from pathlib import Path

#prepare all used files to be os independent
project_path = Path(__file__).parents[1]
testfile = Path.joinpath(project_path, "data/titanic3_test.csv")
trainfile = Path.joinpath(project_path, "data/titanic3_train.csv")
plotfile = Path.joinpath(project_path, "presentation/Rose_and_Jack_post.out")
submissionfile = Path.joinpath(project_path, "submissions/Rose_and_Jack_post.csv")

#load data files for test and train 
test = pd.read_csv(testfile, sep=";")
train = pd.read_csv(trainfile, sep=";")

#plot a first overview of the data
print("\nFirst look at the Data")
plot.ts_mapplot(dataset=train, grid_rv='boat', grid_cv='pclass', grid_hue='survived', map_v='age')
plot.ts_surv_prop(dataset=train, xv='age', yv='survived', colv='sex', huev='sex')

#to have better results, clean up the data
print ("\nCleaning up the data")
utils.clean_data(train,testfile,trainfile)
utils.clean_data(test,testfile,trainfile)

print ("\nExtracting target and features")

print(train.shape)
target = train["survived"].values
feature_names = ["pclass", "age", "sex", "fare", "sibsp", "parch", "embarked", "boat"]
features = train[feature_names].values
decision_tree = tree.DecisionTreeClassifier(
    max_depth = 3,
    min_samples_split = 2,
    random_state = 1)
decision_tree = decision_tree.fit(features, target)

print(decision_tree.feature_importances_)
print(decision_tree.score(features, target))
plotfilehandle = open(plotfile, "w")
tree.export_graphviz(decision_tree, feature_names=feature_names, out_file=plotfilehandle, rounded = True, proportion = False, precision = 2, filled = True)

scores = model_selection.cross_val_score(decision_tree, features, target, scoring='accuracy', cv=10)
print (scores)
print (scores.mean())

print ("\nWrite new predicition")

test_features = test[["pclass", "age", "sex", "fare", "sibsp", "parch", "embarked", "boat"]].values
prediction = decision_tree.predict(test_features)
utils.write_prediction(prediction, testfile, submissionfile)
# Resultat -> 0.973158