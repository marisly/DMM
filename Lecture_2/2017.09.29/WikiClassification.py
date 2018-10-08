from __future__ import print_function
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from random import random
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

print("Start..")
X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
Y_True = [0,    0,    0,    0,    0,    0,    1,    0,    1,    0,    1,    0,    1,    0,    1,    1,    1,    1,    1,   1]

X = [[x] for x in X]
Y = [y for y in Y_True]

def RunModel(model,model_name=""):
    print(model_name)
    for i in range(1,6):
        predict_value = model.predict(i)
        print(i,predict_value)
    print("\n")

    proba_list_sm = []
    for x in X:
        predict_value = model.predict(x[0])
        proba_list_sm.append(predict_value)
    return proba_list_sm


def getMetrics(Y_True,proba_list,text=""):
    print("\n{0}".format(text))
    print("R^2 score {0}:".format(text), r2_score(Y_True, proba_list))
    print("Roc Auc score:", roc_auc_score(Y_True, proba_list))
    print("Matthews corrcoef", matthews_corrcoef(Y_True, [x[0] for x  in proba_list]))

rand_forest = RandomForestClassifier().fit(X, Y)
ada_boost = AdaBoostClassifier().fit(X, Y)
kneighb = KNeighborsClassifier().fit(X, Y)
dtree = DecisionTreeClassifier().fit(X, Y)

proba_rf = RunModel(rand_forest,"RandomForest")
proba_ab = RunModel(ada_boost,"AdaBoost")
proba_kn = RunModel(kneighb,"KNeighbors")
proba_dt = RunModel(dtree,"dtree")

getMetrics(Y_True, proba_rf,"RandomForest")
getMetrics(Y_True, proba_ab,"AdaBoost")
getMetrics(Y_True, proba_kn,"KNeighbors")
getMetrics(Y_True, proba_dt,"DecisionTree")