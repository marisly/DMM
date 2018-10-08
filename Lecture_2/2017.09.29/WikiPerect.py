from __future__ import print_function
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit, probit, poisson, ols
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from random import random
from random import randint
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr

print("Start..")
X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
Y_True = [0,    0,    0,    0,    0,    0,    1,    0,    1,    0,    1,    0,    1,    0,    1,    1,    1,    1,    1,    1]
randVals = [ random() for i in range(len(Y_True))]

N = 10000
randVals1 = [ randint(0,1) for i in range(N)]
randVals2 = [ randint(0,1) for i in range(N)]

def getLogLos(Y_True,proba_list,text=""):
    print(text)
    print("Log loss value: ".format(text), log_loss(Y_True, proba_list))
    print("R^2 score:", r2_score(Y_True, proba_list))
    print("Roc Auc score:", roc_auc_score(Y_True, proba_list))
    print("Correlation:", pearsonr(Y_True, proba_list)[0])

getLogLos(Y_True,Y_True,"PERFECT")
print("\n")
#getLogLos(Y_True,randVals,"RANDOM")
getLogLos(randVals2,randVals1,"RANDOM")