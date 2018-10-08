from __future__ import print_function
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit, probit, poisson, ols
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from random import random
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr

print("Start..")
X = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
Y_True = [0,    0,    0,    0,    0,    0,    1,    0,    1,    0,    1,    0,    1,    0,    1,    1,    1,    1,    1,    1]

X = [[x] for x in X]
Y = [y for y in Y_True]

def SkRunModel(model,model_name=""):
    print(model_name)
    proba_list = []
    for x in X:
        proba = model.predict_proba(x[0])
        proba_list.append(proba[0])

    for i in range(1,6):
        proba = model.predict_proba(i)
        print(i,"{0:.2f}".format(proba[0][1]))
    print("\n")

    #print("sklearn koeff: ", logit_mod.coef_)
    return proba_list

def getLogLos(Y_True,proba_list,text=""):
    print("\nLog loss value {0}: ".format(text), log_loss(Y_True, proba_list))
    if len(proba_list[0])>1:
        xmod = [x[1] for x in proba_list]
        print("R^2 score {0}:".format(text), r2_score(Y_True,xmod))
        print("Roc Auc score:",roc_auc_score(Y_True,xmod))
        print("Correlation:", pearsonr(Y_True, xmod)[0])
    else:
        print("R^2 score {0}:".format(text), r2_score(Y_True, proba_list))
        print("Roc Auc score:", roc_auc_score(Y_True, proba_list))
        print("Correlation:", pearsonr(Y_True, [x[0] for x in proba_list])[0])

#----------------------------------------
def RunModel(model,model_name=""):
    print(model_name)
    for i in range(1,6):
        predict_value = model.predict(i)
        print(i,"{0:.2f}".format(predict_value[0]))
    print("\n")

    proba_list_sm = []
    for x in X:
        predict_value = model.predict(x[0])
        proba_list_sm.append(predict_value)
    return proba_list_sm

# MODELS

# --- LOGIT & PROBIT REGRESSIONS ---
logit_mod_sk1 = LogisticRegression().fit(X, Y)
logit_mod_sk2 = LogisticRegression(C=1e5).fit(X, Y)#C=1e5

logit_mod_sm = sm.Logit(Y, X).fit()
probit_mod_sm = sm.Probit(Y, X).fit()

# ---  poission regression and 'A simple ordinary least squares model'
logit_mod_poisson = sm.Poisson(Y, X).fit()
probit_mod_ols = sm.OLS(Y, X).fit()

proba_list_sk1 = SkRunModel(logit_mod_sk1,"sklearn1")
proba_list_sk2 = SkRunModel(logit_mod_sk2,"sklearn2")
proba_list_logit = RunModel(logit_mod_sm,"stats models logit")
proba_list_probit = RunModel(probit_mod_sm,"stats models probit")
proba_list_poisson = RunModel(logit_mod_poisson,"Poisson")
proba_list_ols = RunModel(probit_mod_ols,"OLS")

getLogLos(Y_True, proba_list_sk1,"sklearn1")
getLogLos(Y_True, proba_list_sk2,"sklearn2")
getLogLos(Y_True, proba_list_logit,"stats models logit")
getLogLos(Y_True, proba_list_probit,"stats models probit")
getLogLos(Y_True, proba_list_poisson,"Poisson")
getLogLos(Y_True, proba_list_ols,"OLS")

# [1] 0.1162429
# [1] 0.3228511
# [1] 0.6334625
# [1] 0.8623444
# [1] 0.9578194

# [1] 0.1579475
# [1] 0.3458795
# [1] 0.5984901
# [1] 0.8077657
# [1] 0.9221509



