#print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from random import random
import statsmodels.api as sm
import statsmodels
import statsmodels.discrete.discrete_model as sm1

N = 1000

def noise(K=10):
    return (random()-0.5)/K

# class 1 - for Political Party "Conservative"
# class 2 - for Political Party "Liberal"

# 0 - for Party 1, 1 - for Party 2
def voteForParty1(p = 70):
    p = p /100.0
    return (not p<random())

def getSklearnProb(X,Y,value,show=False):
    logit_mod = LogisticRegression().fit(X, Y)
    if show:
        print("sklearn koeff: ", logit_mod.coef_)
    prob = logit_mod.predict_proba([value])
    return prob[0][1]*100

def getStatsmodelProb(X,Y,value,show=False):
    logit_mod_sm = sm.Logit(Y, X).fit()
    if show:
        print("statsmodels koeff: ", logit_mod_sm.params)    
    prob_sm = logit_mod_sm.predict(value)
    rez = logit_mod_sm.predict(value)
    print("rez==",rez)

    return prob_sm[0]*100

def CompareProba(exog,endog,value,mdP,header = None):
    if header is not None:
        print(header)
    p1 = getSklearnProb(exog, endog, value, True)
    p2 = getStatsmodelProb(exog, endog, value, True)
    print("Model probability (sklearn) vote for Party 1: {0:.2f}, Error: {1:.2f}".format(p1, abs(p1 - mdP)))
    print("Model probability (statsmodels) vote for Party 1: {0:.2f}, Error: {1:.2f}".format(p2, abs(p2 - mdP)))

def GenerateGroup(value,index = None,index1=None,K=10):
    rez = []
    for i in range(int(N)):
        value1 = value[:]
        if index is not None:
            value1[index] = value1[index] + noise(K)
        if index1 is not None:
            value1[index1] = value1[index1] + noise(K)
        rez.append(value1)
    return rez

def GenerateDecision(mdP):
    return [ voteForParty1(mdP) for i in range(int(N)) ]

def OneVarTwoTypes():
    mdP_men = 20
    mdP_woman = 90
    # 1 - men, 2 - woman
    men = GenerateGroup([1])
    woman = GenerateGroup([2])
    menDecision = GenerateDecision(mdP_men)
    womanDecision = GenerateDecision(mdP_woman)

    exog = men + woman                      # exogenous
    endog = menDecision + womanDecision     # endogenous

    print('exogenous', exog)
    print('endogenous', endog)

    value = [1]
    CompareProba(exog, endog, value, mdP_men,"\nMen")
    value = [2]
    CompareProba(exog, endog, value, mdP_woman,"\nWoman")

OneVarTwoTypes()