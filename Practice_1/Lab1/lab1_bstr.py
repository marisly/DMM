import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd

# # generate data
# K = 1
# N = 100*K
# np.random.seed(1)
# x = np.arange(N)/K
# y = (x * 0.5 + np.random.normal(size=N,scale=10)>30)
# #[print(v) for v in y]


print("Start..")
x = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
y = [0,    0,    0,    0,    0,    0,    1,    0,    1,    0,    1,    0,    1,    0,    1,    1,    1,    1,    1,    1]

print(x,y)

X_test = [0, 1, 2, 3, 4, 5]


def sklearn(x,y):

    X = sm.add_constant(x)
    model = linear_model.LogisticRegression()
    proba = model.fit(X,y) # predicted probability
    # print(proba)
    x_test = sm.add_constant(X_test)
    Y_sklearn = proba.predict_proba(x_test)
    return list(x[1] for x in Y_sklearn),proba,model


def statsmod(x,y):
    # estimate the model
    X = sm.add_constant(x)
    model = sm.Logit(y, X).fit()
    x_test = sm.add_constant(X_test)
    proba = model.predict(x_test)  # predicted probability
    return proba
    # estimate confidence interval for predicted probabilities
    cov = model.cov_params()
    gradient = (proba * (1 - proba) * X.T).T  # matrix of gradients for each observation
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
    c = 1.96  # multiplier for confidence interval
    upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
    lower = np.maximum(0, np.minimum(1, proba - std_errors * c))

    plt.plot(x, proba)
    plt.plot(x, lower, color='g')
    plt.plot(x, upper, color='g')
    plt.show()

# print(statsmod(x,y))

X = sm.add_constant(x)

logit = sm.Logit(y,X).fit_regularized(disp=False)
proba = (logit.predict(X))

# estimate confidence interval for predicted probabilities
cov = logit.cov_params()
gradient = (proba * (1 - proba) * X.T).T # matrix of gradients for each observation
std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])



c = 1.0 # multiplier for confidence interval
upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
lower = np.maximum(0, np.minimum(1, proba - std_errors * c))

# plt.plot(x, proba, label ='Probability delta method',alpha=0.25)
# plt.plot(x, lower, color='r',label='lower 95% CI delta method',alpha=0.25)
# plt.plot(x, upper, color='r', label = 'upper 95% CI delta method',alpha=0.25)
# plt.legend()
# plt.savefig("logit.png")
# plt.show()


#bootstrap
#Generate large sample from Exam task using uniform distribution generator and logit regression
print("Bootstrap")
rnd_hrs = np.random.uniform(0,5,1000)
rnd_hrs = np.array(rnd_hrs)

# rnd_X = sm.add_constant(rnd_hrs)
# rnd_proba = (logit.predict(rnd_X))

preds = []

for i in range(10000):
    try:

        pred = logit.predict(X)
        new_y = [1 if np.random.random() < p else 0 for p in pred]
        new_logit = sm.Logit(new_y, X).fit_regularized(disp=False)
        proba_new = new_logit.predict(X)
        # print(proba_new)
        preds.append(proba_new)

    except:
        #         print(proba_new)
        pass

print(len(preds))
df_preds = pd.DataFrame(preds)
# print(df_preds)

q_0_75 = df_preds.quantile(q=0.16, axis=0)
q_0_975 = df_preds.quantile(q=0.84, axis=0)

# less = p[:10]
print(q_0_75)

plt.plot(X[:, 1], q_0_75, label="2.5%")
plt.plot(X[:, 1], df_preds.mean(), label="probability")
plt.plot(X[:, 1], q_0_975, label="97.5")
# plt.plot(x, proba, label ='Probability delta method',alpha=0.25)
plt.plot(x, lower, color='r',label='lower 95% CI delta method',alpha=0.25)
plt.plot(x, upper, color='r', label = 'upper 95% CI delta method',alpha=0.25)
plt.legend()


# plt.plot(x, mean, color='r',label='bootstrap mean')
# plt.plot(upper_bootstrap, color='b',label='bootstrap high')
# plt.plot(lower_bootstrap, color='b',label='bootstrap low')

plt.show()




# cov = model.cov_params()
# gradient = (pred * (1 - pred) * X.T).T
# std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
# c = 1.96  # multiplier for confidence interval
# upper = np.maximum(0, np.minimum(1, pred + std_errors * c))
# lower = np.maximum(0, np.minimum(1, pred - std_errors * c))
# mean = (upper + lower)/2
# print(mean)
#
# plt.plot(x, np.mean(p[:, 1, :], 0),color = 'r', label='mean bootstrap')
# plt.plot(x, lower, color='b',label='bootstrap high')
# plt.plot(x, upper, color='b', label = 'bootstrap low%')
#
#
# print(lower)
# print(upper)
# for i in range[len(preds)]:
#
# print(p[:,0])
# p = np.nan_to_num()
#
# plt.plot(x, np.mean(p[:, 1, :], 0),color = 'r', label='mean bootstrap')
# plt.plot(x, np.mean(p[:, 0, :], 0),color= 'g',label='97.5%')
# plt.plot(x, np.mean(p[:, 2, :], 0),color = 'g',label='2.5%')
# plt.legend()
# plt.show()
#
# print((preds[0]))
#

# p = np.array(preds)
#
# plt.plot(x, proba, label ='Probability')

# plt.plot(x, np.percentile(p[:, 1, :], 97.5, axis=0),color='g',label='97.5%')
# plt.plot(x, np.percentile(p[:, 1, :], 2.5, axis=0),color='r',label='2.5%')
# plt.legend()
# plt.savefig("bootstrap.png")
# plt.show()


