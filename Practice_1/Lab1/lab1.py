import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


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
model = sm.Logit(y, X).fit()
proba = model.predict(X) # predicted probability

logit = sm.Logit(y,X).fit_regularized(alpha=0.2)
# proba = model.fit_regularized(alpha=0.2)
proba = (logit.predict(X))

# estimate confidence interval for predicted probabilities
cov = logit.cov_params()

gradient = (proba * (1 - proba) * X.T).T # matrix of gradients for each observation

std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
c = 1.96 # multiplier for confidence interval
upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
lower = np.maximum(0, np.minimum(1, proba - std_errors * c))

plt.plot(X, proba)
plt.plot(X, lower, color='g')
plt.plot(X, upper, color='g')
plt.show()



#bootstrap
preds = []
for i in range(1000):
    boot_idx = np.random.choice(len(X), replace=True, size=len(X))
    model = sm.Logit(y[boot_idx], X[boot_idx]).fit(disp=0)
    preds.append(model.predict(X))
p = np.array(preds)
plt.plot(X[:, 1], np.percentile(p, 97.5, axis=0))
plt.plot(X[:, 1], np.percentile(p, 2.5, axis=0))
plt.show()


# wage = pd.read_csv('../../data/Wage.csv', index_col=0)
# wage['wage250'] = 0
# wage.loc[wage['wage'] > 250, 'wage250'] = 1
# http://programmerz.ru/questions/105732/confidence-interval-of-probability-prediction-from-logistic-regression-statsmode-question.html
# poly = Polynomialfeatures(degree=4)
# age = poly.fit_transform(wage['age'].values.reshape(-1, 1))
# age_range_poly = poly.fit_transform(np.arange(18, 81).reshape(-1, 1))
