import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# generate data
K = 1
N = 100*K
np.random.seed(1)
x = np.arange(N)/K
y = (x * 0.5 + np.random.normal(size=N,scale=10)>30)
#[print(v) for v in y]

# estimate the model
X = sm.add_constant(x)
model = sm.Logit(y, X).fit()
proba = model.predict(X) # predicted probability


# estimate confidence interval for predicted probabilities
cov = model.cov_params()
gradient = (proba * (1 - proba) * X.T).T # matrix of gradients for each observation
std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
c = 1.96 # multiplier for confidence interval
upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
lower = np.maximum(0, np.minimum(1, proba - std_errors * c))

plt.plot(x, proba)
plt.plot(x, lower, color='g')
plt.plot(x, upper, color='g')
plt.show()