import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.preprocessing as sk

wage = pd.read_csv('Wage.csv', index_col=0)
wage['wage250'] = 0
wage.loc[wage['wage'] > 250, 'wage250'] = 1

poly = sk.PolynomialFeatures(degree=4)
age = poly.fit_transform(wage['age'].values.reshape(-1, 1))

logit = sm.Logit(wage['wage250'], age).fit()

age_range_poly = poly.fit_transform(np.arange(18, 81).reshape(-1, 1))

proba = logit.predict(age_range_poly)
cov = logit.cov_params()
gradient = (proba * (1 - proba) * age_range_poly.T).T
std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
c = 1.96
upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
lower = np.maximum(0, np.minimum(1, proba - std_errors * c))

plt.plot(age_range_poly[:, 1], proba)
plt.plot(age_range_poly[:, 1], lower, color='g')
plt.plot(age_range_poly[:, 1], upper, color='g')
plt.show()


print(wage['wage250'])
print(age)
preds = []
for i in range(1000):
    boot_idx = np.random.choice(len(age), replace=True, size=len(age))
    model = sm.Logit(wage['wage250'].iloc[boot_idx], age[boot_idx]).fit(disp=0)
    preds.append(model.predict(age_range_poly))
p = np.array(preds)
plt.plot(age_range_poly[:, 1], np.percentile(p, 97.5, axis=0))
plt.plot(age_range_poly[:, 1], np.percentile(p, 2.5, axis=0))
plt.show()


xb = np.dot(age_range_poly, logit.params)
std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in age_range_poly])
upper_xb = xb + c * std_errors
lower_xb = xb - c * std_errors
upper = np.exp(upper_xb) / (1 + np.exp(upper_xb))
lower = np.exp(lower_xb) / (1 + np.exp(lower_xb))
plt.plot(age_range_poly[:, 1], upper)
plt.plot(age_range_poly[:, 1], lower)
plt.show()

