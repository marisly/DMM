# The National Institute of Diabetes and Digestive and Kidney Diseases conducted a study on 768 adult female Pima Indians living near Phoenix. The purpose of the study was to investigate factors related to diabetes. The data may be found in the dataset ‘pima.txt’.
#
#  Perform simple graphical and numerical summaries of the data. Can you find any obvious irregularities in the data? If you do, take appropriate steps to correct the problems.
# Fit a model with the result of the diabetes test as the response and all the other variables as predictors. Can you tell whether this model fits the data?
# What is the difference in the odds of testing positive for diabetes for a woman with a BMI at the first quartile compared with a woman at the third quartile, if all other factors are held constant? Give a confidence interval for this difference.
# Do women who test positive have higher diastolic blood pressures? Is the diastolic blood pressure significant in the regression model? Explain the distinction between the two questions and discuss why the answers are only apparently contradictory.
# Predict the outcome for a woman with predictor values 1, 99, 64, 22, 76, 27, 0.25, 25 (same order as in the dataset). Give a confidence interval for your prediction.
# Use different regressions (binary choice) models and quantify the quality of predictions with different metrics. Which solution is the best?
# For confidence interval estimation: (Texts in statistical science) Faraway, Julian James-Extending the linear model with R _ generalized linear, mixed effects and nonparametric regression models-Chapman & Hall_CR (attached)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection

data = pd.read_csv('DATA/pima.txt',delimiter="	")
columns = ['pregnant','glucose','diastolic','triceps','insulin','bmi','diabetes','age','test']

non_zero_columns = ['glucose','diastolic','triceps','bmi','diabetes','age']


# subst zero values for the mean values
for column in non_zero_columns:
    median = data[column].median()
# Substitute it in the BMI column of the
# dataset where values are 0
    data[column] = data[column].replace(
    to_replace=0, value=median)


print(data.describe())

# histogram of the whole data
hist = data.hist(figsize=(15,15))
plt.show()

#correlation check
# corr = data.corr()
# sns.heatmap(corr, annot = True)
# plt.show()


train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

train_set_test = train_set["test"].copy()
train_set = train_set.drop("test", axis=1)

test_set_test = test_set["test"].copy()
test_set = test_set.drop("test", axis=1)


scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)
test_set_scaled = scaler.transform(test_set)

models = [('LR', LogisticRegression()),('KNN', KNeighborsClassifier()),('NB', GaussianNB()),('SVC', SVC()),('LSVC', LinearSVC()),('RFC', RandomForestClassifier()),('DTR', DecisionTreeRegressor())]

seed = 7
results = []
names = []
X = train_set_scaled
Y = train_set_test

for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, Y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (
        name, cv_results.mean(), cv_results.std())
    print(msg)


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




