import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import sklearn as skl
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

print("Scikit Learn version: " + skl.__version__)
print(__doc__)


df = pd.read_csv("../data/INDIAVIX.csv", index_col = "Date")
df.index.freq = 'd'

train = np.log(df["Close"]).dropna().loc[:"2020-01-01"]
test = np.log(df["Close"]).dropna().loc["2020-01-02":]

plt.figure(figsize=(17,10))
plt.plot(train, 'b-', test, 'g-')
plt.xticks(df.index.values[::345])
plt.show()

# Perform on training set
plot_acf(train.diff().dropna())
plt.show()

plot_pacf(train.diff().dropna())
plt.show()

# Make model
model = ARIMA(train, order=(1, 1, 1), freq = 'D')
model_fit = model.fit(disp=0)
print(model_fit.summary())

# plot residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

#history = [x for x in train]
#predictions = []

history = train
predictions = pd.DataFrame(columns = ["Close"])

for t in test.index:
    print(t)
#for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1), freq = 'D')
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    print(output)
    yhat = output[0]
    predictions.loc[t] = yhat
    #predictions.append(yhat) # Turn into DF with t as time index
    obs = test[t]
    history[t] = obs
    #history.append(obs)
    print("predicted={:.3f}, expected={:.3f}".format(yhat[0], obs))
error = mean_squared_error(test, predictions)
print("Test MSE: %.3f" % error)

print("TEST: ", test, predictions)
plt.figure(figsize=(17,10))
plt.plot(np.exp(test), 'r-', np.exp(predictions), 'b-')
plt.xticks(test.index.values[::10])
plt.show()

"""
df.index = pd.to_datetime(df.index)

Y = df["Close"] # target
X = df.index

# Design matrix:
columns = 1000
XArr = np.array((X - X[0]).days).reshape(-1,1)
Xm = np.zeros((X.shape[0], columns))
print(Xm.shape, XArr.shape)
for i in range(columns):
    Xm[(columns - 1 - i):,i] = XArr[:(X.shape[0] - (columns - 1) + i),0]

Xtrain, Xtest = train_test_split(X, shuffle = False)

train = sns.lineplot(data = Y[Xtrain])
test = sns.lineplot(data = Y[Xtest])

plt.legend(["Train", "Test"])
plt.title("Train set and test set")
plt.show()

# Data engineering
Xengineered = pd.DataFrame({#'Day': (df.index - df.index[0]).days,
                            'Weekday': df.index.weekday,
                            'Week': df.index.week,
                            'Month': df.index.month,
                            'Year': df.index.year,
                            'Day of year': df.index.dayofyear,
                            'Day in month': df.index.day,
                            #'is month start': df.index.day == 1,
                            #'is month end': df.index.day == df.index.daysinmonth,
                            #'start of year': df.index.dayofyear == 1,
                            #'end of year': df.index.strftime('%d-%m') == '31-12',
                            'quarter': df.index.quarter
}, index = df.index)


# Make the pipeline
YtrainArr = Y[Xtrain].values.astype('float')
YtestArr = Y[Xtest].values.astype('float')
XtrainArr = np.array((Xtrain - Xtrain[0]).days).reshape(-1,1)
XtestArr = np.array((Xtest - Xtrain[0]).days).reshape(-1,1)
Xmtrain = Xm[:XtrainArr.shape[0]]
Xmtest = Xm[XtrainArr.shape[0] + 1:]
#XArr = np.array((X - X[0]).days).reshape(-1,1)

print(XtrainArr.shape, YtrainArr.shape)
print(XtrainArr.dtype)
print(YtrainArr.dtype)

print(XtestArr.shape, YtestArr.shape)
print(XtestArr.dtype)
print(YtestArr.dtype)

# Linear regression using simply the dates
reg = LinearRegression()
reg.fit(XtrainArr, YtrainArr)
prediction = reg.predict(XArr)

dfLinePred = pd.concat([Y[Xtrain], Y[Xtest], pd.DataFrame(prediction, index = X)], axis=1, sort=False)
linePred = sns.lineplot(data = dfLinePred)

plt.legend(["Data", "Prediction"])
plt.title("Simple regression")
plt.show()

# Using linear regression with design matrix
reg = LinearRegression()
reg.fit(Xmtrain, YtrainArr)
prediction = reg.predict(Xm)

dfLinePred = pd.concat([Y[Xtrain], Y[Xtest], pd.DataFrame(prediction, index = X)], axis=1, sort=False)
linePred = sns.lineplot(data = dfLinePred)

plt.legend(["Data", "Prediction"])
plt.title("Regression using time shift")
plt.show()

# Using linear regression with feature engineering
reg = LinearRegression()
reg.fit(Xengineered.loc[Xtrain].values, YtrainArr)
prediction = reg.predict(Xengineered.values)

dfLinePred = pd.concat([Y[Xtrain], Y[Xtest], pd.DataFrame(prediction, index = X)], axis=1, sort=False)
linePred = sns.lineplot(data = dfLinePred)

plt.legend(["Data", "Prediction"])
plt.title("Regression using feature engineering")
plt.show()

# Decision trees

clf = RandomForestRegressor(n_estimators=100, max_depth=20)
clf.fit(Xengineered.loc[Xtrain].values, YtrainArr)
prediction = clf.predict(Xengineered.values)

dfLinePred = pd.concat([Y[Xtrain], Y[Xtest], pd.DataFrame(prediction, index = X)], axis=1, sort=False)
linePred = sns.lineplot(data = dfLinePred)

plt.legend(["Train", "Test", "Prediction"])
plt.title("Decision trees using feature engineering")
plt.show()

"""
