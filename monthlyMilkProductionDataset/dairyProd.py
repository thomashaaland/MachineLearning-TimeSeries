"""
Fitting arima model to milk production and making a forecast
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import r2_score

# Set the style to be used in the plots
sns.set(style="darkgrid")

# Import the data
df = pd.read_csv('./CADairyProduction.txt')

# Need to make datetime index from the month and year columns
index = pd.to_datetime(df["Month"] + df["Year"].astype(str), format="%b%Y")
df["Date"] = index
df = df[df.columns[2:]]

# Give the date column as the index and throw away the date column
df.index = df["Date"]
df = df.drop(columns = ["Date"], axis = 1)

# Seperate the cheese, icecream and milk df
cotagecheese_df = df["Cotagecheese.Prod"]
icecream_df = df["Icecream.Prod"]
milk_df = df["Milk.Prod"]

# Print sample
print(milk_df.head())

# plot the milk production
milk_df.plot(title = "Milk production")
plt.show()

decomposition = seasonal_decompose(milk_df, period = 12)
decomposition.plot()
plt.show()

for i in range(12):
    plt.plot(milk_df[milk_df.index.month == i])
plt.show()

fig, (ax1, ax2) = plt.subplots(2)
plot_acf(milk_df, lags = 50, ax = ax1)
plot_pacf(milk_df, lags = 50, ax = ax2)
plt.tight_layout()
plt.show()

dftest = adfuller(milk_df)
dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value',
                                           '#Lags used',
                                           'Number of Observations Used'])

for key, value in dftest[4].items():
    dfoutput['Critical Value (%s) '%key] = value
print(dfoutput)

milk_df_diff = milk_df.diff().diff(12).dropna()
milk_df_diff.plot()
plt.show()


fig, (ax1, ax2) = plt.subplots(2)
plot_acf(milk_df_diff, lags = 50, ax = ax1)
plot_pacf(milk_df_diff, lags = 50, ax = ax2)
plt.tight_layout()
plt.show()

dftest = adfuller(milk_df_diff)
dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value',
                                           '#Lags used',
                                           'Number of Observations Used'])

for key, value in dftest[4].items():
    dfoutput['Critical Value (%s) '%key] = value
print(dfoutput)

train = milk_df[:int(len(milk_df) * 0.95)]
test = milk_df.drop(train.index)

model = pm.auto_arima(train, d=1, D=1,
                      m = 12, trend = 'c', seasonal = True,
                      start_p=0, start_q=0, max_order=6, test='adf',
                      stepwise=True, trace=True)


print("Best model: ", model.order, "x", model.seasonal_order, "Intercept: ",
      model.with_intercept)
model = SARIMAX(train, order = model.order,
                seasonal_order = model.seasonal_order,
                with_intercept = model.with_intercept) #order = (1,1,1), seasonal_order=(0,1,2,12))
results = model.fit()
results.summary()

results.plot_diagnostics()
plt.tight_layout()
plt.show()

forecast_object = results.get_forecast(steps=len(test))
mean = forecast_object.predicted_mean
conf_int = forecast_object.conf_int()
dates = mean.index

plt.figure()
milk_df.plot(legend="Real")
mean.plot(legend="Predicted")
plt.fill_between(conf_int.index, conf_int[conf_int.columns[0]],
                 conf_int[conf_int.columns[1]], alpha = 0.5)
plt.show()

#from sklearn.utils import check_array
def mean_absolute_percentage_error(y_true, y_pred):
    #y_true, y_pred = check_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(r2_score(test, mean))
print(mean_absolute_percentage_error(test, mean))
