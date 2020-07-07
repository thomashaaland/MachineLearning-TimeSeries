from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pandas as pd
import seaborn as sns; sns.set(style="whitegrid", color_codes=True)
import seaborn_qqplot as sqp

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

from statsmodels.tsa.stattools import acf, pacf

print("Pandas version: " + pd.__version__)
print("Seaborn version: " + sns.__version__)
print("Seaborn qqplot version: " + sqp.__version__)
print("Numpy version: " + np.__version__)

df = pd.read_csv("../data/INDIAVIX.csv", index_col = "Date")
df.index = pd.to_datetime(df.index)
print(df.head())

"""
df_values = df[["Open", "High", "Low", "Close"]]
df_changes = df[["Change", "%Change"]]

plotValues = sns.lineplot(data=df_values)
plt.title("Stock performance")
plt.show()

plotChanges = sns.lineplot(data=df_changes)
plt.title("Stock change")
plt.show()

plotBoxValues = sns.boxplot(x = "variable", y = "value", data=pd.melt(df_values))
plt.title("Box plot for stock value")
plt.show()

plotBoxChanges = sns.boxplot(x = "variable", y = "value", data=pd.melt(df_changes))
plt.title("Box plot for change")
plt.show()

plotQQValues = sqp.qqplot(df_values, x = "High", y = "Low", height = 4, aspect = 1.6)
plt.title("QQplot for high and low")
plt.show()

plotQQOpenChange = sqp.qqplot(df, x = "Open", y = "Change", height = 4, aspect = 1.6)
plt.title("QQplot for Open and Change")
plt.show()

# The correlation matrix
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11,9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask = mask, cmap = cmap,
            vmax = 1, vmin = 0,
            center = 0, square = True,
            linewidth=.5, cbar_kws=({"shrink": .5}))
plt.title("Correlation matrix")
plt.show()
"""

# Check the data first
print(df["Close"].head())
plt.figure(figsize=(16,7))
plt.plot(df["Close"])
plt.title("Stocks")
plt.show()

lag_acf = acf(df["Close"], nlags = 50, fft=True)
plt.figure(figsize=(16,7))
plt.subplot(121)
plt.bar(range(len(lag_acf)), lag_acf, 0.5)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df.index)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df.index)), linestyle='--', color='gray')
plt.title("Autocorrelation function")
plt.xlabel("Number of lags")
plt.ylabel("Correlation")

lag_pacf = pacf(df["Close"])
plt.subplot(122)
plt.bar(range(len(lag_pacf)), lag_pacf, 0.5)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df.index)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df.index)), linestyle='--', color='gray')
plt.title("PACF")
plt.xlabel("Number of lags")
plt.ylabel("Correlation")
plt.tight_layout()
plt.show()

# Looking at the difference instead
lag_acf_diff = acf(np.log(df["Close"]).diff().loc["2009-03-03":], nlags = 50, fft = True)
plt.figure(figsize=(16,7))
plt.subplot(121)
plt.bar(range(len(lag_acf_diff)), lag_acf_diff, 0.5)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df.index)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df.index)), linestyle='--', color='gray')
plt.title("Autocorrelation function on diff")
plt.xlabel("Number of lags")
plt.ylabel("Correlation")

lag_pacf_diff = pacf(np.log(df["Close"]).diff().loc["2009-03-03":])
plt.subplot(122)
plt.bar(range(len(lag_pacf_diff)), lag_pacf_diff, 0.5)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df.index)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(df.index)), linestyle='--', color='gray')
plt.title("PACF on diff")
plt.xlabel("Number of lags")
plt.ylabel("Correlation")
plt.tight_layout()
plt.show()

