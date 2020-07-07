# -*python*-
# Python 3.7

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rc('xtick', labelsize=40)
matplotlib.rc('ytick', labelsize=40)

import seaborn as sns
sns.set(style="darkgrid", color_codes=True)
from statsmodels.tsa.stattools import acf, pacf

t = np.linspace(0, 10, 500)
# noise
ys = np.random.normal(0, 5, 500)
# trend
ye = np.exp(t**0.5)

# data
y = ys + ye

# plot
plt.figure(figsize=(16,7))
plt.plot(t, y)
plt.show()

# ACF
lag_acf = acf(y, nlags=300)
# plot ACF
plt.figure(figsize=(16, 7))
plt.plot(lag_acf, marker='+')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(y)), linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(y)), linestyle='--', color='gray')
plt.title('ACF')
plt.xlabel('number of lags')
plt.ylabel('correlation')
plt.tight_layout()
plt.show()

# PACF
lag_pacf = pacf(y, nlags=30, method='ols')
# plot PACF
plt.figure(figsize=(16,7))
plt.plot(lag_pacf, marker='+')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(y)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(y)), linestyle='--', color='gray')
plt.title('PACF')
plt.xlabel('Number of lags')
plt.ylabel('correlation')
plt.tight_layout()
plt.show()

