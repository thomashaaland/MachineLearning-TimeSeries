#-*Python*-
"""
From Kaggle. Inspired by user pratik1120
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas_profiling import ProfileReport

data = pd.read_csv('../data/NIFTY 50.csv')
print(data.head())
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

#report = ProfileReport(data)
#report.to_file(output_file='dataExploreNifty50.html')

plt.figure(figsize=(10,7))
plt.plot(data['Date'], data['Close'])
plt.xlabel('Years')
plt.ylabel('Closing values')
plt.title('Closing values vs Years')
plt.show()

peaks = data[['year', 'High']].copy()
peaks['max_high'] = peaks.groupby('year')['High'].transform('max')
peaks.drop('High', axis = 1, inplace=True)
peaks = peaks.drop_duplicates()
peaks = peaks.sort_values('max_high', ascending=False)
peaks = peaks.head()

fig = plt.figure(figsize = (10, 7))
plt.pie(peaks['max_high'], labels = peaks['year'], autopct='%1.1f%%',
        shadow=True)
centre_circle = plt.Circle((0,0), 0.45, color='black', fc='white',
                           linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.show()

sns.kdeplot(data=data['High'], shade=True)
plt.title('Distribution of highest values')
plt.show()

sns.kdeplot(data=data['Volume'], shade = True)
plt.title('Transaction volume')
plt.show()

top_5_genres = [1, 3, 5, 7, 9, 11]
perc = data[['year', 'month', 'Volume']].copy()
perc['new_volume'] = perc.groupby([perc.month, perc.year])['Volume'].transform('mean')
perc.drop('Volume', axis=1, inplace=True)
perc = perc[perc.year<2020]
perc = perc.drop_duplicates()
perc = perc.loc[perc['month'].isin(top_5_genres)]
perc = perc.sort_values('year')

fig = px.bar(perc, x = 'month', y = 'new_volume', animation_frame='year',
             animation_group='month', color='month', hover_name='month',
             range_y = [perc['new_volume'].min(), perc['new_volume'].max()])
fig.update_layout(showlegend=False)
fig.show()

sns.scatterplot(data=data, x = 'Volume', y = 'Close', hue = 'year')
plt.title('Relation of volume to stock prices')
plt.show()

sns.kdeplot(data=data['Turnover'], shade=True)
plt.title('Turnover Distribution')
plt.show()

sns.scatterplot(data = data, x = 'Turnover', y = 'Volume')
plt.title('Relation of Turnover to Volume')
plt.show()

turn = data.loc[:, ['year', 'month', 'Turnover']]
turn['monthly_turnover'] = turn.groupby([turn.year, turn.month])['Turnover'].transform('mean')
turn.drop('Turnover', axis=1, inplace=True)
turn = turn.drop_duplicates()
fig = px.scatter(turn, x = 'month', y = 'monthly_turnover', animation_frame='year',
                 animation_group='month', color = 'month', size_max=1000, \
                 range_y = [turn['monthly_turnover'].min(), \
                            turn['monthly_turnover'].max()])
fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()

# Price to earning ratio
sns.kdeplot(data=data['P/E'], shade = True)
plt.title('P/E Distribution')
plt.show()

# Relation of P/E ratio to Close values
sns.scatterplot(data=data, x = 'P/E', y = 'Close')
plt.show()

# Yearly graph of P/E
df = data.loc[:, ['year', 'P/E']]
df['meanPE'] = df.groupby('year')['P/E'].transform('mean')
df['maxPE'] = df.groupby('year')['P/E'].transform('max')
df['minPE'] = df.groupby('year')['P/E'].transform('min')
df.drop('P/E', axis = 1, inplace = True)
df = df.drop_duplicates().sort_values('year')

plt.figure(figsize=(10,7))
plt.plot(df['year'], df['meanPE'])
plt.plot(df['year'], df['maxPE'])
plt.plot(df['year'], df['minPE'])
plt.xlabel('P/E values')
plt.ylabel('Years')
plt.title('P/E values vs Years')
plt.legend(['mean', 'max', 'min'])
plt.show()

# Distribution of P/B
sns.kdeplot(data=data['P/B'], shade=True)
plt.title('P/B Distribution')
plt.show()

# Relation of P/B to Close values
sns.scatterplot(data=data, x = 'P/B', y = 'Close', hue = 'year')
plt.title('Relation of P/B to Close')
plt.show()

# Yearly graph of P/B
df = data.loc[:, ['year', 'P/B']]
df['meanPB'] = df.groupby('year')['P/B'].transform('mean')
df['maxPB'] = df.groupby('year')['P/B'].transform('max')
df['minPB'] = df.groupby('year')['P/B'].transform('min')
df.drop('P/B', axis = 1, inplace = True)
df = df.drop_duplicates().sort_values('year')

plt.figure(figsize=(10,7))
plt.plot(df['year'], df['meanPB'])
plt.plot(df['year'], df['maxPB'])
plt.plot(df['year'], df['minPB'])
plt.xlabel('Year')
plt.ylabel('P/B values')
plt.legend(['mean', 'max', 'min'])
plt.title('P/B values vs Years')
plt.show()

# Dividend yield
sns.kdeplot(data=data['Div Yield'], shade = True)
plt.title('Div Yield Distribution')
plt.show()

sns.scatterplot(data=data, x = 'Div Yield', y = 'Close', hue = 'year')
plt.title('Relation of Div Yield to Close')
plt.show()

df = data.loc[:, ['year', 'Div Yield']]
df['meanDiv'] = df.groupby('year')['Div Yield'].transform('mean')
df['maxDiv'] = df.groupby('year')['Div Yield'].transform('max')
df['minDiv'] = df.groupby('year')['Div Yield'].transform('min')
df.drop('Div Yield', axis = 1, inplace = True)
df = df.drop_duplicates().sort_values('year')

plt.figure(figsize=(10,7))
plt.plot(df['year'], df['meanDiv'])
plt.plot(df['year'], df['maxDiv'])
plt.plot(df['year'], df['minDiv'])
plt.xlabel('Years')
plt.ylabel('Div Yield values')
plt.title('Div Yield values vs Years')
plt.legend(['mean', 'max', 'min'])
plt.show()

df = data.loc[:, ['year', 'P/B', 'P/E', 'Div Yield']]
df[['meanPE', 'meanPB', 'meanDiv']] = df.groupby('year')[['P/B', 'P/E', 'Div Yield']].transform('mean')
df[['maxPE', 'maxPB', 'maxDiv']] = df.groupby('year')[['P/B', 'P/E', 'Div Yield']].transform('max')
df[['minPE', 'minPB', 'minDiv']] = df.groupby('year')[['P/B', 'P/E', 'Div Yield']].transform('min')

plt.figure(figsize=(10,7))
plt.plot(df['year'], df['meanPE'], color='b')
plt.plot(df['year'], df['meanPB'], color='r')
plt.plot(df['year'], df['meanDiv'], color='g')
plt.fill_between(df['year'], df['minPE'], df['maxPE'], alpha=0.25, color='b')
plt.fill_between(df['year'], df['minPB'], df['maxPB'], alpha=0.25, color='r')
plt.fill_between(df['year'], df['minDiv'], df['maxDiv'], alpha=0.25, color='g')
plt.xlabel('Years')
plt.legend(['meanPE', 'meanPB', 'meanDiv'])
plt.show()
