'''
Author: Scott Underwood
Date: 03/18/2023

This script contains code to analyze the policies and performance 
of the algorithm in plant.py, comparing it to a baseline
'''
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('policies.csv')
# curtail if revenue would be negative
df['revenue_no_storage'] = np.maximum(df['Power (MW)']/4*df['LMP'],0)
df['cumulative_revenue_no_storage'] = df['revenue_no_storage'].cumsum()

# calculate revenue with storage
df['revenue_with_storage'] = 0
for index, row in df.iterrows():
    wind_energy = row['Power (MW)']/4
    lmp = row['LMP']
    if row['policy'] == 0:
        df.loc[index, 'revenue_with_storage'] = (wind_energy+10)*lmp
    elif row['policy'] == 1:
        df.loc[index, 'revenue_with_storage'] = wind_energy*lmp
    elif row['policy'] == 2:
        df.loc[index, 'revenue_with_storage'] = (wind_energy-10)*lmp
df['cumulative_revenue_with_storage'] = df['revenue_with_storage'].cumsum()

# plot cumulative revenues
dates = pd.to_datetime(df['Date'])
plt.figure(0)
plt.plot(dates,df['cumulative_revenue_no_storage']/1e7, label = 'No storage')
plt.plot(dates,df['cumulative_revenue_with_storage']/1e7, label = 'Storage')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Cumulative Revenue (Million $)')
plt.savefig('revenue.jpg')

# plot time step revenues for first few hours
plt.figure(1)
plt.plot(dates[0:30],df['revenue_no_storage'][0:30], label='No storage')
plt.plot(dates[0:30],df['revenue_with_storage'][0:30], label='Storage')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Timestep Revenue ($)')
plt.savefig('hourly_rev.jpg')

# plot pie chart of actions
action_counts = df['policy'].value_counts()
plt.figure(2)
plt.pie(action_counts, labels = ['Hold', 'Charge', 'Discharge'])
plt.savefig('action_pie.jpg')
