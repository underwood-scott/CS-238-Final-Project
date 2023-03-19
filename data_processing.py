'''
Author: Scott Underwood
Date: 02/28/2023

This script takes in ERCOT time series data for generation by 
fuel type and LMP for calendar year 2022. It then cleans the 
data, discretizes LMP and power into bins, and outputs the 
data to be used in plant.py.
'''
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd


# read in lmp data
lmp_sheets = pd.read_excel("ERCOT_LMP_2022.xlsx", sheet_name = None)
lmp = pd.DataFrame(columns = ['Date', 'LMP'])

# add each month to dataframe
for month in lmp_sheets.values():
    month = month[(month['Settlement Point Name'] == 'HB_WEST') & (month['Repeated Hour Flag'] == 'N')] 
    month['Delivery Date'] = pd.to_datetime(month['Delivery Date'])

    # get datetime for each row
    for index, row in month.iterrows():
        time = datetime.time(row['Delivery Hour']-1, (row['Delivery Interval']-1)*15)
        date_time = datetime.datetime.combine(row['Delivery Date'], time) + timedelta(minutes = 15)
        month.loc[index, 'Delivery Date'] = date_time
    
    month = month[['Delivery Date', 'Settlement Point Price']].rename(columns = {'Delivery Date': 'Date', 'Settlement Point Price': 'LMP'})
    lmp = pd.concat([lmp, month])

# read in generation data and delete unwanted sheets
gen_sheets = pd.read_excel("ERCOT_Gen_by_Type_2022.xlsx", sheet_name = None)
del gen_sheets['Summary']
del gen_sheets['Disclaimer']
del gen_sheets['data_Summary_1']
del gen_sheets['data_Summary_2']

# initialize dataframe
gen = pd.DataFrame(columns = ['Date', 'Power (MW)'])

# add data for each month
for month in gen_sheets.values():
    month = month[month['Fuel'] == 'Wind']
    month = month.drop(columns = ['Fuel', 'Settlement Type', 'Total'])

    # formulate entry for each day
    for index, row in month.iterrows():
        row_array = row.to_numpy()
        date_time = np.arange(row_array[0] + timedelta(minutes = 15), row_array[0] + timedelta(days=1, minutes = 15), timedelta(minutes=15))
        # calculate power of our turbine, assumed to have same capacity factor as average over all of ERCOT, and convert from MWh to MW (15 minute period)
        power = row_array[1:97]*100/37000/0.25 
        d = np.array([date_time, power]).T
        entry = pd.DataFrame(data=d, columns = ['Date', 'Power (MW)'])
        gen = pd.concat([gen, entry], ignore_index=True)

df = gen.join(lmp.set_index('Date'), on = 'Date')
df.dropna(inplace = True)
df['Battery Charge'] = 5 # initialize charge to 0.5 (minimum charge state) for all hours
df.loc[0, 'Battery Charge'] = 45 # first charging state will be half charged

# bin continous data
power_bins = np.arange(0, 73.5, 1)
power_averages = np.arange(0.5, 73.5, 1)
power_labels = np.arange(power_averages.size)
df['power_binned'] = pd.cut(df['Power (MW)'], bins=power_bins, labels=power_labels)
lmp_bins = np.arange(-252, -251, 2)
lmp_bins = np.append(lmp_bins, np.arange(-30, 305, 5))
lmp_averages = np.arange(-32.5, 305, 5)
lmp_labels = np.arange(lmp_averages.size)
lmp_bins = np.append(lmp_bins, np.arange(5600, 5650, 100))
df['lmp_binned'] = pd.cut(df['LMP'], bins=lmp_bins, labels=lmp_labels)
charge_bins = np.arange(0, 110, 10)
charge_averages = np.arange(5, 100, 10)
charge_labels = np.arange(charge_averages.size)
df['charge_binned'] = pd.cut(df['Battery Charge'], bins=charge_bins, labels=charge_labels)
df.to_csv('state_space.csv', index=False)

# store bins
np.savez('bins.npz', power=power_averages, lmp=lmp_averages, charge=charge_averages)
