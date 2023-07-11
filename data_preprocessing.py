import pandas as pd
import numpy as np

criterion = 3

df = pd.read_csv('./vectors/datapoints.csv')

columns_to_filter = ['rssi0', 'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8', 'rssi9', 'rssi10',
                     'rssi11']

grouped = df.groupby(['label'])
mean = grouped[columns_to_filter].transform('mean')
std = grouped[columns_to_filter].transform('std')

# Replace zero RSSI values with the minimum RSSI value of its data point
# min_values = grouped[columns_to_filter].transform(lambda x: x.replace(0, x.min()))
# df[columns_to_filter] = min_values

# Filter using criterion
filter_condition = (df[columns_to_filter] > mean - criterion * std) & (df[columns_to_filter] < mean + criterion * std)
filtered_df = df[filter_condition.all(axis=1)]
filtered_df.reset_index(drop=True, inplace=True)


filtered_df.to_csv("./data/datapoints.csv", index=False)
