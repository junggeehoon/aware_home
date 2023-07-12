import pandas as pd

criterion = 3

df = pd.read_csv('./vectors/all.csv')

columns_to_filter = ['rssi0', 'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8', 'rssi9', 'rssi10',
                     'rssi11']
# Replace RSSI values less than -100 with -100
df[columns_to_filter] = df[columns_to_filter].where(df[columns_to_filter] >= -100, -100)

grouped = df.groupby(['label'])
mean = grouped[columns_to_filter].transform('mean')
std = grouped[columns_to_filter].transform('std')

filter_condition = (df[columns_to_filter] > mean - criterion * std) & (df[columns_to_filter] < mean + criterion * std)
filtered_df = df[filter_condition.all(axis=1)]
filtered_df.reset_index(drop=True, inplace=True)


filtered_df.to_csv("./data/datapoints.csv", index=False)
