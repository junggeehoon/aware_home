import pandas as pd
import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"]()

WINDOW_SIZE = 5

df = pd.read_csv('../../data/datapoints.csv')
columns_to_filter = ['rssi0', 'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8', 'rssi9', 'rssi10',
                     'rssi11']


filtered_df = df.groupby(['x', 'y'])[columns_to_filter].transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
filtered_df['x'] = df['x']
filtered_df['y'] = df['y']
filtered_df['label'] = df['label']

filtered_df.to_csv('./data/datapoints_exponential_filtered.csv', index=False)
