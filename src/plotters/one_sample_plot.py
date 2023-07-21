from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sb

LABEL = 'H-04'
CHANNELS = [0, 2]

df = pd.read_csv('../../data/sample.csv')
# filtered_df = pd.read_csv('./data/datapoints_mean_filtered.csv')

raw_data = df.loc[df['label'] == LABEL]
# filtered_data = filtered_df.loc[filtered_df['label'] == LABEL]

for channel in CHANNELS:
    rssi = raw_data[f"rssi{channel}"].values
    plt.plot(rssi, label=f"Channel {channel}")

# kf = KalmanFilter(initial_state_mean=rssi0[0], n_dim_obs=1)
# (filtered_state_means, filtered_state_covariances) = kf.filter(rssi0)
# print(filtered_state_means)
# rssi0_filtered = filtered_data['rssi0'].values


plt.xlabel("Number of samples")
plt.ylabel("RSSI (dBm)")
plt.ylim([-105, -65])

# plt.plot(rssi0_filtered, label="Filtered")

plt.legend()
# plt.savefig("./figures/avg.png")
plt.show()
