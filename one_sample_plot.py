from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sb

LABEL = 'L-04'

df = pd.read_csv('./data/sample.csv')
filtered_df = pd.read_csv('./data/datapoints_mean_filtered.csv')

raw_data = df.loc[df['label'] == LABEL]
filtered_data = filtered_df.loc[filtered_df['label'] == LABEL]

rssi0 = raw_data['rssi0'].values
# kf = KalmanFilter(initial_state_mean=rssi0[0], n_dim_obs=1)
# (filtered_state_means, filtered_state_covariances) = kf.filter(rssi0)
# print(filtered_state_means)
rssi0_filtered = filtered_data['rssi0'].values
# rssi1 = raw_data['rssi4'].values
# rssi2 = raw_data['rssi9'].values

plt.xlabel("Number of samples")
plt.ylabel("RSSI (dBm)")
plt.ylim([-105, -70])
plt.plot(rssi0, label="Raw")
plt.plot(rssi0_filtered, label="Filtered")
# plt.plot(rssi1, label="Channel 4")
# plt.plot(rssi2, label="Channel 9")
plt.legend()
plt.savefig("./figures/avg.png")
plt.show()
