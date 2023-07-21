import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd


sns.set_style("whitegrid")
df = pd.read_csv('../../data/single_plot.csv')
data = df.loc[df['distance'] == 1]

cut_off_data = data[(np.abs(stats.zscore(data['rssi'])) < 2)]['rssi']

moving_avg = cut_off_data.rolling(5).mean()
moving_median = cut_off_data.rolling(5).median()
exponential_avg = cut_off_data.ewm(alpha=0.3, adjust=False).mean()

fig, ax = plt.subplots()
sns.kdeplot(data=cut_off_data.values.flatten(),
            color='crimson', label='Raw', fill=True, ax=ax, linewidth=0)
sns.kdeplot(data=moving_avg.values.flatten(),
            color='limegreen', label='Filtered', fill=True, ax=ax, linewidth=0)
ax.legend()
plt.xlabel("RSSI (dBm)")
plt.ylabel("Probability Density")
plt.tight_layout()
# plt.savefig("./figures/rssi kde plot.png")
plt.show()