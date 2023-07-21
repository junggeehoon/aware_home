import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

LABEL = 'H-04'
CHANNEL = 'rssi5'

df = pd.read_csv('../../data/sample.csv')
data = df.loc[df['label'] == LABEL]

rssi = data[CHANNEL].values
mu, std = norm.fit(rssi)

sns.kdeplot(data=rssi,
            color='purple', label='Raw', fill=True, linewidth=0)

title = "µ: {:.2f} σ: {:.2f}".format(mu, std)
plt.title(title)
plt.xlabel("RSSI (dBm)")

plt.show()
