import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set_style("whitegrid")
df = pd.read_csv('data/single_plot.csv')
grouped_data = df.groupby(['distance'])
mean = grouped_data.mean()
std = grouped_data.std()

p = sns.stripplot(x="distance", y="rssi", data=df, hue="distance", legend=False)

sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="distance",
            y="rssi",
            data=df,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=p)
plt.xlabel("Distance (m)")
plt.ylabel("RSSI (dBm)")
plt.savefig("./figures/rssi by distance.png")
plt.show()