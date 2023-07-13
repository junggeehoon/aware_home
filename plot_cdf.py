import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
df = pd.read_csv("./result/rsme.csv")

sns.kdeplot(data=df, x='rsme', hue="method", cumulative=True, common_norm=False)

plt.xlabel("Error (m)")
plt.ylabel("CDF")
plt.xlim([0, 5])
plt.show()