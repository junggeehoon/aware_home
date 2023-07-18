import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
df = pd.read_csv("./result/rmse.csv")

sns.kdeplot(data=df, x='rmse', hue="method", cumulative=True, common_norm=False)

plt.xlabel("RMSE (m)")
plt.ylabel("CDF")
plt.xlim([0, 5])
plt.tight_layout()
plt.savefig("./figures/rmse_cdf.png")
plt.show()