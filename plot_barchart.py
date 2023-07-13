import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
df = pd.read_csv("./result/rsme.csv")

sns.barplot(data=df, x="method", y="rsme")

plt.ylabel("RSME")
plt.xlabel("Method")
plt.tight_layout()
plt.savefig("./figures/rmse_barchart.png")
plt.show()

print(df.groupby(['method']).mean())
