import pandas as pd

df = pd.read_csv("./data/datapoints.csv")
df = df.dropna()
random_data = df.groupby(['x', 'y']).apply(lambda x: x.sample(1000)).reset_index(drop=True)

random_data.to_csv("./data/sample.csv", index=False)