import pandas as pd

df = pd.read_csv('./data/datapoints.csv')
df = df.dropna()

train_df, test_df = df.sample(frac=0.8, random_state=42), df.sample(frac=0.2, random_state=42)

train_df.to_csv('./train/datapoints.csv', index=False)
test_df.to_csv('./test/datapoints.csv', index=False)