import pandas as pd

df = pd.read_csv('./data/sample.csv')

train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)

train_df.to_csv('./train/datapoints.csv', index=False)
test_df.to_csv('./test/datapoints.csv', index=False)