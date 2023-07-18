import pandas as pd

# Load the data
df = pd.read_csv('result/rsme.csv')

# Errors for which to compute cumulative probabilities
errors = [0, 1, 2, 3, 4, 5]

# Group by method
grouped = df.groupby('method')

for error in errors:
    print(f"\nError: {error}")
    for method, data in grouped:
        prob = (data['rsme'] <= error).mean()
        print(f"{method}: {prob:.3f}")