import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

labels = {
    'K-01': [0, 0],
    'K-02': [1, 0],
    'K-03': [2, 0],
    'K-04': [3, 0],
    'K-05': [1, 1],
    'K-06': [2, 1],
    'K-07': [3, 1],
    'K-08': [1, 2],
    'K-09': [2, 2],
    'K-10': [3, 2],
    'K-11': [0, 3],
    'K-12': [1, 3],
    'K-13': [2, 3],
    'H-01': [0, 4],
    'H-02': [1, 4],
    'H-03': [2, 4],
    'H-04': [3, 4],
    'H-05': [4, 4],
    'H-06': [5, 4],
    'H-07': [6, 4],
    'H-08': [7, 4],
    'H-09': [8, 4],
    'H-10': [9, 4],
    'H-11': [10, 4],
    'H-12': [11, 4],
    'L-01': [3, 5],
    'L-02': [4, 5],
    'L-03': [3, 6],
    'L-04': [4, 6],
    'L-05': [3, 7],
    'L-06': [2, 6],
    'L-07': [1, 7],
    'L-08': [2, 7],
    'L-09': [1, 8],
    'L-10': [2, 8]
}

df = pd.read_csv("./test/datapoints.csv")
df = df.dropna()

X = df.iloc[:, 0: -3].values
y = df['label'].values

rf = pickle.load(open("./models/rf.pickle", "rb"))

predicted_label = rf.predict(X)
predicted_coordinate = [labels[label] for label in predicted_label]
actual_coordinate = [labels[label] for label in y]

mean_square_error = np.sqrt(np.square(np.subtract(predicted_coordinate, actual_coordinate)).sum(axis=1))

print(np.count_nonzero(mean_square_error))
# sns.ecdfplot(data=mean_square_error)
# plt.show()