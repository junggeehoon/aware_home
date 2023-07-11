from scipy.spatial import distance
import numpy as np
import pandas as pd
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

train = pd.read_csv('./train/datapoints.csv')
test = pd.read_csv('./test/datapoints.csv')

columns = ['rssi0', 'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8', 'rssi9', 'rssi10',
                     'rssi11']

centroids = train.groupby('label').mean()[columns]


def predict_labels(rssi_vectors):
    # Compute distances to each centroid
    distances = distance.cdist(rssi_vectors, centroids.values, 'euclidean')
    
    # Find the indices of the closest centroids
    closest_centroids = np.argmin(distances, axis=1)
    
    # Return the labels of the closest centroids
    return centroids.index[closest_centroids]


actual = [labels[label] for label in test['label']]

rf = pickle.load(open("./models/rf.pickle", "rb"))
rf_predict = [labels[label] for label in rf.predict(test[columns].values)]

traditional_predict = [labels[label] for label in predict_labels(test[columns]).values]

traditional_rmse = np.sqrt(np.square(np.subtract(actual, traditional_predict)).sum(axis=1)).mean()
rf_rmse = np.sqrt(np.square(np.subtract(actual, rf_predict)).sum(axis=1)).mean()

print("Traditional RMSE: {:.3f}, Random Forest RMSE: {:.3f}".format(traditional_rmse, rf_rmse))
