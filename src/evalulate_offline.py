from scipy.spatial import distance
from labels import labels
import numpy as np
import pandas as pd
import pickle

train = pd.read_csv('../train/datapoints.csv')
test = pd.read_csv('../test/datapoints.csv')

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

rf = pickle.load(open("../models/rf.pickle", "rb"))
knn = pickle.load(open("../models/knn.pickle", "rb"))
svm = pickle.load(open("../models/svm.pickle", "rb"))

rf_predict = [labels[label] for label in rf.predict(test[columns].values)]
knn_predict = [labels[label] for label in knn.predict(test[columns].values)]
svm_predict = [labels[label] for label in svm.predict(test[columns].values)]

traditional_predict = [labels[label] for label in predict_labels(test[columns]).values]

traditional_rmse = np.sqrt(np.square(np.subtract(actual, traditional_predict)).sum(axis=1))
rf_rmse = np.sqrt(np.square(np.subtract(actual, rf_predict)).sum(axis=1))
svm_rmse = np.sqrt(np.square(np.subtract(actual, svm_predict)).sum(axis=1))
knn_rmse = np.sqrt(np.square(np.subtract(actual, knn_predict)).sum(axis=1))

traditional_data = {
    "rsme": traditional_rmse,
    "method": "traditional",
    "label": test['label']
}

rf_data = {
    "rsme": rf_rmse,
    "method": "random forest",
    "label": test['label']
}

svm_data = {
    "rsme": svm_rmse,
    "method": "svm",
    "label": test['label']
}

knn_data = {
    "rsme": knn_rmse,
    "method": "knn",
    "label": test['label']
}

df_traditional = pd.DataFrame(traditional_data)
df_rf = pd.DataFrame(rf_data)
df_svm = pd.DataFrame(svm_data)
df_knn = pd.DataFrame(knn_data)

df = pd.concat([df_traditional, df_rf, df_svm, df_knn], ignore_index=True)
df.to_csv("./result/rmse.csv", index=False)

print(df.groupby(['method']).mean())
