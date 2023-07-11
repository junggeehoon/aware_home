import pandas as pd
import numpy as np
import pickle
import serial
import time

ACTUAL_LABEL = 'K-12'
NUMBER_OF_DATA = 400

random_forest_predict = []
knn_predict = []
svm_predict = []
traditional_predict = []

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
PORT = '/dev/cu.usbserial-020F8794'

df = pd.read_csv('./data/datapoints.csv')
df = df.dropna()

X = df.iloc[:, 0: -3].values
y = df['label'].values

mean = df.groupby(['x', 'y']).mean()

rf = pickle.load(open("./models/rf.pickle", "rb"))
knn = pickle.load(open("./models/knn.pickle", "rb"))
svm = pickle.load(open("./models/svm.pickle", "rb"))

ser = serial.Serial(PORT, 115200)
ser.setDTR(False)
ser.setRTS(False)


def validate(arr):
    if arr.count(0) > 0:
        return False
    return True


def parse_int(arr):
    for i in range(len(arr)):
        arr[i] = int(arr[i])
    return arr


def check_numbers(arr):
    if len(arr) <= 0:
        return False
    for n in arr:
        if not n.lstrip('-').isdigit():
            return False
    return True

count = 0
start_time = time.time()
data = []
while len(data) < NUMBER_OF_DATA * 4:
    string = ser.readline().decode()
    s = string.split()
    if check_numbers(s):
        arr = parse_int(s)
        if validate(arr):
            count += 1
            predict = (np.square(mean - arr).mean(axis=1)).idxmin()

            td_rmse = np.sqrt(np.square(np.subtract(predict, labels[ACTUAL_LABEL])).sum())

            knn_rmse = np.sqrt(np.square(np.subtract(labels[knn.predict([arr])[0]], labels[ACTUAL_LABEL])).sum())
            rf_rmse = np.sqrt(np.square(np.subtract(labels[rf.predict([arr])[0]], labels[ACTUAL_LABEL])).sum())
            svm_rmse = np.sqrt(np.square(np.subtract(labels[svm.predict([arr])[0]], labels[ACTUAL_LABEL])).sum())
            data.append([td_rmse, "traditional", ACTUAL_LABEL])
            data.append([knn_rmse, "knn", ACTUAL_LABEL])
            data.append([rf_rmse, "random forest", ACTUAL_LABEL])
            data.append([svm_rmse, "svm", ACTUAL_LABEL])

            
            current_time = time.time()
            elapsed_time = current_time - start_time
            estimated_time = (NUMBER_OF_DATA - count) * elapsed_time / count
            print("{}/{} : Elapsed: {:.1f}s , ETA: {:.1f}s || Label: {}, RMSE: {}".format(
                count, NUMBER_OF_DATA, elapsed_time, estimated_time, rf.predict([arr]), rf_rmse))
        else:
            print("Data is invalid", arr)
ser.close()

print("Done! Took {:.1f}s to complete".format(time.time() - start_time))

df = pd.DataFrame(data, columns=['rsme', 'method', 'label'])

mean = df.groupby(['method']).mean()
print(mean)

df.to_csv("./result/rmse.csv", mode='a', index=False, header=False)

# data = {
#     "traditional": np.sqrt(np.square(np.subtract(traditional_predict, labels[ACTUAL_LABEL])).sum(axis=1)),
#     "knn": np.sqrt(np.square(np.subtract(knn_predict, labels[ACTUAL_LABEL])).sum(axis=1)),
#     "svm": np.sqrt(np.square(np.subtract(svm_predict, labels[ACTUAL_LABEL])).sum(axis=1)),
#     "random forest": np.sqrt(np.square(np.subtract(random_forest_predict, labels[ACTUAL_LABEL])).sum(axis=1)),
#     "label": [ACTUAL_LABEL for i in range(NUMBER_OF_DATA)]
# }

# print("Traditional MSE: {:.3f}".format(data['traditional'].mean()))
# print("KNN MSE: {:.3f}".format(data['knn'].mean()))
# print("SVM MSE: {:.3f}".format(data['svm'].mean()))
# print("Random Forest MSE: {:.3f}".format(data['random forest'].mean()))
#
# df = pd.DataFrame(data, columns=data.keys())
# df.to_csv("./result/mse.csv", mode='a', index=False, header=False)

