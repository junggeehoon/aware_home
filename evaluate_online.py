from labels import labels
import pandas as pd
import numpy as np
import pickle
import serial
import time

ACTUAL_LABEL = 'K-06'
NUMBER_OF_DATA = 300

random_forest_predict = []
knn_predict = []
svm_predict = []
traditional_predict = []

PORT = '/dev/cu.usbserial-020F8794'

df = pd.read_csv('./train/datapoints.csv')

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

def convert(arr):
    array = np.array(arr)
    array[array < -100] = -100
    return array

count = 0
start_time = time.time()
data = []
while len(data) < NUMBER_OF_DATA * 4:
    string = ser.readline().decode()
    s = string.split()
    if check_numbers(s):
        arr = parse_int(s)
        if validate(arr):
            arr = convert(arr)
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

print(df.groupby(['method']).mean())

# df.to_csv("./result/rmse.csv", mode='a', index=False, header=False)