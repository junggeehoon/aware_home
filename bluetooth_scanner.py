import serial
import pandas as pd
import numpy as np
import time

LABEL = "H-10"
NUMBER_OF_DATA = 1000
PORT = '/dev/cu.usbserial-020F8794'

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

x = []
count = 0
start_time = time.time()
while len(x) < NUMBER_OF_DATA:
    string = ser.readline().decode()
    s = string.split()
    if check_numbers(s):
        arr = parse_int(s)
        if validate(arr):
            x.append(arr)
            count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            estimated_time = (NUMBER_OF_DATA - count) * elapsed_time / count
            print("{}/{} : Elapsed: {:.1f}s , ETA: {:.1f}s : {} (dBm)".format(count, NUMBER_OF_DATA, elapsed_time,
                                                                 estimated_time, x[-1]))
        else:
            print("Waiting for valid data...", arr)

ser.close()

y = np.array([labels[LABEL] for i in range(NUMBER_OF_DATA)])
label = np.array([LABEL for i in range(NUMBER_OF_DATA)])

columns = []
for i in range(len(x[0])):
    columns.append("rssi{}".format(i))
columns.append("x")
columns.append("y")
columns.append("label")

df = pd.DataFrame(np.concatenate((x, y, label.reshape(-1, 1)), axis=1), columns=columns)
print("Done! for label: {} Took {:.1f}s to complete".format(LABEL, time.time() - start_time))
print("Saving data...")
df.to_csv('./test/datapoints.csv', mode='a', index=False, header=False)
