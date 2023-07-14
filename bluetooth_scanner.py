from labels import labels
import serial
import pandas as pd
import numpy as np
import time

LABEL = "H-01"
NUMBER_OF_DATA = 1000
PORT = '/dev/cu.usbserial-020F8794'


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
print("\nDone! Label: {} Took {:.1f}s to complete".format(LABEL, time.time() - start_time))
print("\nSaving data...")
df.to_csv('./vectors/ninth.csv', mode='a', index=False, header=False)
