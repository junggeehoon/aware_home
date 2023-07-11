import serial
import pandas as pd
import numpy as np
import time

x = []
NUMBER_OF_DATA = 300
DISTANCE = 5
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


count = 0
start_time = time.time()
while len(x) < NUMBER_OF_DATA:
    string = ser.readline().decode()
    s = string.split()
    if check_numbers(s):
        arr = parse_int(s)
        x.append(arr[0])
        count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        estimated_time = (NUMBER_OF_DATA - count) * elapsed_time / count
        print("{}/{} : Elapsed: {:.1f}s , ETA: {:.1f}s : {} (dBm)".format(count, NUMBER_OF_DATA, elapsed_time, estimated_time, x[-1]))

ser.close()

y = np.array([DISTANCE for i in range(NUMBER_OF_DATA)])
x = np.array(x)

print("Done! Took {:.1f}s to complete".format(time.time() - start_time))
print("Saving data...")

data = {
    "rssi": x,
    "distance": y
}

df = pd.DataFrame(data)

df.to_csv('./data/single_plot.csv', index=False, mode='a', header=False)