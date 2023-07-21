from matplotlib.animation import FuncAnimation
from files.labels import labels
import numpy as np
import serial
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd

matplotlib.use("TkAgg")

rf = pickle.load(open("../../models/rf.pickle", "rb"))

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


def convert(arr):
    array = np.array(arr)
    array[array < -100] = -100
    return array


sns.set_style("whitegrid")
fig, ax = plt.subplots()

# plot reference points
classes = rf.classes_
coordinates = [labels[label] for label in rf.classes_]
x, y = zip(*coordinates)

x = np.subtract(8, x)
y = np.array(y)

ax.scatter(x, y, color='blue')

predicted_point, = ax.plot([], [], 'ro')  # Initialize a red dot for the predicted point

ax.set_xlim(min(x) - 1, max(x) + 1)
ax.set_ylim(min(y) - 1, max(y) + 1)
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])

fig.show()

while True:
    string = ser.readline().decode()
    s = string.split()
    if check_numbers(s):
        arr = parse_int(s)
        if validate(arr):
            arr = convert(arr)
            predict = rf.predict([arr])
            print("Label: {}, Coordinate: {}".format(predict, labels[predict[0]]))
            predicted_x = 8 - labels[predict[0]][0]
            predicted_y = labels[predict[0]][1]
            predicted_point.set_data(predicted_x, predicted_y)  # Update the red dot position
            fig.canvas.draw()
        else:
            print("Data is invalid", arr)