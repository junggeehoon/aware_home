from src.labels import labels
import numpy as np
import serial
import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
x_offset = 7
y_offset = -3

rf = pickle.load(open("../../models/rf.pickle", "rb"))

img = plt.imread('../../figures/test.png')

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


fig, ax = plt.subplots(figsize=(8, 6))


# plot reference points
classes = rf.classes_
coordinates = [labels[label] for label in rf.classes_]
x, y = zip(*coordinates)

x = np.multiply(np.subtract(8, x), 10)
y = np.multiply(np.array(y), 10)

ax.imshow(img, extent=[min(x) - 16, max(x) + 16, min(y) - 16, max(y) + 16], zorder=0)
ax.scatter(x + x_offset, y + y_offset, color='blue')

predicted_point, = ax.plot([], [], 'ro')  # Initialize a red dot for the predicted point

ax.set_xlim(min(x) - 20, max(x) + 20)
ax.set_ylim(min(y) - 20, max(y) + 20)
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])

fig.show()

last_predictions = []

while True:
    string = ser.readline().decode()
    s = string.split()
    if check_numbers(s):
        arr = parse_int(s)
        if validate(arr):
            arr = convert(arr)
            predict = rf.predict([arr])
            last_predictions.append(predict[0])
            if len(last_predictions) > 5:
                last_predictions.pop(0)
            unique_labels, counts = np.unique(last_predictions, return_counts=True)
            most_frequent = unique_labels[np.argmax(counts)]
            print("Label: {}, Coordinate: {}".format(most_frequent, labels[most_frequent]))
            predicted_x = np.multiply(8 - labels[most_frequent][0], 10) + x_offset
            predicted_y = np.multiply(labels[most_frequent][1], 10) + y_offset
            predicted_point.set_data(predicted_x, predicted_y)
            fig.canvas.draw()
        else:
            print("Data is invalid", arr)