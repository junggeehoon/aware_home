import serial
import pickle
import time

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

rf = pickle.load(open("./models/rf.pickle", "rb"))

x = []
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


start_time = time.time()
while True:
    string = ser.readline().decode()
    s = string.split()
    if check_numbers(s):
        arr = parse_int(s)
        if validate(arr):
            predict = rf.predict([arr])
            print("Label: {}, Coordinate: {}".format(predict, labels[predict[0]]))
        else:
            print("Data is invalid", arr)