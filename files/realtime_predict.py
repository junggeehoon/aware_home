from files.labels import labels
import numpy as np
import serial
import pickle
import time

rf = pickle.load(open("../models/rf.pickle", "rb"))

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


def convert(arr):
    array = np.array(arr)
    array[array < -100] = -100
    return array


start_time = time.time()
while True:
    string = ser.readline().decode()
    s = string.split()
    if check_numbers(s):
        arr = parse_int(s)
        if validate(arr):
            arr = convert(arr)
            predict = rf.predict([arr])
            print("Label: {}, Coordinate: {}".format(predict, labels[predict[0]]))
        else:
            print("Data is invalid", arr)
