# iport kum
import csv
import math
import operator
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import joblib
import csv
import random
from machine_learn_basic.ku import *
import  pandas as pd
# def load_model():
train_label = []
trains = [-1]


def train_model():
    train_label=[]
    with open("trainlabel.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            thislabel = [0 for _ in range(10)]
            num = int(float(row[0]))
            thislabel[num] = 1
            trains.append(num)
            train_label.append(thislabel)
    train_label=np.array(train_label)
    date = Image.open('train_img/{}.jpg'.format(1))
    train_data = np.array(date)
    train_data = train_data.reshape((28 * 28 * 3, 1))
    max1 = max(train_data)
    min1 = min(train_data)
    train_data = train_data / (max1 - min1)
    for i in range(2, 600):
        date = Image.open('train_img/{}.jpg'.format(i))
        date1 = np.array(date)
        date1 = date1.reshape((28 * 28 * 3, 1))
        max1 = max(date1)
        min1 = min(date1)
        date1 = date1 / (max1 - min1)
        # print(date1)
        train_data = np.hstack((train_data, date1))
    # print(np.shape(train_data))
    train_data = train_data.T
    bp = new_backpropagation(28 * 28 * 3, 10, 10)
    # print(train_data,train_label)
    bp.fit(train_data, train_label[:600])
    joblib.dump(bp, "bp.pkl")
    print("done")
def predict(bp):
    sum = 0
    # # print(train_label)
    train_label = []
    trains=[-1]
    with open("trainlabel.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            thislabel = [0 for _ in range(10)]
            num = int(float(row[0]))
            thislabel[num] = 1
            trains.append(num)
            train_label.append(thislabel)


    for i in range(1, 60001):
        date = Image.open('train_img/{}.jpg'.format(i))
        train_data = np.array(date)
        train_data = train_data.reshape((28 * 28 * 3, 1))
        train_data = train_data.T.reshape((28 * 28 * 3))
        max1 = max(train_data)
        min1 = min(train_data)
        train_data = train_data / (max1 - min1)
        pre = np.argmax(np.array(bp.predict(train_data)))
        # print(np.array(bp.predict(train_data)))
        # print(i, pre)
        if pre==trains[i]:
            sum = sum + 1
    print("准确率：" + str(sum/60000) )

if __name__ == '__main__':
    # train_model()
    try:
        bp = joblib.load("bp.pkl")
    except FileNotFoundError:
        train_model()
    predict(bp)

