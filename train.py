from machine_learn_basic.ku import backpropagation
from PIL import Image
import numpy as np
import csv
import joblib
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    date = Image.open('train_img/{}.jpg'.format(1))
    train_data = np.array(date)
    train_data = train_data.reshape((28 * 28 * 3, 1))
    max1=max(train_data)
    min1=min(train_data)
    train_data=train_data/(max1-min1)
        # train_data[date]=d
    # print(train_data)

    for i in range(2,6000):
        date=Image.open('train_img/{}.jpg'.format(i))
        date1=np.array(date)
        date1=date1.reshape((28*28*3,1))
        max1 = max(date1)
        min1 = min(date1)
        date1=date1/(max1-min1)
        # print(date1)
        train_data=np.hstack((train_data,date1))
    # print(train_data)
    train_data=train_data.T
    train_data=train_data.tolist()
    train_label=[]
    with open("trainlabel.csv",'r') as f:
        reader=csv.reader(f)
        for row in reader:
            thislabel = [0 for _ in range(10)]
            num=int(float(row[0]))
            thislabel[num]=1
            train_label.append(thislabel)

    # print(np.shape(train_data))
    bp=backpropagation(28*28*3,10,10)
    bp.fit(train_data,train_label[:599])

    joblib.dump(bp,"bp.pkl")
    sum=0
    # print(train_label)

    sum=0
    for i in range(1,60001):
        date = Image.open('train_img/{}.jpg'.format(i))
        train_data = np.array(date)
        train_data = train_data.reshape((28 * 28 * 3, 1))
        train_data=train_data.T.reshape((28 * 28 * 3))
        max1 = max(train_data)
        min1 = min(train_data)
        train_data=train_data/(max1-min1)
        pre=np.argmax(np.array(bp.predict(train_data)))
        print(np.array(bp.predict(train_data)))
        print(i,pre)
        if pre==trains[i]:
            sum=sum+1
    print("准确率",str(sum/60000))

