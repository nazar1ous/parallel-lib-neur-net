import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import cv2
import random as rd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, MaxPool2D, Conv2D, Dense, Reshape, Dropout
from keras.utils import np_utils
from keras.datasets import mnist
import warnings
warnings.filterwarnings('ignore')

rd.seed(200)


def gen_data(path, width, height):
    data = []
    for sub_dir,dirs,files in os.walk(path):
        for file in files:
            fname = sub_dir + '/' + file
            label = 0 if sub_dir.startswith('data/dog') else 1
            img = cv2.imread(fname)
            img = cv2.resize(img, (width, height))
            img_data = np.array(img).reshape((height * width * 3, 1))
            data.append((label, img_data))
    rd.shuffle(data)
    return data


def write_data(data, path):
    with open(path, 'w') as f:
        f.write(str(data[0][1].shape[0]) + ' ' + str(len(data)) + '\n')

        for label, img_data in data:
            str_ = str(label) + ' ' + ' '.join(map(lambda x: str(x[0]), img_data)) + '\n'
            f.write(str_)


def test_network(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(10, activation='linear'))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.fit(X_train,Y_train, epochs=50, batch_size=24)
    res = model.evaluate(X_test, Y_test)
    print(res)



if __name__ == "__main__":
    data_path = 'data'
    cpp_dataset = 'gen_data/cpp_dataset.txt'
    data = gen_data(data_path, 64, 64)
    N = len(data)
    X = np.array([list(elem[1]) for elem in data]).reshape((data[0][1].shape[0], N)).T / 255
    Y = np.array([[elem[0]] for elem in data]).reshape((1, N)).T

    # write_data(data, cpp_dataset)
    diff = int(len(data) * 0.15)
    test_network(X[:N-diff, :], Y[:N-diff, :], X[N-diff:, :], Y[N-diff:, :])

# cv2.imshow("img", img)
# cv2.waitKey(0)