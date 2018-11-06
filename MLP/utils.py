from __future__ import division
from __future__ import print_function
import numpy as np
from datetime import datetime


def onehot_encoding(label, max_num_class):
    # get the onehot vector of this model
    encoding = np.eye(max_num_class)
    encoding = encoding[label]
    return encoding


def calculate_acc(output, label):
    # use the max dimension as the output, compare the accuracy of this batch
    correct = np.sum(np.argmax(output, axis=1) == label)
    return correct / len(label)


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)
