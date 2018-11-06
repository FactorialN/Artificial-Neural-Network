from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # input and target are the output-Layer vector of a batch of input data, or the deltaav(n)
        cur = np.sum(np.square(input - target), axis=1)
        return 0.5 * np.sum(cur) / len(cur)

    def backward(self, input, target):
        # calculate the local gradient contribution e of the output layer
        return (input - target) / len(input)

class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        expx = np.exp(input)
        saved = (1 / np.sum(expx, axis=1))
        h = []
        for i in range(0, len(expx)):
            h.append(expx[i] * saved[i])
        h = np.matrix(h)
        h = np.maximum(0.00001, h)
        h = np.minimum(0.99999, h)
        h1 = 1 - h
        target1 = 1 - target
        e = []
        for i in range(0, len(h)):
            e.append(np.log(h[i]).dot(-target[i])+np.log(h1[i]).dot(-target1[i]))
            e[i] = np.array(e[i])[0]
        e = np.array(e)
        return np.mean(e)

    def backward(self, input, target):
        expx = np.exp(input)
        saved = (1 / np.sum(expx, axis=1))
        h = []
        for i in range(0, len(expx)):
            h.append(expx[i] * saved[i])
        h = np.array(h)
        h = 1 - h
        e1 = []
        for i in range(0, len(h)):
            e1.append(h[i]*(-target[i]))
            e1[i] = np.array(e1[i])
        e1 = np.array(e1)
        h = 1 - h
        target1 = 1 - target
        e2 = []
        for i in range(0, len(h)):
            e2.append(h[i] * (-target1[i]))
            e2[i] = np.array(e2[i])
        e2 = np.array(e2)
        e = e1 - e2
        return e / len(e)