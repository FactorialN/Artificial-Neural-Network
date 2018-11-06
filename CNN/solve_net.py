from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np
from time import time
import matplotlib.pyplot as plt


def vis_square(data): # from caffe's tutorial
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imsave("a.png", data)
    plt.axis('off')

def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq, Loss, Acur):

    iter_counter = 0
    loss_list = []
    acc_list = []
    ll = []
    ac = []

    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient
        model.backward(grad)

        if loss_value > 1:
            config['learning_rate'] = 0.2
        elif loss_value > 0.5:
            config['learning_rate'] = 0.1
        elif loss_value > 0.2:
            config['learning_rate'] = 0.05
        else:
            config['learning_rate'] = max(loss_value / 5.0, 0.005)

        # update layers' weights
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)
        ll.append(loss_value)
        ac.append(acc_value)

        if iter_counter % disp_freq == 0:
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f' % (iter_counter, np.mean(loss_list), np.mean(acc_list))
            Loss.append(np.mean(loss_list))
            Acur.append(np.mean(acc_list))
            loss_list = []
            acc_list = []
            LOG_INFO(msg)

    Loss.append(np.mean(ll))
    Acur.append(np.mean(ac))


def test_net(model, loss, inputs, labels, batch_size):
    loss_list = []
    acc_list = []
    a = 0

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        opp = model.fforward(input)
        a += 1
        if a == 1:
            vis_square(opp[0])
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    LOG_INFO(msg)
