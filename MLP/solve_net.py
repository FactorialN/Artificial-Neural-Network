from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np


def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    #  mix data per batch and generate
    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq, Loss, Acur):

    iter_counter = 0
    loss_list = []
    acc_list = []
    ll = []
    ac = []

    # train model with
    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss value of the whole batch
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss, this is actually the local gradient contribution of the output layer
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)

        # update layers' weights: recount after the whole backward procedure
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        ll.append(loss_value)
        acc_list.append(acc_value)
        ac.append(acc_value)

        if iter_counter % disp_freq == 0:
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f' % (iter_counter, np.mean(loss_list), np.mean(acc_list))
            loss_list = []
            acc_list = []
            LOG_INFO(msg)
    Loss.append(np.mean(ll))
    Acur.append(np.mean(ac))


def test_net(model, loss, inputs, labels, batch_size):
    loss_list = []
    acc_list = []

    # test model with all the test data
    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        # get the expected value of this batch of input
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        # calculate loss of this batch
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

    # use the mean of all batch's loss and accuracy as the final result
    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    LOG_INFO(msg)
