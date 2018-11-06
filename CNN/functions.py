import numpy as np
from scipy import signal


def conv2d_forward(input, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_out (#output channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after convolution
    '''
    xx = np.pad(input, pad, 'constant', constant_values=0)[pad: pad + input.shape[0],pad:pad + input.shape[1], :, :]
    shape = xx.shape
    output = np.zeros((shape[0], W.shape[0], shape[2] - kernel_size + 1, shape[3] - kernel_size + 1))
    for i in range(shape[0]):
        for ic in range(shape[1]):
            for oc in range(W.shape[0]):
                output[i][oc] += signal.convolve2d(xx[i][ic], np.rot90(W[oc][ic], 2), mode='valid')
    for i in range(shape[0]):
        for j in range(shape[1]):
            output[i][j] += b[j]
    return output


def conv2d_backward(input, grad_output, W, b, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_out (#output channel) x h_out x w_out
        W: weight, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        b: bias, shape = c_out
        kernel_size: size of the convolving kernel (or filter)
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_W: gradient of W, shape = c_out (#output channel) x c_in (#input channel) x k (#kernel_size) x k (#kernel_size)
        grad_b: gradient of b, shape = c_out
    '''
    xx = np.pad(input, pad, 'constant', constant_values=0)[pad: pad + input.shape[0], pad:pad + input.shape[1], :, :]
    grad_input1 = np.zeros(xx.shape)
    grad_W = np.zeros(W.shape)
    for i in range(grad_input1.shape[0]):
        for oc in range(grad_output.shape[1]):
            Wp = np.rot90(grad_output[i][oc], 2)
            for ic in range(grad_input1.shape[1]):
                grad_input1[i][ic] += signal.convolve2d(grad_output[i][oc], W[oc][ic], mode='full')
                grad_W[oc][ic] += signal.convolve2d(xx[i][ic], Wp, mode='valid')
    grad_input = grad_input1[:, :, pad:pad + grad_input1.shape[2], pad:pad + grad_input1.shape[3]]
    grad_b = np.sum(grad_output, axis=(0, 2, 3))
    return grad_input, grad_W, grad_b


def avgpool2d_forward(input, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        output: shape = n (#sample) x c_in (#input channel) x h_out x w_out,
            where h_out, w_out is the height and width of output, after average pooling over input
    '''
    xx = np.pad(input, pad, 'constant', constant_values=0)[pad: pad + input.shape[0], pad:pad + input.shape[1], :, :]
    shape = xx.shape
    output = np.zeros((shape[0], shape[1], int(shape[2]/kernel_size), int(shape[3]/kernel_size)))
    for x in range(output.shape[2]):
        for y in range(output.shape[3]):
            x1 = x * kernel_size
            y1 = y * kernel_size
            output[:, :, x, y] += np.sum(xx[:, :, x1: x1 + kernel_size, y1: y1 + kernel_size], axis=(2, 3))
    output /= kernel_size ** 2
    return output


def avgpool2d_backward(input, grad_output, kernel_size, pad):
    '''
    Args:
        input: shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
        grad_output: shape = n (#sample) x c_in (#input channel) x h_out x w_out
        kernel_size: size of the window to take average over
        pad: number of zero added to both sides of input

    Returns:
        grad_input: gradient of input, shape = n (#sample) x c_in (#input channel) x h_in (#height) x w_in (#width)
    '''
    grad_input = (np.kron(grad_output, np.ones((kernel_size, kernel_size))) / (kernel_size ** 2))[:, :, pad:pad + input.shape[2], pad:pad + input.shape[3]]
    return grad_input
