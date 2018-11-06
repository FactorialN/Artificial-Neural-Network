from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import time
import matplotlib.pyplot as plt

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# I may add a lot of comments to help me understand what this code is doing...

# Generate a model here and add layers, split linear calculation and activation functions
# Find out the vector you are actually using to calculate in the training procedure
model = Network()
model.add(Linear('fc1', 784, 256, 0.01))
# My model under testing
model.add(Relu('relu1'))
model.add(Linear('fc2', 256, 10, 0.01))
#model.add(Relu ('relu2'))

# defining a loss calculator for loss calculation
#loss = EuclideanLoss(name='loss')
loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes numaaber of iterations in one epoch to display information.


# config is a dictionary directing the training process
config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0002,
    'momentum': 0.85,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 5
}

# train the model by epoch, in each epoch all the input data are used to generate a input set by random
Loss = [1]
Acur = [0]
nowTime = time.time()
for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], Loss, Acur)

    # test the model performance when necessary
    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])

test_net(model, loss, test_data, test_label, config['batch_size'])
print(time.time() - nowTime)
plt.plot(range(0, len(Acur)), Acur)
plt.plot(range(0, len(Acur)), Loss)
plt.savefig('a.jpg')
plt.show()
