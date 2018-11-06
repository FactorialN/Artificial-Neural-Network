from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
import time



train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', 1, 4, 3, 1, 0.2))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
model.add(Conv2D('conv2', 4, 4, 3, 1, 0.2))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
model.add(Reshape('flatten', (-1, 196)))
model.add(Linear('fc3', 196, 10, 0.1))


loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.015,
    'weight_decay': 0.00005,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 1,
    'disp_freq': 10,
    'test_epoch': 5
}

Loss = [1]
Acur = [0]
nowTime = time.time()

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], Loss, Acur)

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])

test_net(model, loss, test_data, test_label, config['batch_size'])
print(time.time() - nowTime)

"""
f = open("a.txt", "a")
f.write(str(Acur)+'#')
g = open("b.txt", "a")
g.write(str(Loss)+"#")

k = []
p = []
for i, j in zip(Acur, Loss):
    if j < 2 and i > 0.3:
        k.append(i)
        p.append(j)
plt.plot(range(0, len(k)), k)
plt.plot(range(0, len(Acur)), Loss)
plt.show()
"""