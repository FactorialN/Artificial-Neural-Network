#What I changed
In this PA I changed the following things:

## main.py
I changed
```cnn_model = Model()``` to
```cnn_model = Model(True)```or```cnn_model = Model(False)```.

## model.py
1. Added a is_train to parameters
2. filled in the blank of arguments in forward
```
self.loss, self.pred, self.acc = self.forward(is_Train, tf.AUTO_REUSE)
self.loss_val, self.pred_val, self.acc_val = self.forward(is_Train, True)
```
3. implemented layer structure in forward
```
k_conv1 = tf.get_variable(name='k_conv1', shape=[3, 3, 1, 4])
b_conv1 = tf.get_variable(name='b_conv1', shape=[4])
h_conv1 = tf.nn.conv2d(self.x_, k_conv1, padding='SAME', strides=[1, 1, 1, 1]) + b_conv1
hr_conv1 = dropout_layer(tf.nn.relu(batch_normalization_layer(h_conv1, is_train)), 0.5, is_train)
#hr_conv1 = dropout_layer(tf.nn.relu(h_conv1), 0.3, is_train)
p_pool1 = tf.nn.max_pool(hr_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
k_conv2 = tf.get_variable(name='k_conv2', shape=[3, 3, 4, 4])
b_conv2 = tf.get_variable(name='b_conv2', shape=[4])
h_conv2 = tf.nn.conv2d(p_pool1, k_conv2, padding='SAME', strides=[1, 1, 1, 1]) + b_conv2
hr_conv2 = dropout_layer(tf.nn.relu(batch_normalization_layer(h_conv2, is_train)), 0.5, is_train)
#hr_conv2 = dropout_layer(tf.nn.relu(h_conv2), 0.3, is_train)
p_pool2 = tf.nn.max_pool(hr_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
# flt = tf.reshape(pool2, shape=[-1, 196])
flt = tf.reshape(p_pool2, shape=[-1, 196])
w_fc3 = tf.get_variable(name='w_fc3', shape=[196, 10])
b_fc3 = tf.get_variable(name='b_fc3', shape=[10])
logits = tf.matmul(flt, w_fc3) + b_fc3
```
4. implemented batch normalization
```
if is_train:
    return tf.layers.batch_normalization(incoming, momentum=0.99, epsilon=1e-5, training=True)
else:
    return tf.layers.batch_normalization(incoming, training=False)
```
5. implemented dropout layer
```
if is_train:
    return tf.layers.dropout(incoming, rate=drop_rate)
else:
    return incoming
```