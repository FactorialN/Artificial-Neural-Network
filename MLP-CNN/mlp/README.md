#What I changed
In this PA I changed the following things:

## main.py
I changed
```mlp_model = Model()``` to
```mlp_model = Model(True)```or```mlp_model = Model(False)```.

## model.py
1. Added a is_train to parameters
2. filled in the blank of arguments in forward
```
self.loss, self.pred, self.acc = self.forward(is_Train, tf.AUTO_REUSE)
self.loss_val, self.pred_val, self.acc_val = self.forward(is_Train, True)
```
3. implemented layer structure in forward
```
w_fc1 = tf.get_variable('w_fc1', shape=[28*28, 256])
b_fc1 = tf.get_variable('b_fc1', shape=[256])
h_fc1 = tf.nn.relu(batch_normalization_layer(tf.matmul(self.x_, w_fc1) + b_fc1, is_train=is_train) )
#h_fc1 = tf.nn.relu(tf.matmul(self.x_, w_fc1) + b_fc1)
ho_fc1 = dropout_layer(h_fc1, 0.3, is_train)
w_fc2 = tf.get_variable('w_fc2', shape=[256, 10])
b_fc2 = tf.get_variable('b_fc2', shape=[10])
logits = tf.matmul(ho_fc1, w_fc2) + b_fc2
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