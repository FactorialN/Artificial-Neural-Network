# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y_ = tf.placeholder(tf.int32, [None])

        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc = self.forward(is_train, tf.AUTO_REUSE)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_train, True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()

        for item in self.params:
            print(item)
        
        # TODO:  maybe you need to update the parameter of batch_normalization?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=None):
    
        with tf.variable_scope("model", reuse=reuse):
            # TODO: implement input -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Conv -- BN -- ReLU -- Dropout -- MaxPool -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            """conv1 = tf.layers.conv2d(
                name='conv1',
                inputs=self.x_,
                filters=4,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)"""
            k_conv1 = tf.get_variable(name='k_conv1', shape=[3, 3, 1, 4])
            b_conv1 = tf.get_variable(name='b_conv1', shape=[4])
            h_conv1 = tf.nn.conv2d(self.x_, k_conv1, padding='SAME', strides=[1, 1, 1, 1]) + b_conv1
            hr_conv1 = dropout_layer(tf.nn.relu(batch_normalization_layer(h_conv1, is_train)), 0.5, is_train)
            #hr_conv1 = dropout_layer(tf.nn.relu(h_conv1), 0.3, is_train)
            """pool1 = tf.layers.max_pooling2d(
                name='pool1',
                inputs=conv1,
                pool_size=[2, 2],
                strides=2)"""
            p_pool1 = tf.nn.max_pool(hr_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            """conv2 = tf.layers.conv2d(
                name='conv2',
                inputs=pool1,
                filters=4,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)"""
            k_conv2 = tf.get_variable(name='k_conv2', shape=[3, 3, 4, 4])
            b_conv2 = tf.get_variable(name='b_conv2', shape=[4])
            h_conv2 = tf.nn.conv2d(p_pool1, k_conv2, padding='SAME', strides=[1, 1, 1, 1]) + b_conv2
            hr_conv2 = dropout_layer(tf.nn.relu(batch_normalization_layer(h_conv2, is_train)), 0.5, is_train)
            #hr_conv2 = dropout_layer(tf.nn.relu(h_conv2), 0.3, is_train)
            """pool2 = tf.layers.max_pooling2d(
                name='pool2',
                inputs=conv2,
                pool_size=[2, 2],
                strides=2)"""
            p_pool2 = tf.nn.max_pool(hr_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # flt = tf.reshape(pool2, shape=[-1, 196])
            flt = tf.reshape(p_pool2, shape=[-1, 196])
            w_fc3 = tf.get_variable(name='w_fc3', shape=[196, 10])
            b_fc3 = tf.get_variable(name='b_fc3', shape=[10])
            logits = tf.matmul(flt, w_fc3) + b_fc3

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch
        
        return loss, pred, acc

def batch_normalization_layer(incoming, is_train=True):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should return calculate mu and sigma by mini-batch
    #       If isTrain is False, you must estimate mu and sigma from training data
    if is_train:
        return tf.layers.batch_normalization(incoming, momentum=0.99, epsilon=1e-5, training=True)
    else:
        return tf.layers.batch_normalization(incoming, training=False)
    
def dropout_layer(incoming, drop_rate, is_train=True):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    if is_train:
        return tf.layers.dropout(incoming, rate=drop_rate)
    else:
        return incoming
