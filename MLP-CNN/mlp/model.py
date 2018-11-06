# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class Model:
    def __init__(self,
                 is_Train=True,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28])
        self.y_ = tf.placeholder(tf.int32, [None])

        # TODO:  fill the blank of the arguments

        self.loss, self.pred, self.acc = self.forward(is_Train, tf.AUTO_REUSE)
        self.loss_val, self.pred_val, self.acc_val = self.forward(is_Train, True)
        
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()

        for item in self.params:
            print(item)

        # TODO:  maybe you need to update the parameter of batch_normalization?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
                                    
    def forward(self, is_train, reuse=None):
    
        with tf.variable_scope("model", reuse=reuse):
            # TODO:  implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Linear Layer
            w_fc1 = tf.get_variable('w_fc1', shape=[28*28, 256])
            b_fc1 = tf.get_variable('b_fc1', shape=[256])
            h_fc1 = tf.nn.relu(batch_normalization_layer(tf.matmul(self.x_, w_fc1) + b_fc1, is_train=is_train) )
            #h_fc1 = tf.nn.relu(tf.matmul(self.x_, w_fc1) + b_fc1)
            ho_fc1 = dropout_layer(h_fc1, 0.3, is_train)
            w_fc2 = tf.get_variable('w_fc2', shape=[256, 10])
            b_fc2 = tf.get_variable('b_fc2', shape=[10])
            logits = tf.matmul(ho_fc1, w_fc2) + b_fc2

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the pr
        ''# ediction result
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

