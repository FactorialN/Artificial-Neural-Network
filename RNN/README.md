# PA4 README

In this PA, I changed the following part of files:

## model.py
1. I implemented embed, Cell and bidirectional rnn model.
```
# todo: implement embedding inputs
self.embed_input = tf.nn.embedding_lookup(embed, self.index_input)  # shape: [batch, length, num_embed_units]

# todo: implement 3 RNNCells (BasicRNNCell, GRUCell, BasicLSTMCell) in a multi-layer setting with #num_units neurons and #num_layers layers
if num_layers == 1:
     cell_fw = BasicLSTMCell(num_units)
     cell_bw = BasicLSTMCell(num_units)
else:
     cell_fw = MultiRNNCell([BasicRNNCell(num_units) for i in range(num_layers)])
     cell_bw = MultiRNNCell([BasicRNNCell(num_units) for i in range(num_layers)])

# todo: implement bidirectional RNN
outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embed_input, sequence_length=self.texts_length,                                                        dtype=tf.float32, scope="rnn")
```
2. I implemented self-attension mechanism.
```
A1 = tf.tanh(tf.matmul(tf.reshape(H, [-1, 2 * num_units]), Ws1))
A = tf.nn.softmax(tf.matmul(A1, Ws2))
A = tf.reshape(A, [batch_size, -1, param_r])
M = tf.matmul(tf.transpose(A, perm=[0, 2, 1]), H)  # shape: [batch, param_r, 2*num_units]
flatten_M = tf.reshape(M, shape=[batch_size, param_r * 2 * num_units])
```
```
self.penalized_term = tf.norm((tf.matmul(tf.transpose(A, perm=[0, 2, 1]), A) - identity)) ** 2
```
3. I normalize the code with Pycharm's default normalizer.

## rnn_cell.py
1. I implemented method GRUCell.call:
```
with vs.variable_scope("candidate"):
    #todo: calculate c and new_h according to GRU
    c = tf.layers.dense(tf.concat([inputs, r * state], 1), self._num_units, activation=self._activation, use_bias=True)
new_h = u * state + (1 - u) * c
```
2. I implemeted method BasicLSTMCell.call:
```
val = tf.layers.dense(tf.concat([inputs, h], 1), 4 * self._num_units, use_bias=True)
i, ct, f, o = tf.split(value=val, num_or_size_splits=4, axis=1)
new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(ct))
new_h = tf.tanh(new_c) * sigmoid(o)
```