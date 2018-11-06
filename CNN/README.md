Everything I changed in the source code:

in loss.py:\
I implemented the class SoftmaxCrossEntropyLoss.

in functions.py:\
I implemented the four functions conv2d_forward, conv2d_backward, pooling_forward, pooling_backward.

in run_cnn.py:\
I added a part to output training time and plot a graph. \
I modified the parameters in config and layers.

in solve_net.py:\
I added a part to change the learning rate dynamically with loss value in train_net method.\
I added vis_square function to visualize the result.

in network.py:\
I added a new method called fforward for visualization.