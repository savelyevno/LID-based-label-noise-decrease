import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from consts import FC_WIDTH


def build_deepnn(x, is_training):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """

    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def batch_norm(input_tnsr, is_training):
        with tf.variable_scope('batch_norm'):
            phase_train = tf.convert_to_tensor(is_training, dtype=tf.bool)

            if len(input_tnsr.get_shape()) > 2:
                n_out = int(input_tnsr.get_shape()[3])
            else:
                n_out = int(input_tnsr.get_shape()[1])

            beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=input_tnsr.dtype),
                               name='beta', trainable=True, dtype=input_tnsr.dtype)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=input_tnsr.dtype),
                                name='gamma', trainable=True, dtype=input_tnsr.dtype)

            axes = list(np.arange(0, len(input_tnsr.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(input_tnsr, axes, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.995)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = control_flow_ops.cond(phase_train,
                                              mean_var_with_update,
                                              lambda: (ema.average(batch_mean), ema.average(batch_var)))

            normed = tf.nn.batch_normalization(input_tnsr, mean, var, beta, gamma, 1e-3)

        return normed

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        a_conv1 = conv2d(x_image, W_conv1) + b_conv1
        b_norm_conv1 = batch_norm(a_conv1, is_training)
        h_conv1 = tf.nn.relu(b_norm_conv1)
        # h_conv1 = tf.nn.relu(a_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        a_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
        b_norm_conv2 = batch_norm(a_conv2, is_training)
        h_conv2 = tf.nn.relu(b_norm_conv2)
        # h_conv2 = tf.nn.relu(a_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 128 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, FC_WIDTH])
        b_fc1 = bias_variable([FC_WIDTH])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        a_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        b_norm_fc1 = batch_norm(a_fc1, is_training)
        h_fc1 = tf.nn.relu(b_norm_fc1)
        # h_fc1 = tf.nn.relu(a_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([FC_WIDTH, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.identity(tf.matmul(h_fc1, W_fc2) + b_fc2, name='probs')
        # y_conv = tf.identity(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='probs')

    # print(keep_prob.name)

    return h_fc1, y_conv, keep_prob
