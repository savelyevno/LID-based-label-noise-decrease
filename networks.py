import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np

from consts import FC_WIDTH, N_CLASSES


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, name=None):
    """weight_variable generates a weight variable of a given shape."""
    n_in = 1
    for i in range(len(shape) - 1):
        n_in *= shape[i]

    initial = tf.truncated_normal(shape, stddev=(2 / n_in)**0.5)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)


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


def build_mnist(x, is_training, n_blocks):
    batch_size = tf.shape(x)[0]

    # Reshape to use within a convolutional neural net.
    # Last dimension is for "lid_features" - there is only one here, since images are
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

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 128 lid_features.
    with tf.name_scope('fc1'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        Ws_fc1 = []
        bs_fc1 = []
        hs_fc1 = []
        acts_fc1 = []
        for i in range(n_blocks):
            Ws_fc1.append(weight_variable([7 * 7 * 64, FC_WIDTH['mnist']], 'W_' + str(i + 1)))
            bs_fc1.append(bias_variable([FC_WIDTH['mnist']], 'b_' + str(i + 1)))
            h = tf.matmul(h_pool2_flat, Ws_fc1[i]) + bs_fc1[i]
            bn_h = tf.identity(batch_norm(h, is_training), name='bn_' + str(i + 1))
            hs_fc1.append(bn_h)
            a = tf.nn.relu(bn_h, name='relu_' + str(i + 1))
            acts_fc1.append(a)

        h_fc1 = tf.reshape(tf.stack(hs_fc1, 1), (-1, FC_WIDTH['mnist'] * n_blocks), name='pre_lid_input')
        a_fc1 = tf.nn.relu(h_fc1, name='lid_input')

    Ws_fc2 = []
    bs_fc2 = []
    logits_fc2 = []

    # Map the 1024 lid_features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        for i in range(n_blocks):
            Ws_fc2.append(weight_variable([FC_WIDTH['mnist'], N_CLASSES], 'W_' + str(i + 1)))
            bs_fc2.append(bias_variable([N_CLASSES], 'b_' + str(i + 1)))
            logits_fc2.append(tf.identity(tf.matmul(acts_fc1[i], Ws_fc2[i]) + bs_fc2[i], 'logits_' + str(i + 1)))

        block_w = tf.reshape(tf.one_hot(tf.random_uniform((batch_size,), 0, n_blocks, dtype=tf.int32), n_blocks),
                             (batch_size, n_blocks, 1))

        blocks_logits = tf.stack(logits_fc2, 1)
        preds = tf.reduce_mean(tf.nn.softmax(blocks_logits), 1)

        logits = tf.reduce_sum(blocks_logits * block_w, 1, name='logits')

    return h_fc1, a_fc1, logits, preds


def build_cifar_10(x, is_training):
    # Block 1
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 3, 64])
        b_conv1 = bias_variable([64])
        a_conv1 = conv2d(x, W_conv1) + b_conv1
        normed_a_conv1 = batch_norm(a_conv1, is_training)
        h_conv1 = tf.nn.relu(normed_a_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 64, 64])
        b_conv2 = bias_variable([64])
        a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2
        normed_a_conv2 = batch_norm(a_conv2, is_training)
        h_conv2 = tf.nn.relu(normed_a_conv2)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv2)

    # Block 2
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        a_conv3 = conv2d(h_pool1, W_conv3) + b_conv3
        normed_a_conv3 = batch_norm(a_conv3, is_training)
        h_conv3 = tf.nn.relu(normed_a_conv3)

    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, 128, 128])
        b_conv4 = bias_variable([128])
        a_conv4 = conv2d(h_conv3, W_conv4) + b_conv4
        normed_a_conv4 = batch_norm(a_conv4, is_training)
        h_conv4 = tf.nn.relu(normed_a_conv4)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv4)

    # Block 3
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([3, 3, 128, 196])
        b_conv5 = bias_variable([196])
        a_conv5 = conv2d(h_pool2, W_conv5) + b_conv5
        normed_a_conv5 = batch_norm(a_conv5, is_training)
        h_conv5 = tf.nn.relu(normed_a_conv5)

    with tf.name_scope('conv6'):
        W_conv6 = weight_variable([3, 3, 196, 196])
        b_conv6 = bias_variable([196])
        a_conv6 = conv2d(h_conv5, W_conv6) + b_conv6
        normed_a_conv6 = batch_norm(a_conv6, is_training)
        h_conv6 = tf.nn.relu(normed_a_conv6)

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv6)

    with tf.name_scope('flatten'):
        h_flattened = tf.reshape(h_pool3, [-1, 4 * 4 * 196])

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 4 * 196, FC_WIDTH['cifar-10']])
        W_l2_reg_sum = tf.nn.l2_loss(W_fc1)
        b_fc1 = bias_variable([FC_WIDTH['cifar-10']])
        b_l2_reg_sum = tf.nn.l2_loss(b_fc1)

        a_fc1 = tf.matmul(h_flattened, W_fc1) + b_fc1
        normed_a_fc1 = tf.identity(batch_norm(a_fc1, is_training), name='pre_lid_input')
        h_fc1 = tf.nn.relu(normed_a_fc1, name='lid_input')

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([FC_WIDTH['cifar-10'], N_CLASSES])
        b_fc2 = bias_variable([N_CLASSES])

        a_fc2 = tf.identity(tf.matmul(h_fc1, W_fc2) + b_fc2, name='logits')

    return normed_a_fc1, h_fc1, a_fc2, W_l2_reg_sum, b_l2_reg_sum
