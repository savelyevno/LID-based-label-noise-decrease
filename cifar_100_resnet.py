import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras import backend
from networks import batch_norm


BN_AXIS = 3


def conv2d(filters, kernel_size, strides, padding, x, reg_loss):
    if strides is None:
        layer = Conv2D(filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_normal", padding=padding)
    else:
        layer = Conv2D(filters=filters, kernel_size=kernel_size,
                       strides=strides,
                       kernel_initializer="he_normal", padding=padding)

    z = layer(x)
    assert layer.weights[0].shape[:2] == kernel_size
    reg_loss = tf.add(reg_loss, tf.nn.l2_loss(layer.weights[0]))

    return z, reg_loss


def build_cifar_100(x, is_training, depth=6, num_classes=100):
    num_conv = 3
    # decay = 2e-3
    reg_loss = tf.Variable(0, trainable=False, dtype=tf.float32)

    # 1 conv + BN + relu
    filters = 16
    b, reg_loss = conv2d(filters, (num_conv, num_conv), None, 'same', x, reg_loss)
    b = batch_norm(b, is_training)
    b = Activation("relu")(b)

    # 1 res, no striding
    b, reg_loss = residual(b, num_conv, filters, reg_loss, is_training, first=True)  # 2 layers inside
    for _ in np.arange(1, depth):  # start from 1 => 2 * depth in total
        b, reg_loss = residual(b, num_conv, filters, reg_loss, is_training)

    filters *= 2

    # 2 res, with striding
    b, reg_loss = residual(b, num_conv, filters, reg_loss, is_training, more_filters=True)
    for _ in np.arange(1, depth):
        b, reg_loss = residual(b, num_conv, filters, reg_loss, is_training)

    filters *= 2

    # 3 res, with striding
    b, reg_loss = residual(b, num_conv, filters, reg_loss, is_training, more_filters=True)
    for _ in np.arange(1, depth):
        b, reg_loss = residual(b, num_conv, filters, reg_loss, is_training)

    pre_lid_input = batch_norm(b, is_training)
    b = Activation("relu")(pre_lid_input)

    b = AveragePooling2D(pool_size=(8, 8), strides=(1, 1),
                         padding="valid")(b)

    out = Flatten(name='lid_input')(b)

    dense_layer = Dense(units=num_classes, input_shape=out.shape[1:], kernel_initializer="he_normal")
    dense = dense_layer(out)
    reg_loss = tf.add(reg_loss, tf.nn.l2_loss(dense_layer.weights[0]))

    act = Activation("softmax")(dense)

    return Flatten(name='pre_lid_input')(pre_lid_input), out, dense, act, reg_loss


def residual(inp, num_conv, filters, reg_loss, is_training, more_filters=False, first=False):
    def f(x, reg_loss):
        # in_channel = input._keras_shape[1]
        out_channel = filters

        if more_filters and not first:
            # out_channel = in_channel * 2
            stride = 2
        else:
            # out_channel = in_channel
            stride = 1

        if not first:
            b = batch_norm(x, is_training)
            b = Activation("relu")(b)
        else:
            b = x

        b, reg_loss = conv2d(out_channel, (num_conv, num_conv), (stride, stride), 'same', b, reg_loss)
        b = batch_norm(b, is_training)
        b = Activation("relu")(b)
        res, reg_loss = conv2d(out_channel, (num_conv, num_conv), None, 'same', b, reg_loss)

        # check and match number of filter for the shortcut
        input_shape = backend.int_shape(x)
        residual_shape = backend.int_shape(res)
        if not input_shape[3] == residual_shape[3]:
            stride_width = int(round(input_shape[1] / residual_shape[1]))
            stride_height = int(round(input_shape[2] / residual_shape[2]))

            x, reg_loss = conv2d(residual_shape[3], (1, 1), (stride_width, stride_height), 'valid', x, reg_loss)

        return Add()([x, res]), reg_loss

    return f(inp, reg_loss)
