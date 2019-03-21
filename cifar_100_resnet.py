import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add
from tensorflow.keras import backend
from networks import batch_norm


BN_AXIS = 3


def build_cifar_100(x, is_training, depth=6, num_classes=100):
    num_conv = 3
    decay = 2e-3

    # 1 conv + BN + relu
    filters = 16
    b = Conv2D(filters=filters, kernel_size=(num_conv, num_conv),
               kernel_initializer="he_normal", padding="same",
               kernel_regularizer=l2(decay), bias_regularizer=l2(0))(x)
    b = batch_norm(b, is_training)
    b = Activation("relu")(b)

    # 1 res, no striding
    b = residual(num_conv, filters, decay, is_training, first=True)(b)  # 2 layers inside
    for _ in np.arange(1, depth):  # start from 1 => 2 * depth in total
        b = residual(num_conv, filters, decay, is_training)(b)

    filters *= 2

    # 2 res, with striding
    b = residual(num_conv, filters, decay, is_training, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(num_conv, filters, decay, is_training)(b)

    filters *= 2

    # 3 res, with striding
    b = residual(num_conv, filters, decay, is_training, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(num_conv, filters, decay, is_training)(b)

    pre_lid_input = batch_norm(b, is_training)
    b = Activation("relu")(pre_lid_input)

    b = AveragePooling2D(pool_size=(8, 8), strides=(1, 1),
                         padding="valid")(b)

    out = Flatten(name='lid_input')(b)

    dense = Dense(units=num_classes, kernel_initializer="he_normal",
                  kernel_regularizer=l2(decay), bias_regularizer=l2(0))(out)

    act = Activation("softmax")(dense)

    return Flatten(name='pre_lid_input')(pre_lid_input), out, dense, act


def residual(num_conv, filters, decay, is_training, more_filters=False, first=False):
    def f(x):
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

        b = Conv2D(filters=out_channel,
                   kernel_size=(num_conv, num_conv),
                   strides=(stride, stride),
                   kernel_initializer="he_normal", padding="same",
                   kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)
        b = batch_norm(b, is_training)
        b = Activation("relu")(b)
        res = Conv2D(filters=out_channel,
                     kernel_size=(num_conv, num_conv),
                     kernel_initializer="he_normal", padding="same",
                     kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)

        # check and match number of filter for the shortcut
        input_shape = backend.int_shape(x)
        residual_shape = backend.int_shape(res)
        if not input_shape[3] == residual_shape[3]:
            stride_width = int(round(input_shape[1] / residual_shape[1]))
            stride_height = int(round(input_shape[2] / residual_shape[2]))

            x = Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                       strides=(stride_width, stride_height),
                       kernel_initializer="he_normal",
                       padding="valid", kernel_regularizer=l2(decay))(x)

        return Add()([x, res])

    return f
