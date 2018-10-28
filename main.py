# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from PIL import Image

from consts import *

from Timer import timer
from preprocessing import read_dataset
from batch_iterate import batch_iterator
from build_model import build_deepnn
from lid import get_LID_calc_op, get_LID_per_element_calc_op


def calc_LID(X, Y, LID_calc_op, x, keep_prob, is_training):
    LID_score = 0

    i_batch = -1
    for batch in batch_iterator(X, Y, LID_BATCH_SIZE, True):
        i_batch += 1

        if i_batch == LID_BATCH_CNT:
            break

        batch_LID_scores = LID_calc_op.eval(feed_dict={x: batch[0], keep_prob: 1, is_training: False})
        LID_score += batch_LID_scores.mean()

    LID_score /= LID_BATCH_CNT

    return LID_score


def calc_LID_per_element(X, Y, LID_input_layer, x, LID_per_element_calc_op, LID_sample_set_pl, is_training):
    print('\ncalculating LID for the whole dataset...')

    #
    # FILL LID SAMPLE SET
    #

    batch_size = 1000

    LID_sample_set = np.empty((0, LID_input_layer.shape[1]), dtype=np.float32)
    i_batch = -1
    for batch in batch_iterator(X, Y, batch_size):
        i_batch += 1

        LID_layer = LID_input_layer.eval(feed_dict={x: batch[0], is_training: False})
        LID_sample_set = np.vstack((LID_sample_set, LID_layer))

        # if (i_batch + 1) % 100 == 0:
        #     print('\t filled LID sample set for %d/%d' % ((i_batch + 1) * BATCH_SIZE, LID_SAMPLE_SET_SIZE))

        if LID_sample_set.shape[0] >= LID_SAMPLE_SET_SIZE:
            break

    #
    # CALCULATE LID PER DATASET ELEMENT
    #

    epoch_LIDs_per_element = np.empty((0,))
    i_batch = -1
    for batch in batch_iterator(X, Y, batch_size, False):
        i_batch += 1

        LIDs_per_batch_element = LID_per_element_calc_op.eval(
            feed_dict={x: batch[0], LID_sample_set_pl: LID_sample_set, is_training: False})
        epoch_LIDs_per_element = np.append(epoch_LIDs_per_element, LIDs_per_batch_element)

        # if (i_batch + 1) % 100 == 0:
        #     print('\t calculated LID for %d/%d' % ((i_batch + 1) * batch_size, DATASET_SIZE))

    epoch_LIDs_per_element = epoch_LIDs_per_element.reshape([-1, 1])

    return epoch_LIDs_per_element


def train_model(model_name='model', train_dataset_name='train', use_LID_based_labels=True):
    # Import data
    X, Y = read_dataset(train_dataset_name)

    X_test, Y_test = read_dataset('test')

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784], name='x')

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    # Build the graph for the deep net
    LID_input_layer, y_conv, keep_prob = build_deepnn(x, is_training)

    #
    # CREATE LOSS & OPTIMIZER
    #

    with tf.name_scope('loss'):
        alpha_var = tf.Variable(1, False, dtype=tf.float32, name='alpha')
        modified_labels = tf.identity(alpha_var * y_ + (1 - alpha_var.value()) * tf.one_hot(tf.argmax(y_conv, 1), 10),
                                      name='modified_labels')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=modified_labels,
                                                                   logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

    #
    # PREPARE LID CALCULATION OP
    #

    LID_per_epoch = np.array([])
    LID_calc_op = get_LID_calc_op(LID_input_layer)

    # LID per dataset element

    LIDs_per_element = np.empty((DATASET_SIZE, 0), dtype=np.float32)

    LID_sample_set_pl = tf.placeholder(dtype=tf.float32, shape=[LID_SAMPLE_SET_SIZE, LID_input_layer.shape[1]],
                                           name='LID_sample_set')
    LID_per_element_calc_op = get_LID_per_element_calc_op(LID_input_layer, LID_sample_set_pl)

    turning_epoch = -1  # number of epoch where we turn from regular loss function to the modified one

    #
    # CREATE SUMMARIES
    #

    tf.summary.scalar(name='cross_entropy', tensor=cross_entropy)
    tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    summary = tf.summary.merge_all()

    LID_summary_scalar = tf.placeholder(tf.float32)
    LID_summary = tf.summary.scalar(name='LID', tensor=LID_summary_scalar)

    LID_per_element_summary_pl = tf.placeholder(dtype=tf.float32, shape=[DATASET_SIZE,])
    LID_per_element_summary_hist = tf.summary.histogram(name='LID_per_element', values=LID_per_element_summary_pl)

    alpha_summary_scalar = tf.placeholder(tf.float32)
    alpha_summary = tf.summary.scalar(name='alpha', tensor=alpha_summary_scalar)

    test_accuracy_summary_scalar = tf.placeholder(tf.float32)
    test_accuracy_summary = tf.summary.scalar(name='test_accuracy', tensor=test_accuracy_summary_scalar)

    per_epoch_summary = tf.summary.merge([LID_summary, alpha_summary, test_accuracy_summary, LID_per_element_summary_hist])

    saver = tf.train.Saver()
    model_path = 'checkpoints/' + model_name + '/'

    #
    # SESSION START
    #

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)

        sess.run(tf.global_variables_initializer())

        #
        # CALCULATE AND LOG INITIAL LID SCORE
        #

        initial_LID_score = calc_LID(X, Y, LID_calc_op, x, keep_prob, is_training)
        LID_per_epoch = np.append(LID_per_epoch, initial_LID_score)

        print('initial LID score:', initial_LID_score)

        LID_summary_str = sess.run(LID_summary, feed_dict={LID_summary_scalar: initial_LID_score})
        summary_writer.add_summary(LID_summary_str, 0)
        summary_writer.flush()

        #
        # CALCULATE INITIAL LID PER DATASET ELEMENT AND LOG IT
        #

        epoch_LIDs_per_element = calc_LID_per_element(X, Y, LID_input_layer, x, LID_per_element_calc_op,
                                                      LID_sample_set_pl, is_training)
        LIDs_per_element = np.hstack((LIDs_per_element, epoch_LIDs_per_element))

        print('LID mean: %g, std. dev.: %g, min: %g, max: %g' % (epoch_LIDs_per_element.mean(),
                                                                 epoch_LIDs_per_element.var() ** 0.5,
                                                                 epoch_LIDs_per_element.min(),
                                                                 epoch_LIDs_per_element.max()))
        LID_per_element_summary_hist_str = LID_per_element_summary_hist.eval(
            feed_dict={LID_per_element_summary_pl: LIDs_per_element[:, -1]}
        )
        summary_writer.add_summary(LID_per_element_summary_hist_str, 0)
        summary_writer.flush()

        #
        # EPOCH LOOP
        #

        i_step = -1
        for i_epoch in range(1, N_EPOCHS + 1):
            print('___________________________________________________________________________')
            print('\nSTARTING EPOCH %d\n' % (i_epoch, ))

            #
            # TRAIN
            #

            print('\nstarting training:')

            i_batch = -1
            for batch in batch_iterator(X, Y, BATCH_SIZE):
                i_batch += 1
                i_step += 1

                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, is_training: True})

                if i_step % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0, is_training: False})
                    print('\tstep %d, training accuracy %g' % (i_step, train_accuracy))

                    summary_str = sess.run(summary, feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0, is_training: False})
                    summary_writer.add_summary(summary_str, i_step)
                    summary_writer.flush()

            #
            # CALCULATE LID
            #

            new_LID_score = calc_LID(X, Y, LID_calc_op, x, keep_prob, is_training)
            LID_per_epoch = np.append(LID_per_epoch, new_LID_score)

            print('\nLID score after %dth epoch: %g' % (i_epoch, new_LID_score,))

            #
            # CALCULATE LID PER DATASET ELEMENT
            #

            epoch_LIDs_per_element = calc_LID_per_element(X, Y, LID_input_layer, x, LID_per_element_calc_op,
                                                          LID_sample_set_pl, is_training)
            LIDs_per_element = np.hstack((LIDs_per_element, epoch_LIDs_per_element))

            print('LID mean: %g, std. dev.: %g, min: %g, max: %g' % (epoch_LIDs_per_element.mean(),
                                                                     epoch_LIDs_per_element.var() ** 0.5,
                                                                     epoch_LIDs_per_element.min(),
                                                                     epoch_LIDs_per_element.max()))

            #
            # CHECK FOR STOPPING INIT PERIOD
            #

            if use_LID_based_labels:
                if turning_epoch == -1 and i_epoch > EPOCH_WINDOW:
                    last_w_LIDs = LID_per_epoch[-EPOCH_WINDOW - 1: -1]

                    lid_check_value = new_LID_score - last_w_LIDs.mean() - 2 * last_w_LIDs.var() ** 0.5
                    print('LID check:', lid_check_value)
                    if lid_check_value > 0:
                        turning_epoch = i_epoch - 1

                        saver.restore(sess, model_path + str(i_epoch - 1))

                        print('Turning point passed, reverting to previous epoch and starting using modified loss')

            #
            # MODIFYING ALPHA
            #

            if turning_epoch != -1:
                new_alpha_value = np.exp(-(i_epoch / N_EPOCHS) * (LID_per_epoch[-1] / LID_per_epoch[:-1].min()))
                print('\nnew alpha value:', new_alpha_value)
            else:
                new_alpha_value = 1

            sess.run(alpha_var.assign(new_alpha_value))

            #
            # TEST ACCURACY
            #

            test_accuracy = 0
            i_batch = -1
            for batch in batch_iterator(X_test, Y_test, 100, False):
                i_batch += 1

                partial_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0, is_training: False})

                test_accuracy = (i_batch * test_accuracy + partial_accuracy) / (i_batch + 1)

            print('\ntest accuracy after %dth epoch: %g' % (i_epoch, test_accuracy))

            summary_str = sess.run(per_epoch_summary, feed_dict={
                LID_summary_scalar: LID_per_epoch[-1],
                alpha_summary_scalar: new_alpha_value,
                test_accuracy_summary_scalar: test_accuracy,
                LID_per_element_summary_pl: LIDs_per_element[:, -1]})
            summary_writer.add_summary(summary_str, i_step + 1)
            summary_writer.flush()

            #
            # SAVE MODEL
            #

            checkpoint_file = model_path + str(i_epoch)
            saver.save(sess, checkpoint_file)

            np.save('LID_matrices/LID_matrix_' + model_name, LIDs_per_element)


def full_test(model_name, epoch):
    with tf.Session() as sess:
        full_model_name = 'checkpoints/' + model_name + '/' + str(epoch) + '.meta'
        saver = tf.train.import_meta_graph(full_model_name)
        saver.restore(sess, full_model_name)

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name('x:0')
        y_ = graph.get_tensor_by_name('y_:0')
        keep_prob = graph.get_tensor_by_name('dropout/keep_prob:0')
        is_training = graph.get_tensor_by_name('is_training:0')
        accuracy_op = graph.get_tensor_by_name('accuracy_1:0')

        X, Y = read_dataset('test')

        test_accuracy = 0
        i_batch = -1
        for batch in batch_iterator(X, Y, 100, False):
            i_batch += 1

            partial_accuracy = accuracy_op.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0, is_training: False})

            test_accuracy = (i_batch * test_accuracy + partial_accuracy) / (i_batch + 1)

        print('test accuracy %g' % test_accuracy)


def test_image(image_path, model_name='model'):
    def soft_max(arg):
        exp = np.exp(arg)
        sm = np.sum(exp)
        return exp/sm

    img = np.float32(Image.open(image_path).convert('L'))/256

    x = np.reshape(img, (1, 784))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('check_point/' + model_name + '.meta')
        saver.restore(sess, 'check_point/' + model_name)

        graph = tf.get_default_graph()

        x_op = graph.get_tensor_by_name('x:0')
        keep_prob_op = graph.get_tensor_by_name('dropout/keep_prob:0')
        probs_op = graph.get_tensor_by_name('fc2/probs:0')

        res = np.float32(probs_op.eval(feed_dict={x_op: x, keep_prob_op: 1.0})[0])

        res = soft_max(res)

        dict_res = {}
        for i in range(10):
            dict_res[i] = res[i]

        print(dict_res)
        print(np.argmax(res))


if __name__ == '__main__':
    train_model('model25_noLID', 'train25', False)
    # full_test('model')
    # test_image('testSample/img_3.jpg')
