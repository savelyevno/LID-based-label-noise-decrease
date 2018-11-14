import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from consts import *
from Timer import timer
from preprocessing import read_dataset
from batch_iterate import batch_iterator, batch_iterator_with_indices
from build_model import build_deepnn
from lid import get_LID_calc_op, get_LID_per_element_calc_op
from tools import bitmask_contains


class Model:
    def __init__(self, model_name, lid_update_mode, lid_log_mask):
        """

        :param model_name:      name of the model
        :param lid_update_mode: 0: vanilla
                                1: as in the paper
                                2: per element
        :param lid_log_mask:    in case it contains
                                    last bit: logs LID data from the paper
                                    second bit: logs LID data per element
        """
        self.lid_update_mode = lid_update_mode
        self.lid_log_mask = lid_log_mask
        self.model_name = model_name

    @staticmethod
    def _build_nn(x, is_training):
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

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
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

    def _build(self):
        # Create the model
        self.nn_input = tf.placeholder(tf.float32, [None, 784], name='x')

        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # Build the graph for the deep net
        self.lid_input_layer, self.y_conv, self.keep_prob = self._build_nn(self.nn_input, self.is_training)

        #
        # CREATE LOSS & OPTIMIZER
        #

        with tf.name_scope('loss'):
            if self.lid_update_mode == 0:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                                                           logits=self.y_conv)
            if self.lid_update_mode == 1 or bitmask_contains(self.lid_log_mask, 0):
                self.alpha_var = tf.Variable(1, False, dtype=tf.float32, name='alpha')

                if self.lid_update_mode == 1:
                    modified_labels = tf.identity(
                        self.alpha_var * self.y_ + (1 - self.alpha_var.value()) * tf.one_hot(tf.argmax(self.y_conv, 1), 10),
                        name='modified_labels')
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=modified_labels,
                                                                               logits=self.y_conv)
            if self.lid_update_mode == 2:
                self.element_weights_pl = tf.placeholder(dtype=np.float32,
                                                         shape=[None, 1],
                                                         name='element_weights')
                modified_labels = tf.identity(
                    self.element_weights_pl * self.y_ +
                    (1 - self.element_weights_pl) * tf.one_hot(tf.argmax(self.y_conv, 1), 10),
                    name='modified_labels_per_element')
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=modified_labels,
                                                                           logits=self.y_conv)

        self.cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

        if self.lid_update_mode == 1 or bitmask_contains(self.lid_log_mask, 0):
            #
            # PREPARE LID CALCULATION OP
            #

            self.lid_per_epoch = np.array([])
            self.lid_calc_op = get_LID_calc_op(self.lid_input_layer)

        if self.lid_update_mode == 2 or bitmask_contains(self.lid_log_mask, 1):
            #
            # PREPARE LID PER DATASET ELEMENT CALCULATION
            #

            self.lid_sample_set_pl = tf.placeholder(dtype=tf.float32, shape=[LID_SAMPLE_SET_SIZE, self.lid_input_layer.shape[1]],
                                                    name='LID_sample_set')
            self.lid_per_element_calc_op = get_LID_per_element_calc_op(self.lid_input_layer, self.lid_sample_set_pl)

            self.element_weights_matrix = np.empty((DATASET_SIZE, 0), dtype=np.float32)

    def train(self, train_dataset_name='train'):
        def calc_lid(X, Y, lid_calc_op, x, keep_prob, is_training):
            lid_score = 0

            i_batch = -1
            for batch in batch_iterator(X, Y, LID_BATCH_SIZE, True):
                i_batch += 1

                if i_batch == LID_BATCH_CNT:
                    break

                batch_lid_scores = lid_calc_op.eval(feed_dict={x: batch[0], keep_prob: 1, is_training: False})
                lid_score += batch_lid_scores.mean()

            lid_score /= LID_BATCH_CNT

            return lid_score

        def calc_lid_per_element(X, Y, lid_input_layer, x, lid_per_element_calc_op, lid_sample_set_pl, is_training):
            print('\ncalculating LID for the whole dataset...')

            #
            # FILL LID SAMPLE SET
            #

            batch_size = 1000

            lid_sample_set = np.empty((0, lid_input_layer.shape[1]), dtype=np.float32)
            i_batch = -1
            for batch in batch_iterator(X, Y, batch_size):
                i_batch += 1

                lid_layer = lid_input_layer.eval(feed_dict={x: batch[0], is_training: False})
                lid_sample_set = np.vstack((lid_sample_set, lid_layer))

                # if (i_batch + 1) % 100 == 0:
                #     print('\t filled LID sample set for %d/%d' % ((i_batch + 1) * BATCH_SIZE, LID_SAMPLE_SET_SIZE))

                if lid_sample_set.shape[0] >= LID_SAMPLE_SET_SIZE:
                    break

            #
            # CALCULATE LID PER DATASET ELEMENT
            #

            epoch_lids_per_element = np.empty((0,))
            i_batch = -1
            for batch in batch_iterator(X, Y, batch_size, False):
                i_batch += 1

                lids_per_batch_element = lid_per_element_calc_op.eval(
                    feed_dict={x: batch[0], lid_sample_set_pl: lid_sample_set, is_training: False})
                epoch_lids_per_element = np.append(epoch_lids_per_element, lids_per_batch_element)

                # if (i_batch + 1) % 100 == 0:
                #     print('\t calculated LID for %d/%d' % ((i_batch + 1) * batch_size, DATASET_SIZE))

            epoch_lids_per_element = epoch_lids_per_element.reshape([-1, 1])

            return epoch_lids_per_element

        # Import data
        X, Y = read_dataset(train_dataset_name)

        if self.lid_update_mode == 2:
            class_element_indices = [[] for c in range(N_CLASSES)]
            for i in range(DATASET_SIZE):
                class_element_indices[int(np.argmax(Y[i]))].append(i)
            class_element_indices = [np.array(it) for it in class_element_indices]

        X_test, Y_test = read_dataset('test')

        self._build()

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        #
        # CREATE SUMMARIES
        #

        tf.summary.scalar(name='cross_entropy', tensor=self.cross_entropy)
        tf.summary.scalar(name='train_accuracy', tensor=self.accuracy)
        summary = tf.summary.merge_all()

        test_accuracy_summary_scalar = tf.placeholder(tf.float32)
        test_accuracy_summary = tf.summary.scalar(name='test_accuracy', tensor=test_accuracy_summary_scalar)

        summaries_to_merge = [test_accuracy_summary]
        if bitmask_contains(self.lid_log_mask, 0):
            lid_summary_scalar = tf.placeholder(tf.float32)
            lid_summary = tf.summary.scalar(name='LID', tensor=lid_summary_scalar)

            alpha_summary_scalar = tf.placeholder(tf.float32)
            alpha_summary = tf.summary.scalar(name='alpha', tensor=alpha_summary_scalar)

            summaries_to_merge.extend([lid_summary, alpha_summary])

        if bitmask_contains(self.lid_log_mask, 1):
            lid_per_element_summary_pl = tf.placeholder(dtype=tf.float32, shape=[DATASET_SIZE, ])
            lid_per_element_summary_hist = tf.summary.histogram(name='LID_per_element',
                                                                values=lid_per_element_summary_pl)

            summaries_to_merge.append(lid_per_element_summary_hist)

        per_epoch_summary = tf.summary.merge(summaries_to_merge)

        saver = tf.train.Saver()
        model_path = 'checkpoints/' + self.model_name + '/'

        #
        # SESSION START
        #

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(model_path, sess.graph)

            sess.run(tf.global_variables_initializer())

            if self.lid_update_mode == 1 or bitmask_contains(self.lid_log_mask, 0):
                #
                # CALCULATE AND LOG INITIAL LID SCORE
                #

                initial_lid_score = calc_lid(X, Y, self.lid_calc_op, self.nn_input, self.keep_prob, self.is_training)
                lid_per_epoch = np.append(self.lid_per_epoch, initial_lid_score)

                print('initial LID score:', initial_lid_score)

                if bitmask_contains(self.lid_log_mask, 0):
                    lid_summary_str = sess.run(lid_summary, feed_dict={lid_summary_scalar: initial_lid_score})
                    summary_writer.add_summary(lid_summary_str, 0)
                    summary_writer.flush()

            if self.lid_update_mode == 2 or bitmask_contains(self.lid_log_mask, 1):
                #
                # CALCULATE INITIAL LID PER DATASET ELEMENT AND LOG IT
                #

                element_weights = np.ones((DATASET_SIZE,), dtype=np.float32)
                lids_per_element = np.empty((DATASET_SIZE, 0), dtype=np.float32)

                epoch_lids_per_element = calc_lid_per_element(X=X, Y=Y,
                                                              lid_input_layer=self.lid_input_layer,
                                                              x=self.nn_input,
                                                              lid_per_element_calc_op=self.lid_per_element_calc_op,
                                                              lid_sample_set_pl=self.lid_sample_set_pl,
                                                              is_training=self.is_training)
                lids_per_element = np.hstack((lids_per_element, epoch_lids_per_element))

                print('LID mean: %g, std. dev.: %g, min: %g, max: %g' % (epoch_lids_per_element.mean(),
                                                                         epoch_lids_per_element.var() ** 0.5,
                                                                         epoch_lids_per_element.min(),
                                                                         epoch_lids_per_element.max()))

                if bitmask_contains(self.lid_log_mask, 1):
                    lid_per_element_summary_hist_str = lid_per_element_summary_hist.eval(
                        feed_dict={lid_per_element_summary_pl: lids_per_element[:, -1]}
                    )
                    summary_writer.add_summary(lid_per_element_summary_hist_str, 0)
                    summary_writer.flush()

            #
            # EPOCH LOOP
            #

            turning_epoch = -1  # number of epoch where we turn from regular loss function to the modified one

            i_step = -1
            for i_epoch in range(1, N_EPOCHS + 1):
                print('___________________________________________________________________________')
                print('\nSTARTING EPOCH %d\n' % (i_epoch,))

                if self.lid_update_mode == 2:
                    #
                    # SET ELEMENT WEIGHTS
                    #

                    if turning_epoch != -1:
                        print('\nmodifying weights...')
                        for c in range(N_CLASSES):
                            class_lids = np.take(lids_per_element[:, -1].reshape(-1), class_element_indices[c])

                            # mean = np.mean(class_lids)
                            # st_dev = np.var(class_lids) ** 0.5

                            old_class_weights = element_weights[class_element_indices[c]]
                            mean = np.average(class_lids, weights=old_class_weights)
                            st_dev = (np.average((class_lids - mean)**2, weights=old_class_weights))**0.5

                            class_element_weights = 1 - np.clip(np.abs(class_lids - mean) / (2 * st_dev), 0, 1)
                            np.put(element_weights, class_element_indices[c], class_element_weights)

                            print('\tclass: %d. mean: %g, st. dev.: %g; weighted mean: %g, weighted st. dev.: %g' %
                                  (c, np.mean(class_lids), np.var(class_lids)**.5, mean, st_dev))

                        self.element_weights_matrix = np.hstack((self.element_weights_matrix, element_weights.reshape((-1, 1))))

                #
                # TRAIN
                #

                print('\nstarting training...')

                i_batch = -1
                for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE):
                    i_batch += 1
                    i_step += 1

                    feed_dict = {self.nn_input: batch[0], self.y_: batch[1], self.keep_prob: 0.5, self.is_training: True}
                    if self.lid_update_mode == 2:
                        feed_dict[self.element_weights_pl] = element_weights[batch[2]].reshape((-1, 1))
                    train_step.run(feed_dict=feed_dict)

                    if i_step % 100 == 0:
                        feed_dict[self.is_training] = False
                        # train_accuracy = self.accuracy.eval(feed_dict=feed_dict)
                        # print('\tstep %d, training accuracy %g' % (i_step, train_accuracy))

                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i_step)
                        summary_writer.flush()

                if self.lid_update_mode == 1 or bitmask_contains(self.lid_log_mask, 0):
                    #
                    # CALCULATE LID
                    #

                    new_lid_score = calc_lid(X, Y, self.lid_calc_op, self.nn_input, self.keep_prob, self.is_training)
                    lid_per_epoch = np.append(lid_per_epoch, new_lid_score)

                    print('\nLID score after %dth epoch: %g' % (i_epoch, new_lid_score,))

                if self.lid_update_mode == 2 or bitmask_contains(self.lid_log_mask, 1):
                    #
                    # CALCULATE LID PER DATASET ELEMENT
                    #

                    epoch_lids_per_element = calc_lid_per_element(X, Y, self.lid_input_layer, self.nn_input,
                                                                  self.lid_per_element_calc_op,
                                                                  self.lid_sample_set_pl, self.is_training)
                    lids_per_element = np.hstack((lids_per_element, epoch_lids_per_element))

                    print('LID mean: %g, std. dev.: %g, min: %g, max: %g' % (epoch_lids_per_element.mean(),
                                                                             epoch_lids_per_element.var() ** 0.5,
                                                                             epoch_lids_per_element.min(),
                                                                             epoch_lids_per_element.max()))

                #
                # CHECK FOR STOPPING INIT PERIOD
                #

                if self.lid_update_mode == 1 or self.lid_update_mode == 2 or bitmask_contains(self.lid_log_mask, 0):
                    if turning_epoch == -1 and i_epoch > EPOCH_WINDOW:
                        last_w_lids = lid_per_epoch[-EPOCH_WINDOW - 1: -1]

                        lid_check_value = new_lid_score - last_w_lids.mean() - 2 * last_w_lids.var() ** 0.5

                        if self.lid_update_mode == 1 or self.lid_update_mode == 2:
                            print('LID check:', lid_check_value)

                        if lid_check_value > 0:
                            turning_epoch = i_epoch - 1

                            if self.lid_update_mode == 1 or self.lid_update_mode == 2:
                                saver.restore(sess, model_path + str(i_epoch - 1))
                                print('Turning point passed, reverting to previous epoch and starting using modified loss')

                    #
                    # MODIFYING ALPHA
                    #

                    if turning_epoch != -1:
                        new_alpha_value = np.exp(-(i_epoch / N_EPOCHS) * (lid_per_epoch[-1] / lid_per_epoch[:-1].min()))
                        if self.lid_update_mode == 1:
                            print('\nnew alpha value:', new_alpha_value)
                    else:
                        new_alpha_value = 1

                    if self.lid_update_mode == 1 or bitmask_contains(self.lid_log_mask, 0):
                        sess.run(self.alpha_var.assign(new_alpha_value))

                #
                # TEST ACCURACY
                #

                test_accuracy = 0
                i_batch = -1
                for batch in batch_iterator_with_indices(X_test, Y_test, 100, False):
                    i_batch += 1

                    feed_dict = {self.nn_input: batch[0], self.y_: batch[1], self.keep_prob: 1.0, self.is_training: False}
                    if self.lid_update_mode == 2:
                        feed_dict[self.element_weights_pl] = element_weights[batch[2]].reshape((-1, 1))
                    partial_accuracy = self.accuracy.eval(feed_dict=feed_dict)

                    test_accuracy = (i_batch * test_accuracy + partial_accuracy) / (i_batch + 1)

                print('\ntest accuracy after %dth epoch: %g' % (i_epoch, test_accuracy))

                #
                # WRITE PER EPOCH SUMMARIES
                #

                feed_dict = {test_accuracy_summary_scalar: test_accuracy}
                if bitmask_contains(self.lid_log_mask, 0):
                    feed_dict[lid_summary_scalar] = lid_per_epoch[-1]
                    feed_dict[alpha_summary_scalar] = new_alpha_value

                if bitmask_contains(self.lid_log_mask, 1):
                    feed_dict[lid_per_element_summary_pl] = lids_per_element[:, -1]

                summary_str = sess.run(per_epoch_summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i_step + 1)
                summary_writer.flush()

                #
                # SAVE MODEL
                #

                checkpoint_file = model_path + str(i_epoch)
                saver.save(sess, checkpoint_file)

                np.save('LID_matrices/' + self.model_name, lids_per_element)
                np.save('weight_matrices/' + self.model_name, self.element_weights_matrix)

    @staticmethod
    def test(model_name, epoch):
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
