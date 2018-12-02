import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from consts import *
from Timer import timer
from preprocessing import read_dataset
from batch_iterate import batch_iterator, batch_iterator_with_indices
from tools import bitmask_contains


def get_LID_calc_op(g_x):
    batch_size = tf.shape(g_x)[0]

    norm_squared = tf.reshape(tf.reduce_sum(g_x * g_x, 1), [-1, 1])
    norm_squared_t = tf.transpose(norm_squared)

    dot_products = tf.matmul(g_x, tf.transpose(g_x))

    distances_squared = tf.maximum(norm_squared - 2 * dot_products + norm_squared_t, 0)
    distances = tf.sqrt(distances_squared) + tf.ones((batch_size, batch_size)) * EPS

    k_nearest_raw, _ = tf.nn.top_k(-distances, k=LID_K + 1, sorted=True)
    k_nearest = -k_nearest_raw[:, 1:]

    distance_ratios = tf.transpose(tf.multiply(tf.transpose(k_nearest), 1 / k_nearest[:, -1]))
    LIDs = - LID_K / tf.reduce_sum(tf.log(distance_ratios + EPS), 1)

    return LIDs


def get_LID_per_element_calc_op(g_x, X):
    def get_norm_squared(arg):
        return tf.reduce_sum(arg*arg, axis=2)

    sample_batch_size = LID_BATCH_SIZE

    batch_size = tf.shape(g_x)[0]

    random_batch_indices = tf.random_uniform((batch_size, sample_batch_size), maxval=tf.shape(X)[0], dtype=tf.int32)

    random_batch = tf.gather(X, random_batch_indices)

    tiled_g_x = tf.tile(tf.expand_dims(g_x, 1), [1, sample_batch_size, 1])

    g_x_norm_squared = tf.expand_dims(get_norm_squared(tiled_g_x), 2)
    rnd_norm_squared = tf.expand_dims(get_norm_squared(random_batch), 1)

    dot_products = tf.matmul(tiled_g_x, tf.transpose(random_batch, [0, 2, 1]))

    distances_squared = tf.maximum(g_x_norm_squared - 2 * dot_products + rnd_norm_squared, 0)
    distances = tf.sqrt(distances_squared) + tf.ones((batch_size, sample_batch_size, sample_batch_size)) * EPS

    k_nearest_raw, _ = tf.nn.top_k(-distances, k=LID_K, sorted=True)
    k_nearest = -k_nearest_raw

    distance_ratios = tf.multiply(k_nearest, tf.expand_dims(1 / k_nearest[:, :, -1], 2))
    LIDs_for_btach = - LID_K / tf.reduce_sum(tf.log(distance_ratios + EPS), 2)
    LIDs = tf.reduce_mean(LIDs_for_btach, 1)

    return LIDs




class Model:
    def __init__(self, model_name, update_mode, log_mask):
        """

        :param model_name:      name of the model
        :param update_mode:     0: vanilla
                                1: as in the paper
                                2: per element
        :param log_mask:    in case it contains
                                    0th bit: logs LID data from the paper
                                    1st bit: logs LID data per element
                                    2nd bit: logs lid features per element
                                    3rd bit: logs pre lid features per element (before relu)
                                    4th bit: logs logits per element
        """
        self.update_mode = update_mode
        self.log_mask = log_mask
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
        # is down to 7x7x64 feature maps -- maps this to 128 lid_features.
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([7 * 7 * 64, FC_WIDTH])
            b_fc1 = bias_variable([FC_WIDTH])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            a_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
            b_norm_fc1 = tf.identity(batch_norm(a_fc1, is_training), name='pre_lid_input')
            h_fc1 = tf.nn.relu(b_norm_fc1, name='lid_input')
            # h_fc1 = tf.nn.relu(a_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # lid_features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 lid_features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([FC_WIDTH, 10])
            b_fc2 = bias_variable([10])

            y_conv = tf.identity(tf.matmul(h_fc1, W_fc2) + b_fc2, name='logits')
            # y_conv = tf.identity(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='probs')

        return b_norm_fc1, h_fc1, y_conv, keep_prob

    def _build(self):
        # Create the model
        self.nn_input = tf.placeholder(tf.float32, [None, 784], name='x')

        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        # Build the graph for the deep net
        self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.keep_prob = self._build_nn(self.nn_input, self.is_training)

        #
        # CREATE LOSS & OPTIMIZER
        #

        with tf.name_scope('loss'):
            if self.update_mode == 0:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                                                           logits=self.logits)
            if self.update_mode == 1 or bitmask_contains(self.log_mask, 0):
                self.alpha_var = tf.Variable(1, False, dtype=tf.float32, name='alpha')

                if self.update_mode == 1:
                    modified_labels = tf.identity(
                        self.alpha_var * self.y_ + (1 - self.alpha_var.value()) * tf.one_hot(tf.argmax(self.logits, 1), 10),
                        name='modified_labels')
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=modified_labels,
                                                                               logits=self.logits)
            if self.update_mode == 2:
                self.element_weights_pl = tf.placeholder(dtype=np.float32,
                                                         shape=[None, 1],
                                                         name='element_weights')
                modified_labels = tf.identity(
                    self.element_weights_pl * self.y_ +
                    (1 - self.element_weights_pl) * tf.one_hot(tf.argmax(self.logits, 1), 10),
                    name='modified_labels_per_element')
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=modified_labels,
                                                                           logits=self.logits)

        self.cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

        if self.update_mode == 1 or bitmask_contains(self.log_mask, 0):
            #
            # PREPARE LID CALCULATION OP
            #

            self.lid_per_epoch = np.array([])
            self.lid_calc_op = get_LID_calc_op(self.lid_layer_op)


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
        if bitmask_contains(self.log_mask, 0):
            lid_summary_scalar = tf.placeholder(tf.float32)
            lid_summary = tf.summary.scalar(name='LID', tensor=lid_summary_scalar)

            alpha_summary_scalar = tf.placeholder(tf.float32)
            alpha_summary = tf.summary.scalar(name='alpha', tensor=alpha_summary_scalar)

            summaries_to_merge.extend([lid_summary, alpha_summary])

        per_epoch_summary = tf.summary.merge(summaries_to_merge)

        saver = tf.train.Saver()
        model_path = 'checkpoints/' + self.model_name + '/'

        if bitmask_contains(self.log_mask, 2):
            lid_features_per_epoch_per_element = np.empty((0, DATASET_SIZE, FC_WIDTH))
        if bitmask_contains(self.log_mask, 3):
            pre_lid_features_per_epoch_per_element = np.empty((0, DATASET_SIZE, FC_WIDTH))
        if bitmask_contains(self.log_mask, 4):
            logits_per_epoch_per_element = np.empty((0, DATASET_SIZE, N_CLASSES))

        #
        # SESSION START
        #

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(model_path, sess.graph)

            sess.run(tf.global_variables_initializer())

            if self.update_mode == 1 or bitmask_contains(self.log_mask, 0):
                #
                # CALCULATE AND LOG INITIAL LID SCORE
                #

                initial_lid_score = calc_lid(X, Y, self.lid_calc_op, self.nn_input, self.keep_prob, self.is_training)
                lid_per_epoch = np.append(self.lid_per_epoch, initial_lid_score)

                print('initial LID score:', initial_lid_score)

                if bitmask_contains(self.log_mask, 0):
                    lid_summary_str = sess.run(lid_summary, feed_dict={lid_summary_scalar: initial_lid_score})
                    summary_writer.add_summary(lid_summary_str, 0)
                    summary_writer.flush()

            #
            # EPOCH LOOP
            #

            turning_epoch = -1  # number of epoch where we turn from regular loss function to the modified one

            i_step = -1
            for i_epoch in range(1, N_EPOCHS + 1):
                print('___________________________________________________________________________')
                print('\nSTARTING EPOCH %d\n' % (i_epoch,))

                #
                # TRAIN
                #

                print('\nstarting training...')

                if bitmask_contains(self.log_mask, 2):
                    lid_features_per_element = np.empty((DATASET_SIZE, FC_WIDTH))
                if bitmask_contains(self.log_mask, 3):
                    pre_lid_features_per_element = np.empty((DATASET_SIZE, FC_WIDTH))
                if bitmask_contains(self.log_mask, 4):
                    logits_per_element = np.empty((DATASET_SIZE, N_CLASSES))

                i_batch = -1
                for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE):
                    i_batch += 1
                    i_step += 1

                    feed_dict = {self.nn_input: batch[0], self.y_: batch[1], self.keep_prob: 0.5, self.is_training: True}
                    train_step.run(feed_dict=feed_dict)

                    feed_dict[self.keep_prob] = 1.0
                    feed_dict[self.is_training] = False

                    if bitmask_contains(self.log_mask, 2):
                        lid_features_per_element[batch[2], ] = self.lid_layer_op.run(feed_dict=feed_dict)
                    if bitmask_contains(self.log_mask, 3):
                        pre_lid_features_per_element[batch[2], ] = self.pre_lid_layer_op.run(feed_dict=feed_dict)
                    if bitmask_contains(self.log_mask, 4):
                        logits_per_element[batch[2], ] = self.logits(feed_dict=feed_dict)

                    if i_step % 100 == 0:
                        feed_dict[self.is_training] = False
                        # train_accuracy = self.accuracy.eval(feed_dict=feed_dict)
                        # print('\tstep %d, training accuracy %g' % (i_step, train_accuracy))

                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i_step)
                        summary_writer.flush()

                if bitmask_contains(self.log_mask, 2):
                    lid_features_per_epoch_per_element = np.append(lid_features_per_epoch_per_element, lid_features_per_element)
                if bitmask_contains(self.log_mask, 3):
                    pre_lid_features_per_epoch_per_element = np.append(pre_lid_features_per_epoch_per_element, pre_lid_features_per_element)
                if bitmask_contains(self.log_mask, 4):
                    logits_per_epoch_per_element = np.append(logits_per_epoch_per_element, logits_per_element)

                if self.update_mode == 1 or bitmask_contains(self.log_mask, 0):
                    #
                    # CALCULATE LID
                    #

                    new_lid_score = calc_lid(X, Y, self.lid_calc_op, self.nn_input, self.keep_prob, self.is_training)
                    lid_per_epoch = np.append(lid_per_epoch, new_lid_score)

                    print('\nLID score after %dth epoch: %g' % (i_epoch, new_lid_score,))

                #
                # CHECK FOR STOPPING INIT PERIOD
                #

                if self.update_mode == 1 or bitmask_contains(self.log_mask, 0):
                    if turning_epoch == -1 and i_epoch > EPOCH_WINDOW:
                        last_w_lids = lid_per_epoch[-EPOCH_WINDOW - 1: -1]

                        lid_check_value = new_lid_score - last_w_lids.mean() - 2 * last_w_lids.var() ** 0.5

                        if self.update_mode == 1:
                            print('LID check:', lid_check_value)

                        if lid_check_value > 0:
                            turning_epoch = i_epoch - 1

                            if self.update_mode == 1:
                                saver.restore(sess, model_path + str(i_epoch - 1))
                                print('Turning point passed, reverting to previous epoch and starting using modified loss')

                    #
                    # MODIFYING ALPHA
                    #

                    if turning_epoch != -1:
                        new_alpha_value = np.exp(-(i_epoch / N_EPOCHS) * (lid_per_epoch[-1] / lid_per_epoch[:-1].min()))
                        if self.update_mode == 1:
                            print('\nnew alpha value:', new_alpha_value)
                    else:
                        new_alpha_value = 1

                    if self.update_mode == 1 or bitmask_contains(self.log_mask, 0):
                        sess.run(self.alpha_var.assign(new_alpha_value))

                #
                # TEST ACCURACY
                #

                test_accuracy = 0
                i_batch = -1
                for batch in batch_iterator_with_indices(X_test, Y_test, 100, False):
                    i_batch += 1

                    feed_dict = {self.nn_input: batch[0], self.y_: batch[1], self.keep_prob: 1.0, self.is_training: False}
                    partial_accuracy = self.accuracy.eval(feed_dict=feed_dict)

                    test_accuracy = (i_batch * test_accuracy + partial_accuracy) / (i_batch + 1)

                print('\ntest accuracy after %dth epoch: %g' % (i_epoch, test_accuracy))

                #
                # WRITE PER EPOCH SUMMARIES
                #

                feed_dict = {test_accuracy_summary_scalar: test_accuracy}
                if bitmask_contains(self.log_mask, 0):
                    feed_dict[lid_summary_scalar] = lid_per_epoch[-1]
                    feed_dict[alpha_summary_scalar] = new_alpha_value

                summary_str = sess.run(per_epoch_summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i_step + 1)
                summary_writer.flush()

                #
                # SAVE MODEL
                #

                checkpoint_file = model_path + str(i_epoch)
                saver.save(sess, checkpoint_file)

                if bitmask_contains(self.log_mask, 2):
                    np.save('lid_features/' + self.model_name, lid_features_per_epoch_per_element)
                if bitmask_contains(self.log_mask, 3):
                    np.save('pre_lid_features/' + self.model_name, pre_lid_features_per_epoch_per_element)
                if bitmask_contains(self.log_mask, 4):
                    np.save('logits/' + self.model_name, logits_per_epoch_per_element)

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