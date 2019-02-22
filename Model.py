import os
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from networks import build_mnist, build_cifar_10
from tf_ops import get_euclid_dist_to_mean_calc_op, get_new_label_op, get_cosine_dist_to_mean_calc_op, get_lid_calc_op,\
    get_update_class_features_sum_and_counts_op_logits, get_update_class_feature_selective_sum_and_counts_op, \
    get_update_class_covariances_selective_sum_op, get_LDA_logits_calc_op, get_Mahalanobis_distance_calc_op
from consts import *
from Timer import timer
from preprocessing import read_dataset
from batch_iterate import batch_iterator, batch_iterator_with_indices
from tools import bitmask_contains


class Model:
    def __init__(self, dataset_name, model_name, update_mode, update_param, update_submode=0, update_subsubmode=0,
                 log_mask=0, reg_coef=5e-4, n_blocks=1):
        """

        :param dataset_name:         dataset name: mnist/cifar-10
        :param model_name:      name of the model
        :param update_mode:     0: vanilla
                                1: as in the paper
                                2: per element
                                3: lda
        :param update_param:    for update_mode = 2 or 3: number of epoch to start using modified labels
        :param update_submode:  for update_mode = 2 or 3:
                                                        0: use pre lid features
                                                        1: use lid features
        :param update_subsubmode: for update_mode = 2:
                                                        0: use cosine distance
                                                        1: use euclid distance
        :param log_mask:    in case it contains
                                    0th bit: logs LID data from the paper
                                    1st bit: logs class feature means
                                    2nd bit: logs lid features per element
                                    3rd bit: logs pre lid features per element (before relu)
                                    4th bit: logs logits per element
        """

        self.dataset_name = dataset_name
        self.update_mode = update_mode
        self.update_param = update_param
        self.update_submode = update_submode
        self.update_subsubmode = update_subsubmode
        self.log_mask = log_mask
        self.model_name = model_name
        self.reg_coef = reg_coef
        self.n_blocks = n_blocks

        self.fc_width = FC_WIDTH[dataset_name]
        self.dataset_size = DATASET_SIZE[dataset_name]

    def _build(self):
        #
        # PREPARE PLACEHOLDERS
        #

        if self.dataset_name == 'mnist':
            self.nn_input = tf.placeholder(tf.float32, [None, 784], name='x')
        elif self.dataset_name == 'cifar-10':
            self.nn_input = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')

        self.y_ = tf.placeholder(tf.float32, [None, N_CLASSES], name='y_')

        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        #
        # PREPARE DATA AUGMENTATION OPERATION
        #

        self.data_augmenter = None
        if self.dataset_name == 'cifar-10':
            self.data_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

        #
        # BUILD NETWORK
        #

        if self.dataset_name == 'mnist':
            self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.preds = build_mnist(self.nn_input,
                                                                                            self.is_training,
                                                                                            self.n_blocks)
        elif self.dataset_name == 'cifar-10':
            self.pre_lid_layer_op, self.lid_layer_op, self.logits, W_l2_reg_sum, b_l2_reg_sum = build_cifar_10(
                self.nn_input, self.is_training)

        #
        # PREPARE FOR LABEL CHANGING
        #

        # LID calculation

        if self.update_mode == 1 or self.to_log(0):
            self.lid_per_epoch = np.array([])
            self.lid_calc_op = get_lid_calc_op(self.lid_layer_op)

        # Distances to class centroids

        if self.update_mode == 2 or self.to_log(1):
            used_lid_layer = None
            if self.update_submode == 0:
                used_lid_layer = self.pre_lid_layer_op
            elif self.update_submode == 1:
                used_lid_layer = self.lid_layer_op

            self.class_feature_sums_var = tf.Variable(np.zeros((N_CLASSES, self.fc_width)), dtype=tf.float32)
            self.class_feature_counts_var = tf.Variable(np.zeros((N_CLASSES, )), dtype=tf.float32)

            self.update_class_features_sum_and_counts_op = get_update_class_features_sum_and_counts_op_logits(
                self.class_feature_sums_var, self.class_feature_counts_var,
                used_lid_layer, self.logits, self.y_)

            self.reset_class_feature_sums_and_counts_op = tf.group(
                tf.assign(self.class_feature_sums_var, tf.zeros((N_CLASSES, self.fc_width))),
                tf.assign(self.class_feature_counts_var, tf.zeros(N_CLASSES, ))
            )

            self.class_feature_means_pl = tf.placeholder(tf.float32, (N_CLASSES, self.fc_width))

            self.features_to_means_dist_op = None
            if self.update_subsubmode == 0:
                self.features_to_means_dist_op = get_cosine_dist_to_mean_calc_op(used_lid_layer, self.class_feature_means_pl)
            elif self.update_subsubmode == 1:
                self.features_to_means_dist_op = get_euclid_dist_to_mean_calc_op(used_lid_layer, self.class_feature_means_pl)

        # LDA predictor

        if self.update_mode == 3:
            self.feature_layer = None
            if self.update_subsubmode == 0:
                self.feature_layer = self.pre_lid_layer_op
            elif self.update_subsubmode == 1:
                self.feature_layer = self.lid_layer_op

            self.class_feature_sums_var = tf.Variable(np.zeros((N_CLASSES, self.fc_width)), dtype=tf.float32)
            self.class_feature_counts_var = tf.Variable(np.zeros((N_CLASSES,)), dtype=tf.float32)
            self.class_covariance_sums_var = tf.Variable(np.zeros((N_CLASSES, self.fc_width, self.fc_width)), dtype=tf.float32)

            self.features_pl = tf.placeholder(tf.float32, (None, self.fc_width))
            self.class_feature_ops_keep_array_pl = tf.placeholder(tf.bool, (None,))

            self.update_class_features_sum_and_counts_op = get_update_class_feature_selective_sum_and_counts_op(
                self.class_feature_sums_var, self.class_feature_counts_var, self.features_pl, self.y_,
                self.class_feature_ops_keep_array_pl)

            self.class_feature_means_pl = tf.placeholder(tf.float32, (N_CLASSES, self.fc_width))

            self.update_class_covariances_op = get_update_class_covariances_selective_sum_op(
                self.class_covariance_sums_var, self.class_feature_means_pl, self.features_pl, self.y_,
                self.class_feature_ops_keep_array_pl)

            self.class_inv_covariances_pl = tf.placeholder(tf.float32, (N_CLASSES, self.fc_width, self.fc_width))
            self.inv_covariance_pl = tf.placeholder(tf.float32, (self.fc_width, self.fc_width))

            self.reset_class_feature_sums_counts_and_covariance_op = tf.group(
                tf.assign(self.class_feature_sums_var, tf.zeros((N_CLASSES, self.fc_width), tf.float32)),
                tf.assign(self.class_feature_counts_var, tf.zeros((N_CLASSES,), tf.float32)),
                tf.assign(self.class_covariance_sums_var, tf.zeros((N_CLASSES, self.fc_width, self.fc_width), tf.float32))
            )

            self.mahalanobis_dist_calc = get_Mahalanobis_distance_calc_op(self.class_feature_means_pl,
                                                                          self.class_inv_covariances_pl,
                                                                          self.features_pl, self.y_)

            self.LDA_logits = get_LDA_logits_calc_op(self.class_feature_means_pl, self.inv_covariance_pl,
                                                     self.feature_layer, np.ones(N_CLASSES, np.float32) / N_CLASSES)
            self.LDA_predictions = tf.one_hot(tf.argmax(self.LDA_logits, 1), N_CLASSES)

        #
        # CREATE LOSS
        #

        with tf.name_scope('loss'):
            if self.update_mode == 0:
                self.modified_labels_op = self.y_
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                                                           logits=self.logits)
            if self.update_mode == 1 or self.to_log(0):
                self.alpha_var = tf.Variable(1, False, dtype=tf.float32, name='alpha')

                if self.update_mode == 1:
                    self.modified_labels_op = tf.identity(
                        self.alpha_var * self.y_ + (1 - self.alpha_var.value()) * tf.one_hot(tf.argmax(self.preds, 1), 10),
                        name='modified_labels')
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=tf.stop_gradient(self.modified_labels_op),
                        logits=self.logits)
            if self.update_mode == 2:
                self.modified_labels_op = get_new_label_op(self.features_to_means_dist_op, self.y_, self.logits)
                self.use_modified_labels_pl = tf.placeholder(tf.bool)

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cond(self.use_modified_labels_pl,
                                                                                          lambda: self.modified_labels_op,
                                                                                          lambda: self.y_),
                                                                           logits=self.logits)

            if self.update_mode == 3:
                self.LDA_labels_weight_pl = tf.placeholder(tf.float32, ())
                self.modified_labels_op = self.LDA_labels_weight_pl * tf.stop_gradient(self.LDA_predictions) + \
                                          (1 - self.LDA_labels_weight_pl) * self.y_

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.modified_labels_op,
                    logits=self.logits)

            self.reg_loss = 0
            if self.dataset_name == 'cifar-10':
                self.reg_loss = self.reg_coef * (W_l2_reg_sum + b_l2_reg_sum)
            cross_entropy += self.reg_loss

        self.cross_entropy = tf.reduce_mean(cross_entropy)

        #
        # CREATE ACCURACY
        #

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.preds, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

        with tf.name_scope('modified_accuracy'):
            new_label = self.y_
            if self.update_mode == 1:
                new_label = self.modified_labels_op
            elif self.update_mode == 2:
                new_label = tf.cond(self.use_modified_labels_pl,
                                    lambda: self.modified_labels_op,
                                    lambda: self.y_)
            elif self.update_mode == 3:
                new_label = self.modified_labels_op

            acc = tf.reduce_sum(tf.one_hot(tf.argmax(self.preds, 1), N_CLASSES) * new_label, 1)
            self.modified_accuracy = tf.reduce_mean(acc, name='accuracy')

    def train(self, train_dataset_type='train'):
        # Import data

        X, Y = read_dataset(name=self.dataset_name, type=train_dataset_type)
        X0, Y0 = read_dataset(name=self.dataset_name, type='train')

        Y_ind = np.argmax(Y, 1)
        Y_ind_per_class = [[] for c in range(N_CLASSES)]
        for i in range(self.dataset_size):
            Y_ind_per_class[Y_ind[i]].append(i)

        X_test, Y_test = read_dataset(name=self.dataset_name, type='test')

        self._build()

        self.epoch_pl = tf.placeholder(tf.int32)

        with tf.name_scope('learning_rate'):
            if self.dataset_name == 'mnist':
                self.lr = tf.cond(self.epoch_pl > 80, lambda: 1e-3,
                                  lambda: tf.cond(self.epoch_pl > 40, lambda: 1e-2,
                                                  lambda: 1e-1)) * 1e-3
                # self.lr = tf.cond(self.epoch_pl > 30, lambda: 1e-6,
                #                                      lambda: 1e-5)
                # self.lr = tf.cond(self.epoch_pl > 40, lambda: 1e-5, lambda: 1e-4)
                # self.lr = 1e-6
            elif self.dataset_name == 'cifar-10':
                self.lr = tf.cond(self.epoch_pl > 80, lambda: 1e-3,
                                  lambda: tf.cond(self.epoch_pl > 40, lambda: 1e-2,
                                                  lambda: 1e-1)) * 1e-1

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

        #
        # CREATE SUMMARIES
        #

        tf.summary.scalar(name='cross_entropy', tensor=self.cross_entropy)
        tf.summary.scalar(name='train_accuracy', tensor=self.accuracy)
        tf.summary.scalar(name='reg_loss', tensor=self.reg_loss)
        tf.summary.scalar(name='modified_train_accuracy', tensor=self.modified_accuracy)
        summary = tf.summary.merge_all()

        test_accuracy_summary_scalar = tf.placeholder(tf.float32)
        test_accuracy_summary = tf.summary.scalar(name='test_accuracy', tensor=test_accuracy_summary_scalar)

        summaries_to_merge = [test_accuracy_summary]

        if self.update_mode == 1 or self.update_mode == 2 or self.update_mode == 3:
            modified_labels_accuracy_summary_scalar = tf.placeholder(tf.float32)
            modified_labels_accuracy_summary = tf.summary.scalar(name='modified_labels_accuracy',
                                                                 tensor=modified_labels_accuracy_summary_scalar)
            summaries_to_merge.append(modified_labels_accuracy_summary)

        if self.to_log(0):
            lid_summary_scalar = tf.placeholder(tf.float32)
            lid_summary = tf.summary.scalar(name='LID', tensor=lid_summary_scalar)

            alpha_summary_scalar = tf.placeholder(tf.float32)
            alpha_summary = tf.summary.scalar(name='alpha', tensor=alpha_summary_scalar)

            summaries_to_merge.extend([lid_summary, alpha_summary])

        per_epoch_summary = tf.summary.merge(summaries_to_merge)

        saver = tf.train.Saver()
        model_path = 'checkpoints/' + self.dataset_name + '/' + self.model_name + '/'

        if self.to_log(1):
            class_feature_means_per_epoch = np.empty((0, N_CLASSES, self.fc_width))
        if self.to_log(2):
            lid_features_per_epoch_per_element = np.empty((0, self.dataset_size, self.fc_width))
        if self.to_log(3):
            pre_lid_features_per_epoch_per_element = np.empty((0, self.dataset_size, self.fc_width))
        if self.to_log(4):
            logits_per_epoch_per_element = np.empty((0, self.dataset_size, N_CLASSES))

        #
        # SESSION START
        #

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='readline')   # run -t n_epochs
            # sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)       # run -f has_inf_or_nan
            # tf.logging.set_verbosity(tf.logging.ERROR)

            summary_writer = tf.summary.FileWriter(model_path, sess.graph)

            sess.run(tf.global_variables_initializer())
            saver.save(sess, model_path + str(0))

            #
            # CALCULATE AND LOG INITIAL LID SCORE
            #

            if self.update_mode == 1 or self.to_log(0):
                initial_lid_score = self.calc_lid(X, Y, self.lid_calc_op, self.nn_input, self.is_training)
                lid_per_epoch = np.append(self.lid_per_epoch, initial_lid_score)

                print('initial LID score:', initial_lid_score)

                if self.to_log(0):
                    lid_summary_str = sess.run(lid_summary, feed_dict={lid_summary_scalar: initial_lid_score})
                    summary_writer.add_summary(lid_summary_str, 0)
                    summary_writer.flush()

            #
            # FIT AUGMENTER
            #

            X_augmented_iter = None
            if self.data_augmenter is not None:
                self.data_augmenter.fit(X)
                X_augmented_iter = self.data_augmenter.flow(X, batch_size=X.shape[0], shuffle=False)

            #
            # LDA INIT
            #
            if self.update_mode == 3:
                self.class_feature_means = np.zeros((N_CLASSES, self.fc_width))
                self.inv_covariance = np.eye(self.fc_width, self.fc_width)

            #
            # EPOCH LOOP
            #

            turning_epoch = -1  # number of epoch where we turn from regular loss function to the modified one

            i_step = -1
            for i_epoch in range(1, N_EPOCHS + 1):
                timer.start()

                def use_modified_labels():
                    return i_epoch >= self.update_param

                print('___________________________________________________________________________')
                print('\nSTARTING EPOCH %d\n' % (i_epoch,))

                if X_augmented_iter is not None:
                    print('Augmenting data...')
                    X_augmented = X_augmented_iter.next()
                else:
                    X_augmented = X

                if self.update_mode == 2 or self.to_log(1):
                    #
                    #   COMPUTE CLASS FEATURE MEANS
                    #

                    print('Computing class feature means')

                    sess.run(self.reset_class_feature_sums_and_counts_op)

                    for batch in batch_iterator(X_augmented, Y, BATCH_SIZE, False):
                        feed_dict = {self.nn_input: batch[0], self.y_: batch[1], self.is_training: False}
                        sess.run(self.update_class_features_sum_and_counts_op, feed_dict=feed_dict)

                    counts = self.class_feature_counts_var.eval(sess)
                    sums = self.class_feature_sums_var.eval(sess)

                    class_feature_means = sums / np.maximum(counts, 1).reshape((-1, 1))

                    if self.to_log(1):
                        class_feature_means_per_epoch = np.append(class_feature_means_per_epoch, class_feature_means)

                print('')

                #
                # TRAIN
                #

                print('starting training...')

                modified_labels_accuracy = 0
                batch_cnt = -1
                for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE):
                    i_step += 1

                    batch_cnt += 1
                    batch_size = batch[0].shape[0]

                    feed_dict = {self.nn_input: batch[0], self.y_: batch[1], self.is_training: True,
                                 self.epoch_pl: i_epoch}

                    if self.update_mode == 2:
                        feed_dict[self.class_feature_means_pl] = class_feature_means
                        feed_dict[self.use_modified_labels_pl] = use_modified_labels()

                    if self.update_mode == 3:
                        feed_dict[self.LDA_labels_weight_pl] = 1 - self.alpha_var.eval(sess)
                        feed_dict[self.class_feature_means_pl] = self.class_feature_means
                        feed_dict[self.inv_covariance_pl] = self.inv_covariance

                    if self.update_mode == 1 or self.update_mode == 2 or self.update_mode == 3:
                        modified_labels = self.modified_labels_op.eval(feed_dict=feed_dict)
                        batch_accs = np.sum(modified_labels * Y0[batch[2]])
                        modified_labels_accuracy = (modified_labels_accuracy * batch_cnt * BATCH_SIZE + batch_accs) / \
                                                   (batch_cnt * BATCH_SIZE + batch_size)

                    train_step.run(feed_dict=feed_dict)
                    feed_dict[self.is_training] = False

                    if i_step % 100 == 0:
                        train_accuracy = self.accuracy.eval(feed_dict=feed_dict)
                        print('\tstep %d, training accuracy %g' % (i_step, train_accuracy))

                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i_step)
                        summary_writer.flush()

                #
                # LOG FEATURES/LOGITS
                #

                if self.to_log(2):
                    lid_features_per_element = np.empty((self.dataset_size, self.fc_width))
                if self.to_log(3):
                    pre_lid_features_per_element = np.empty((self.dataset_size, self.fc_width))
                if self.to_log(4):
                    logits_per_element = np.empty((self.dataset_size, N_CLASSES))

                if self.to_log(2) or self.to_log(3) or self.to_log(4):
                    for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE, False):
                        feed_dict = {self.nn_input: batch[0], self.y_: batch[1], self.is_training: False}
                        if self.to_log(2):
                            lid_features_per_element[batch[2]] = self.lid_layer_op.eval(feed_dict=feed_dict)
                        if self.to_log(3):
                            pre_lid_features_per_element[batch[2]] = self.pre_lid_layer_op.eval(feed_dict=feed_dict)
                        if self.to_log(4):
                            logits_per_element[batch[2]] = self.logits.eval(feed_dict=feed_dict)

                if self.update_mode == 3:
                    #
                    # COMPUTE LDA PARAMETERS
                    #

                    print('Computing LDA parameters')

                    features = np.empty((self.dataset_size, self.fc_width))
                    for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE, False):
                        feed_dict = {self.nn_input: batch[0], self.is_training: False}
                        features[batch[2]] = self.feature_layer.eval(feed_dict=feed_dict)

                    # initially select random subset of samples
                    keep_arr = np.random.binomial(1, 0.5, self.dataset_size).astype(bool)
                    for i in range(3 + 1):
                        sess.run(self.reset_class_feature_sums_counts_and_covariance_op)

                        # compute class feature means
                        for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE * 8, False):
                            feed_dict = {self.features_pl: features[batch[2]],
                                         self.y_: batch[1],
                                         self.class_feature_ops_keep_array_pl: keep_arr[batch[2]]}
                            sess.run(self.update_class_features_sum_and_counts_op, feed_dict=feed_dict)

                        counts = self.class_feature_counts_var.eval(sess)
                        sums = self.class_feature_sums_var.eval(sess)

                        self.class_feature_means = sums / counts.reshape((-1, 1))

                        # compute class feature covariance matrices
                        for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE * 8, False):
                            feed_dict = {self.features_pl: features[batch[2]], self.y_: batch[1],
                                         self.class_feature_means_pl: self.class_feature_means,
                                         self.class_feature_ops_keep_array_pl: keep_arr[batch[2]]}
                            sess.run(self.update_class_covariances_op, feed_dict=feed_dict)

                        class_covariance_sums = self.class_covariance_sums_var.eval(sess)
                        class_covariances = class_covariance_sums / counts.reshape((-1, 1, 1)) +\
                                            np.eye(self.fc_width, self.fc_width) * EPS
                        class_inv_covariances = np.array([np.linalg.inv(it) for it in class_covariances])

                        print('covariance determinants per class:',
                              [np.linalg.det(it) for it in class_covariances])
                        print('min covariance eigenvalues per class:',
                              [np.linalg.svd(it)[1][-1] for it in class_covariances])

                        # compute tied covariance matrix
                        covariance = np.sum(class_covariance_sums, 0) / np.sum(counts) +\
                                     np.eye(self.fc_width, self.fc_width) * EPS
                        self.inv_covariance = np.linalg.inv(covariance)

                        print('tied covariance determinant:',
                              np.linalg.det(covariance))
                        print('min tied covariance eigenvalue:',
                              np.linalg.svd(covariance)[1][-1])

                        # compute Mahalanobis distance based anomaly scores
                        mahalanobis_distances = np.empty(self.dataset_size)
                        for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE * 8, False):
                            feed_dict = {self.features_pl: features[batch[2]],
                                         self.class_feature_means_pl: self.class_feature_means,
                                         self.class_inv_covariances_pl: class_inv_covariances,
                                         self.y_: batch[1]}
                            mah_dists = sess.run(self.mahalanobis_dist_calc, feed_dict=feed_dict)
                            mahalanobis_distances[batch[2]] = mah_dists

                        mahalanobis_distances_per_class = [np.take(mahalanobis_distances, Y_ind_per_class[c]) for c in range(N_CLASSES)]

                        # select less anomalous samples for the next subset
                        medians = np.array([np.quantile(arr, 0.5) for arr in mahalanobis_distances_per_class])
                        print('Mahalanobis distance medians per class', [round(it**0.5, 1) for it in medians])

                        keep_arr = mahalanobis_distances < np.take(medians, Y_ind)

                if self.update_mode == 1 or self.to_log(0):
                    #
                    # CALCULATE LID
                    #

                    new_lid_score = self.calc_lid(X, Y, self.lid_calc_op, self.nn_input, self.is_training)
                    lid_per_epoch = np.append(lid_per_epoch, new_lid_score)

                    print('\nLID score after %dth epoch: %g' % (i_epoch, new_lid_score,))

                #
                # CHECK FOR STOPPING INIT PERIOD
                #

                if self.update_mode == 1 or self.update_mode == 3 or self.to_log(0):
                    if turning_epoch == -1 and i_epoch > EPOCH_WINDOW:
                        last_w_lids = lid_per_epoch[-EPOCH_WINDOW - 1: -1]

                        lid_check_value = new_lid_score - last_w_lids.mean() - 2 * last_w_lids.var() ** 0.5

                        print('LID check:', lid_check_value)

                        if lid_check_value > 0:
                            turning_epoch = i_epoch - 1

                            if self.update_mode == 1 or self.update_mode == 3:
                                saver.restore(sess, model_path + str(i_epoch - 1))
                                print('Turning point passed, reverting to previous epoch and starting using modified loss')

                    #
                    # MODIFYING ALPHA
                    #

                    if turning_epoch != -1:
                        new_alpha_value = np.exp(-(i_epoch / N_EPOCHS) * (lid_per_epoch[-1] / lid_per_epoch[:-1].min()))
                        print('\nnew alpha value:', new_alpha_value)
                    else:
                        new_alpha_value = 1

                    if self.update_mode == 1 or self.update_mode == 3 or self.to_log(0):
                        sess.run(self.alpha_var.assign(new_alpha_value))

                #
                # TEST ACCURACY
                #

                test_accuracy = 0
                i_batch = -1
                tested_cnt = 0
                for batch in batch_iterator_with_indices(X_test, Y_test, BATCH_SIZE, False):
                    i_batch += 1

                    feed_dict = {self.nn_input: batch[0], self.y_: batch[1], self.is_training: False}
                    if self.update_mode == 2:
                        feed_dict[self.class_feature_means_pl] = class_feature_means
                        feed_dict[self.use_modified_labels_pl] = use_modified_labels()

                    partial_accuracy = self.accuracy.eval(feed_dict=feed_dict)

                    test_batch_size = len(batch[0])
                    test_accuracy = (tested_cnt * test_accuracy + partial_accuracy * test_batch_size) / (tested_cnt + test_batch_size)
                    tested_cnt += test_batch_size

                print('\ntest accuracy after %dth epoch: %g' % (i_epoch, test_accuracy))

                #
                # WRITE PER EPOCH SUMMARIES
                #

                feed_dict = {test_accuracy_summary_scalar: test_accuracy}

                if self.update_mode == 1 or self.update_mode == 2 or self.update_mode == 3:
                    print('accuracy of modified labels compared to true labels on a train set: %g' % (modified_labels_accuracy, ))
                    feed_dict[modified_labels_accuracy_summary_scalar] = modified_labels_accuracy

                if self.to_log(0):
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

                if self.to_log(1):
                    np.save('class_feature_means/' + self.dataset_name + '/' + self.model_name, class_feature_means_per_epoch)
                if self.to_log(2):
                    lid_features_per_epoch_per_element = np.append(
                        lid_features_per_epoch_per_element,
                        np.expand_dims(lid_features_per_element, 0),
                        0)
                    np.save('lid_features/' + self.dataset_name + '/' + self.model_name, lid_features_per_epoch_per_element)
                if self.to_log(3):
                    pre_lid_features_per_epoch_per_element = np.append(
                        pre_lid_features_per_epoch_per_element,
                        np.expand_dims(pre_lid_features_per_element, 0),
                        0)
                    np.save('pre_lid_features/' + self.dataset_name + '/' + self.model_name, pre_lid_features_per_epoch_per_element)
                if self.to_log(4):
                    logits_per_epoch_per_element = np.append(
                        logits_per_epoch_per_element,
                        np.expand_dims(logits_per_element, 0),
                        0)
                    np.save('logits/' + self.dataset_name + '/' + self.model_name, logits_per_epoch_per_element)

                print(timer.stop())

    def to_log(self, bit):
        return bitmask_contains(self.log_mask, bit)

    @staticmethod
    def compute_features(dataset_name, model_name, epoch, dataset_type, compute_pre_relu):
        with tf.Session() as sess:
            full_model_name = 'checkpoints/' + dataset_name + '/' + model_name + '/' + str(epoch)
            saver = tf.train.import_meta_graph(full_model_name + '.meta')
            saver.restore(sess, full_model_name)

            graph = tf.get_default_graph()

            x = graph.get_tensor_by_name('x:0')
            y_ = graph.get_tensor_by_name('y_:0')
            is_training = graph.get_tensor_by_name('is_training:0')
            # accuracy_op = graph.get_tensor_by_name('accuracy:0')
            if compute_pre_relu:
                features_op = graph.get_tensor_by_name('fc1/pre_lid_input:0')
            else:
                features_op = graph.get_tensor_by_name('fc1/lid_input:0')

            X, Y = read_dataset(name=dataset_name, type=dataset_type)

            features = np.empty((X.shape[0], features_op.shape[1]))

            for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE, False):
                batch_res = features_op.eval(feed_dict={x: batch[0], y_: batch[1], is_training: False})
                features[batch[2]] = batch_res

            if compute_pre_relu:
                np.save('pre_lid_features/' + dataset_name + '/' + model_name + '_' + dataset_type, np.expand_dims(features, 0))
            else:
                np.save('lid_features/' + dataset_name + '/' + model_name + '_' + dataset_type, np.expand_dims(features, 0))

    @staticmethod
    def test(dataset_name, model_name, epoch):
        with tf.Session() as sess:
            full_model_name = 'checkpoints/' + dataset_name + '/' + model_name + '/' + str(epoch) + '.meta'
            saver = tf.train.import_meta_graph(full_model_name)
            saver.restore(sess, full_model_name)

            graph = tf.get_default_graph()

            x = graph.get_tensor_by_name('x:0')
            y_ = graph.get_tensor_by_name('y_:0')
            is_training = graph.get_tensor_by_name('is_training:0')
            accuracy_op = graph.get_tensor_by_name('accuracy:0')

            X, Y = read_dataset(name=dataset_name, type='test')

            test_accuracy = 0
            i_batch = -1
            for batch in batch_iterator(X, Y, BATCH_SIZE, False):
                i_batch += 1

                partial_accuracy = accuracy_op.eval(feed_dict={x: batch[0], y_: batch[1], is_training: False})

                test_accuracy = (i_batch * test_accuracy + partial_accuracy) / (i_batch + 1)

            print('test accuracy %g' % test_accuracy)

    @staticmethod
    def calc_lid(X, Y, lid_calc_op, x, is_training):
        lid_score = 0

        i_batch = -1
        for batch in batch_iterator(X, Y, LID_BATCH_SIZE, True):
            i_batch += 1

            if i_batch == LID_BATCH_CNT:
                break

            batch_lid_scores = lid_calc_op.eval(feed_dict={x: batch[0], is_training: False})
            if batch_lid_scores.min() < 0:
                print('negative lid!', list(batch_lid_scores))
                i_batch -= 1
                continue
            if batch_lid_scores.max() > 10000:
                print('too big lig!', list(batch_lid_scores))
                i_batch -= 1
                continue

            lid_score += batch_lid_scores.mean()

        lid_score /= LID_BATCH_CNT

        return lid_score
