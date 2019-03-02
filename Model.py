import os
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from networks import build_mnist, build_cifar_10
from tf_ops import get_lid_calc_op, get_update_class_feature_selective_sum_and_counts_op, \
    get_update_class_covariances_selective_sum_op, get_LDA_logits_calc_op, get_Mahalanobis_distance_calc_op, \
    get_LDA_based_labels_calc_op
from consts import *
from Timer import timer
from preprocessing import read_dataset
from batch_iterate import batch_iterator, batch_iterator_with_indices
from tools import bitmask_contains, softmax


class Model:
    def __init__(self, dataset_name, model_name, update_mode, update_param, n_epochs, lr_segments=None,
                 log_mask=0, lid_use_pre_relu=False, lda_use_pre_relu=True,
                 n_blocks=1, block_width=256,
                 n_epochs_to_transition=None):
        """

        :param dataset_name:         dataset name: mnist/cifar-10
        :param model_name:      name of the model
        :param update_mode:     0: vanilla
                                1: as in the LID paper
                                2: lda
        :param update_param:    for update_mode = 2: number of sampling iterations
        :param log_mask:    in case it contains
                                    0th bit: logs LID data from the paper
                                    1st bit: logs relu features per element
                                    2nd bit: logs pre relu features per element
                                    3rd bit: logs logits per element
        """

        self.dataset_name = dataset_name
        self.update_mode = update_mode
        self.update_param = update_param
        self.lid_use_pre_relu = lid_use_pre_relu
        self.lda_use_pre_relu = lda_use_pre_relu
        self.log_mask = log_mask
        self.model_name = model_name
        self.n_blocks = n_blocks
        self.n_epochs = n_epochs
        self.lr_segments = lr_segments
        self.n_epochs_to_transition = n_epochs_to_transition

        self.block_width = block_width
        self.total_hidden_width = block_width * n_blocks
        self.dataset_size = DATASET_SIZE[dataset_name]

    def _build(self):
        #
        # PREPARE PLACEHOLDERS
        #

        if self.dataset_name == 'mnist':
            self.nn_input = tf.placeholder(tf.float32, [None, 784], name='x')
        elif self.dataset_name == 'cifar-10':
            self.nn_input = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')

        self.dataset_labels_pl = tf.placeholder(tf.float32, [None, N_CLASSES], name='y_')

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
                                                                                            self.n_blocks,
                                                                                            self.block_width)
        elif self.dataset_name == 'cifar-10':
            self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.preds = build_cifar_10(self.nn_input,
                                                                                               self.is_training,
                                                                                               self.n_blocks,
                                                                                               self.block_width)

        #
        # PREPARE FOR LABEL CHANGING
        #

        # LID calculation

        if self.update_mode == 1 or self.to_log(0):
            self.lid_per_epoch = np.array([])
            if self.lid_use_pre_relu:
                self.lid_calc_op = get_lid_calc_op(self.pre_lid_layer_op)
            else:
                self.lid_calc_op = get_lid_calc_op(self.lid_layer_op)

        # LDA predictor

        if self.update_mode == 2:
            self.feature_layer = None
            if self.lda_use_pre_relu:
                self.feature_layer = self.pre_lid_layer_op
            else:
                self.feature_layer = self.lid_layer_op

            self.class_feature_sums_var = tf.Variable(np.zeros((N_CLASSES, self.block_width)), dtype=tf.float32)
            self.class_feature_counts_var = tf.Variable(np.zeros((N_CLASSES,)), dtype=tf.float32)
            self.class_covariance_sums_var = tf.Variable(np.zeros((N_CLASSES, self.block_width, self.block_width)), dtype=tf.float32)

            self.features_pl = tf.placeholder(tf.float32, (None, self.block_width))
            self.class_feature_ops_keep_array_pl = tf.placeholder(tf.bool, (None,))

            self.update_class_features_sum_and_counts_op = get_update_class_feature_selective_sum_and_counts_op(
                self.class_feature_sums_var, self.class_feature_counts_var, self.features_pl, self.dataset_labels_pl,
                self.class_feature_ops_keep_array_pl)

            self.class_feature_means_pl = tf.placeholder(tf.float32, (N_CLASSES, self.block_width))

            self.update_class_covariances_op = get_update_class_covariances_selective_sum_op(
                self.class_covariance_sums_var, self.class_feature_means_pl, self.features_pl, self.dataset_labels_pl,
                self.class_feature_ops_keep_array_pl)

            self.reset_class_feature_sums_counts_and_covariance_op = tf.group(
                tf.assign(self.class_feature_sums_var, tf.zeros((N_CLASSES, self.block_width), tf.float32)),
                tf.assign(self.class_feature_counts_var, tf.zeros((N_CLASSES,), tf.float32)),
                tf.assign(self.class_covariance_sums_var, tf.zeros((N_CLASSES, self.block_width, self.block_width), tf.float32))
            )

            self.class_inv_covariances_pl = tf.placeholder(tf.float32, (N_CLASSES, self.block_width, self.block_width))

            self.mahalanobis_dist_calc = get_Mahalanobis_distance_calc_op(self.class_feature_means_pl,
                                                                          self.class_inv_covariances_pl,
                                                                          self.features_pl, self.dataset_labels_pl)

            self.block_class_feature_means_pl = tf.placeholder(tf.float32, (self.n_blocks, N_CLASSES, self.block_width))
            self.block_inv_covariance_pl = tf.placeholder(tf.float32, (self.n_blocks, self.block_width, self.block_width))

            self.LDA_logits = get_LDA_logits_calc_op(self.block_class_feature_means_pl, self.block_inv_covariance_pl,
                                                     self.feature_layer, np.ones(N_CLASSES, np.float32) / N_CLASSES)
            self.LDA_probs = get_LDA_based_labels_calc_op(self.LDA_logits)
            self.LDA_labels = tf.one_hot(tf.argmax(self.LDA_probs, 1), N_CLASSES)

        #
        # CREATE LOSS
        #

        with tf.name_scope('loss'):
            cross_entropy = None

            if self.update_mode == 0:
                self.new_labels_op = self.dataset_labels_pl
                self.modified_labels_op = self.dataset_labels_pl
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.dataset_labels_pl,
                                                                           logits=self.logits)
            if self.update_mode == 1 or self.to_log(0):
                self.alpha_var = tf.Variable(1, False, dtype=tf.float32, name='alpha')

                if self.update_mode == 1:
                    self.new_labels_op = tf.one_hot(tf.argmax(self.preds, 1), 10)
                    self.modified_labels_op = tf.identity(
                        self.alpha_var * self.dataset_labels_pl + (1 - self.alpha_var.value()) * self.new_labels_op,
                        name='modified_labels')
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=tf.stop_gradient(self.modified_labels_op),
                        logits=self.logits)

            if self.update_mode == 2:
                self.LDA_labels_weight_pl = tf.placeholder(tf.float32, ())
                self.new_labels_op = self.LDA_labels
                self.modified_labels_op = self.LDA_labels_weight_pl * tf.stop_gradient(self.new_labels_op) + \
                                          (1 - self.LDA_labels_weight_pl) * self.dataset_labels_pl

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.modified_labels_op,
                    logits=self.logits)

            self.reg_loss = 0
            cross_entropy += self.reg_loss

        self.cross_entropy = tf.reduce_mean(cross_entropy)

        #
        # CREATE ACCURACY
        #

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.preds, 1), tf.argmax(self.dataset_labels_pl, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

        with tf.name_scope('modified_accuracy'):
            mod_label = self.dataset_labels_pl
            if self.update_mode == 1:
                mod_label = self.modified_labels_op
            elif self.update_mode == 2:
                mod_label = self.modified_labels_op

            acc = tf.reduce_sum(tf.one_hot(tf.argmax(self.preds, 1), N_CLASSES) * mod_label, 1)
            self.modified_labels_accuracy = tf.reduce_mean(acc, name='accuracy')

        with tf.name_scope('new_labels_accuracy'):
            acc = tf.reduce_sum(tf.one_hot(tf.argmax(self.preds, 1), N_CLASSES) * self.new_labels_op, 1)
            self.new_labels_accuracy = tf.reduce_mean(acc, name='accuracy')

        self.epoch_pl = tf.placeholder(tf.int32)

        with tf.name_scope('learning_rate'):
            def produce_lr_tensor(segment_start=0, segment_i=1):
                if segment_i == len(self.lr_segments):
                    return self.lr_segments[-1][1]
                else:
                    segment_end = segment_start + self.lr_segments[segment_i - 1][0]
                    return tf.cond(pred=tf.cast(self.epoch_pl, tf.float32) < segment_end * self.n_epochs,
                                   true_fn=lambda: self.lr_segments[segment_i - 1][1],
                                   false_fn=lambda: produce_lr_tensor(segment_end, segment_i + 1))

            self.lr = produce_lr_tensor()

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

    def train(self, train_dataset_type='train'):
        #
        # LOAD
        #

        X, Y = read_dataset(name=self.dataset_name, type=train_dataset_type)
        X0, Y0 = read_dataset(name=self.dataset_name, type='train')

        Y_ind = np.argmax(Y, 1)
        Y0_ind = np.argmax(Y0, 1)
        is_sample_clean = (Y_ind == Y0_ind).astype(int)

        Y_ind_per_class = [[] for c in range(N_CLASSES)]
        for i in range(self.dataset_size):
            Y_ind_per_class[Y_ind[i]].append(i)

        X_test, Y_test = read_dataset(name=self.dataset_name, type='test')

        self._build()

        #
        # CREATE SUMMARIES
        #
        if True:
            tf.summary.scalar(name='cross_entropy', tensor=self.cross_entropy)
            tf.summary.scalar(name='train_accuracy', tensor=self.accuracy)
            tf.summary.scalar(name='reg_loss', tensor=self.reg_loss)
            tf.summary.scalar(name='modified_train_accuracy', tensor=self.modified_labels_accuracy)
            tf.summary.scalar(name='learning_rate', tensor=self.lr)
            summary = tf.summary.merge_all()

            test_accuracy_summary_scalar = tf.placeholder(tf.float32)
            test_accuracy_summary = tf.summary.scalar(name='test_accuracy', tensor=test_accuracy_summary_scalar)

            summaries_to_merge = [test_accuracy_summary]

            if self.update_mode == 1 or self.update_mode == 2:
                modified_labels_accuracy_summary_scalar = tf.placeholder(tf.float32)
                modified_labels_accuracy_summary = tf.summary.scalar(name='modified_labels_accuracy',
                                                                     tensor=modified_labels_accuracy_summary_scalar)
                summaries_to_merge.append(modified_labels_accuracy_summary)

                # similarity of new labels (i.e. with weight 0) compared with true labels without noise

                new_labels_accuracy_summary_scalar = tf.placeholder(tf.float32)
                new_labels_accuracy_summary = tf.summary.scalar(name='new_labels_accuracy',
                                                                tensor=new_labels_accuracy_summary_scalar)
                summaries_to_merge.append(new_labels_accuracy_summary)

                # similarity of new labels (i.e. with weight 0) compared with labels with noise

                new_labels_accuracy_with_noise_summary_scalar = tf.placeholder(tf.float32)
                new_labels_accuracy_with_noise_summary = tf.summary.scalar(
                    name='new_labels_accuracy_with_noise',
                    tensor=new_labels_accuracy_with_noise_summary_scalar)
                summaries_to_merge.append(new_labels_accuracy_with_noise_summary)

                # similarity of new labels (i.e. with weight 0) compared with true labels without noise but only on
                # clean samples

                new_labels_accuracy_on_clean_only_summary_scalar = tf.placeholder(tf.float32)
                new_labels_accuracy_on_clean_only_summary = tf.summary.scalar(
                    name='new_labels_accuracy_on_clean_only',
                    tensor=new_labels_accuracy_on_clean_only_summary_scalar)
                summaries_to_merge.append(new_labels_accuracy_on_clean_only_summary)

                # similarity of new labels (i.e. with weight 0) compared with true labels without noise but only on
                # noised samples

                new_labels_accuracy_on_noised_only_summary_scalar = tf.placeholder(tf.float32)
                new_labels_accuracy_on_noised_only_summary = tf.summary.scalar(
                    name='new_labels_accuracy_on_noised_only',
                    tensor=new_labels_accuracy_on_noised_only_summary_scalar)
                summaries_to_merge.append(new_labels_accuracy_on_noised_only_summary)

            lid_summary = None
            lid_summary_scalar = None
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
            lid_features_per_epoch_per_element = np.empty((0, self.dataset_size, self.total_hidden_width))
        if self.to_log(2):
            pre_lid_features_per_epoch_per_element = np.empty((0, self.dataset_size, self.total_hidden_width))
        if self.to_log(3):
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

            lid_per_epoch = None
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
            if self.update_mode == 2:
                self.block_class_feature_means = np.zeros((self.n_blocks, N_CLASSES, self.block_width))
                self.block_inv_covariances = np.array([np.eye(self.block_width, self.block_width) for i in range(self.n_blocks)])

            #
            # EPOCH LOOP
            #

            turning_epoch = -1  # number of epoch where we turn from regular loss function to the modified one

            i_step = -1
            for i_epoch in range(1, self.n_epochs + 1):
                timer.start()

                print('___________________________________________________________________________')
                print('\nSTARTING EPOCH %d, learning rate: %g\n' % (i_epoch, self.lr.eval({self.epoch_pl: i_epoch})))

                if X_augmented_iter is not None:
                    print('Augmenting data...')
                    X_augmented = X_augmented_iter.next()
                else:
                    X_augmented = X

                print('')

                #
                # TRAIN
                #

                print('starting training...')

                modified_labels_accuracy = 0
                new_labels_accuracy = 0
                new_labels_accuracy_with_noise = 0
                new_labels_accuracy_on_clean_only = 0
                new_labels_accuracy_on_noised_only = 0
                clean_samples_total_cnt = 0
                noised_samples_total_cnt = 0
                batch_cnt = -1
                for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE):
                    i_step += 1

                    batch_cnt += 1
                    batch_size = batch[0].shape[0]

                    feed_dict = {self.nn_input: batch[0], self.dataset_labels_pl: batch[1], self.is_training: True,
                                 self.epoch_pl: i_epoch}

                    if self.update_mode == 2:
                        feed_dict[self.LDA_labels_weight_pl] = 1 - self.alpha_var.eval(sess)
                        feed_dict[self.block_class_feature_means_pl] = self.block_class_feature_means
                        feed_dict[self.block_inv_covariance_pl] = self.block_inv_covariances

                    # Calculate different label accuracies
                    if self.update_mode == 1 or self.update_mode == 2:
                        modified_labels = self.modified_labels_op.eval(feed_dict=feed_dict)
                        batch_accs = np.sum(modified_labels * Y0[batch[2]])
                        modified_labels_accuracy = (modified_labels_accuracy * batch_cnt * BATCH_SIZE + batch_accs) / \
                                                   (batch_cnt * BATCH_SIZE + batch_size)

                        new_labels = self.new_labels_op.eval(feed_dict=feed_dict)
                        new_labels_ind = np.argmax(new_labels, 1)

                        new_labels_accs = (new_labels_ind == Y0_ind[batch[2]]).astype(int)
                        new_labels_acc_sum = np.sum(new_labels_accs)

                        new_labels_accuracy = (new_labels_accuracy * batch_cnt * BATCH_SIZE + new_labels_acc_sum) / \
                                              (batch_cnt * BATCH_SIZE + batch_size)

                        batch_accs = np.sum(new_labels_ind == Y_ind[batch[2]]).astype(int)
                        new_labels_accuracy_with_noise = (new_labels_accuracy_with_noise * batch_cnt * BATCH_SIZE +
                                                          batch_accs) / (batch_cnt * BATCH_SIZE + batch_size)

                        clean_samples = is_sample_clean[batch[2]]
                        clean_samples_cnt = np.sum(clean_samples)
                        batch_accs = np.sum(new_labels_accs * clean_samples)
                        new_labels_accuracy_on_clean_only = (
                                                        new_labels_accuracy_on_clean_only * clean_samples_total_cnt +
                                                        batch_accs) / (clean_samples_total_cnt + clean_samples_cnt)
                        clean_samples_total_cnt += clean_samples_cnt

                        noised_samples = 1 - clean_samples
                        noised_samples_cnt = np.sum(noised_samples)
                        batch_accs = np.sum(new_labels_accs * noised_samples)
                        new_labels_accuracy_on_noised_only = (
                                                        new_labels_accuracy_on_noised_only * noised_samples_total_cnt +
                                                        batch_accs) / (noised_samples_total_cnt + noised_samples_cnt)
                        noised_samples_total_cnt += noised_samples_cnt

                    self.train_step.run(feed_dict=feed_dict)
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

                lid_features_per_element = None
                if self.to_log(1):
                    lid_features_per_element = np.empty((self.dataset_size, self.total_hidden_width))

                pre_lid_features_per_element = None
                if self.to_log(2):
                    pre_lid_features_per_element = np.empty((self.dataset_size, self.total_hidden_width))

                logits_per_element = None
                if self.to_log(3):
                    logits_per_element = np.empty((self.dataset_size, N_CLASSES))

                if self.to_log(1) or self.to_log(2) or self.to_log(3):
                    for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE, False):
                        feed_dict = {self.nn_input: batch[0], self.dataset_labels_pl: batch[1], self.is_training: False}
                        if self.to_log(1):
                            lid_features_per_element[batch[2]] = self.lid_layer_op.eval(feed_dict=feed_dict)
                        if self.to_log(2):
                            pre_lid_features_per_element[batch[2]] = self.pre_lid_layer_op.eval(feed_dict=feed_dict)
                        if self.to_log(3):
                            logits_per_element[batch[2]] = self.logits.eval(feed_dict=feed_dict)

                if self.update_mode == 2:
                    #
                    # COMPUTE LDA PARAMETERS
                    #

                    self._compute_LDA(sess, X_augmented, Y, Y_ind, Y_ind_per_class)

                if self.update_mode == 1 or self.update_mode == 2 or self.to_log(0):
                    #
                    # CALCULATE LID
                    #

                    new_lid_score = self.calc_lid(X, Y, self.lid_calc_op, self.nn_input, self.is_training)
                    lid_per_epoch = np.append(lid_per_epoch, new_lid_score)

                    print('\nLID score after %dth epoch: %g' % (i_epoch, new_lid_score,))

                    #
                    # CHECK FOR STOPPING INIT PERIOD
                    #

                    if turning_epoch == -1 and i_epoch > EPOCH_WINDOW:
                        last_w_lids = lid_per_epoch[-EPOCH_WINDOW - 1: -1]

                        lid_check_value = new_lid_score - last_w_lids.mean() - 2 * last_w_lids.var() ** 0.5

                        print('LID check:', lid_check_value)

                        if lid_check_value > 0:
                            turning_epoch = i_epoch - 1

                            if self.update_mode == 1 or self.update_mode == 2:
                                saver.restore(sess, model_path + str(i_epoch - 1))
                                print('Turning point passed, reverting to previous epoch and starting'
                                      'using modified labels')

                    #
                    # MODIFY ALPHA
                    #

                    if turning_epoch != -1:
                        new_alpha_value = np.exp(-(i_epoch / self.n_epochs) * (lid_per_epoch[-1] / lid_per_epoch[:-1].min()))
                        # new_alpha_value = max(0, 1 - (i_epoch - turning_epoch) / self.n_epochs_to_transition)
                        print('\nnew alpha value:', new_alpha_value)
                    else:
                        new_alpha_value = 1

                    sess.run(self.alpha_var.assign(new_alpha_value))

                #
                # TEST ACCURACY
                #

                test_accuracy = 0
                i_batch = -1
                tested_cnt = 0
                for batch in batch_iterator_with_indices(X_test, Y_test, BATCH_SIZE, False):
                    i_batch += 1

                    feed_dict = {self.nn_input: batch[0], self.dataset_labels_pl: batch[1], self.is_training: False}

                    partial_accuracy = self.accuracy.eval(feed_dict=feed_dict)

                    test_batch_size = len(batch[0])
                    test_accuracy = (tested_cnt * test_accuracy + partial_accuracy * test_batch_size) / (tested_cnt + test_batch_size)
                    tested_cnt += test_batch_size

                print('\ntest accuracy after %dth epoch: %g' % (i_epoch, test_accuracy))

                #
                # WRITE PER EPOCH SUMMARIES
                #

                feed_dict = {test_accuracy_summary_scalar: test_accuracy}

                if self.update_mode == 1 or self.update_mode == 2:
                    print('accuracy of modified labels compared to true labels on a train set: %g' %
                          (modified_labels_accuracy, ))
                    feed_dict[modified_labels_accuracy_summary_scalar] = modified_labels_accuracy

                    print('accuracy of new labels compared to true labels on a train set: %g' %
                          (new_labels_accuracy,))
                    feed_dict[new_labels_accuracy_summary_scalar] = new_labels_accuracy

                    print('accuracy of new labels compared to noise labels on a train set: %g' %
                          (new_labels_accuracy_with_noise,))
                    feed_dict[new_labels_accuracy_with_noise_summary_scalar] = new_labels_accuracy_with_noise

                    print('accuracy of new labels on clean samples only on a train set: %g' %
                          (new_labels_accuracy_on_clean_only,))
                    feed_dict[new_labels_accuracy_on_clean_only_summary_scalar] = new_labels_accuracy_on_clean_only

                    print('accuracy of new labels on noised samples only on a train set: %g' %
                          (new_labels_accuracy_on_noised_only,))
                    feed_dict[new_labels_accuracy_on_noised_only_summary_scalar] = new_labels_accuracy_on_noised_only

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
                    lid_features_per_epoch_per_element = np.append(
                        lid_features_per_epoch_per_element,
                        np.expand_dims(lid_features_per_element, 0),
                        0)
                    np.save('lid_features/' + self.dataset_name + '/' + self.model_name, lid_features_per_epoch_per_element)
                if self.to_log(2):
                    pre_lid_features_per_epoch_per_element = np.append(
                        pre_lid_features_per_epoch_per_element,
                        np.expand_dims(pre_lid_features_per_element, 0),
                        0)
                    np.save('pre_lid_features/' + self.dataset_name + '/' + self.model_name, pre_lid_features_per_epoch_per_element)
                if self.to_log(3):
                    logits_per_epoch_per_element = np.append(
                        logits_per_epoch_per_element,
                        np.expand_dims(logits_per_element, 0),
                        0)
                    np.save('logits/' + self.dataset_name + '/' + self.model_name, logits_per_epoch_per_element)

                print(timer.stop())

    def _compute_LDA(self, sess, X_augmented, Y, Y_ind, Y_ind_per_class):
        print('Computing LDA parameters')

        features = np.empty((self.dataset_size, self.total_hidden_width))
        for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE, False):
            feed_dict = {self.nn_input: batch[0], self.is_training: False}
            features[batch[2]] = self.feature_layer.eval(feed_dict=feed_dict)
        features = np.reshape(features, (-1, self.n_blocks, self.block_width))

        self.block_class_feature_means = np.empty((0, N_CLASSES, self.block_width))
        self.block_inv_covariances = np.empty((0, self.block_width, self.block_width))

        for i_block in range(self.n_blocks):
            print('block', i_block + 1)
            print('__________________________________________')
            block_features = features[:, i_block, :]

            class_feature_means = None
            inv_covariance = None

            # initially select random subset of samples
            keep_arr = np.random.binomial(1, 0.5, self.dataset_size).astype(bool)
            for i in range(self.update_param + 1):
                sess.run(self.reset_class_feature_sums_counts_and_covariance_op)

                # compute class feature means
                for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE * 8, False):
                    feed_dict = {self.features_pl: block_features[batch[2]],
                                 self.dataset_labels_pl: batch[1],
                                 self.class_feature_ops_keep_array_pl: keep_arr[batch[2]]}
                    sess.run(self.update_class_features_sum_and_counts_op, feed_dict=feed_dict)

                counts = self.class_feature_counts_var.eval(sess)
                sums = self.class_feature_sums_var.eval(sess)

                class_feature_means = sums / counts.reshape((-1, 1))

                # compute class feature covariance matrices
                for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE * 8, False):
                    feed_dict = {self.features_pl: block_features[batch[2]], self.dataset_labels_pl: batch[1],
                                 self.class_feature_means_pl: class_feature_means,
                                 self.class_feature_ops_keep_array_pl: keep_arr[batch[2]]}
                    sess.run(self.update_class_covariances_op, feed_dict=feed_dict)

                class_covariance_sums = self.class_covariance_sums_var.eval(sess)
                class_covariances = class_covariance_sums / counts.reshape((-1, 1, 1)) + \
                                    np.eye(self.block_width, self.block_width) * EPS
                class_inv_covariances = np.array([np.linalg.inv(it) for it in class_covariances])

                # print('covariance determinants per class:',
                #       [np.linalg.det(it) for it in class_covariances])
                # print('min covariance eigenvalues per class:',
                #       [np.linalg.svd(it)[1][-1] for it in class_covariances])

                # compute tied covariance matrix
                covariance = np.sum(class_covariance_sums, 0) / np.sum(counts) + \
                             np.eye(self.block_width, self.block_width) * EPS
                inv_covariance = np.linalg.inv(covariance)

                print('tied covariance determinant:',
                      np.linalg.det(covariance))
                print('min tied covariance eigenvalue:',
                      np.linalg.svd(covariance)[1][-1])

                # compute Mahalanobis distance based anomaly scores
                mahalanobis_distances = np.empty(self.dataset_size)
                for batch in batch_iterator_with_indices(X_augmented, Y, BATCH_SIZE * 8, False):
                    feed_dict = {self.features_pl: block_features[batch[2]],
                                 self.class_feature_means_pl: class_feature_means,
                                 self.class_inv_covariances_pl: class_inv_covariances,
                                 self.dataset_labels_pl: batch[1]}
                    mah_dists = sess.run(self.mahalanobis_dist_calc, feed_dict=feed_dict)
                    mahalanobis_distances[batch[2]] = mah_dists

                mahalanobis_distances_per_class = [np.take(mahalanobis_distances, Y_ind_per_class[c]) for c in
                                                   range(N_CLASSES)]

                # select less anomalous samples for the next subset
                medians = np.array([np.median(arr) for arr in mahalanobis_distances_per_class])
                print('Mahalanobis distance medians per class', [round(it ** 0.5, 1) for it in medians])

                keep_arr = mahalanobis_distances < np.take(medians, Y_ind)

            self.block_class_feature_means = np.append(self.block_class_feature_means,
                                                       np.expand_dims(class_feature_means, 0), 0)
            self.block_inv_covariances = np.append(self.block_inv_covariances,
                                                   np.expand_dims(inv_covariance, 0), 0)

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
    def compute_block_features(dataset_name, model_name, epoch, dataset_type, n_blocks):
        with tf.Session() as sess:
            full_model_name = 'checkpoints/' + dataset_name + '/' + model_name + '/' + str(epoch)
            saver = tf.train.import_meta_graph(full_model_name + '.meta')
            saver.restore(sess, full_model_name)

            graph = tf.get_default_graph()

            x = graph.get_tensor_by_name('x:0')
            y_ = graph.get_tensor_by_name('y_:0')
            is_training = graph.get_tensor_by_name('is_training:0')
            # accuracy_op = graph.get_tensor_by_name('accuracy:0')
            block_features_ops = []
            block_logit_ops = []
            for i in range(n_blocks):
                block_features_ops.append(graph.get_tensor_by_name('fc1/bn_' + str(i + 1) + ':0'))
                block_logit_ops.append(graph.get_tensor_by_name('fc2/logits_' + str(i + 1) + ':0'))

            n_dims = block_features_ops[0].shape[1]

            X, Y = read_dataset(name=dataset_name, type=dataset_type)

            def list_round(arr, d=2):
                from collections.abc import Iterable

                res = ''
                if isinstance(arr[0], Iterable):
                    for it in arr:
                        res += list_round(it, d) + '\n'
                else:
                    return str(list(np.round(arr, d)))

                return res

            for batch in batch_iterator_with_indices(X, Y, 1):
                # for i in range(n_blocks):
                #     features = block_features_ops[i].eval({x: batch[0], is_training: False})
                #     for j in range(batch_size):
                #         print(list_round(features[j]))
                block_preds = []
                for i in range(n_blocks):
                    logits = block_logit_ops[i].eval({x: batch[0], is_training: False})
                    preds = softmax(logits[0])
                    block_preds.append(preds)
                    # print(list_round(logits[j]), list_round(preds))
                    # print(list_round(preds))

                preds_co_acc = np.zeros((n_blocks, n_blocks))
                for i in range(n_blocks):
                    for j in range(n_blocks):
                        preds_co_acc[i, j] = np.dot(block_preds[i], block_preds[j])
                print(list_round(preds_co_acc))

                print('\n\n')



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
