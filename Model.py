import os
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from networks import build_mnist, build_cifar_10
from cifar_100_resnet import build_cifar_100
from tf_ops import get_lid_calc_op, get_update_class_feature_selective_sum_and_counts_op, \
    get_update_class_covariances_selective_sum_op, get_LDA_logits_calc_op, get_Mahalanobis_distance_calc_op, \
    get_LDA_based_labels_calc_op
from consts import *
from Timer import timer
from preprocessing import read_dataset
from batch_iterate import batch_iterator, batch_iterator_with_indices
from tools import bitmask_contains, softmax
from noise_dataset import introduce_symmetric_noise


class Model:
    def __init__(self, dataset_name, model_name, update_mode, init_epochs, n_epochs, reg_coef, lr_segments=None,
                 log_mask=0, lid_use_pre_relu=False, lda_use_pre_relu=True,
                 n_blocks=1, block_width=256,
                 n_label_resets=0, cut_train_set=False, mod_labels_after_last_reset=True, use_loss_weights=False):
        """

        :param dataset_name:         dataset name: mnist/cifar-10/cifar-100
        :param model_name:      name of the model
        :param update_mode:     0: vanilla
                                1: as in the LID paper
                                2: lda
        :param log_mask:    in case it contains
                                    0th bit: logs LID data from the paper
                                    1st bit: logs relu features per element
                                    2nd bit: logs pre relu features per element
                                    3rd bit: logs logits per element
        """

        self.dataset_name = dataset_name
        self.update_mode = update_mode
        self.lid_use_pre_relu = lid_use_pre_relu
        self.lda_use_pre_relu = lda_use_pre_relu
        self.log_mask = log_mask
        self.model_name = model_name
        self.n_blocks = n_blocks
        self.n_epochs = n_epochs
        self.lr_segments = lr_segments
        self.n_label_resets = n_label_resets
        self.cut_train_set = cut_train_set
        self.mod_labels_after_last_reset = mod_labels_after_last_reset
        self.use_loss_weights = use_loss_weights
        self.init_epochs = init_epochs
        self.reg_coef = reg_coef

        self.block_width = block_width
        self.total_hidden_width = block_width * n_blocks

        if self.dataset_name == 'mnist' or self.dataset_name == 'cifar-10':
            self.n_classes = 10
        elif self.dataset_name == 'cifar-100':
            self.n_classes = 100

    def _build(self):
        #
        # PREPARE PLACEHOLDERS
        #

        if self.dataset_name == 'mnist':
            self.nn_input_pl = tf.placeholder(tf.float32, [None, 784], name='x')
        elif self.dataset_name == 'cifar-10' or self.dataset_name == 'cifar-100':
            self.nn_input_pl = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')

        self.labels_pl = tf.placeholder(tf.float32, [None, self.n_classes], name='y_')

        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        #
        # PREPARE DATA AUGMENTATION OPERATION
        #

        self.data_augmenter = None
        if self.dataset_name == 'cifar-10' or self.dataset_name == 'cifar-100':
            self.data_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

        #
        # BUILD NETWORK
        #

        reg_loss_unscaled = 0.0
        if self.dataset_name == 'mnist':
            self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.preds = build_mnist(self.nn_input_pl,
                                                                                            self.is_training,
                                                                                            self.n_blocks,
                                                                                            self.block_width)
        elif self.dataset_name == 'cifar-10':
            self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.preds, reg_loss_unscaled = build_cifar_10(
                self.nn_input_pl, self.is_training, self.n_blocks, self.block_width)

        elif self.dataset_name == 'cifar-100':
            self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.preds, reg_loss_unscaled = build_cifar_100(
                self.nn_input_pl,
                self.is_training)

        #
        # PREPARE FOR LABEL CHANGING
        #

        # LID calculation

        if self.update_mode == 1 or self.to_log(0):
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

            self.class_feature_sums_var = tf.Variable(np.zeros((self.n_classes, self.block_width)), dtype=tf.float32)
            self.class_feature_counts_var = tf.Variable(np.zeros((self.n_classes,)), dtype=tf.float32)
            self.class_covariance_sums_var = tf.Variable(np.zeros((self.n_classes, self.block_width, self.block_width)),
                                                         dtype=tf.float32)

            self.features_pl = tf.placeholder(tf.float32, (None, self.block_width))
            self.class_feature_ops_keep_array_pl = tf.placeholder(tf.bool, (None,))

            self.update_class_features_sum_and_counts_op = get_update_class_feature_selective_sum_and_counts_op(
                self.class_feature_sums_var, self.class_feature_counts_var, self.features_pl, self.labels_pl,
                self.class_feature_ops_keep_array_pl)

            self.class_feature_means_pl = tf.placeholder(tf.float32, (self.n_classes, self.block_width))

            self.update_class_covariances_op = get_update_class_covariances_selective_sum_op(
                self.class_covariance_sums_var, self.class_feature_means_pl, self.features_pl, self.labels_pl,
                self.class_feature_ops_keep_array_pl)

            self.reset_class_feature_sums_counts_and_covariance_op = tf.group(
                tf.assign(self.class_feature_sums_var, tf.zeros((self.n_classes, self.block_width), tf.float32)),
                tf.assign(self.class_feature_counts_var, tf.zeros((self.n_classes,), tf.float32)),
                tf.assign(self.class_covariance_sums_var, tf.zeros((self.n_classes, self.block_width, self.block_width),
                                                                   dtype=tf.float32))
            )

            self.class_inv_covariances_pl = tf.placeholder(tf.float32, (self.n_classes, self.block_width, self.block_width))

            self.mahalanobis_dist_calc = get_Mahalanobis_distance_calc_op(self.class_feature_means_pl,
                                                                          self.class_inv_covariances_pl,
                                                                          self.features_pl, self.labels_pl)

            self.block_class_feature_means_pl = tf.placeholder(tf.float32, (self.n_blocks, self.n_classes, self.block_width))
            self.block_inv_covariance_pl = tf.placeholder(tf.float32,
                                                          (self.n_blocks, self.block_width, self.block_width))

            self.LDA_logits = get_LDA_logits_calc_op(self.block_class_feature_means_pl, self.block_inv_covariance_pl,
                                                     self.feature_layer, np.ones(self.n_classes, np.float32) / self.n_classes)
            self.LDA_probs = get_LDA_based_labels_calc_op(self.LDA_logits)
            self.LDA_labels = tf.one_hot(tf.argmax(self.LDA_probs, 1), self.n_classes)

        #
        # CREATE LOSS
        #

        with tf.name_scope('loss'):
            self.loss_weights_pl = tf.placeholder(tf.float32, (None,), 'loss_weights')

            cross_entropy = None

            if self.update_mode == 0:
                self.new_labels_op = tf.one_hot(tf.argmax(self.preds, 1), self.n_classes)
                self.modified_labels_op = self.labels_pl
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels_pl,
                                                                           logits=self.logits)
            if self.update_mode == 1 or self.to_log(0):
                self.alpha_var = tf.Variable(1, False, dtype=tf.float32, name='alpha')

                if self.update_mode == 1:
                    self.new_labels_op = tf.one_hot(tf.argmax(self.preds, 1), self.n_classes)
                    self.modified_labels_op = tf.identity(
                        self.alpha_var * self.labels_pl + (1 - self.alpha_var) * self.new_labels_op,
                        name='modified_labels')
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=tf.stop_gradient(self.modified_labels_op),
                        logits=self.logits)

            if self.update_mode == 2:
                self.LDA_labels_weight_pl = tf.placeholder(tf.float32, ())
                self.new_labels_op = self.LDA_labels
                self.modified_labels_op = (1 - self.LDA_labels_weight_pl) * tf.stop_gradient(self.new_labels_op) + \
                    self.LDA_labels_weight_pl * self.labels_pl

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self.modified_labels_op,
                    logits=self.logits)

            self.reg_loss = tf.identity(2 * self.reg_coef * reg_loss_unscaled, 'reg_loss')

        self.cross_entropy = self.reg_loss + tf.reduce_mean(cross_entropy * self.loss_weights_pl)

        #
        # CREATE ACCURACY
        #

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.preds, 1), tf.argmax(self.labels_pl, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

        with tf.name_scope('modified_accuracy'):
            mod_label = self.labels_pl
            if self.update_mode == 1 or self.update_mode == 2:
                mod_label = self.modified_labels_op

            acc = tf.reduce_sum(tf.one_hot(tf.argmax(self.preds, 1), self.n_classes) * mod_label, 1)
            self.modified_labels_accuracy = tf.reduce_mean(acc, name='accuracy')

        self.rel_epoch_pl = tf.placeholder(tf.int32)

        with tf.name_scope('learning_rate'):
            def produce_lr_tensor(segment_start=0, segment_i=1):
                if segment_i == len(self.lr_segments):
                    return self.lr_segments[-1][1]
                else:
                    segment_end = segment_start + self.lr_segments[segment_i - 1][0]
                    return tf.cond(pred=tf.cast(self.rel_epoch_pl, tf.float32) < segment_end * self.n_epochs,
                                   true_fn=lambda: self.lr_segments[segment_i - 1][1],
                                   false_fn=lambda: produce_lr_tensor(segment_end, segment_i + 1))

            self.lr = produce_lr_tensor()

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

    def train(self, train_dataset_type='train_without_val', noise_ratio=0, noise_seed=None):
        #
        # LOAD
        #

        X, Y0 = read_dataset(name=self.dataset_name, type=train_dataset_type)
        Y = np.copy(Y0)
        introduce_symmetric_noise(Y, noise_ratio, noise_seed)

        Y_ind = np.argmax(Y, 1)
        Y0_cls_ind = np.argmax(Y0, 1)
        self.is_sample_clean = (Y_ind == Y0_cls_ind).astype(int)

        Y_ind_per_class = [[] for c in range(self.n_classes)]
        for i in range(X.shape[0]):
            Y_ind_per_class[Y_ind[i]].append(i)

        X_current = np.array(X)
        self.current_dataset_size = X.shape[0]
        X_current_ind = np.arange(self.current_dataset_size)

        Y_current = np.array(Y)
        Y_current_cls_ind = np.array(Y_ind)
        Y_current_ind_per_class = np.array(Y_ind_per_class)

        Y0_current_cls_ind = np.array(Y0_cls_ind)
        Y0_current = np.array(Y0)

        X_test, Y_test = read_dataset(name=self.dataset_name, type='test')
        X_validation, Y_validation = read_dataset(name=self.dataset_name, type='validation')

        current_loss_weights = np.full((self.current_dataset_size,), 1, np.float32)

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

            clean_train_accuracy_summary_scalar = tf.placeholder(tf.float32)
            clean_train_accuracy_summary = tf.summary.scalar(name='clean_train_accuracy',
                                                             tensor=clean_train_accuracy_summary_scalar)
            clean_train_accuracy_on_clean_summary_scalar = tf.placeholder(tf.float32)
            clean_train_accuracy_on_clean_summary = tf.summary.scalar(name='clean_train_accuracy_on_clean',
                                                                    tensor=clean_train_accuracy_on_clean_summary_scalar)
            clean_train_accuracy_on_noised_summary_scalar = tf.placeholder(tf.float32)
            clean_train_accuracy_on_noised_summary = tf.summary.scalar(name='clean_train_accuracy_on_noised',
                                                             tensor=clean_train_accuracy_on_noised_summary_scalar)
            test_accuracy_summary_scalar = tf.placeholder(tf.float32)
            test_accuracy_summary = tf.summary.scalar(name='test_accuracy', tensor=test_accuracy_summary_scalar)
            validation_accuracy_summary_scalar = tf.placeholder(tf.float32)
            validation_accuracy_summary = tf.summary.scalar(name='validation_accuracy',
                                                            tensor=validation_accuracy_summary_scalar)

            summaries_to_merge = [test_accuracy_summary, validation_accuracy_summary, clean_train_accuracy_summary,
                                  clean_train_accuracy_on_clean_summary, clean_train_accuracy_on_noised_summary]

            modified_labels_accuracy_summary_scalar = None
            new_labels_accuracy_summary_scalar = None
            new_labels_accuracy_with_noise_summary_scalar = None
            new_labels_accuracy_on_clean_only_summary_scalar = None
            new_labels_accuracy_on_noised_only_summary_scalar = None
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
            alpha_summary_scalar = None
            if self.to_log(0):
                lid_summary_scalar = tf.placeholder(tf.float32)
                lid_summary = tf.summary.scalar(name='LID', tensor=lid_summary_scalar)

                alpha_summary_scalar = tf.placeholder(tf.float32)
                alpha_summary = tf.summary.scalar(name='alpha', tensor=alpha_summary_scalar)

                summaries_to_merge.extend([lid_summary, alpha_summary])

            per_epoch_summary = tf.summary.merge(summaries_to_merge)

        saver = tf.train.Saver(max_to_keep=1)
        saver0 = tf.train.Saver(max_to_keep=1000)
        model_path = 'checkpoints/' + self.dataset_name + '/' + self.model_name + '/'

        lid_features_per_epoch_per_element = None
        if self.to_log(1):
            lid_features_per_epoch_per_element = np.empty((0, self.current_dataset_size, self.total_hidden_width))
        pre_lid_features_per_epoch_per_element = None
        if self.to_log(2):
            pre_lid_features_per_epoch_per_element = np.empty((0, self.current_dataset_size, self.total_hidden_width))
        logits_per_epoch_per_element = None
        if self.to_log(3):
            logits_per_epoch_per_element = np.empty((0, self.current_dataset_size, self.n_classes))

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
            saver0.save(sess, model_path + 'start')

            #
            # CALCULATE AND LOG INITIAL LID SCORE
            #

            lid_per_epoch = None
            if self.update_mode == 1 or self.to_log(0):
                initial_lid_score = self._calc_lid(X_current, Y_current, self.lid_calc_op, self.nn_input_pl, self.is_training)
                if initial_lid_score is None:
                    lid_per_epoch = []
                else:
                    lid_per_epoch = [initial_lid_score]

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
                self.data_augmenter.fit(X_current)
                X_augmented_iter = self.data_augmenter.flow(X_current,
                                                            batch_size=self.current_dataset_size,
                                                            shuffle=False)

            #
            # LDA INIT
            #
            if self.update_mode == 2:
                self.block_class_feature_means = np.zeros((self.n_blocks, self.n_classes, self.block_width))
                self.block_inv_covariances = np.array([np.eye(self.block_width, self.block_width) for i in range(self.n_blocks)])

            #
            # EPOCH LOOP
            #

            alpha_value = 1
            turning_rel_epoch = -1  # number of epoch where we turn from regular loss function to the modified one

            n_label_resets_done = 0
            i_epoch_tot = 0
            i_epoch_rel = 0
            i_step = -1
            while i_epoch_rel < self.n_epochs:
                timer.start()

                i_epoch_rel += 1
                i_epoch_tot += 1

                print('___________________________________________________________________________')
                print('\nSTARTING EPOCH relative: %d, total: %d; learning rate: %g\n' %
                      (i_epoch_rel, i_epoch_tot, self.lr.eval({self.rel_epoch_pl: i_epoch_rel})))

                if X_augmented_iter is not None:
                    print('Augmenting data...')
                    X_augmented = X_augmented_iter.next()
                else:
                    X_augmented = X_current

                print('')

                if self.update_mode == 2:
                    #
                    # COMPUTE LDA PARAMETERS
                    #

                    # TODO: would unaugmented data change LDA algo performance?
                    self._compute_LDA(sess, X_augmented, Y_current, Y_current_cls_ind, Y_current_ind_per_class)

                #
                # TRAIN
                #

                if self.update_mode == 1 or self.update_mode == 2:
                    sess.run(self.alpha_var.assign(alpha_value))

                print('starting training...')

                for batch in batch_iterator_with_indices(X_augmented, Y_current, BATCH_SIZE):
                    i_step += 1

                    feed_dict = {self.nn_input_pl: batch[0], self.labels_pl: batch[1], self.is_training: True,
                                 self.rel_epoch_pl: i_epoch_rel,
                                 self.loss_weights_pl: current_loss_weights[batch[2]]}

                    if self.update_mode == 2:
                        feed_dict[self.LDA_labels_weight_pl] = self.alpha_var.eval(sess)
                        feed_dict[self.block_class_feature_means_pl] = self.block_class_feature_means
                        feed_dict[self.block_inv_covariance_pl] = self.block_inv_covariances

                    self.train_step.run(feed_dict=feed_dict)

                    if i_step % 100 == 0:
                        feed_dict[self.is_training] = False
                        train_accuracy = self.accuracy.eval(feed_dict=feed_dict)
                        print('\tstep %d, training accuracy %g' % (i_step, train_accuracy))

                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i_step)
                        summary_writer.flush()

                #
                # LOG THINGS
                #

                lid_features_per_element = None
                if self.to_log(1):
                    lid_features_per_element = np.empty((self.current_dataset_size, self.total_hidden_width))
                pre_lid_features_per_element = None
                if self.to_log(2):
                    pre_lid_features_per_element = np.empty((self.current_dataset_size, self.total_hidden_width))
                logits_per_element = None
                if self.to_log(3):
                    logits_per_element = np.empty((self.current_dataset_size, self.n_classes))

                modified_labels_accuracy = 0
                new_labels_accuracy = 0
                new_labels_accuracy_with_noise = 0
                new_labels_accuracy_on_clean_only = 0
                new_labels_accuracy_on_noised_only = 0
                clean_samples_total_cnt = 0
                noised_samples_total_cnt = 0
                batch_cnt = -1
                for batch in batch_iterator_with_indices(X_augmented, Y_current, BATCH_SIZE, False):
                    batch_size = batch[0].shape[0]
                    batch_cnt += 1
                    feed_dict = {self.nn_input_pl: batch[0], self.labels_pl: batch[1], self.is_training: False}
                    if self.update_mode == 2:
                        feed_dict[self.LDA_labels_weight_pl] = self.alpha_var.eval(sess)
                        feed_dict[self.block_class_feature_means_pl] = self.block_class_feature_means
                        feed_dict[self.block_inv_covariance_pl] = self.block_inv_covariances

                    # Calculate different label accuracies
                    if self.update_mode == 1 or self.update_mode == 2:
                        modified_labels = self.modified_labels_op.eval(feed_dict=feed_dict)
                        batch_accs = np.sum(modified_labels * Y0_current[batch[2]])
                        modified_labels_accuracy = (modified_labels_accuracy * batch_cnt * BATCH_SIZE + batch_accs) / \
                                                   (batch_cnt * BATCH_SIZE + batch_size)

                        new_labels = self.new_labels_op.eval(feed_dict=feed_dict)
                        new_labels_ind = np.argmax(new_labels, 1)

                        new_labels_accs = (new_labels_ind == Y0_current_cls_ind[batch[2]]).astype(int)
                        new_labels_acc_sum = np.sum(new_labels_accs)

                        new_labels_accuracy = (new_labels_accuracy * batch_cnt * BATCH_SIZE + new_labels_acc_sum) / \
                                              (batch_cnt * BATCH_SIZE + batch_size)

                        batch_accs = np.sum(new_labels_ind == Y_current_cls_ind[batch[2]]).astype(int)
                        new_labels_accuracy_with_noise = (new_labels_accuracy_with_noise * batch_cnt * BATCH_SIZE +
                                                          batch_accs) / (batch_cnt * BATCH_SIZE + batch_size)

                        clean_samples = self.is_sample_clean[batch[2]]
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
                            new_labels_accuracy_on_noised_only * noised_samples_total_cnt + batch_accs) / \
                                max((noised_samples_total_cnt + noised_samples_cnt), 1)
                        noised_samples_total_cnt += noised_samples_cnt

                    if self.to_log(1):
                        lid_features_per_element[batch[2]] = self.lid_layer_op.eval(feed_dict=feed_dict)
                    if self.to_log(2):
                        pre_lid_features_per_element[batch[2]] = self.pre_lid_layer_op.eval(feed_dict=feed_dict)
                    if self.to_log(3):
                        logits_per_element[batch[2]] = self.logits.eval(feed_dict=feed_dict)

                if self.to_log(1):
                    lid_features_per_epoch_per_element = np.append(
                        lid_features_per_epoch_per_element,
                        np.expand_dims(lid_features_per_element, 0),
                        0)
                    np.save(file='lid_features/' + self.dataset_name + '/' + self.model_name,
                            arr=lid_features_per_epoch_per_element)

                if self.to_log(2):
                    pre_lid_features_per_epoch_per_element = np.append(
                        pre_lid_features_per_epoch_per_element,
                        np.expand_dims(pre_lid_features_per_element, 0),
                        0)
                    np.save(file='pre_lid_features/' + self.dataset_name + '/' + self.model_name,
                            arr=pre_lid_features_per_epoch_per_element)

                if self.to_log(3):
                    logits_per_epoch_per_element = np.append(
                        logits_per_epoch_per_element,
                        np.expand_dims(logits_per_element, 0),
                        0)
                    np.save(file='logits/' + self.dataset_name + '/' + self.model_name,
                            arr=logits_per_epoch_per_element)

                #
                # TEST/VALIDATION ACCURACY
                #

                clean_train_accuracy = self.calc_accuracy_on_dataset(X_current, Y0_current)

                cleans_indices = np.nonzero(self.is_sample_clean)[0]
                clean_train_accuracy_on_clean = self.calc_accuracy_on_dataset(X_current[cleans_indices],
                                                                              Y0_current[cleans_indices])
                noised_indices = np.nonzero(1 - self.is_sample_clean)[0]
                if len(noised_indices) > 0:
                    clean_train_accuracy_on_noised = self.calc_accuracy_on_dataset(X_current[noised_indices],
                                                                                   Y0_current[noised_indices])
                else:
                    clean_train_accuracy_on_noised = 0

                test_accuracy = self.calc_accuracy_on_dataset(X_test, Y_test)
                validation_accuracy = self.calc_accuracy_on_dataset(X_validation, Y_validation)

                print('\nclean train accuracy after %dth epoch: %g' % (i_epoch_tot, clean_train_accuracy))
                print('clean train accuracy on clean samples after %dth epoch: %g' % (i_epoch_tot,
                                                                                      clean_train_accuracy_on_clean))
                print('clean train accuracy on noised samples after %dth epoch: %g' % (i_epoch_tot,
                                                                                       clean_train_accuracy_on_noised))
                print('\ntest accuracy after %dth epoch: %g' % (i_epoch_tot, test_accuracy))
                print('validation accuracy after %dth epoch: %g' % (i_epoch_tot, validation_accuracy))

                #
                # WRITE PER EPOCH SUMMARIES
                #

                feed_dict = {test_accuracy_summary_scalar: test_accuracy,
                             validation_accuracy_summary_scalar: validation_accuracy,
                             clean_train_accuracy_summary_scalar: clean_train_accuracy,
                             clean_train_accuracy_on_clean_summary_scalar: clean_train_accuracy_on_clean,
                             clean_train_accuracy_on_noised_summary_scalar: clean_train_accuracy_on_noised}

                if self.update_mode == 1 or self.update_mode == 2:
                    print('accuracy of modified labels compared to true labels on a train set: %g' %
                          (modified_labels_accuracy, ))
                    feed_dict[modified_labels_accuracy_summary_scalar] = modified_labels_accuracy

                    print('accuracy of new labels compared to true labels on a train set: %g' %
                          (new_labels_accuracy,))
                    feed_dict[new_labels_accuracy_summary_scalar] = new_labels_accuracy

                    print('accuracy of new labels compared to current labels on a train set: %g' %
                          (new_labels_accuracy_with_noise,))
                    feed_dict[new_labels_accuracy_with_noise_summary_scalar] = new_labels_accuracy_with_noise

                    print('accuracy of new labels on clean samples only on a train set: %g' %
                          (new_labels_accuracy_on_clean_only,))
                    feed_dict[new_labels_accuracy_on_clean_only_summary_scalar] = new_labels_accuracy_on_clean_only

                    print('accuracy of new labels on noised samples only on a train set: %g' %
                          (new_labels_accuracy_on_noised_only,))
                    feed_dict[new_labels_accuracy_on_noised_only_summary_scalar] = new_labels_accuracy_on_noised_only

                if self.update_mode == 1 or self.update_mode == 2 or self.to_log(0):
                    #
                    # CALCULATE LID
                    #

                    # TODO: would augmented data change LIDs?
                    new_lid_score = self._calc_lid(X_current, Y_current, self.lid_calc_op, self.nn_input_pl, self.is_training)
                    lid_per_epoch.append(new_lid_score)

                    print('\nLID score after %dth epoch: %g' % (i_epoch_tot, new_lid_score,))

                if self.to_log(0):
                    feed_dict[lid_summary_scalar] = lid_per_epoch[-1]
                    feed_dict[alpha_summary_scalar] = alpha_value

                summary_str = sess.run(per_epoch_summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i_step + 1)
                summary_writer.flush()

                if self.update_mode == 1 or self.update_mode == 2:
                    #
                    # CHECK FOR STOPPING INIT PERIOD
                    #

                    if turning_rel_epoch == -1 and len(lid_per_epoch) > self.init_epochs + EPOCH_WINDOW:
                        if self.mod_labels_after_last_reset or not(0 < self.n_label_resets == n_label_resets_done):
                            last_w_lids = lid_per_epoch[-EPOCH_WINDOW - 1: -1]

                            lid_check_value = lid_per_epoch[-1] - np.mean(last_w_lids) - 2 * np.std(last_w_lids)

                            print('LID check:', lid_check_value)

                            if lid_check_value > 0:
                                turning_rel_epoch = i_epoch_rel - 1
                                print('Turning point passed, starting using modified labels')

                    #
                    # MODIFY ALPHA
                    #

                    if turning_rel_epoch != -1:
                        alpha_value = np.exp(
                            -(i_epoch_rel / self.n_epochs) * (lid_per_epoch[-1] /
                                                              np.min(lid_per_epoch[self.init_epochs:-1])))
                        print('\nnext alpha value:', alpha_value)

                if i_epoch_rel == self.n_epochs and n_label_resets_done < self.n_label_resets:
                    print('reached the end, resetting current labels')

                    Y_current_new = np.empty((self.current_dataset_size, self.n_classes), np.float32)
                    for batch in batch_iterator_with_indices(X_augmented, Y_current, BATCH_SIZE, False):
                        feed_dict = {self.nn_input_pl: batch[0], self.is_training: False}
                        if self.update_mode == 2:
                            feed_dict[self.block_class_feature_means_pl] = self.block_class_feature_means
                            feed_dict[self.block_inv_covariance_pl] = self.block_inv_covariances
                        if self.update_mode == 0:
                            feed_dict[self.labels_pl] = batch[1]
                        new_labels = self.new_labels_op.eval(feed_dict)
                        Y_current_new[batch[2]] = new_labels

                    if self.cut_train_set:
                        Y_current_cls_ind_new = np.argmax(Y_current_new, 1)
                        are_labels_equal_ind = np.where(Y_current_cls_ind == Y_current_cls_ind_new)[0]

                        X_current = np.array(X_current[are_labels_equal_ind])
                        X_current_ind = np.array(X_current_ind[are_labels_equal_ind])

                        Y_current = np.array(Y_current_new[are_labels_equal_ind])
                        Y_current_cls_ind = np.array(Y_current_cls_ind_new[are_labels_equal_ind])

                        Y0_current_cls_ind = np.array(Y0_cls_ind[X_current_ind])
                        Y0_current = np.array(Y0[X_current_ind])

                        self.current_dataset_size = X_current.shape[0]
                        self.is_sample_clean = (Y_current_cls_ind == Y0_current_cls_ind).astype(int)

                        Y_current_ind_per_class = [[] for c in range(self.n_classes)]
                        for i in range(self.current_dataset_size):
                            Y_current_ind_per_class[Y_current_cls_ind[i]].append(i)

                        if self.data_augmenter is not None:
                            self.data_augmenter.fit(X_current)
                            X_augmented_iter = self.data_augmenter.flow(X_current,
                                                                        batch_size=self.current_dataset_size,
                                                                        shuffle=False)

                        print('new train set size: %d; clean: %d; noisy: %d' %
                              (self.current_dataset_size,
                               self.is_sample_clean.sum(),
                               self.current_dataset_size - self.is_sample_clean.sum())
                              )
                    else:
                        if self.use_loss_weights:
                            # Compute new loss weights

                            new_loss_weights = np.empty((self.current_dataset_size,), np.float32)
                            for batch in batch_iterator_with_indices(X_current, Y_current, BATCH_SIZE, False):
                                feed_dict = {self.nn_input_pl: batch[0], self.labels_pl: batch[1],
                                             self.is_training: False}
                                if self.update_mode == 2:
                                    feed_dict[self.block_class_feature_means_pl] = self.block_class_feature_means
                                    feed_dict[self.block_inv_covariance_pl] = self.block_inv_covariances

                                # preds = self.preds.eval(feed_dict, sess)
                                # weights = np.sum(preds * batch[1], 1) * 3
                                mod_labels = self.modified_labels_op.eval(feed_dict, sess)
                                weights = np.sum(mod_labels * batch[1], 1) * 3

                                # new_loss_weights[batch[2]] = np.clip(weights, 0, 1)
                                new_loss_weights[batch[2]] = weights
                            current_loss_weights = new_loss_weights

                            def print_arr_stats(arr):
                                print('\tMean:', arr.mean())
                                print('\tStd:', arr.std())
                                print('\tMin:', arr.min())
                                print('\tMax:', arr.max())

                            print('New loss weights stats:')
                            print_arr_stats(current_loss_weights)
                            print('New loss weight stats for clean samples:')
                            print_arr_stats(current_loss_weights[np.nonzero(self.is_sample_clean)])
                            print('New loss weight stats for noised samples:')
                            print_arr_stats(current_loss_weights[np.nonzero(1 - self.is_sample_clean)])

                        Y_current = np.array(Y_current_new)
                        Y_current_ind = np.argmax(Y_current, 1)
                        Y_current_ind_per_class = [[] for c in range(self.n_classes)]
                        for i in range(self.current_dataset_size):
                            Y_current_ind_per_class[Y_current_ind[i]].append(i)
                        self.is_sample_clean = (Y_current_ind == Y0_cls_ind).astype(int)

                    alpha_value = 1
                    n_label_resets_done += 1

                    i_epoch_rel = 0
                    turning_rel_epoch = -1
                    lid_per_epoch = []
                    # print('resetting epoch number')

                #
                # SAVE MODEL
                #

                if i_epoch_rel == 0:
                    # saver0.restore(sess, model_path + 'start')
                    sess.run(tf.global_variables_initializer())
                    print('restarting from scratch')
                elif (self.update_mode == 1 or self.update_mode == 2) and turning_rel_epoch == i_epoch_tot - 1:
                    saver.restore(sess, model_path + str(i_epoch_tot - 1))
                    print('restoring model from previous epoch')
                else:
                    checkpoint_file = model_path + str(i_epoch_tot)
                    saver.save(sess, checkpoint_file)

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
    def _calc_lid(X, Y, lid_calc_op, x, is_training):
        lid_score = 0

        i_batch = -1
        for batch in batch_iterator(X, Y, LID_BATCH_SIZE, True):
            i_batch += 1

            if i_batch == LID_BATCH_CNT:
                break

            batch_lid_scores = lid_calc_op.eval(feed_dict={x: batch[0], is_training: False})
            if batch_lid_scores.min() < 0:
                print('negative lid!', list(batch_lid_scores))
                return None
                i_batch -= 1
                continue
            if batch_lid_scores.max() > 10000:
                print('lid is too big!', list(batch_lid_scores))
                return None
                i_batch -= 1
                continue

            lid_score += batch_lid_scores.mean()

        lid_score /= LID_BATCH_CNT

        return lid_score

    def _compute_LDA(self, sess, X, Y, Y_ind, Y_ind_per_class):
        print('Computing LDA parameters')

        features = np.empty((self.current_dataset_size, self.total_hidden_width))
        for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE, False):
            feed_dict = {self.nn_input_pl: batch[0], self.is_training: False}
            features[batch[2]] = self.feature_layer.eval(feed_dict=feed_dict)
        features = np.reshape(features, (-1, self.n_blocks, self.block_width))

        self.block_class_feature_means = np.empty((0, self.n_classes, self.block_width))
        self.block_inv_covariances = np.empty((0, self.block_width, self.block_width))

        for i_block in range(self.n_blocks):
            print('block', i_block + 1)
            print('__________________________________________')
            block_features = features[:, i_block, :]

            class_feature_means = None
            inv_covariance = None

            # initially select random subset of samples
            keep_arr = np.random.binomial(1, 0.5, self.current_dataset_size).astype(bool)
            for i in range(3 + 1):
                sess.run(self.reset_class_feature_sums_counts_and_covariance_op)

                # compute class feature means
                for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE * 8, False):
                    feed_dict = {self.features_pl: block_features[batch[2]],
                                 self.labels_pl: batch[1],
                                 self.class_feature_ops_keep_array_pl: keep_arr[batch[2]]}
                    sess.run(self.update_class_features_sum_and_counts_op, feed_dict=feed_dict)

                counts = self.class_feature_counts_var.eval(sess)
                sums = self.class_feature_sums_var.eval(sess)

                class_feature_means = sums / counts.reshape((-1, 1))

                # compute class feature covariance matrices
                for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE * 8, False):
                    feed_dict = {self.features_pl: block_features[batch[2]], self.labels_pl: batch[1],
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
                mahalanobis_distances = np.empty(self.current_dataset_size)
                for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE * 8, False):
                    feed_dict = {self.features_pl: block_features[batch[2]],
                                 self.class_feature_means_pl: class_feature_means,
                                 self.class_inv_covariances_pl: class_inv_covariances,
                                 self.labels_pl: batch[1]}
                    mah_dists = sess.run(self.mahalanobis_dist_calc, feed_dict=feed_dict)
                    mahalanobis_distances[batch[2]] = mah_dists

                mahalanobis_distances_per_class = [np.take(mahalanobis_distances, Y_ind_per_class[c]) for c in
                                                   range(self.n_classes)]

                # select less anomalous samples for the next subset
                medians = np.array([np.median(arr) for arr in mahalanobis_distances_per_class])
                print('Mahalanobis distance medians per class', [round(it ** 0.5, 1) for it in medians])

                keep_arr = mahalanobis_distances < np.take(medians, Y_ind)

                keep_arr_int = keep_arr.astype(int)
                kept_and_clean_rat = np.sum(keep_arr_int * self.is_sample_clean) / keep_arr_int.sum()
                print(round(kept_and_clean_rat * 100, 1), '% of kept samples are clean')

            self.block_class_feature_means = np.append(self.block_class_feature_means,
                                                       np.expand_dims(class_feature_means, 0), 0)
            self.block_inv_covariances = np.append(self.block_inv_covariances,
                                                   np.expand_dims(inv_covariance, 0), 0)

    def calc_accuracy_on_dataset(self, X, Y):
        accuracy = 0
        i_batch = -1
        cnt = 0
        for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE, False):
            i_batch += 1

            feed_dict = {self.nn_input_pl: batch[0], self.labels_pl: batch[1], self.is_training: False}

            partial_accuracy = self.accuracy.eval(feed_dict=feed_dict)

            batch_size = len(batch[0])
            accuracy = (cnt * accuracy + partial_accuracy * batch_size) / (cnt + batch_size)
            cnt += batch_size

        return accuracy
