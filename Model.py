import os
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import itertools

from networks import build_mnist, build_cifar_10, linear_layer
from cifar_100_resnet import build_cifar_100
from tf_ops import get_lid_calc_op, get_update_class_feature_selective_sum_and_counts_op, \
    get_update_class_covariances_selective_sum_op, get_LDA_logits_calc_op, get_Mahalanobis_distance_calc_op, \
    get_LDA_based_labels_calc_op, get_lid_per_element_calc_op
from consts import *
from Timer import timer
from preprocessing import read_dataset
from batch_iterate import batch_iterator, batch_iterator_with_indices
from tools import bitmask_contains, softmax
from noise_dataset import introduce_uniform_noise, introduce_noise
from dataset_metric_calculator import DatasetMetricCalculator
from confusion_matrix_plot import get_confusion_matrix_image


# noinspection PyAttributeOutsideInit
class Model:
    def __init__(self, dataset_name, model_name, update_mode, init_epochs, reg_coef, lr_segments=None,
                 log_mask=0, lid_use_pre_relu=False, lda_use_pre_relu=True,
                 n_blocks=1, block_width=256,
                 n_label_resets=0, cut_train_set=False, mod_labels_after_last_reset=True, use_loss_weights=False,
                 calc_lid_min_before_init_epoch=False, start_after_init_epoch=False,
                 train_separate_ll=False, separate_ll_class_count=2, separate_ll_count=10, separate_ll_fc_width=16,
                    separate_ll_lr=None, separate_ll_reg_coef=0):
        """

        :param dataset_name:         dataset name: mnist/cifar-10/cifar-100
        :param model_name:      name of the model
        :param update_mode:     0: vanilla
                                1: as in the LID paper
                                2: lda
                                3: my
                                4: paper with separate LL LIDs
                                5: class pairwise weights based on separate LL LIDs
        :param log_mask:    in case it contains
                                    0th bit: logs LID data from the paper
                                    1st bit: logs relu features per element
                                    2nd bit: logs pre relu features per element
                                    3rd bit: logs logits per element
                                    4rd bit: logs LID per class
                                    5th bit: logs LID class wise (choosing reference points from the same class)
                                    6th bit: logs additional accuracies/label similarities
                                    7th bit: logs accuracy confusion matrix
        """

        self.dataset_name = dataset_name
        self.update_mode = update_mode
        self.lid_use_pre_relu = lid_use_pre_relu
        self.lda_use_pre_relu = lda_use_pre_relu
        self.log_mask = log_mask
        self.model_name = model_name
        self.n_blocks = n_blocks
        self.n_epochs = sum(it[0] for it in lr_segments)
        self.lr_segments = lr_segments
        self.n_label_resets = n_label_resets
        self.cut_train_set = cut_train_set
        self.mod_labels_after_last_reset = mod_labels_after_last_reset
        self.use_loss_weights = use_loss_weights
        self.calc_lid_min_before_init_epoch = calc_lid_min_before_init_epoch
        self.start_after_init_epoch = start_after_init_epoch
        self.init_epochs = init_epochs
        self.reg_coef = reg_coef

        self.train_separate_ll = train_separate_ll
        self.separate_ll_class_count = separate_ll_class_count
        self.separate_ll_count = separate_ll_count
        self.separate_ll_fc_width = separate_ll_fc_width
        self.separate_ll_lr = separate_ll_lr
        self.separate_ll_reg_coef = separate_ll_reg_coef

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
            self.nn_input_pl = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
        elif self.dataset_name == 'cifar-10' or self.dataset_name == 'cifar-100':
            self.nn_input_pl = tf.placeholder(tf.float32, [None, 32, 32, 3], name='x')

        self.labels_pl = tf.placeholder(tf.float32, [None, self.n_classes], name='y_')
        self.labels_argmaxed = tf.argmax(self.labels_pl, -1)

        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        self.separate_lls_classes = None
        self.separate_ll_classes_inv_map = None
        if self.train_separate_ll:
            if self.separate_ll_count is not None:
                all_possible_class_groups = list(itertools.combinations(range(self.n_classes),
                                                                        self.separate_ll_class_count))
                if len(all_possible_class_groups) < self.separate_ll_count:
                    raise Exception("Not enough class combinations for separate linear layers")

                after_seed = np.random.randint(1 << 30)
                np.random.seed(0)
                self.separate_lls_classes = np.random.permutation(all_possible_class_groups)[:self.separate_ll_count]
                np.random.seed(after_seed)
            else:
                self.separate_lls_classes = [(i, (i + 1) % self.n_classes) for i in range(self.n_classes)]
                self.separate_ll_count = len(self.separate_lls_classes)

            self.separate_ll_classes_inv_map = []
            for i in range(self.separate_ll_count):
                inv_map = np.zeros(self.n_classes, np.int64)
                for j in range(self.separate_ll_class_count):
                    inv_map[self.separate_lls_classes[i][j]] = j
                self.separate_ll_classes_inv_map.append(inv_map)


        #
        # PREPARE DATA AUGMENTATION OPERATION
        #

        self.data_augmenter = None
        if self.dataset_name == 'cifar-10' or self.dataset_name == 'cifar-100':
            self.data_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)
        elif self.dataset_name == 'mnist':
            self.data_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True)

        #
        # BUILD NETWORK
        #

        reg_loss_unscaled = 0.0
        if self.dataset_name == 'mnist':
            self.separate_ll_input, self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.preds = build_mnist(
                self.nn_input_pl, self.is_training, self.n_blocks, self.block_width)
        elif self.dataset_name == 'cifar-10':
            self.separate_ll_input, self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.preds, reg_loss_unscaled =\
                build_cifar_10(self.nn_input_pl, self.is_training, self.n_blocks, self.block_width, 10)

        elif self.dataset_name == 'cifar-100':
            self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.preds, reg_loss_unscaled = build_cifar_100(
                            self.nn_input_pl,
                            self.is_training)
            self.separate_ll_input = self.lid_layer_op
            # conv_res, self.pre_lid_layer_op, self.lid_layer_op, self.logits, self.preds, reg_loss_unscaled = \
            #     build_cifar_10(self.nn_input_pl, self.is_training, self.n_blocks, self.block_width, 100)

        self.separate_lls_preds = None
        self.separate_lls_lid_calc_op = None
        self.separate_lls_logits = None
        self.separate_lls_is_training = None
        self.separate_lls_reg_sum = None
        if self.train_separate_ll:
            self.separate_lls_labels = []
            self.separate_lls_preds = []
            self.separate_lls_lid_calc_op = []
            self.separate_lls_logits = []
            self.separate_lls_is_training = []
            self.separate_lls_reg_sum = []

            self.separate_ll_input_pl = tf.placeholder(tf.float32, [None, self.separate_ll_input.shape[1]],
                                                       'separate_ll_input_pl')
            for i in range(self.separate_ll_count):
                is_training = tf.placeholder(dtype=tf.bool, name='is_training')
                self.separate_lls_is_training.append(is_training)
                batch_elements_belonging_to_selected_classes_indices = tf.reshape(
                    tf.where(
                        tf.reduce_any(
                            input_tensor=tf.equal(x=tf.expand_dims(self.labels_argmaxed, 1),
                                                  y=self.separate_lls_classes[i]),
                            axis=1)),
                    (-1, ))

                # separate_ll_input = tf.gather(tf.stop_gradient(self.separate_ll_input),
                #                               batch_elements_belonging_to_selected_classes_indices)

                self.separate_lls_labels.append(tf.one_hot(
                        tf.gather(
                            tf.gather(self.separate_ll_classes_inv_map[i],
                                      self.labels_argmaxed),
                            batch_elements_belonging_to_selected_classes_indices),
                        self.separate_ll_class_count))

                separate_ll_hidden_op, separate_lls_logits, separate_lls_reg_sum = linear_layer(
                    self.separate_ll_input_pl, is_training, self.separate_ll_fc_width, self.separate_ll_class_count)
                self.separate_lls_logits.append(separate_lls_logits)
                self.separate_lls_preds.append(tf.nn.softmax(separate_lls_logits))
                self.separate_lls_lid_calc_op.append(get_lid_calc_op(separate_ll_hidden_op))
                self.separate_lls_reg_sum.append(separate_lls_reg_sum)

        #
        # PREPARE FOR LABEL CHANGING
        #

        # LID calculation

        if self.update_mode == 1 or self.to_log(0):
            if self.lid_use_pre_relu:
                self.lid_calc_op = get_lid_calc_op(self.pre_lid_layer_op)
            else:
                self.lid_calc_op = get_lid_calc_op(self.lid_layer_op)

        if self.to_log(5):
            self.lid_per_element_calc_op_features_pl = tf.placeholder(tf.float32,
                                                                      (None, self.lid_layer_op.shape[-1]))
            self.lid_per_element_calc_op_ref_features_pl = tf.placeholder(tf.float32,
                                                                          (None, self.lid_layer_op.shape[-1]))
            self.lid_per_element_calc_op = get_lid_per_element_calc_op(self.lid_per_element_calc_op_features_pl,
                                                                       self.lid_per_element_calc_op_ref_features_pl)

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
            if self.update_mode == 1 or self.update_mode == 3 or self.update_mode == 4 or self.to_log(0):
                self.alpha_var = tf.Variable(1, False, dtype=tf.float32, name='alpha')

                if self.update_mode == 1 or self.update_mode == 3 or self.update_mode == 4:
                    self.new_labels_op = tf.one_hot(tf.argmax(self.preds, 1), self.n_classes)
                    self.modified_labels_op = tf.identity(
                        self.alpha_var * self.labels_pl + (1 - self.alpha_var) * self.new_labels_op,
                        name='modified_labels')
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=tf.stop_gradient(self.modified_labels_op),
                        logits=self.logits)

            if self.update_mode == 5:
                self.weights_matrix_pl = tf.placeholder(tf.float32, (self.n_classes, self.n_classes), 'weights_matrix')

                labels_argmaxed = tf.arg_max(self.labels_pl, -1)
                preds_argmaxed = tf.argmax(self.preds, -1)

                gather_indices = tf.concat((tf.expand_dims(labels_argmaxed, -1),
                                            tf.expand_dims(preds_argmaxed, -1)), -1)

                weights_gathered = tf.expand_dims(tf.gather_nd(self.weights_matrix_pl, gather_indices), -1)

                self.new_labels_op = tf.one_hot(preds_argmaxed, self.n_classes)
                self.modified_labels_op = tf.identity(
                    weights_gathered * self.labels_pl + (1 - weights_gathered) * self.new_labels_op,
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

            self.computed_labels_similarity_reference_pl = tf.placeholder(tf.float32, (None, self.n_classes))
            self.modified_labels_similarity = tf.reduce_sum(
                self.modified_labels_op * self.computed_labels_similarity_reference_pl, -1)
            self.new_labels_similarity = tf.reduce_sum(
                self.new_labels_op * self.computed_labels_similarity_reference_pl, -1)

            self.reg_loss = tf.identity(2 * self.reg_coef * reg_loss_unscaled, 'reg_loss')

            self.separate_lls_loss = None
            if self.train_separate_ll:
                self.separate_lls_loss = []
                for i in range(self.separate_ll_count):
                    loss = self.separate_lls_reg_sum[i] * self.separate_ll_reg_coef + \
                           tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=self.separate_lls_labels[i], logits=self.separate_lls_logits[i])
                    self.separate_lls_loss.append(loss)

        self.cross_entropy = tf.reduce_mean(cross_entropy * self.loss_weights_pl)

        #
        # CREATE ACCURACY
        #

        with tf.name_scope('accuracy'):
            self.batch_accuracies = tf.cast(tf.equal(tf.argmax(self.preds, 1), self.labels_argmaxed), tf.float32)
            self.accuracy = tf.reduce_mean(self.batch_accuracies, name='accuracy')

            self.separate_lls_accuracy = None
            if self.train_separate_ll:
                self.separate_lls_accuracy = []
                for i in range(self.separate_ll_count):
                    self.separate_lls_accuracy.append(tf.cast(
                            tf.equal(tf.argmax(self.separate_lls_preds[i], 1),
                                     tf.argmax(self.separate_lls_labels[i], 1)),
                            tf.float32))

        with tf.name_scope('modified_accuracy'):
            mod_label = self.labels_pl
            if self.update_mode != 0:
                mod_label = self.modified_labels_op

            acc = tf.reduce_sum(tf.one_hot(tf.argmax(self.preds, 1), self.n_classes) * mod_label, 1)
            self.modified_labels_accuracy = tf.reduce_mean(acc, name='accuracy')

        self.rel_epoch_pl = tf.placeholder(tf.int32)

        with tf.name_scope('learning_rate'):
            def produce_lr_tensor(segment_start=0, segment_i=0):
                if segment_i == len(self.lr_segments) - 1:
                    return self.lr_segments[-1][1]
                else:
                    segment_end = segment_start + self.lr_segments[segment_i][0]
                    return tf.cond(pred=self.rel_epoch_pl < segment_end,
                                   true_fn=lambda: self.lr_segments[segment_i][1],
                                   false_fn=lambda: produce_lr_tensor(segment_end + 1, segment_i + 1))

            self.lr = produce_lr_tensor()

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy + self.reg_loss)

            self.separate_lls_train_step = None
            if self.train_separate_ll:
                self.separate_lls_train_step = []
                for i in range(self.separate_ll_count):
                    self.separate_lls_train_step.append(
                        tf.train.AdamOptimizer(self.lr).minimize(self.separate_lls_loss[i]))

    def train(self, train_dataset_type='train_without_val', noise_ratio=0, noise_seed=0, noise_matrix=None):
        #
        # LOAD
        #

        X, Y0 = read_dataset(name=self.dataset_name, type=train_dataset_type)
        Y = np.copy(Y0)
        if noise_matrix is None:
            introduce_uniform_noise(Y, noise_ratio, noise_seed)
            noise_matrix = np.eye(self.n_classes, self.n_classes) * (1 - noise_ratio)
            noise_matrix += np.ones((self.n_classes, self.n_classes)) * noise_ratio / (self.n_classes - 1)
            noise_matrix -= np.eye(self.n_classes, self.n_classes) * noise_ratio / (self.n_classes - 1)
        else:
            introduce_noise(Y, noise_matrix, noise_seed)

        Y_ind = np.argmax(Y, 1)
        Y0_cls_ind = np.argmax(Y0, 1)
        self.is_sample_clean = (Y_ind == Y0_cls_ind).astype(int)

        X_current = np.array(X)
        self.current_dataset_size = X.shape[0]
        X_current_ind = np.arange(self.current_dataset_size)

        Y_current = np.array(Y)
        Y_current_cls_ind = np.array(Y_ind)
        Y_current_ind_per_class = [[] for c in range(self.n_classes)]
        for i in range(X.shape[0]):
            Y_current_ind_per_class[Y_ind[i]].append(i)

        Y0_current = np.array(Y0)

        self.clean_indices = np.nonzero(self.is_sample_clean)[0]
        self.noised_indices = np.nonzero(1 - self.is_sample_clean)[0]

        self.X_current_clean = X_current[self.clean_indices]
        self.Y_current_clean = Y_current[self.clean_indices]
        self.X_current_noised = X_current[self.noised_indices]
        self.Y_current_noised = Y_current[self.noised_indices]
        self.Y0_current_noised = Y0_current[self.noised_indices]

        X_test, Y_test = read_dataset(name=self.dataset_name, type='test')
        X_validation, Y_validation = read_dataset(name=self.dataset_name, type='validation')

        Y_validation_ind = np.argmax(Y_validation, 1)
        Y_validation_ind_per_class = [[] for c in range(self.n_classes)]
        for i in range(X_validation.shape[0]):
            Y_validation_ind_per_class[Y_validation_ind[i]].append(i)

        current_loss_weights = np.full((self.current_dataset_size,), 1, np.float32)

        self._build()

        #
        # CREATE SUMMARIES
        #

        tf.summary.scalar(name='cross_entropy', tensor=self.cross_entropy)
        tf.summary.scalar(name='train_accuracy', tensor=self.accuracy)
        tf.summary.scalar(name='reg_loss', tensor=self.reg_loss)
        tf.summary.scalar(name='modified_train_accuracy', tensor=self.modified_labels_accuracy)
        tf.summary.scalar(name='learning_rate', tensor=self.lr)
        train_summaries = tf.summary.merge_all()

        self.pairwise_LID_confusion_matrix_summary_pl = None
        self.pairwise_LID_confusion_matrix_summary = None
        if self.update_mode == 5:
            self.pairwise_LID_confusion_matrix_summary_pl = tf.placeholder(tf.uint8, (1, None, None, 3))
            self.pairwise_LID_confusion_matrix_summary = tf.summary.image(
                'pairwise_LID_cm', self.pairwise_LID_confusion_matrix_summary_pl)

            self.pairwise_weights_summary_pl = tf.placeholder(tf.uint8, (1, None, None, 3))
            self.pairwise_weights_summary = tf.summary.image(
                'pairwise_weights', self.pairwise_weights_summary_pl)

        if self.to_log(7):
            self.validation_accuracy_LID_confusion_matrix_summary_pl = tf.placeholder(tf.uint8, (1, None, None, 3))
            self.validation_accuracy_LID_confusion_matrix_summary = tf.summary.image(
                'validation_accuracy_cm', self.validation_accuracy_LID_confusion_matrix_summary_pl)

        if noise_matrix is not None:
            self.noise_confusion_matrix_summary_pl = tf.placeholder(tf.uint8, (1, None, None, 3))
            self.noise_confusion_matrix_summary = tf.summary.image(
                'noise_cm', self.noise_confusion_matrix_summary_pl)

        saver = tf.train.Saver(max_to_keep=1)
        # saver0 = tf.train.Saver(max_to_keep=1000)
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
        with tf.Session(config=config) as self.sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='readline')   # run -t n_epochs
            # sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)       # run -f has_inf_or_nan
            # tf.logging.set_verbosity(tf.logging.ERROR)

            summary_writer = tf.summary.FileWriter(model_path, self.sess.graph)

            noise_matrix_image = get_confusion_matrix_image(noise_matrix, self.n_classes, False)
            noise_matrix_summary_str = self.sess.run(
                self.noise_confusion_matrix_summary,
                {self.noise_confusion_matrix_summary_pl: noise_matrix_image[np.newaxis, :, :, :]})
            summary_writer.add_summary(noise_matrix_summary_str, 0)

            self.sess.run(tf.global_variables_initializer())
            # saver0.save(self.sess, model_path + 'start')

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

            lid_per_epoch = []
            separate_ll_lid_per_epoch = []
            separate_ll_lid_per_epoch_per_class_pair = []

            weight_matrix = np.ones((self.n_classes, self.n_classes), np.float32)
            turning_rel_epoch_matrix = -np.ones((self.n_classes, self.n_classes), np.int32)
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
                    self.compute_LDA(X_augmented, Y_current, Y_current_cls_ind, Y_current_ind_per_class)

                #
                # TRAIN
                #

                if self.update_mode == 1 or self.update_mode == 2 or self.update_mode == 3 or self.update_mode == 4:
                    self.sess.run(self.alpha_var.assign(alpha_value))

                print('starting training...')

                for batch in batch_iterator_with_indices(X_augmented, Y_current, BATCH_SIZE):
                    i_step += 1

                    feed_dict = {self.nn_input_pl: batch[0], self.labels_pl: batch[1], self.is_training: True,
                                 self.rel_epoch_pl: i_epoch_rel,
                                 self.loss_weights_pl: current_loss_weights[batch[2]]}

                    if self.update_mode == 2:
                        feed_dict[self.LDA_labels_weight_pl] = self.alpha_var.eval(self.sess)
                        feed_dict[self.block_class_feature_means_pl] = self.block_class_feature_means
                        feed_dict[self.block_inv_covariance_pl] = self.block_inv_covariances

                    if self.update_mode == 5:
                        feed_dict[self.weights_matrix_pl] = weight_matrix

                    self.train_step.run(feed_dict=feed_dict)

                    if i_step % 100 == 0:
                        feed_dict[self.is_training] = False
                        summary_str, train_accuracy = self.sess.run([train_summaries, self.accuracy],
                                                                    feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i_step)
                        summary_writer.flush()

                        print('\tstep %d, training accuracy %g' % (i_step, train_accuracy))

                #
                # TRAIN SEPARATE LINEAR LAYERS
                #

                if self.train_separate_ll:
                    print('\nPrecomputing separate linear layer inputs...')

                    separate_ll_inputs_current = np.empty((X_current.shape[0], self.separate_ll_input.shape[1]), np.float32)
                    for batch in batch_iterator_with_indices(X_current, Y_current, BATCH_SIZE):
                        feed_dict = {self.nn_input_pl: batch[0], self.is_training: False}
                        separate_ll_inputs_current[batch[2]] = self.separate_ll_input.eval(feed_dict=feed_dict)

                    # separate_ll_inputs_augmented = np.empty((X_augmented.shape[0], self.separate_ll_input.shape[1]),
                    #                                         np.float32)
                    # for batch in batch_iterator_with_indices(X_augmented, Y_current, BATCH_SIZE):
                    #     feed_dict = {self.nn_input_pl: batch[0], self.is_training: False}
                    #     separate_ll_inputs_augmented[batch[2]] = self.separate_ll_input.eval(feed_dict=feed_dict)

                    separate_ll_inputs_validation = np.empty((X_validation.shape[0], self.separate_ll_input.shape[1]),
                                                             np.float32)
                    for batch in batch_iterator_with_indices(X_validation, Y_validation, BATCH_SIZE):
                        feed_dict = {self.nn_input_pl: batch[0], self.is_training: False}
                        separate_ll_inputs_validation[batch[2]] = self.separate_ll_input.eval(feed_dict=feed_dict)

                    print('\nTraining separate linear layers...')
                    feed_dict = {self.rel_epoch_pl: i_epoch_rel}
                    if self.update_mode == 5:
                        feed_dict[self.weights_matrix_pl] = weight_matrix
                    self.train_separate_lls(separate_ll_inputs_current, Y_current, Y_current_ind_per_class,
                                            current_loss_weights, feed_dict)

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

                for batch in batch_iterator_with_indices(X_current, Y_current, BATCH_SIZE, False):
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
                    np.save(file='logs/lid_features/' + self.dataset_name + '/' + self.model_name,
                            arr=lid_features_per_epoch_per_element)

                if self.to_log(2):
                    pre_lid_features_per_epoch_per_element = np.append(
                        pre_lid_features_per_epoch_per_element,
                        np.expand_dims(pre_lid_features_per_element, 0),
                        0)
                    np.save(file='logs/pre_lid_features/' + self.dataset_name + '/' + self.model_name,
                            arr=pre_lid_features_per_epoch_per_element)

                if self.to_log(3):
                    logits_per_epoch_per_element = np.append(
                        logits_per_epoch_per_element,
                        np.expand_dims(logits_per_element, 0),
                        0)
                    np.save(file='logs/logits/' + self.dataset_name + '/' + self.model_name,
                            arr=logits_per_epoch_per_element)

                #
                # CALCULATE LID
                #

                if self.update_mode == 1 or self.update_mode == 2 or self.to_log(0) or self.to_log(4):
                    # TODO: would augmented data change LIDs?
                    lid_calculator = self.calc_lid(self.lid_calc_op, batch_iterator(X_current, Y_current,
                                                                                    LID_BATCH_SIZE, True))
                    new_lid_score = lid_calculator.mean
                    lid_per_epoch.append(new_lid_score)

                    print('\nRegular LID score after %dth epoch: %g' % (i_epoch_tot, new_lid_score,))

                    if self.to_log(4):
                        folder_to_save = os.path.join('logs/LID/per_class', self.dataset_name, self.model_name)
                        np.save(os.path.join(folder_to_save, str(i_epoch_tot)), lid_calculator.mean_per_class)

                if self.train_separate_ll:
                    print('\nComputing LIDs for separate linear layers...')
                    separate_lls_combined_lid_calculator, separate_lls_lid_calculators = \
                        self.calc_separate_lls_lid(separate_ll_inputs_current, Y_current, Y_current_ind_per_class)

                    new_lid_score = separate_lls_combined_lid_calculator.mean
                    separate_ll_lid_per_epoch.append(new_lid_score)

                    lid_per_pair = []
                    for separate_lls_lid_calculator in separate_lls_lid_calculators:
                        lid_per_pair.append(separate_lls_lid_calculator.mean)
                    separate_ll_lid_per_epoch_per_class_pair.append(lid_per_pair)

                    print('\nSeparate linear layer LID score after %dth epoch: %g' % (i_epoch_tot, new_lid_score,))

                #
                # WRITE PER EPOCH SUMMARIES
                #

                summary = tf.Summary()
                self.add_simple_value_summary(summary, 'train_set_size', len(X_current))
                self.add_simple_value_summary(summary, 'clean_train_set_size', len(self.clean_indices))
                print()

                #
                # 1. LID related stuff
                #

                if self.to_log(0):
                    self.add_simple_value_summary(summary, 'LID', lid_per_epoch[-1])
                    self.add_simple_value_summary(summary, 'alpha', alpha_value)
                    if self.train_separate_ll:
                        self.add_simple_value_summary(summary, 'separate lls mean LID',
                                                      separate_lls_combined_lid_calculator.mean)
                        self.add_simple_value_summary(summary, 'separate lls median LID',
                                                      separate_lls_combined_lid_calculator.median)
                        self.add_simple_value_summary(summary, 'separate lls median mean LID',
                                                      np.mean([it.median for it in separate_lls_lid_calculators]))

                if self.update_mode == 5:
                    confusion_matrix = np.zeros((self.n_classes, self.n_classes))
                    for i in range(len(self.separate_lls_classes)):
                        c1, c2 = self.separate_lls_classes[i]

                        lid_value = separate_ll_lid_per_epoch_per_class_pair[-1][i]
                        confusion_matrix[c1, c2] = lid_value
                        confusion_matrix[c2, c1] = lid_value

                        c1_ = min(c1, c2) + 1
                        c2 = max(c1, c2) + 1
                        c1 = c1_

                        c1_formatted = str(c1) if c1 > 9 else '0' + str(c1)
                        c2_formatted = str(c2) if c2 > 9 else '0' + str(c2)
                        self.add_simple_value_summary(
                            summary,
                            'separate ll mean LID/{} {}'.format(c1_formatted, c2_formatted),
                            lid_value)

                    confusion_matrix_image = get_confusion_matrix_image(confusion_matrix, self.n_classes, False)
                    confusion_matrix_summary_str = self.sess.run(
                        self.pairwise_LID_confusion_matrix_summary,
                        {self.pairwise_LID_confusion_matrix_summary_pl: confusion_matrix_image[np.newaxis, :, :, :]})
                    summary_writer.add_summary(confusion_matrix_summary_str, i_step)

                    weights_cm_matrix_image = get_confusion_matrix_image(weight_matrix, self.n_classes, False)
                    weights_cm_summary_str = self.sess.run(
                        self.pairwise_weights_summary,
                        {self.pairwise_weights_summary_pl: weights_cm_matrix_image[np.newaxis, :, :, :]}
                    )
                    summary_writer.add_summary(weights_cm_summary_str, i_step)

                    summary_writer.flush()

                #
                # 2. train/test/validation accuracy
                #

                test_accuracy_calculator = self.calc_metric_on_dataset(X_test, Y_test, self.batch_accuracies)
                test_accuracy = test_accuracy_calculator.mean
                self.add_simple_value_summary(summary, 'test_accuracy', test_accuracy)
                print('test accuracy:', test_accuracy)

                validation_accuracy_calculator = self.calc_metric_on_dataset(X_validation, Y_validation,
                                                                             self.batch_accuracies)
                validation_accuracy = validation_accuracy_calculator.mean
                self.add_simple_value_summary(summary, 'validation_accuracy', validation_accuracy)
                print('validation accuracy:', validation_accuracy)

                if self.to_log(7):
                    validation_accuracy_cm = self.calc_confusion_matrix(
                        batch_iterator_with_indices(X_validation, Y_validation, BATCH_SIZE),
                        self.preds)
                    validation_accuracy_cm_image = get_confusion_matrix_image(
                        validation_accuracy_cm, self.n_classes, True
                    )
                    summary_str = self.sess.run(
                        self.validation_accuracy_LID_confusion_matrix_summary,
                        {self.validation_accuracy_LID_confusion_matrix_summary_pl:
                             validation_accuracy_cm_image[np.newaxis, :, :, :]})
                    summary_writer.add_summary(summary_str, i_step)

                if self.to_log(6):
                    clean_train_accuracy_calculator = self.calc_metric_on_dataset(X_current, Y0_current,
                                                                                  self.batch_accuracies)
                    clean_train_accuracy = clean_train_accuracy_calculator.mean
                    self.add_simple_value_summary(summary, 'clean_train_accuracy', clean_train_accuracy)
                    print('clean train accuracy:', clean_train_accuracy)

                    train_accuracy_on_clean_calculator = self.calc_metric_on_dataset(self.X_current_clean,
                                                                                     self.Y_current_clean,
                                                                                     self.batch_accuracies)
                    train_accuracy_on_clean = train_accuracy_on_clean_calculator.mean
                    self.add_simple_value_summary(summary, 'train_accuracy_on_clean', train_accuracy_on_clean)
                    print('train accuracy on clean samples:', train_accuracy_on_clean)

                    if len(self.X_current_noised) > 0:
                        clean_train_accuracy_on_noised_calculator = self.calc_metric_on_dataset(self.X_current_noised,
                                                                                                self.Y0_current_noised,
                                                                                                self.batch_accuracies)
                        clean_train_accuracy_on_noised = clean_train_accuracy_on_noised_calculator.mean
                        self.add_simple_value_summary(summary, 'clean_train_accuracy_on_noised',
                                                      clean_train_accuracy_on_noised)
                        print('clean train accuracy on noised samples:', clean_train_accuracy_on_noised)

                        noised_train_accuracy_on_noised_calculator = self.calc_metric_on_dataset(self.X_current_noised,
                                                                                                 self.Y_current_noised,
                                                                                                 self.batch_accuracies)
                        noised_train_accuracy_on_noised = noised_train_accuracy_on_noised_calculator.mean
                        self.add_simple_value_summary(summary, 'noised_train_accuracy_on_noised',
                                                      noised_train_accuracy_on_noised)
                        print('noised train accuracy on noised samples:', noised_train_accuracy_on_noised)

                    print()

                #
                # 3. Separate LL layer accuracies
                #

                if self.train_separate_ll:
                    print()

                    separate_ll_train_accuracy_calculator = self.calc_separate_ll_metric_on_dataset(
                        separate_ll_inputs_current, Y_current, Y_current_ind_per_class, self.separate_lls_accuracy,
                        False, {self.weights_matrix_pl: weight_matrix} if self.update_mode == 5 else None)
                    separate_ll_train_accuracy = separate_ll_train_accuracy_calculator.mean
                    self.add_simple_value_summary(summary, 'separate_ll_train_accuracy', separate_ll_train_accuracy)
                    print('Separate linear layer train accuracy:', separate_ll_train_accuracy)

                    separate_ll_validation_accuracy_calculator = self.calc_separate_ll_metric_on_dataset(
                        separate_ll_inputs_validation, Y_validation, Y_validation_ind_per_class,
                        self.separate_lls_accuracy, False, {self.weights_matrix_pl: weight_matrix} if self.update_mode == 5 else None)
                    separate_ll_validation_accuracy = separate_ll_validation_accuracy_calculator.mean
                    self.add_simple_value_summary(summary, 'separate_ll_validation_accuracy',
                                                  separate_ll_validation_accuracy)
                    print('Separate linear layer validation accuracy:', separate_ll_validation_accuracy)

                #
                # 4. Modified/new labels accuracies(similarities)
                #

                if self.to_log(6):
                    base_feed_dict = {}

                    if self.update_mode == 2:
                        base_feed_dict[self.LDA_labels_weight_pl] = self.alpha_var.eval(self.sess)
                        base_feed_dict[self.block_class_feature_means_pl] = self.block_class_feature_means
                        base_feed_dict[self.block_inv_covariance_pl] = self.block_inv_covariances

                    if self.update_mode == 5:
                        base_feed_dict[self.weights_matrix_pl] = weight_matrix

                    metric_calculators = self.calc_metric_on_dataset(
                        X_current, Y_current, keep_values=False, base_feed_dict=base_feed_dict,
                        metric_ops=[
                            self.modified_labels_similarity,
                            self.new_labels_similarity,
                        ],
                        Y_ref=Y0_current)
                    modified_labels_similarity_calculator, new_labels_similarity_calculator = metric_calculators

                    modified_labels_accuracy = modified_labels_similarity_calculator.mean
                    self.add_simple_value_summary(summary, 'modified_labels_accuracy', modified_labels_accuracy)
                    print('accuracy of modified labels compared to clean labels on a train set:',
                          modified_labels_accuracy)

                    new_labels_accuracy = new_labels_similarity_calculator.mean
                    self.add_simple_value_summary(summary, 'new_labels_accuracy', new_labels_accuracy)
                    print('accuracy of new labels compared to clean labels on a train set:', new_labels_accuracy)

                    if len(self.noised_indices) > 0:
                        new_labels_similarity_with_noise_calculator = self.calc_metric_on_dataset(
                            X_current, Y_current, self.new_labels_similarity, keep_values=False,
                            base_feed_dict=base_feed_dict, Y_ref=Y_current)
                        new_labels_accuracy_with_noise = new_labels_similarity_with_noise_calculator.mean
                    else:
                        new_labels_accuracy_with_noise = new_labels_accuracy
                    self.add_simple_value_summary(summary, 'new_labels_accuracy_with_noise',
                                                  new_labels_accuracy_with_noise)
                    print('accuracy of new labels compared to noised labels on a train set:',
                          new_labels_accuracy_with_noise)

                    if len(self.noised_indices) > 0:
                        new_labels_similarity_on_clean_only_calculator = self.calc_metric_on_dataset(
                            self.X_current_clean, self.Y_current_clean, self.new_labels_similarity, keep_values=False,
                            base_feed_dict=base_feed_dict, Y_ref=self.Y_current_clean)
                        new_labels_accuracy_on_clean_only = new_labels_similarity_on_clean_only_calculator.mean
                    else:
                        new_labels_accuracy_on_clean_only = new_labels_accuracy
                    self.add_simple_value_summary(summary, 'new_labels_accuracy_on_clean_only',
                                                  new_labels_accuracy_on_clean_only)
                    print('accuracy of new labels on clean samples only on a train set:',
                          new_labels_accuracy_on_clean_only)

                    if len(self.noised_indices) > 0:
                        new_labels_similarity_on_noised_only_calculator = self.calc_metric_on_dataset(
                            self.X_current_noised, self.Y_current_noised, self.new_labels_similarity, keep_values=False,
                            base_feed_dict=base_feed_dict, Y_ref=Y0_current[self.noised_indices])
                        new_labels_accuracy_on_noised_only = new_labels_similarity_on_noised_only_calculator.mean
                    else:
                        new_labels_accuracy_on_noised_only = 0
                    self.add_simple_value_summary(summary, 'new_labels_accuracy_on_noised_only',
                                                  new_labels_accuracy_on_noised_only)
                    print('accuracy of new labels on noised samples only on a train set:',
                          new_labels_accuracy_on_noised_only)

                if self.to_log(5):
                    class_wise_lid_calculator = self.calc_lid_class_wise(X_current, Y_current,
                                                                         Y_current_ind_per_class)

                    folder_to_save = os.path.join('logs/LID/class_wise', self.dataset_name, self.model_name)
                    np.save(os.path.join(folder_to_save, str(i_epoch_tot)), class_wise_lid_calculator.mean_per_class)

                summary_writer.add_summary(summary, i_step + 1)
                summary_writer.flush()

                if self.update_mode == 1 or self.update_mode == 2 or self.update_mode == 4:
                    #
                    # CHECK FOR STOPPING INIT PERIOD
                    #

                    if self.start_after_init_epoch and i_epoch_rel == self.init_epochs + 1:
                        turning_rel_epoch = i_epoch_rel - 1

                    lid_values = separate_ll_lid_per_epoch if self.update_mode == 4 else lid_per_epoch

                    if turning_rel_epoch == -1 and len(lid_values) > self.init_epochs + EPOCH_WINDOW:
                        if self.mod_labels_after_last_reset or not(0 < self.n_label_resets == n_label_resets_done):
                            last_w_lids = lid_values[-EPOCH_WINDOW - 1: -1]

                            lid_check_value = lid_values[-1] - np.mean(last_w_lids) - 2 * np.std(last_w_lids)

                            print('LID check:', lid_check_value)

                            if lid_check_value > 0:
                                turning_rel_epoch = i_epoch_rel - 1
                                print('Turning point passed, starting using modified labels')

                    #
                    # Modify alpha
                    #

                    if turning_rel_epoch != -1:
                        min_value = np.min(lid_values[:-1]) if self.calc_lid_min_before_init_epoch else \
                            np.min(lid_values[max(0, self.init_epochs - 1):-1])
                        alpha_value = np.exp(-(i_epoch_rel / self.n_epochs) * (lid_values[-1] / min_value))
                        print('\nnext alpha value:', alpha_value)

                #
                # SEPARATE LINEAR LAYER LID BASED METHOD
                #
                if self.update_mode == 3:
                    if turning_rel_epoch == -1 and len(separate_ll_lid_per_epoch) > self.init_epochs + EPOCH_WINDOW:
                        if self.mod_labels_after_last_reset or not(0 < self.n_label_resets == n_label_resets_done):
                            last_w_lids = separate_ll_lid_per_epoch[-EPOCH_WINDOW - 1: -1]

                            last_lid = separate_ll_lid_per_epoch[-1]
                            stable_lid_check_value = abs(last_lid - np.mean(last_w_lids)) - 0.5 * np.std(last_w_lids)
                            growing_lid_check_value = last_lid - np.mean(last_w_lids) - 2 * np.std(last_w_lids)

                            print('Stabilization LID check:', stable_lid_check_value)
                            print('Growth LID check:', growing_lid_check_value)

                            if growing_lid_check_value > 0 or stable_lid_check_value < 0:
                            # if stable_lid_check_value < 0:
                                turning_rel_epoch = i_epoch_rel - 1
                                print('Turning point passed, starting using modified labels with reference lid value '
                                      'equal to', separate_ll_lid_per_epoch[turning_rel_epoch])

                    #
                    # Modify alpha
                    #

                    if turning_rel_epoch != -1:
                        reference_lid_value = np.min(separate_ll_lid_per_epoch[turning_rel_epoch:])
                        # alpha_value = np.exp(-(i_epoch_rel / self.n_epochs) *
                        #                      separate_ll_lid_per_epoch[-1] / reference_lid_value)
                        # alpha_value = min(1, np.exp(-separate_ll_lid_per_epoch[-1] / reference_lid_value))
                        # beta = 1 / (1 - turning_rel_epoch / self.n_epochs)
                        ratio = reference_lid_value / separate_ll_lid_per_epoch[-1]
                        alpha_value = np.clip(1 - (1 - ratio) - turning_rel_epoch / self.n_epochs, 0.2, 1)
                        print('\nnext alpha value:', alpha_value)

                if self.update_mode == 5:
                    mean_lid_values = [np.mean(it) for it in separate_ll_lid_per_epoch_per_class_pair]

                    for i in range(len(self.separate_lls_classes)):
                        c1, c2 = self.separate_lls_classes[i]

                        pair_lid_values = [it[i] for it in separate_ll_lid_per_epoch_per_class_pair]

                        if turning_rel_epoch_matrix[c1, c2] == -1 and \
                            len(pair_lid_values) > self.init_epochs + EPOCH_WINDOW and \
                                (self.mod_labels_after_last_reset or
                                 not(0 < self.n_label_resets == n_label_resets_done)):

                            last_w_lids = pair_lid_values[-EPOCH_WINDOW - 1: -1]
                            # last_w_lids = mean_lid_values[-EPOCH_WINDOW - 1: -1]

                            growing_lid_check_value = pair_lid_values[-1] - np.mean(last_w_lids) - \
                                                      2 * np.std(last_w_lids)

                            if growing_lid_check_value > 0:
                                turning_rel_epoch_matrix[c1, c2] = i_epoch_rel - 1
                                turning_rel_epoch_matrix[c2, c1] = i_epoch_rel - 1

                                print('Turning epoch passed for {}/{}'.format(c1, c2), 'class pair')

                        if turning_rel_epoch_matrix[c1, c2] != -1:
                            min_value = np.min(mean_lid_values[:-1]) if self.calc_lid_min_before_init_epoch else \
                                np.min(mean_lid_values[max(0, self.init_epochs - 1):-1])
                            weight = np.exp(-(i_epoch_rel / self.n_epochs) * (pair_lid_values[-1] / min_value))
                            weight = max(0.2, weight)
                            weight_matrix[c1, c2] = weight
                            weight_matrix[c2, c1] = weight

                if i_epoch_rel == self.n_epochs and n_label_resets_done < self.n_label_resets:
                    print('Reached the end, resetting current labels')

                    Y_current_new = np.empty((self.current_dataset_size, self.n_classes), np.float32)
                    for batch in batch_iterator_with_indices(X_current, Y_current, BATCH_SIZE, False):
                        feed_dict = {self.nn_input_pl: batch[0], self.is_training: False}
                        if self.update_mode == 2:
                            feed_dict[self.block_class_feature_means_pl] = self.block_class_feature_means
                            feed_dict[self.block_inv_covariance_pl] = self.block_inv_covariances
                        if self.update_mode == 0:
                            feed_dict[self.labels_pl] = batch[1]
                        if self.update_mode == 5:
                            feed_dict[self.weights_matrix_pl] = weight_matrix
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
                                mod_labels = self.modified_labels_op.eval(feed_dict, self.sess)
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

                    self.clean_indices = np.nonzero(self.is_sample_clean)[0]
                    self.noised_indices = np.nonzero(1 - self.is_sample_clean)[0]

                    self.X_current_clean = X_current[self.clean_indices]
                    self.Y_current_clean = Y_current[self.clean_indices]
                    self.X_current_noised = X_current[self.noised_indices]
                    self.Y_current_noised = Y_current[self.noised_indices]
                    self.Y0_current_noised = Y0_current[self.noised_indices]

                    alpha_value = 1
                    n_label_resets_done += 1

                    i_epoch_rel = 0
                    turning_rel_epoch = -1
                    turning_rel_epoch_matrix = -np.ones((self.n_classes, self.n_classes), np.int32)
                    lid_per_epoch = []
                    separate_ll_lid_per_epoch = []
                    separate_ll_lid_per_epoch_per_class_pair = []
                    # print('resetting epoch number')

                #
                # SAVE MODEL
                #

                if i_epoch_rel == 0:
                    # saver0.restore(sess, model_path + 'start')
                    self.sess.run(tf.global_variables_initializer())
                    print('restarting from scratch')
                elif (self.update_mode != 1 or self.update_mode == 2 or self.update_mode == 3 or self.update_mode == 4)\
                        and turning_rel_epoch == i_epoch_tot - 1:
                    saver.restore(self.sess, model_path + str(i_epoch_tot - 1))
                    print('restoring model from previous epoch')
                else:
                    checkpoint_file = model_path + str(i_epoch_tot)
                    saver.save(self.sess, checkpoint_file)

                print(timer.stop())

    def add_simple_value_summary(self, summary, tag, simple_value):
        summary.value.add(tag=tag, simple_value=simple_value)

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
                np.save('logs/pre_lid_features/' + dataset_name + '/' + model_name + '_' + dataset_type,
                        np.expand_dims(features, 0))
            else:
                np.save('logs/lid_features/' + dataset_name + '/' + model_name + '_' + dataset_type,
                        np.expand_dims(features, 0))

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

    def calc_lid(self, lid_calc_op, batch_iter):
        lid_calculator = DatasetMetricCalculator(class_count=self.n_classes)

        i_batch = -1
        for batch in batch_iter:
            i_batch += 1

            if i_batch == LID_BATCH_CNT:
                break

            feed_dict = {self.nn_input_pl: batch[0], self.labels_pl: batch[1],
                         self.is_training: False}

            batch_lid_scores = lid_calc_op.eval(feed_dict)
            if batch_lid_scores.min() < 0:
                print('negative lid!', list(batch_lid_scores))
                break
            if batch_lid_scores.max() > 10000:
                print('lid is too big!', list(batch_lid_scores))
                break

            lid_calculator.add_batch_values_with_labels(batch_lid_scores, batch[1])

        return lid_calculator

    def calc_separate_lls_lid(self, separate_ll_inputs, Y, Y_ind_per_class):
        combined_lid_calculator = DatasetMetricCalculator(class_count=self.n_classes)
        lid_calculators = []

        for i in range(self.separate_ll_count):
            lid_calculator = DatasetMetricCalculator(class_count=self.n_classes)
            ll_class_indices = []
            for c in self.separate_lls_classes[i]:
                ll_class_indices.extend(Y_ind_per_class[c])

            i_batch = -1
            for batch in batch_iterator(separate_ll_inputs[ll_class_indices], Y[ll_class_indices], LID_BATCH_SIZE, True):
                i_batch += 1

                if i_batch == LID_BATCH_CNT or batch[0].shape[0] < LID_K + 1:
                    break

                feed_dict = {self.separate_ll_input_pl: batch[0],
                             self.labels_pl: batch[1],
                             self.separate_lls_is_training[i]: False,
                             self.is_training: False}

                batch_lid_scores = self.separate_lls_lid_calc_op[i].eval(feed_dict)

                correct_values = np.logical_and(np.greater(batch_lid_scores, 0), np.less(batch_lid_scores, 1e3))
                correct_values_indices = np.nonzero(correct_values)[0]
                # print(list(sorted(np.round(batch_lid_scores[correct_values_indices], 1), key=lambda x: -x)))

                correct_batch_lid_scores = batch_lid_scores[correct_values_indices]
                combined_lid_calculator.add_batch_values(correct_batch_lid_scores)
                lid_calculator.add_batch_values(correct_batch_lid_scores)

            lid_calculators.append(lid_calculator)

        return combined_lid_calculator, lid_calculators

    def calc_lid_class_wise(self, X, Y, Y_ind_per_class):
        lid_calculator = DatasetMetricCalculator(class_count=self.n_classes)

        features = np.empty((X.shape[0], self.lid_layer_op.shape[-1]), np.float32)

        for batch in batch_iterator_with_indices(X, Y, LID_BATCH_SIZE, False):
            lid_features = self.lid_layer_op.eval({self.nn_input_pl: batch[0], self.is_training: False})
            features[batch[2]] = lid_features

        for c in range(len(Y_ind_per_class)):
            class_indices = np.array(Y_ind_per_class[c])
            class_features = features[class_indices]

            for i in range(LID_BATCH_CNT):
                random_element_features_indices = np.random.choice(np.arange(len(class_features)), BATCH_SIZE, False)
                batch_lid_scores = self.lid_per_element_calc_op.eval(feed_dict={
                    self.lid_per_element_calc_op_features_pl:class_features[random_element_features_indices],
                    self.lid_per_element_calc_op_ref_features_pl: class_features,
                    self.is_training: False})

                lid_calculator.add_batch_values_with_labels_argmaxed(batch_lid_scores,
                                                                     np.full((len(batch_lid_scores), ), c, np.int32))

        return lid_calculator

    def compute_LDA(self, X, Y, Y_ind, Y_ind_per_class):
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
                self.sess.run(self.reset_class_feature_sums_counts_and_covariance_op)

                # compute class feature means
                for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE * 8, False):
                    feed_dict = {self.features_pl: block_features[batch[2]],
                                 self.labels_pl: batch[1],
                                 self.class_feature_ops_keep_array_pl: keep_arr[batch[2]]}
                    self.sess.run(self.update_class_features_sum_and_counts_op, feed_dict=feed_dict)

                counts = self.class_feature_counts_var.eval(self.sess)
                sums = self.class_feature_sums_var.eval(self.sess)

                class_feature_means = sums / counts.reshape((-1, 1))

                # compute class feature covariance matrices
                for batch in batch_iterator_with_indices(X, Y, BATCH_SIZE * 8, False):
                    feed_dict = {self.features_pl: block_features[batch[2]], self.labels_pl: batch[1],
                                 self.class_feature_means_pl: class_feature_means,
                                 self.class_feature_ops_keep_array_pl: keep_arr[batch[2]]}
                    self.sess.run(self.update_class_covariances_op, feed_dict=feed_dict)

                class_covariance_sums = self.class_covariance_sums_var.eval(self.sess)
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
                    mah_dists = self.sess.run(self.mahalanobis_dist_calc, feed_dict=feed_dict)
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

    def calc_metric_on_dataset(self, X, Y, metric_ops, keep_values=True, base_feed_dict=None, batch_iter=None,
                               Y_ref=None):
        if not isinstance(metric_ops, list):
            metric_ops = [metric_ops]

        feed_dict = {}
        if base_feed_dict is not None:
            for k, v in base_feed_dict.items():
                feed_dict[k] = v

        metric_calculators = []
        for _ in range(len(metric_ops)):
            metric_calculators.append(DatasetMetricCalculator(keep_values=keep_values, class_count=self.n_classes))

        if batch_iter is None:
            batch_iter = batch_iterator_with_indices(X, Y, BATCH_SIZE, False)

        for batch in batch_iter:
            feed_dict[self.nn_input_pl] = batch[0]
            feed_dict[self.labels_pl] = batch[1]
            if Y_ref is not None:
                feed_dict[self.computed_labels_similarity_reference_pl] = Y_ref[batch[2]]
            feed_dict[self.is_training] = False

            metrics = self.sess.run(metric_ops, feed_dict)

            labels_argmaxed = np.argmax(batch[1], -1)
            for i in range(len(metrics)):
                metric_calculators[i].add_batch_values_with_labels_argmaxed(metrics[i], labels_argmaxed)

        if len(metric_calculators) == 1:
            metric_calculators = metric_calculators[0]

        return metric_calculators

    def calc_separate_ll_metric_on_dataset(self, separate_ll_inputs, Y, Y_ind_per_class, metric_ops, keep_values=True,
                                           base_feed_dict=None):
        feed_dict = {}
        if base_feed_dict is not None:
            for k, v in base_feed_dict.items():
                feed_dict[k] = v

        metric_calculator = DatasetMetricCalculator(keep_values=keep_values, class_count=self.n_classes)

        for i in range(self.separate_ll_count):
            ll_class_indices = []
            for c in self.separate_lls_classes[i]:
                ll_class_indices.extend(Y_ind_per_class[c])

            for batch in batch_iterator(separate_ll_inputs[ll_class_indices], Y[ll_class_indices], BATCH_SIZE, False):
                feed_dict[self.separate_ll_input_pl] = batch[0]
                feed_dict[self.labels_pl] = batch[1]
                feed_dict[self.separate_lls_is_training[i]] = False
                feed_dict[self.is_training] = False

                metric = self.sess.run(metric_ops[i], feed_dict)
                metric_calculator.add_batch_values_with_labels(metric, batch[1])

        return metric_calculator

    def train_separate_lls(self, separate_ll_inputs, Y, Y_ind_per_class, current_loss_weights, base_feed_dict):
        for i in range(self.separate_ll_count):
            ll_class_indices = []
            for c in self.separate_lls_classes[i]:
                ll_class_indices.extend(Y_ind_per_class[c])

            for batch in batch_iterator_with_indices(separate_ll_inputs[ll_class_indices], Y[ll_class_indices],
                                                     BATCH_SIZE, True):
                feed_dict = {self.separate_ll_input_pl: batch[0],
                             self.labels_pl: batch[1],
                             self.separate_lls_is_training[i]: True,
                             self.is_training: False,
                             self.loss_weights_pl: current_loss_weights[batch[2]]}
                for k, v in base_feed_dict.items():
                    feed_dict[k] = v
                self.separate_lls_train_step[i].run(feed_dict)

    def calc_confusion_matrix(self, batch_iter, predictions_op, base_feed_dict=None):
        prediction_calculator = DatasetMetricCalculator(class_count=self.n_classes, compute_confusion_matrix=True)

        if base_feed_dict is None:
            base_feed_dict = {}

        for batch in batch_iter:
            feed_dict = {self.nn_input_pl: batch[0], self.is_training: False}
            for k, v in base_feed_dict.items():
                feed_dict[k] = v

            predictions = self.sess.run(predictions_op, feed_dict)

            prediction_calculator.add_batch_values_with_labels_argmaxed(np.argmax(predictions, -1),
                                                                        np.argmax(batch[1], -1))

        return prediction_calculator.confusion_matrix
