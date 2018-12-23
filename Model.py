import os
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from networks import build_mnist, build_cifar_10
from tf_ops import get_euclid_dist_to_mean_calc_op, get_new_label_op, get_cosine_dist_to_mean_calc_op, get_lid_calc_op,\
    get_update_class_features_sum_and_counts_op
from consts import *
from Timer import timer
from preprocessing import read_dataset
from batch_iterate import batch_iterator, batch_iterator_with_indices
from tools import bitmask_contains


class Model:
    def __init__(self, dataset_name, model_name, update_mode, update_param, update_submode=0, update_subsubmode=0,
                 log_mask=0, reg_coef=5e-4):
        """

        :param dataset_name:         dataset name: mnist/cifar-10
        :param model_name:      name of the model
        :param update_mode:     0: vanilla
                                1: as in the paper
                                2: per element
        :param update_param:    for update_mode = 2: number of epoch to start using modified labels
        :param update_submode:  for update_mode = 2:
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

        self.FC_WIDTH = FC_WIDTH[dataset_name]
        self.DATASET_SIZE = DATASET_SIZE[dataset_name]

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
            self.pre_lid_layer_op, self.lid_layer_op, self.logits = build_mnist(self.nn_input, self.is_training)
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

            self.class_feature_sums_var = tf.Variable(np.zeros((N_CLASSES, self.FC_WIDTH)), dtype=tf.float32)
            self.class_feature_counts_var = tf.Variable(np.zeros((N_CLASSES, )), dtype=tf.float32)

            self.update_class_features_sum_and_counts_op = get_update_class_features_sum_and_counts_op(
                self.class_feature_sums_var, self.class_feature_counts_var,
                used_lid_layer, self.logits, self.y_)

            self.reset_class_feature_sums_and_counts_op = tf.group(
                tf.assign(self.class_feature_sums_var, tf.zeros((N_CLASSES, self.FC_WIDTH))),
                tf.assign(self.class_feature_counts_var, tf.zeros(N_CLASSES, ))
            )

            self.class_feature_means_pl = tf.placeholder(tf.float32, (N_CLASSES, self.FC_WIDTH))

            self.features_to_means_dist_op = None
            if self.update_subsubmode == 0:
                self.features_to_means_dist_op = get_cosine_dist_to_mean_calc_op(used_lid_layer, self.class_feature_means_pl)
            elif self.update_subsubmode == 1:
                self.features_to_means_dist_op = get_euclid_dist_to_mean_calc_op(used_lid_layer, self.class_feature_means_pl)

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
                        self.alpha_var * self.y_ + (1 - self.alpha_var.value()) * tf.one_hot(tf.argmax(self.logits, 1), 10),
                        name='modified_labels')
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.modified_labels_op,
                                                                               logits=self.logits)
            if self.update_mode == 2:
                self.modified_labels_op = get_new_label_op(self.features_to_means_dist_op, self.y_, self.logits)
                self.use_modified_labels_pl = tf.placeholder(tf.bool)

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cond(self.use_modified_labels_pl,
                                                                                          lambda: self.modified_labels_op,
                                                                                          lambda: self.y_),
                                                                           logits=self.logits)
            if self.dataset_name == 'cifar-10':
                cross_entropy += self.reg_coef * (W_l2_reg_sum + b_l2_reg_sum)

        self.cross_entropy = tf.reduce_mean(cross_entropy)

        #
        # CREATE ACCURACY
        #

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
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

            acc = tf.reduce_sum(tf.one_hot(tf.argmax(self.logits, 1), self.logits.shape[1]) * new_label, 1)
            self.modified_accuracy = tf.reduce_mean(acc, name='accuracy')

    def train(self, train_dataset_type='train'):
        # Import data

        X, Y = read_dataset(name=self.dataset_name, type=train_dataset_type)
        X0, Y0 = read_dataset(name=self.dataset_name, type='train')

        X_test, Y_test = read_dataset(name=self.dataset_name, type='test')

        self._build()

        self.epoch_pl = tf.placeholder(tf.int32)

        with tf.name_scope('learning_rate'):
            if self.dataset_name == 'mnist':
                # self.lr = tf.cond(self.epoch_pl > 40, lambda: 1e-3,
                #                   lambda: tf.cond(self.epoch_pl > 20, lambda: 1e-2,
                #                                   lambda: 1e-1)) * 1e-1
                self.lr = tf.cond(self.epoch_pl > 40, lambda: 1e-5, lambda: 1e-4)
                # self.lr = 1e-4
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
        tf.summary.scalar(name='modified_train_accuracy', tensor=self.modified_accuracy)
        summary = tf.summary.merge_all()

        test_accuracy_summary_scalar = tf.placeholder(tf.float32)
        test_accuracy_summary = tf.summary.scalar(name='test_accuracy', tensor=test_accuracy_summary_scalar)

        summaries_to_merge = [test_accuracy_summary]

        if self.update_mode == 1 or self.update_mode == 2:
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
            class_feature_means_per_epoch = np.empty((0, N_CLASSES, self.FC_WIDTH))
        if self.to_log(2):
            lid_features_per_epoch_per_element = np.empty((0, self.DATASET_SIZE, self.FC_WIDTH))
        if self.to_log(3):
            pre_lid_features_per_epoch_per_element = np.empty((0, self.DATASET_SIZE, self.FC_WIDTH))
        if self.to_log(4):
            logits_per_epoch_per_element = np.empty((0, self.DATASET_SIZE, N_CLASSES))

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

                #
                # TRAIN
                #

                print('starting training...')

                if self.to_log(2):
                    lid_features_per_element = np.empty((self.DATASET_SIZE, self.FC_WIDTH))
                if self.to_log(3):
                    pre_lid_features_per_element = np.empty((self.DATASET_SIZE, self.FC_WIDTH))
                if self.to_log(4):
                    logits_per_element = np.empty((self.DATASET_SIZE, N_CLASSES))

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

                    if self.update_mode == 1 or self.update_mode == 2:
                        feed_dict[self.is_training] = False

                        modified_labels = self.modified_labels_op.eval(feed_dict=feed_dict)
                        batch_accs = np.sum(modified_labels * Y0[batch[2]])
                        modified_labels_accuracy = (modified_labels_accuracy * batch_cnt * BATCH_SIZE + batch_accs) / \
                                                   (batch_cnt * BATCH_SIZE + batch_size)

                        feed_dict[self.is_training] = True

                    train_step.run(feed_dict=feed_dict)

                    feed_dict[self.is_training] = False

                    if self.to_log(2):
                        lid_features_per_element[batch[2], ] = self.lid_layer_op.eval(feed_dict=feed_dict)
                    if self.to_log(3):
                        pre_lid_features_per_element[batch[2], ] = self.pre_lid_layer_op.eval(feed_dict=feed_dict)
                    if self.to_log(4):
                        logits_per_element[batch[2], ] = self.logits.eval(feed_dict=feed_dict)

                    if i_step % 100 == 0:
                        train_accuracy = self.accuracy.eval(feed_dict=feed_dict)
                        print('\tstep %d, training accuracy %g' % (i_step, train_accuracy))

                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i_step)
                        summary_writer.flush()

                if self.to_log(2):
                    lid_features_per_epoch_per_element = np.append(
                        lid_features_per_epoch_per_element,
                        np.expand_dims(lid_features_per_element, 0),
                        0)
                if self.to_log(3):
                    pre_lid_features_per_epoch_per_element = np.append(
                        pre_lid_features_per_epoch_per_element,
                        np.expand_dims(pre_lid_features_per_element, 0),
                        0)
                if self.to_log(4):
                    logits_per_epoch_per_element = np.append(
                        logits_per_epoch_per_element,
                        np.expand_dims(logits_per_element, 0),
                        0)

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

                if self.update_mode == 1 or self.to_log(0):
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

                    if self.update_mode == 1 or self.to_log(0):
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

                if self.update_mode == 1 or self.update_mode == 2:
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
                    np.save('lid_features/' + self.dataset_name + '/' + self.model_name, lid_features_per_epoch_per_element)
                if self.to_log(3):
                    np.save('pre_lid_features/' + self.dataset_name + '/' + self.model_name, pre_lid_features_per_epoch_per_element)
                if self.to_log(4):
                    np.save('logits/' + self.dataset_name + '/' + self.model_name, logits_per_epoch_per_element)

                print(timer.stop())

    def to_log(self, bit):
        return bitmask_contains(self.log_mask, bit)

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

    # def calc_lid_per_element(X, Y, lid_input_layer, x, lid_per_element_calc_op, lid_sample_set_pl, is_training):
    #     print('\ncalculating LID for the whole dataset...')
    #
    #     #
    #     # FILL LID SAMPLE SET
    #     #
    #
    #     batch_size = 1000
    #
    #     lid_sample_set = np.empty((0, lid_input_layer.shape[1]), dtype=np.float32)
    #     i_batch = -1
    #     for batch in batch_iterator(X, Y, batch_size):
    #         i_batch += 1
    #
    #         lid_layer = lid_input_layer.eval(feed_dict={x: batch[0], is_training: False})
    #         lid_sample_set = np.vstack((lid_sample_set, lid_layer))
    #
    #         # if (i_batch + 1) % 100 == 0:
    #         #     print('\t filled LID sample set for %d/%d' % ((i_batch + 1) * BATCH_SIZE, LID_SAMPLE_SET_SIZE))
    #
    #         if lid_sample_set.shape[0] >= LID_SAMPLE_SET_SIZE:
    #             break
    #
    #     #
    #     # CALCULATE LID PER DATASET ELEMENT
    #     #
    #
    #     epoch_lids_per_element = np.empty((0,))
    #     i_batch = -1
    #     for batch in batch_iterator(X, Y, batch_size, False):
    #         i_batch += 1
    #
    #         lids_per_batch_element = lid_per_element_calc_op.eval(
    #             feed_dict={x: batch[0], lid_sample_set_pl: lid_sample_set, is_training: False})
    #         epoch_lids_per_element = np.append(epoch_lids_per_element, lids_per_batch_element)
    #
    #         # if (i_batch + 1) % 100 == 0:
    #         #     print('\t calculated LID for %d/%d' % ((i_batch + 1) * batch_size, DATASET_SIZE))
    #
    #     epoch_lids_per_element = epoch_lids_per_element.reshape([-1, 1])
    #
    #     return epoch_lids_per_element