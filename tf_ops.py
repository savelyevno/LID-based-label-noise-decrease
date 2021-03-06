import tensorflow as tf
import numpy as np

from consts import LID_K, EPS, LID_BATCH_SIZE


def get_lid_calc_op(g_x):
    norm_squared = tf.reshape(tf.reduce_sum(g_x * g_x, 1), [-1, 1])
    norm_squared_t = tf.transpose(norm_squared)

    dot_products = tf.matmul(g_x, tf.transpose(g_x))

    distances_squared = tf.maximum(norm_squared - 2 * dot_products + norm_squared_t, 0)
    # distances = tf.sqrt(distances_squared + tf.ones((batch_size, batch_size)) * EPS)
    # distances = tf.sqrt(distances_squared) + tf.ones((batch_size, batch_size)) * EPS
    distances = tf.sqrt(distances_squared + EPS)

    k_nearest_raw, _ = tf.nn.top_k(-distances, k=LID_K + 1, sorted=True)
    k_nearest = -k_nearest_raw[:, 1:]

    max_distance_non_zero = tf.greater(k_nearest[:,  -1], EPS)
    max_distance_non_zero_indices = tf.reshape(tf.where(max_distance_non_zero), (-1, ))
    k_nearest = tf.gather(k_nearest, max_distance_non_zero_indices)

    k_nearest = tf.cond(tf.equal(tf.shape(max_distance_non_zero_indices)[0], 0),
                        lambda: np.array([[0]], dtype=np.float32),
                        lambda: k_nearest)

    distance_ratios = tf.transpose(tf.multiply(tf.transpose(k_nearest), 1 / k_nearest[:, -1]))
    LIDs = - LID_K / tf.reduce_sum(tf.log(distance_ratios + EPS), 1)

    return LIDs


def get_lid_per_element_calc_op(g_x, X):
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
    LIDs_for_batch = - LID_K / tf.reduce_sum(tf.log(distance_ratios + EPS), 2)
    LIDs = tf.reduce_mean(LIDs_for_batch, 1)

    return LIDs


# logits based
def get_update_class_features_sum_and_counts_op_logits(class_feature_sums_var, class_feature_counts_var, feature_op, logits_op, labels_var):
    n_classes = labels_var.shape[1]
    pred_labels = tf.argmax(logits_op, 1)
    labels = tf.argmax(labels_var, 1)

    are_labels_equal = tf.equal(pred_labels, labels)
    equal_labels_indices = tf.reshape(tf.where(are_labels_equal), (-1,))

    gathered_features = tf.gather(feature_op, equal_labels_indices)
    gathered_labels = tf.gather(labels, equal_labels_indices)
    ones = tf.ones(tf.shape(equal_labels_indices))

    # gathered_features = feature_op
    # gathered_labels = labels
    # ones = tf.ones(tf.shape(labels))

    feature_sum = tf.unsorted_segment_sum(gathered_features, gathered_labels, n_classes)
    feature_counts = tf.unsorted_segment_sum(ones, gathered_labels, n_classes)

    result_op = tf.group(
        tf.assign_add(class_feature_sums_var, feature_sum),
        tf.assign_add(class_feature_counts_var, feature_counts)
    )

    return result_op


def get_cosine_dist_to_mean_calc_op(features, class_means):
    features_norm2 = tf.reduce_sum(features ** 2, 1)
    means_norm2 = tf.reduce_sum(class_means ** 2, 1)
    dot_product = tf.matmul(features, tf.transpose(class_means))

    norm_mult2 = tf.matmul(tf.reshape(features_norm2, (-1, 1)), tf.reshape(means_norm2, (1, -1)))
    norm_mult = tf.sqrt(norm_mult2 + EPS)

    dists = 1 - dot_product / norm_mult

    return dists


def get_euclid_dist_to_mean_calc_op(features, class_means):
    features_norm2 = tf.reduce_sum(features ** 2, 1)
    means_norm2 = tf.reduce_sum(class_means ** 2, 1)
    dot_product = tf.matmul(features, tf.transpose(class_means))

    dists2 = tf.reshape(features_norm2, (-1, 1)) - 2 * dot_product + means_norm2

    dists = tf.sqrt(dists2 + EPS)

    return dists


# def get_lid_to_set_calc_op(x, S):
#     def get_norm_squared(arg):
#         return tf.reduce_sum(arg*arg, axis=2)
#
#     sample_batch_size = LID_BATCH_SIZE
#     x_batch_size = tf.shape(x)[0]
#
#     random_batch_indices = tf.random_uniform((x_batch_size, sample_batch_size), maxval=tf.shape(S)[0], dtype=tf.int32)
#
#     random_batch = tf.gather(S, random_batch_indices)
#
#     tiled_x = tf.tile(tf.expand_dims(x, 1), [1, sample_batch_size, 1])
#
#     g_x_norm_squared = tf.expand_dims(get_norm_squared(tiled_x), 2)
#     rnd_norm_squared = tf.expand_dims(get_norm_squared(random_batch), 1)
#
#     dot_products = tf.matmul(tiled_x, tf.transpose(random_batch, [0, 2, 1]))
#
#     distances_squared = tf.maximum(g_x_norm_squared - 2 * dot_products + rnd_norm_squared, 0)
#     distances = tf.sqrt(distances_squared) + tf.ones((x_batch_size, sample_batch_size, sample_batch_size)) * EPS
#
#     k_nearest_raw, _ = tf.nn.top_k(-distances, k=LID_K, sorted=True)
#     k_nearest = -k_nearest_raw
#
#     distance_ratios = tf.multiply(k_nearest, tf.expand_dims(1 / k_nearest[:, :, -1], 2))
#     lids_for_batch = - LID_K / tf.reduce_sum(tf.log(distance_ratios + EPS), 2)
#     lids = tf.reduce_mean(lids_for_batch, 1)
#
#     return lids


def get_new_label_op(distance_to_class_mean_op, labels, logits):
    label_cls = tf.argmax(labels, 1)
    pred_cls = tf.stop_gradient(tf.argmax(logits, 1))

    n_classes = labels.shape[1]

    label_cls_mask = tf.one_hot(label_cls, n_classes, True, False, dtype=tf.bool)
    label_dist = tf.boolean_mask(distance_to_class_mean_op, label_cls_mask)     # distances to label class

    pred_cls_mask = tf.one_hot(pred_cls, n_classes, True, False, dtype=tf.bool)
    pred_dist = tf.boolean_mask(distance_to_class_mean_op, pred_cls_mask)       # distances to prediction class

    cond_mask = tf.to_int64(label_dist <= pred_dist)

    new_label_cls = cond_mask * label_cls + (1 - cond_mask) * pred_cls

    new_labels = tf.one_hot(new_label_cls, n_classes)

    return new_labels


def get_update_class_feature_selective_sum_and_counts_op(class_feature_sums_var, class_feature_counts_var, feature_op, labels_one_hot, to_keep_arr):
    n_classes = labels_one_hot.shape[1]
    labels = tf.argmax(labels_one_hot, 1)

    keep_labels_indices = tf.reshape(tf.where(to_keep_arr), (-1,))

    gathered_features = tf.gather(feature_op, keep_labels_indices)
    gathered_labels = tf.gather(labels, keep_labels_indices)
    ones = tf.ones(tf.shape(keep_labels_indices))

    feature_sum = tf.unsorted_segment_sum(gathered_features, gathered_labels, n_classes)
    feature_counts = tf.unsorted_segment_sum(ones, gathered_labels, n_classes)

    result_op = tf.group(
        tf.assign_add(class_feature_sums_var, feature_sum),
        tf.assign_add(class_feature_counts_var, feature_counts)
    )

    return result_op


def get_update_class_covariances_selective_sum_op(class_covariance_sums, class_feature_means, features, labels_one_hot, to_keep_arr):
    n_classes = labels_one_hot.shape[1]
    features_dim = features.shape[1]
    labels = tf.argmax(labels_one_hot, 1)

    keep_labels_indices = tf.reshape(tf.where(to_keep_arr), (-1,))

    gathered_features = tf.gather(features, keep_labels_indices)
    gathered_labels = tf.gather(labels, keep_labels_indices)
    gathered_means = tf.gather(class_feature_means, gathered_labels)

    gathered_diff = tf.reshape(gathered_features - gathered_means, (-1, features_dim, 1))
    gathered_diff_T = tf.reshape(gathered_diff, (-1, 1, features_dim))

    gathered_covariances = tf.matmul(gathered_diff, gathered_diff_T)

    new_covariance_sums = tf.unsorted_segment_sum(gathered_covariances, gathered_labels, n_classes)

    result_op = tf.assign_add(class_covariance_sums, new_covariance_sums)

    return result_op


def get_LDA_logits_calc_op(block_class_feature_means, block_inv_covariance_matrix, features, class_priors):
    n_blocks = block_class_feature_means.shape[0]
    n_classes = block_class_feature_means.shape[1]
    n_dims = block_class_feature_means.shape[2]
    batch_size = tf.shape(features)[0]

    means_reshaped = tf.tile(
        input=tf.reshape(block_class_feature_means, (1, n_blocks, n_classes, n_dims, 1)),
        multiples=[batch_size, 1, 1, 1, 1])
    means_reshaped_T = tf.reshape(means_reshaped, (batch_size, n_blocks, n_classes, 1, n_dims))

    features_reshaped = tf.tile(
        input=tf.reshape(features, (batch_size, n_blocks, 1, n_dims, 1)),
        multiples=[1, 1, n_classes, 1, 1])

    inv_covariance_matrix_reshaped = tf.tile(
        input=tf.reshape(block_inv_covariance_matrix, (1, n_blocks, 1, n_dims, n_dims)),
        multiples=[batch_size, 1, n_classes, 1, 1])

    class_priors_reshaped = tf.tile(
        input=tf.reshape(class_priors, (1, 1, n_classes, 1, 1)),
        multiples=[batch_size, n_blocks, 1, 1, 1])

    sm1 = tf.matmul(means_reshaped_T, tf.matmul(inv_covariance_matrix_reshaped, features_reshaped))
    sm2 = 0.5 * tf.matmul(means_reshaped_T, tf.matmul(inv_covariance_matrix_reshaped, means_reshaped))
    sm3 = tf.log(class_priors_reshaped)

    logits = tf.reshape(sm1 - sm2 + sm3, (-1, n_blocks, n_classes))

    return logits


def get_LDA_based_labels_calc_op(LDA_logits):
    block_predictions = tf.nn.softmax(LDA_logits)
    predictions = tf.reduce_mean(block_predictions, 1)

    return predictions


def get_Mahalanobis_distance_calc_op(class_feature_means, class_inv_covariances, features, labels_one_hot):
    features_dim = features.shape[1]
    labels = tf.argmax(labels_one_hot, 1)

    gathered_means = tf.gather(class_feature_means, labels)
    gathered_inv_covariances = tf.gather(class_inv_covariances, labels)

    diff = tf.reshape(features - gathered_means, (-1, features_dim, 1))
    diff_T = tf.reshape(diff, (-1, 1, features_dim))

    result = tf.matmul(diff_T, tf.matmul(gathered_inv_covariances, diff))

    return tf.reshape(result, (-1,))
