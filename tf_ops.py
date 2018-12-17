import tensorflow as tf

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
    LIDs_for_btach = - LID_K / tf.reduce_sum(tf.log(distance_ratios + EPS), 2)
    LIDs = tf.reduce_mean(LIDs_for_btach, 1)

    return LIDs


def get_update_class_features_sum_and_counts_op(class_feature_sums_var, class_feature_counts_var, feature_op, logits_op, labels_var):
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
