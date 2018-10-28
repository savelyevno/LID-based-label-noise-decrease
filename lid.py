import tensorflow as tf
from consts import LID_K, EPS, LID_BATCH_SIZE


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