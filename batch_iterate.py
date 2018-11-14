import numpy as np


def batch_iterator(X, Y, batch_size, is_random=True):
    for X_batch, Y_batch, index_batch in batch_iterator_with_indices(X, Y, batch_size, is_random):
        yield X_batch, Y_batch


def batch_iterator_with_indices(X, Y, batch_size, is_random=True):
    N = X.shape[0]

    if is_random:
        perm = np.random.permutation(N)

        X_shuffled = X[perm]
        Y_shuffled = Y[perm]
    else:
        perm = np.arange(N)
        X_shuffled = X
        Y_shuffled = Y

    l = 0
    end = False
    while not end:
        r = min(l + batch_size, N)

        X_batch = X_shuffled[l:r]
        Y_batch = Y_shuffled[l:r]

        yield X_batch, Y_batch, perm[l:r]

        l = r
        end = r == N