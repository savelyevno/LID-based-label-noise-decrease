import numpy as np


def batch_iterator(X, Y, batch_size, is_random=True):
    N = X.shape[0]

    if is_random:
        perm = np.random.permutation(N)

        X_shuffled = X[perm]
        Y_shuffled = Y[perm]
    else:
        X_shuffled = X
        Y_shuffled = Y

    l = 0
    end = False
    while not end:
        r = min(l + batch_size, N)

        X_batch = X_shuffled[l:r]
        Y_batch = Y_shuffled[l:r]

        yield X_batch, Y_batch

        l = r
        end = r == N