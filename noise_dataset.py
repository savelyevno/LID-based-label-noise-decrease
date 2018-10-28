import numpy as np

from preprocessing import write_dataset, read_dataset


def introduce_symmetric_noise(X, Y, noise_ratio, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = X.shape[0]

    M = int(round(N * noise_ratio))

    perm = np.random.permutation(N)[:M]

    for it in perm:
        old_label = np.argmax(Y[it])
        new_label = old_label

        while old_label == new_label:
            new_label = np.random.randint(10)

        Y[it][old_label] = 0
        Y[it][new_label] = 1

    return X, Y


if __name__ == '__main__':
    X, Y = read_dataset('train')
    introduce_symmetric_noise(X, Y, 0.3)
    write_dataset('train30', X, Y)
