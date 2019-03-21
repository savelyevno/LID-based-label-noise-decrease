import numpy as np

from preprocessing import write_dataset, read_dataset


def introduce_symmetric_noise(Y, noise_ratio, seed=None):
    if seed is None:
        seed = 0

    N = Y.shape[0]
    M = int(round(N * noise_ratio))

    after_seed = np.random.randint(1 << 30)
    np.random.seed(seed)
    perm = np.random.permutation(N)[:M]
    np.random.seed(after_seed)

    for it in perm:
        old_label = np.argmax(Y[it])
        new_label = old_label

        while old_label == new_label:
            new_label = np.random.randint(10)

        Y[it][old_label] = 0
        Y[it][new_label] = 1

    return Y


if __name__ == '__main__':
    dataset_name = 'mnist'
    # dataset_name = 'cifar-10'
    # X, Y = read_dataset(dataset_name, 'train')
    # introduce_symmetric_noise(X, Y, 0.6)
    # write_dataset(dataset_name, 'train60', X, Y)
