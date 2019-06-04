import numpy as np
from scipy.stats import multinomial
from preprocessing import write_dataset, read_dataset


def introduce_uniform_noise(Y, noise_ratio, seed=0):
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


def introduce_noise(Y, noise_matrix, seed=0):
    after_seed = np.random.randint(1 << 30)
    np.random.seed(seed)

    distributions = []
    for c in range(noise_matrix.shape[0]):
        distributions.append(multinomial(1, noise_matrix[c]))

    for i in range(len(Y)):
        old_label = int(np.argmax(Y[i]))
        new_label = np.argmax(distributions[old_label].rvs())

        if new_label != old_label:
            Y[i][old_label] = 0
            Y[i][new_label] = 1

    np.random.seed(after_seed)


def get_pair_confusion_noise_matrix(class_count, noise_rate, symmetric=False):
    noise_matrix = np.eye(class_count, class_count)
    for c in range(class_count):
        noise_matrix[c, c] -= noise_rate
        if symmetric:
            if c % 2 == 0:
                noise_matrix[c, (c + 1) % class_count] += noise_rate
            else:
                noise_matrix[c, (c - 1) % class_count] += noise_rate
        else:
            noise_matrix[c, (c + 1) % class_count] += noise_rate

    return noise_matrix


if __name__ == '__main__':
    dataset_name = 'mnist'
    # dataset_name = 'cifar-10'
    # X, Y = read_dataset(dataset_name, 'train')
    # introduce_uniform_noise(X, Y, 0.6)
    # write_dataset(dataset_name, 'train60', X, Y)
