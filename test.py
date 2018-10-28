# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import read_dataset


def test_single_epoch(epoch=-1):
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train40')

    # LID = np.load('LID_matrices/LID_matrix_model25.npy')
    LID = np.load('LID_matrices/LID_matrix_model25_noLID.npy')
    N = LID.shape[0]

    y = [np.argmax(Y[it]) for it in range(N)]
    y1 = [np.argmax(Y1[it]) for it in range(N)]

    cls = 9
    cls2 = 8


    LID_epoch = np.array(LID[:, epoch])
    mean = LID_epoch.mean()
    st_dev = LID_epoch.var()**0.5

    sample1 = []
    sample2 = []
    for j in range(N):
        if y1[j] == cls:                    # select elements based on noised label
        # if y[j] == cls:                     # select elements based on true label
            if y[j] != y1[j]:
            # if y[j] == cls2:                 # select elements that belonged cls2

                sample2.append(LID_epoch[j])

            sample1.append(LID_epoch[j])
    bins = int(max(sample1))

    # add fake samples for sample1/sample2 hists correspondence
    sample2.append(min(sample1))
    sample2.append(max(sample1))

    a1, b1 = np.histogram(sample1, bins)
    a2, b2 = np.histogram(sample2, bins)

    # remove fake samples
    a2[0] -= 1
    a2[-1] -= 1

    a = a2 / np.maximum(a1, 1)

    plt.bar(np.arange(bins), a1/max(a1), width=1)
    plt.bar(np.arange(bins), a2/max(a1), width=1)
    plt.bar(np.arange(bins), a, width=1, alpha=0.5)

    plt.title(str(epoch))
    plt.grid()
    plt.show()


def test_epoch_progress():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train40')

    # LID = np.load('LID_matrices/old/LID_matrix_model25.npy')
    LID = np.load('LID_matrices/LID_matrix_model25_noLID.npy')
    N = LID.shape[0]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    cls1 = 5

    indices1 = []
    indices_comb = []
    indices_per_comb = [[] for cls in range(10)]
    for i in range(N):
        if y1[i] == cls1:
            indices1.append(i)

            if y0[i] != y1[i]:
                indices_comb.append(i)
                indices_per_comb[y0[i]].append(i)

    samples1 = []
    samples_comb = []
    samples_per_comb = [[] for cls in range(10)]
    for i_epoch in range(41):
        samples1.append([])
        for i in indices1:
            samples1[-1].append(LID[i][i_epoch])

        samples_comb.append([])
        for i in indices_comb:
            samples_comb[-1].append(LID[i][i_epoch])

        for cls in range(10):
            samples_per_comb[cls].append([])
            for i in indices_per_comb[cls]:
                samples_per_comb[cls][-1].append(LID[i][i_epoch])

    samples1 = np.array(samples1)
    samples_comb = np.array(samples_comb)
    samples_per_comb = [np.array(it) for it in samples_per_comb]

    means1 = np.mean(samples1, 1)
    means_comb = np.mean(samples_comb, 1)
    means_per_comb = [np.mean(it, 1) for it in samples_per_comb]
    std_devs1 = np.var(samples1, 1) ** 0.5
    std_devs_comb = np.var(samples_comb, 1) ** 0.5
    std_devs_per_comb = [np.var(it, 1) ** 0.5 for it in samples_per_comb]

    plt.plot(means1, color='blue', label='LID mean of elems labeled as %d' % cls1)
    plt.plot(means1 - std_devs1, '-.', color='blue', linewidth=.5)
    plt.plot(means1 + std_devs1, '-.', color='blue', linewidth=.5)

    plt.plot(means_comb, color='black', label='LID mean of elems labeled as %d, but belonging to other class' % (cls1, ))
    # plt.plot(means_comb - std_devs_comb, '-.', color='black', linewidth=.5)
    # plt.plot(means_comb + std_devs_comb, '-.', color='black', linewidth=.5)

    for cls in range(10):
        if cls != cls1:
            plt.plot(means_per_comb[cls], color='C' + str(cls), label='LID mean of elems labeled as %d, but belonging to %d' % (cls1, cls))

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # test_single_epoch(40)
    test_epoch_progress()