# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.spatial
import bisect

from EM import em_solve, e_step
from Timer import timer

from preprocessing import read_dataset


def test_single_epoch(epoch=-1):

    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train40')

    # LID = np.load('LID_matrices/LID_matrix_model25.npy')
    LID = np.load('LID_matrices/old/LID_matrix_model25_noLID.npy')
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

    # plt.bar(np.arange(bins), a1/max(a1), width=1)
    # plt.bar(np.arange(bins), a2/max(a1), width=1)
    # plt.bar(np.arange(bins), a, width=1, alpha=0.5)

    all_lids_bins = int(np.max(LID_epoch))
    a3, b3 = np.histogram(LID_epoch, all_lids_bins)
    plt.bar(np.arange(all_lids_bins), a3, width=1)

    plt.title(str(epoch))
    plt.grid()
    plt.show()


def test_LID_with_epoch_progress():
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


def test_weights_with_epoch_progress():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    # weights = np.load('LID_matrices/old/LID_matrix_model25.npy')
    weights = np.load('weight_matrices/model25_2_clipped_2stdev_weighted_mean.npy')
    N = weights.shape[0]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    cls1 = 1

    indices1 = []
    indices1_true = []
    indices_comb = []
    indices_per_comb = [[] for cls in range(10)]
    for i in range(N):
        if y1[i] == cls1:
            indices1.append(i)

            if y0[i] != y1[i]:
                indices_comb.append(i)
                indices_per_comb[y0[i]].append(i)
            else:
                indices1_true.append(i)

    samples1 = []
    samples1_true = []
    samples_comb = []
    samples_per_comb = [[] for cls in range(10)]
    for i_epoch in range(weights.shape[1]):
        samples1.append([])
        for i in indices1:
            samples1[-1].append(weights[i][i_epoch])

        samples1_true.append([])
        for i in indices1_true:
            samples1_true[-1].append(weights[i][i_epoch])

        samples_comb.append([])
        for i in indices_comb:
            samples_comb[-1].append(weights[i][i_epoch])

        for cls in range(10):
            samples_per_comb[cls].append([])
            for i in indices_per_comb[cls]:
                samples_per_comb[cls][-1].append(weights[i][i_epoch])

    samples1 = np.array(samples1)
    samples1_true = np.array(samples1_true)
    samples_comb = np.array(samples_comb)
    samples_per_comb = [np.array(it) for it in samples_per_comb]

    means1 = np.mean(samples1, 1)
    means1_true = np.mean(samples1_true, 1)
    means_comb = np.mean(samples_comb, 1)
    means_per_comb = [np.mean(it, 1) for it in samples_per_comb]
    std_devs1 = np.var(samples1, 1) ** 0.5
    std_devs1_true = np.var(samples1_true, 1) ** 0.5
    std_devs_comb = np.var(samples_comb, 1) ** 0.5
    std_devs_per_comb = [np.var(it, 1) ** 0.5 for it in samples_per_comb]

    plt.plot(means1, color='blue', label='weights mean of elems labeled as %d' % cls1)
    # plt.plot(means1 - std_devs1, '-.', color='blue', linewidth=.5)
    # plt.plot(means1 + std_devs1, '-.', color='blue', linewidth=.5)

    plt.plot(means1_true, color='red', label='weights mean of elems labeled as %d and belonging to %d' % (cls1, cls1))
    plt.plot(means1_true - std_devs1_true, '-.', color='red', linewidth=.5)
    plt.plot(means1_true + std_devs1_true, '-.', color='red', linewidth=.5)

    plt.plot(means_comb, color='black', label='weights mean of elems labeled as %d, but belonging to other class' % (cls1, ))
    plt.plot(means_comb - std_devs_comb, '-.', color='black', linewidth=.5)
    plt.plot(means_comb + std_devs_comb, '-.', color='black', linewidth=.5)

    # for cls in range(10):
    #     if cls != cls1:
    #         plt.plot(means_per_comb[cls], color='C' + str(cls), label='weights mean of elems labeled as %d, but belonging to %d' % (cls1, cls))



    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def test_weights_for_single_element_with_epoch_progress():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    # weights = np.load('LID_matrices/old/LID_matrix_model25.npy')
    weights = np.load('weight_matrices/model25_2_clipped_2stdev_weighted_mean.npy')
    N = weights.shape[0]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    cls1 = 1

    indices1 = []
    indices_comb = []
    indices_per_comb = [[] for cls in range(10)]
    for i in range(N):
        if y1[i] == cls1:
            indices1.append(i)

            indices_comb.append(i)
            indices_per_comb[y0[i]].append(i)

    samples1 = []
    samples_comb = []
    samples_per_comb = [[] for cls in range(10)]
    for i_epoch in range(weights.shape[1]):
        samples1.append([])
        for i in indices1:
            samples1[-1].append(weights[i][i_epoch])

        samples_comb.append([])
        for i in indices_comb:
            samples_comb[-1].append(weights[i][i_epoch])

        for cls in range(10):
            samples_per_comb[cls].append([])
            for i in indices_per_comb[cls]:
                samples_per_comb[cls][-1].append(weights[i][i_epoch])

    samples1 = np.array(samples1)
    samples_comb = np.array(samples_comb)
    samples_per_comb = [np.array(it) for it in samples_per_comb]

    arr = samples_per_comb[1]

    # np.random.seed(0)
    plt.plot(arr[:, np.random.randint(arr.shape[1])])

    # plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def test_distr():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    features = np.load('lid_features/model25_none_pre_lid.npy')
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    def plot_hist(data, alpha=1.):
        mean = np.mean(data)
        st_dev = np.var(data) ** 0.5

        step = 1e-1
        k = 3
        bins = np.arange(mean - k * st_dev, mean + k * st_dev, step)
        a1, b1 = np.histogram(data, bins)
        # print(a1[0])
        # a1[0] = 0
        plt.bar(b1[:-1] + step / 2, a1, align='center', width=step, alpha=alpha)

    epoch = 30
    cls1 = 0
    cls0 = 0
    for epoch in range(0, 40):
        ps = []
        for dim in range(0, 129):
            data = []
            for i in range(N):
                # if y1[i] != cls1:
                if not(y1[i] == cls1 and y0[i] == cls0):
                # if not(y1[i] == cls1 and y1[i] != y0[i]):
                    continue

                sample = features[epoch, i, dim]
                # if sample < 1e-2:
                #     continue

                data.append(sample)

            data = np.array(data)
            # plot_hist(data, 0.5)

            k2, p = stats.normaltest(data)
            ps.append(p)
            # print(dim, p)
            plt.plot(dim, p, 'o', color='red')

        print(epoch, np.mean(ps))

        plt.tight_layout()
        plt.axes().set_yscale('log')
        plt.grid()
        # plt.show()


def test_distr_per_class():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    features = np.load('lid_features/model25_none_pre_lid.npy')
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    def plot_hist(data, alpha=1.):
        mean = np.mean(data)
        st_dev = np.var(data) ** 0.5

        step = 1e-1
        k = 3
        bins = np.arange(mean - k * st_dev, mean + k * st_dev, step)
        a1, b1 = np.histogram(data, bins)
        # print(a1[0])
        # a1[0] = 0
        plt.bar(b1[:-1] + step / 2, a1, align='center', width=step, alpha=alpha)

    epoch = 5
    cls1 = 0
    cls0 = 1
    for epoch in range(epoch, epoch + 1):
        ps = []
        for dim in range(0, 129):
            data1 = []
            data0 = []
            for i in range(N):
                sample = features[epoch, i, dim]
                # if sample < 1e-2:
                #     continue

                if y1[i] == cls1:
                    data1.append(sample)
                    if y0[i] == cls0:
                        data0.append(sample)

            data1 = np.array(data1)
            data0 = np.array(data0)
            plot_hist(data1, 0.5)
            plot_hist(data0, 0.5)

            # k2, p = stats.normaltest(data)
            # ps.append(p)
            # print(dim, p)
            # plt.plot(dim, p, 'o', color='red')

        # print(epoch, np.mean(ps))

            plt.tight_layout()
            # plt.axes().set_yscale('log')
            plt.grid()
            plt.show()


def test_mahalanobis():
    def M_dist(x, mean, inv_sigma):
        diff = (x - mean).reshape(-1, 1)
        return np.dot(np.transpose(diff), np.dot(inv_sigma, diff))**0.5

    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    features = np.load('lid_features/model25_none_pre_lid.npy')
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    y = y1

    def calc_for_epoch(i_epoch):
        means = [np.zeros(d) for i in range(10)]
        cnts_for_epoch = np.zeros(10)
        for i in range(N):
            means[y[i]] += features[i_epoch, i]
            cnts_for_epoch[y[i]] += 1

        means /= cnts_for_epoch[:, None]

        sigma = np.zeros((d, d))
        for i in range(N):
            diff = (features[i_epoch, i] - means[y[i]]).reshape((-1, 1))
            sigma += np.dot(diff, np.transpose(diff))
        sigma /= N

        return means, np.linalg.inv(sigma)

    def plot_hist(data, alpha=1.):
        mean = np.mean(data)
        st_dev = np.var(data) ** 0.5

        print(mean, st_dev)

        step = 1
        k = 3
        bins = np.arange(0, mean + k * st_dev, step)
        a1, b1 = np.histogram(data, bins)
        plt.bar(b1[:-1] + step / 2, a1, align='center', width=step, alpha=alpha)

    mode = 1
    if mode == 0:
        epoch = 10
        means, inv_sigma = calc_for_epoch(epoch)

        cls0 = 0
        cls1 = 2

        dist0 = []
        dist1 = []
        for i in range(N):
            if cls1 == y1[i]:
                dist = M_dist(features[epoch, i], means[y1[i]], inv_sigma)
                dist1.append(dist)
                if cls0 == y0[i]:
                # if y0[i] != y1[i]:
                    dist0.append(dist)

        dist1 = np.array(dist1)
        dist0 = np.array(dist0)

        plot_hist(dist1)
        plot_hist(dist0, 0.7)

        plt.grid()
        plt.tight_layout()
        plt.show()
    elif mode == 1:
        cls = 2
        for epoch in range(40):
            means, inv_sigma = calc_for_epoch(epoch)
            cnt = 0
            acc = 0
            for i in range(N):
                if not((cls == -1 or y1[i] == cls) and y1[i] != y0[i]):
                    continue

                dists = np.empty(10)
                x = features[epoch, i]
                for c in range(10):
                    dists[c] = M_dist(x, means[c], inv_sigma)
                acc += int(np.argmin(dists) == y0[i])
                cnt += 1
            acc /= cnt
            print(epoch, acc)


def test_lid_em():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train40')

    LID = np.load('LID_matrices/model_none_atmpt2.npy')
    # LID = np.load('LID_matrices/old/LID_matrix_model25_noLID.npy')
    N = LID.shape[0]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    cls1 = 5

    indices = [[] for i in range(10)]       # indices of elements marked as cls
    true_indices = [[] for i in range(10)]  # indices of elements actually from cls
    true_false_indices = [[[] for j in range(10)] for i in range(10)]   # arr[c1][c2] indices of elements from c1 but marked as c2
    for i in range(N):
        indices[y1[i]].append(i)
        true_indices[y0[i]].append(i)
        true_false_indices[y0[i]][y1[i]].append(i)


    samples = []                # (n_epochs, 10, ?)
    true_samples = []           # (n_epochs, 10, ?)
    true_false_samples = []     # (n_epochs, 10, 10, ?)
    for epoch in range(41):
        epoch_samples = []
        true_epoch_samples = []
        true_false_epoch_samples = []

        for cls in range(10):
            epoch_cls_samples = []
            for i in indices[cls]:
                epoch_cls_samples.append(LID[i][epoch])
            epoch_samples.append(np.array(epoch_cls_samples))

            true_epoch_cls_samples = []
            for i in indices[cls]:
                true_epoch_cls_samples.append(LID[i][epoch])
            true_epoch_samples.append(np.array(true_epoch_cls_samples))

            true_false_epoch_cls_samples = []
            for cls1 in range(10):
                true_false_epoch_cls_cls1_samples = []
                for i in true_false_indices[cls][cls1]:
                    true_false_epoch_cls_cls1_samples.append(LID[i][epoch])
                true_false_epoch_cls_samples.append(np.array(true_false_epoch_cls_cls1_samples))
            true_false_epoch_samples.append(true_false_epoch_cls_samples)

        samples.append(epoch_samples)
        true_samples.append(true_epoch_samples)
        true_false_samples.append(true_false_epoch_samples)

    def solve(train_data, train_means, epoch, cls1, cls2):
        train_data = np.append(train_data[cls1], train_data[cls2], 0).reshape((-1, 1))

        step = 1
        bins = np.arange(0, 100, step)
        a1, b1 = np.histogram(train_data, bins)
        plt.bar(b1[:-1] + step / 2, a1, align='center', width=step)
        plt.grid()
        plt.show()
        # plt.savefig('figs/%d %d.jpg' % (cls1, cls2))
        # plt.cla()

        # np.random.shuffle(train_data)
        #
        # mu_0 = np.array([
        #     [train_means[cls1]],
        #     [train_means[cls2]]
        # ])
        # mus, sigmas, pi, conf = em_solve(2, train_data, max_iter=50, mu_0=mu_0, verbose=1)
        #
        # test_1_2 = true_false_samples[epoch][cls1][cls2].reshape((-1, 1))  # from cls1 but marked as cls2
        # conf_1_2 = e_step(2, mus, sigmas, pi, test_1_2)
        #
        # acc_1_2 = 0
        # conf_mean_1_2 = 0
        # for conf in conf_1_2:
        #     pred_dst = np.argmax(conf)
        #     pred_val = conf[pred_dst]
        #
        #     acc_1_2 += pred_dst == 0
        #     conf_mean_1_2 += pred_val
        # acc_1_2 /= len(test_1_2)
        # conf_mean_1_2 /= len(test_1_2)
        # print('\t accuracy of finding elements that belong to %d but marked as %d: %g; confidence mean: %g' %
        #       (cls1, cls2, acc_1_2, conf_mean_1_2))
        #
        # test_2_1 = true_false_samples[epoch][cls2][cls1].reshape((-1, 1))  # from cls2 but marked as cls1
        # conf_2_1 = e_step(2, mus, sigmas, pi, test_2_1)
        #
        # acc_2_1 = 0
        # conf_mean_2_1 = 0
        # for conf in conf_2_1:
        #     pred_dst = np.argmax(conf)
        #     pred_val = conf[pred_dst]
        #
        #     acc_2_1 += pred_dst == 1
        #     conf_mean_2_1 += pred_val
        # acc_2_1 /= len(test_2_1)
        # conf_mean_2_1 /= len(test_2_1)
        # print('\t accuracy of finding elements that belong to %d but marked as %d: %g; confidence mean: %g' %
        #       (cls2, cls1, acc_2_1, conf_mean_2_1))

    for epoch in range(0, 41):
        print('epoch', epoch)
        train_data = []
        train_means = []
        for cls in range(10):
            data_cls = true_false_samples[epoch][cls][cls]
            mean_cls = np.mean(data_cls)

            train_data.append(data_cls)
            train_means.append(mean_cls)

        solve(train_data, train_means, epoch, 3, 2)

        # for cls1 in range(10):
        #     for cls2 in range(0, cls1):
        #
        #         print('\t', cls1, cls2)
        #         solve(train_data, train_means, epoch, cls1, cls2)


def calc_lid(x, S, b=100, k=20):
    B = S[np.random.choice(S.shape[0], min(b, S.shape[0]), False), :]

    B_norm2 = np.sum(B * B, 1)
    x_norm2 = np.sum(x**2)
    dot_prod = np.matmul(x, B.T)

    dists2 = B_norm2 - 2 * dot_prod + x_norm2

    dists2 = dists2[np.argsort(dists2)][:k + 1]

    if dists2[0] < 1e-10:
        dists2 = dists2[1:]
    else:
        dists2 = dists2[:-1]

    mx = dists2[-1]

    lid = -k / (0.5 * np.sum(np.log(dists2/mx)))

    return lid


def test_relative_lid_calc_clean():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    # features = np.load('lid_feature_matrices/model_none_atmpt3.npy')
    features = np.load('lid_features/model25_none.npy')
    n_epochs = features.shape[0]
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    ind = [[] for c in range(10)]
    for i in range(N):
        ind[y0[i]].append(i)

    features_per_class = []
    for epoch in range(n_epochs):
        features_per_class.append([features[epoch, ind[c], :] for c in range(10)])

    epochs = list(range(n_epochs))
    # epochs = [10]
    for epoch in epochs:
        print('epoch', epoch)

        cls = 0

        lid_means = [0]*10
        lid_to_self_means = [0]*10
        lid_to_cls_means = [0]*10
        for c in range(10):
            cnt = 0
            for it in features_per_class[epoch][c]:
                lid_means[c] += calc_lid(it, features[epoch])
                # lid_to_self_means[c] += calc_lid(it, features_per_class[epoch][c])
                # lid_to_cls_means[c] += calc_lid(it, features_per_class[epoch][cls])

                cnt += 1
                if cnt > 1000:
                    break

            lid_means[c] /= cnt
            # lid_to_self_means[c] /= cnt
            # lid_to_cls_means[c] /= cnt

        print('\t', lid_means, np.mean(lid_means))
        # print('\t', lid_to_self_means, np.mean(lid_to_self_means))
        # print('\t', lid_to_cls_means, np.mean(lid_to_cls_means))


        # for c0 in range(10):
        #     S0 = features_per_class[epoch][c0]
        #
        #     lids_to_self = []
        #     for it in S0:
        #         lids_to_self.append(calc_lid(it, S0))
        #
        #     print(c0, np.mean(lids_to_self), np.var(lids_to_self)**0.5)
        #
        #     for c1 in range(c0 + 1, 10):
        #         lids_to_c1 = []
        #
        #         S1 = features_per_class[epoch][c1]
        #         for it in S0:
        #             lids_to_c1.append(calc_lid(it, S1))
        #
        #         print('\t', c1, np.mean(lids_to_c1), np.var(lids_to_c1)**0.5)


def test_relative_lid_calc_noise():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    features = np.load('lid_features/model25_none.npy')
    # features = np.load('lid_features/model25_none_pre_lid.npy')
    # features = np.load('lid_feature_matrices/model_none_atmpt3.npy')
    n_epochs = features.shape[0]
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    indices = [[] for i in range(10)]  # indices of elements marked as cls
    true_indices = [[] for i in range(10)]  # indices of elements actually from cls
    true_false_indices = [[[] for j in range(10)] for i in
                          range(10)]  # arr[c1][c2] indices of elements from c1 but marked as c2
    for i in range(N):
        indices[y1[i]].append(i)
        true_indices[y0[i]].append(i)
        true_false_indices[y0[i]][y1[i]].append(i)

    # epochs = list(range(n_epochs))
    epochs = [7]
    for epoch in epochs:
        print('epoch', epoch)

        step = 1
        bins = np.arange(0, 80, step)

        for c0 in range(10):
            S_c0_c0 = features[epoch, true_false_indices[c0][c0], :]    # elements that are marked as c0 and actually belong to it

            lids_c0_to_c0 = []
            for it in S_c0_c0:
                lids_c0_to_c0.append(calc_lid(it, S_c0_c0))

            # print(c0, np.mean(lids_c0_to_c0), np.var(lids_c0_to_c0) ** 0.5)

            a0, b0 = np.histogram(lids_c0_to_c0, bins, density=True)

            for c1 in range(c0 + 1, 10):
                lids_c10_to_c0 = []
                lids_c10_to_c1 = []

                S_c1_c0 = features[epoch, true_false_indices[c1][c0], :]    # elements that are marked as c0, but belong to c1
                S_c1_c1 = features[epoch, true_false_indices[c1][c1], :]    # elements that are marked as c1 and actually belong to it
                for it in S_c1_c0:
                    lids_c10_to_c0.append(calc_lid(it, S_c0_c0))
                    lids_c10_to_c1.append(calc_lid(it, S_c1_c1))

                lids_c1_to_c1 = []

                for it in S_c1_c1:
                    lids_c1_to_c1.append(calc_lid(it, S_c1_c1))

                # print('\t', c1, np.mean(lids_c10_to_c0), np.var(lids_c10_to_c0) ** 0.5)

                a1, b1 = np.histogram(lids_c10_to_c0, bins, density=True)
                a2, b2 = np.histogram(lids_c10_to_c1, bins, density=True)
                a3, b3 = np.histogram(lids_c1_to_c1, bins, density=True)

                plt.title(str(c0) + ' ' + str(c1))
                plt.bar(b0[:-1] + step / 2, a0, align='center', width=step, alpha=0.8, label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c0, c0, c0, c0))
                plt.bar(b1[:-1] + step / 2, a1, align='center', width=step, alpha=0.5, label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c1, c0, c0, c0))
                # plt.bar(b3[:-1] + step / 2, a3, align='center', width=step, alpha=0.8, label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c1, c1, c1, c1))
                # plt.bar(b2[:-1] + step / 2, a2, align='center', width=step, alpha=0.5, label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c1, c0, c1, c1))

                plt.grid()
                plt.legend()
                # plt.show()
                plt.savefig('figs/lid/' + str(c0) + ' ' + str(c1))

                plt.cla()


def calc_dist_to_mean(X, mu):
    X_norm2 = np.sum(X * X, 1)
    mu_norm2 = np.sum(mu ** 2)
    dot_prod = np.matmul(mu, X.T)

    dists2 = X_norm2 - 2 * dot_prod + mu_norm2 + 1e-12

    dists = dists2 ** 0.5

    return dists


def calc_cosine_distance_to_mean(X, mu, scaled=True):
    X_norm2 = np.sum(X * X, 1)
    mu_norm2 = np.sum(mu ** 2)
    dot_prod = np.matmul(mu, X.T)

    if scaled:
        dists = 1 - dot_prod / (X_norm2 * mu_norm2 + 1e-14) ** 0.5
    else:
        dists = (X_norm2 * mu_norm2) ** 0.5 - dot_prod

    return dists


def calc_dist_to_random_batch(x, S, b=100):
    B = S[np.random.choice(S.shape[0], b, False), :]

    B_norm2 = np.sum(B * B, 1)
    x_norm2 = np.sum(x**2)
    dot_prod = np.matmul(x, B.T)

    dists2 = B_norm2 - 2 * dot_prod + x_norm2 + 1e-10

    return dists2 ** 0.5


def test_distances():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    features = np.load('lid_features/model25_relu.npy')
    # features = np.load('lid_features/model25_none_pre_lid.npy')
    # features = np.load('lid_feature_matrices/model_none_atmpt3.npy')
    n_epochs = features.shape[0]
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    indices = [[] for i in range(10)]  # indices of elements marked as cls
    true_indices = [[] for i in range(10)]  # indices of elements actually from cls
    true_false_indices = [[[] for j in range(10)] for i in range(10)]  # arr[c1][c2] indices of elements from c1 but marked as c2
    for i in range(N):
        indices[y1[i]].append(i)
        true_indices[y0[i]].append(i)
        true_false_indices[y0[i]][y1[i]].append(i)

    # epochs = list(range(n_epochs))
    epochs = [7]
    for epoch in epochs:
        print('epoch', epoch)

        step = 0.01
        bins = np.arange(0, 2, step)

        for c0 in range(10):
            S_c0_c0 = features[epoch, true_false_indices[c0][c0], :]  # elements that are marked as c0 and actually belong to it
            mu_c0_c0 = np.mean(S_c0_c0, 0)

            dists_c0_to_c0 = calc_cosine_distance_to_mean(S_c0_c0, mu_c0_c0)
            # dists_c0_to_c0 = np.array([calc_lid(it, S_c0_c0) for it in S_c0_c0])

            hist_c0_to_c0, _ = np.histogram(dists_c0_to_c0, bins, density=True)

            for c1 in range(c0 + 1, 10):
                S_c1_c0 = features[epoch, true_false_indices[c1][c0],
                          :]  # elements that are marked as c0, but belong to c1
                S_c1_c1 = features[epoch, true_false_indices[c1][c1],
                          :]  # elements that are marked as c1 and actually belong to it
                mu_c1_c1 = np.mean(S_c1_c1, 0)

                dists_c1_to_c1 = calc_cosine_distance_to_mean(S_c1_c1, mu_c1_c1)
                # dists_c1_to_c1 = np.array([calc_lid(it, S_c1_c1) for it in S_c1_c1])
                hist_c1_to_c1, _ = np.histogram(dists_c1_to_c1, bins, density=True)

                dists_c10_to_c0 = calc_cosine_distance_to_mean(S_c1_c0, mu_c0_c0)
                # dists_c10_to_c0 = np.array([calc_lid(it, S_c0_c0) for it in S_c1_c0])
                dists_c10_to_c1 = calc_cosine_distance_to_mean(S_c1_c0, mu_c1_c1)
                # dists_c10_to_c1 = np.array([calc_lid(it, S_c1_c1) for it in S_c1_c0])
                hist_c10_to_c0, _ = np.histogram(dists_c10_to_c0, bins, density=True)
                hist_c10_to_c1, _ = np.histogram(dists_c10_to_c1, bins, density=True)

                plt.title(str(c0) + ' ' + str(c1))
                # plt.bar(bins[:-1] + step / 2, hist_c0_to_c0, align='center', width=step, alpha=0.8, label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c0, c0, c0, c0))
                # plt.bar(bins[:-1] + step / 2, hist_c10_to_c0, align='center', width=step, alpha=0.5, label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c1, c0, c0, c0))
                plt.bar(bins[:-1] + step / 2, hist_c1_to_c1, align='center', width=step, alpha=0.8, label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c1, c1, c1, c1))
                plt.bar(bins[:-1] + step / 2, hist_c10_to_c1, align='center', width=step, alpha=0.5, label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c1, c0, c1, c1))

                plt.grid()
                plt.legend()
                plt.show()
                # plt.savefig('figs/lid/' + str(c0) + ' ' + str(c1))

                plt.cla()


def test_distances_with_predictions():
    dataset_name = 'cifar-10'

    X, Y = read_dataset(name=dataset_name, type='train')
    X1, Y1 = read_dataset(name=dataset_name, type='train25')

    model_name = '25_lam_1e-2_lr_1e-2_v2'
    features = np.load('lid_features/' + dataset_name + '/' + model_name + '.npy')
    logits = np.load('logits/' + dataset_name + '/' + model_name + '.npy')
    n_epochs = features.shape[0]
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    indices = [set() for i in range(10)]  # indices of elements marked as cls
    true_indices = [set() for i in range(10)]  # indices of elements actually from cls
    true_false_indices = [[set() for j in range(10)] for i in range(10)]  # arr[c1][c2] indices of elements from c1 but marked as c2
    for i in range(N):
        indices[y1[i]].add(i)
        true_indices[y0[i]].add(i)
        true_false_indices[y0[i]][y1[i]].add(i)

    def internal_distance_to_mean(X, mu, S):
        # return calc_dist_to_mean(X, mu)
        return calc_cosine_distance_to_mean(X, mu)

    # epochs = list(range(n_epochs))
    epochs = [10]
    for epoch in epochs:
        print('epoch', epoch)

        pred_indices = [set() for c in range(10)]
        for i in range(N):
            pred = int(np.argmax(logits[epoch, i, :]))
            pred_indices[pred].add(i)
        pred_indices = true_indices

        step = 0.025
        bins = np.arange(0, 1, step)
        density = True

        for c0 in range(10):
            ind_c0_c0 = list(pred_indices[c0].intersection(indices[c0]))  # elements marked as c1 and predicted the same
            S_c0_c0 = features[epoch, ind_c0_c0, :]
            mu_c0_c0 = np.mean(S_c0_c0, 0)

            dists_c0_to_c0 = internal_distance_to_mean(S_c0_c0, mu_c0_c0, S_c0_c0)
            hist_c0_to_c0, _ = np.histogram(dists_c0_to_c0, bins, density=density)

            for c1 in range(c0 + 1, 10):
                ind_c1_c1 = list(pred_indices[c1].intersection(indices[c1]))  # elements marked as c1 and predicted the same
                S_c1_c1 = features[epoch, ind_c1_c1, :]
                mu_c1_c1 = np.mean(S_c1_c1, 0)

                ind_c1_c0 = list(pred_indices[c1].intersection(indices[c0]))  # elements marked as c0, but predicted as c1
                S_c1_c0 = features[epoch, ind_c1_c0, :]

                plt.title(str(c0) + ' ' + str(c1))
                if bool(1):
                    dists_c10_to_c0 = internal_distance_to_mean(S_c1_c0, mu_c0_c0, S_c0_c0)

                    hist_c10_to_c0, _ = np.histogram(dists_c10_to_c0, bins, density=density)

                    plt.bar(bins[:-1] + step / 2, hist_c0_to_c0, align='center', width=step, alpha=0.8,
                            label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c0, c0, c0, c0))
                    plt.bar(bins[:-1] + step / 2, hist_c10_to_c0, align='center', width=step, alpha=0.5,
                            label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c1, c0, c0, c0))

                    # def cdf(x):
                    #     return np.sum(hist_c0_to_c0[:int(round(x/step))]) / len(S_c0_c0)
                    #
                    # weights = []
                    # for it in dists_c10_to_c0:
                    #     cdf_it = cdf(it)
                    #     weight = min(2 * (1 - cdf_it), 2 * cdf_it)
                    #     weights.append(weight)

                    # dists_c0_to_c0_1 = dists_c0_to_c0[np.random.choice(len(S_c0_c0), len(dists_c10_to_c0), False)]
                    # em_data = np.append(dists_c0_to_c0_1, dists_c10_to_c0).reshape((-1, 1))
                    # np.random.shuffle(em_data)
                    # mus, sigmas, pi, conf = em_solve(2, em_data, threshold=0.1,
                    #                                  mu_0=[[np.mean(dists_c0_to_c0)], [np.mean(dists_c10_to_c0)]])
                    # confs = e_step(2, mus, sigmas, pi, dists_c10_to_c0.reshape((-1, 1)))
                    # weights = 1 - confs[:, 1]

                    # step_w = 0.01
                    # bins_w = np.arange(0, 1, step_w)
                    # hist_w, _ = np.histogram(weights, bins_w, density=True)
                    # plt.bar(bins_w[:-1] + step_w/2, hist_w, align='center', width=step_w)
                else:
                    dists_c1_to_c1 = internal_distance_to_mean(S_c1_c1, mu_c1_c1, S_c1_c1)
                    dists_c10_to_c1 = internal_distance_to_mean(S_c1_c0, mu_c1_c1, S_c1_c1)

                    hist_c1_to_c1, _ = np.histogram(dists_c1_to_c1, bins, density=density)
                    hist_c10_to_c1, _ = np.histogram(dists_c10_to_c1, bins, density=density)

                    plt.bar(bins[:-1] + step / 2, hist_c1_to_c1, align='center', width=step, alpha=0.8,
                            label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c1, c1, c1, c1))
                    plt.bar(bins[:-1] + step / 2, hist_c10_to_c1, align='center', width=step, alpha=0.5,
                            label='{TL=%d,FL=%d} to {TL=%d,FL=%d}' % (c1, c0, c1, c1))

                plt.grid()
                plt.legend()
                plt.show()
                # plt.savefig('figs/distance_m/' + str(c0) + ' ' + str(c1))

                plt.cla()


def test_distances_based_weights():
    dataset_name = 'cifar-10'
    X, Y = read_dataset(name=dataset_name, type='train')
    X1, Y1 = read_dataset(name=dataset_name, type='train25')

    model_name = '25_lam_1e-2_lr_1e-2_v2'
    features = np.load('lid_features/' + dataset_name + '/' + model_name + '.npy')
    logits = np.load('logits/' + dataset_name + '/' + model_name + '.npy')
    n_epochs = features.shape[0]
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    indices = [set() for i in range(10)]  # indices of elements marked as cls
    for i in range(N):
        indices[y1[i]].add(i)

    err = [[] for i in range(7)]

    # epochs = list(range(n_epochs))
    epochs = list(range(5, n_epochs))
    # epochs = [15]
    for epoch in epochs:
        print('epoch', epoch)

        pred_indices = [set() for c in range(10)]
        for i in range(N):
            pred = int(np.argmax(logits[epoch, i, :]))
            pred_indices[pred].add(i)

        means = []
        S = []
        # sigma = np.zeros((d, d))
        for c0 in range(10):
            ind_c0_c0 = list(pred_indices[c0].intersection(indices[c0]))  # elements marked as c0 and predicted the same
            S.append(features[epoch, ind_c0_c0, :])
            mu_c0_c0 = np.mean(S[c0], 0)
            means.append(mu_c0_c0)
        #
        #     for it in S[c0]:
        #         diff = it - mu_c0_c0
        #         sigma += np.matmul(diff.reshape((-1, 1)), diff)
        # sigma /= sum([len(S[c]) for c in range(10)])
        # inv_sigma = np.linalg.inv(sigma)
        means = np.array(means)
        # for i in range(10):
        #     print(means[i, :])

        def softmax(x, w=None):
            if w is None:
                w = np.ones(len(x))
            sm = np.sum(np.exp(x) * w)
            return (np.exp(x) * w) / sm

        def err_f(x, y):
            # return cross_entropy(x, y)
            return 1 - np.dot(x, y)

        def one_hot(c):
            a = np.zeros(10)
            a[c] = 1
            return a

        def get_distances(i, mode=2, distance_scale=d ** 0.5):
            x = features[epoch, i, :]

            c0 = int(np.argmax(logits[epoch, i, :]))
            c1 = y1[i]

            distances = None

            if mode == 0:
                distances = calc_dist_to_mean(means, x)
            elif mode == 1:
                distances = np.array([calc_lid(x, S[c]) for c in range(10)])
            elif mode == 2:
                distances = np.ones(10) * distance_scale
                distances[c0] = np.linalg.norm(x - means[c0])
                distances[c1] = np.linalg.norm(x - means[c1])
            elif mode == 3:
                distances = np.ones(10) * distance_scale
                distances[c0] = calc_lid(x, S[c0])
                distances[c1] = calc_lid(x, S[c1])
            elif mode == 4:
                distances = calc_cosine_distance_to_mean(means, x, False)
            elif mode == 5:
                distances = np.ones(10) * distance_scale
                distances[c0] = calc_cosine_distance_to_mean(x.reshape((1, -1)), means[c0])
                distances[c1] = calc_cosine_distance_to_mean(x.reshape((1, -1)), means[c1])
            elif mode == 6:
                distances = np.ones(10) * distance_scale
                d0 = x - means[c0]
                distances[c0] = np.matmul(d0, np.matmul(inv_sigma, d0.reshape((-1, 1))))
                d1 = x - means[c1]
                distances[c1] = np.matmul(d1, np.matmul(inv_sigma, d1.reshape((-1, 1))))

            # distances_to_mean = distances_to_mean / np.max(distances_to_mean) * distance_scale

            return distances

        cnt = 0
        err_sm = [0]*len(err)
        # for i in range(N):
        for i in np.random.choice(N, 10000, False):
            lgts = logits[epoch, i, :]
            c0 = int(np.argmax(lgts))
            c1 = y1[i]

            one_hot_c0 = one_hot(c0)

            cmp_distances = get_distances(i, 4, d)
            # soft_max_dist = get_distances(i, 0, d)
            soft_max_dist = cmp_distances
            # cmp_distances = soft_max_dist

            w = int(cmp_distances[c0] < cmp_distances[c1])
            new_labels1 = one_hot_c0 * w + Y1[i] * (1 - w)

            min_dist_c = np.argmin(cmp_distances)
            one_hot_min_dist = one_hot(min_dist_c)

            w = int(cmp_distances[min_dist_c] < cmp_distances[c1])
            new_labels2 = one_hot_min_dist * w + Y1[i] * (1 - w)

            w = softmax(- np.array([soft_max_dist[min_dist_c], soft_max_dist[c1]]))
            new_labels3 = one_hot_min_dist * w[0] + Y1[i] * w[1]

            w = softmax(- np.array([soft_max_dist[c0], soft_max_dist[c1]]))
            new_labels4 = one_hot_c0 * w[0] + Y1[i] * w[1]

            # w = softmax(- np.array([soft_max_dist[c0], soft_max_dist[c1], soft_max_dist[min_dist_c]]))
            # new_labels5 = one_hot_c0 * w[0] + Y1[i] * w[1] + one_hot_min_dist * w[2]
            #
            # w = softmax(np.array([lgts[c0], lgts[c1], lgts[min_dist_c]]))
            # new_labels6 = one_hot_c0 * w[0] + Y1[i] * w[1] + one_hot_min_dist * w[2]

            err_sm[1] += err_f(new_labels1, Y[i])
            err_sm[2] += err_f(new_labels2, Y[i])
            err_sm[3] += err_f(new_labels3, Y[i])
            err_sm[4] += err_f(new_labels4, Y[i])
            # err_sm[5] += err_f(new_labels5, Y[i])
            # err_sm[6] += err_f(new_labels6, Y[i])

            c_tr = np.argmax(Y[i])
            err_sm[0] += 1 - int(min_dist_c == c_tr or c0 == c_tr or c1 == c_tr)

            cnt += 1

        for i in range(len(err)):
            err[i].append(err_sm[i]/cnt)

    for i in range(len(err)):
        plt.plot(err[i], '.-', label=str(i + 1))

    plt.legend()
    plt.grid()
    plt.show()


def test_search_for_scale():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    model_name = 'model25_relu'
    features = np.load('lid_features/' + model_name + '.npy')
    logits = np.load('logits/' + model_name + '.npy')
    n_epochs = features.shape[0]
    N = features.shape[1]
    d = features.shape[2]

    # y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    epoch = 15

    indices = [set() for i in range(10)]  # indices of elements marked as cls
    pred_indices = [set() for c in range(10)]
    for i in range(N):
        indices[y1[i]].add(i)

        pred = int(np.argmax(logits[epoch, i, :]))
        pred_indices[pred].add(i)

    means = []
    S = []
    for c0 in range(10):
        ind_c0_c0 = list(pred_indices[c0].intersection(indices[c0]))  # elements marked as c0 and predicted the same
        S.append(features[epoch, ind_c0_c0, :])
        mu_c0_c0 = np.mean(S[c0], 0)
        means.append(mu_c0_c0)
    means = np.array(means)

    def softmax(x):
        exp = np.exp(x)
        return exp / np.sum(exp)

    def F(scale):
        err_sm = 0
        for i in range(N):
            c0 = int(np.argmax(logits[epoch, i, :]))
            c1 = y1[i]

            one_hot_pred = np.zeros(10)
            one_hot_pred[c0] = 1

            x = features[epoch, i, :].reshape((1, -1))

            distances = np.array([
                calc_cosine_distance_to_mean(x, means[c0]),
                calc_cosine_distance_to_mean(x, means[c1])
            ])

            w = softmax(-distances * scale)

            new_label = one_hot_pred * w[0] + Y1[i] * w[1]

            err_sm += 1 - np.dot(new_label, Y[i])

        err_sm /= N

        return err_sm

    l = 0
    r = d

    while r - l > 1e-1:
        mid1 = l + (r - l)/3
        mid2 = r - (r - l)/3
        res1 = F(mid1)
        res2 = F(mid2)

        print(l, r, res1, res2)

        if res1 < res2:
            r = mid1
        else:
            l = mid2


def test_mean_distances_change():
    dataset_name = 'cifar-10'
    # dataset_name = 'mnist'
    X, Y = read_dataset(name=dataset_name, type='train')
    X1, Y1 = read_dataset(name=dataset_name, type='train25')

    model_name = '25_lam_1e-3_lr_times_1e-1_augmented_try2'
    # model_name = 'clean_lam_1e-3_lr_times_1e-1_aug'
    # model_name = 'model25_relu'
    features = np.load('lid_features/' + dataset_name + '/' + model_name + '.npy')
    logits = np.load('logits/' + dataset_name + '/' + model_name + '.npy')

    # np.save('lid_features/' + dataset_name + '/' + model_name + '_', features.reshape((-1, 50000, 256)))
    # np.save('pre_lid_features/' + dataset_name + '/' + model_name + '_', features.reshape((-1, 50000, 256)))
    # np.save('logits/' + dataset_name + '/' + model_name + '_', logits.reshape((-1, 50000, 10)))
    # exit(0)

    n_epochs = features.shape[0]
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    if bool(0):
        distances_to_mean = np.empty((N, n_epochs, 10))
        for epoch in range(n_epochs):
            print(epoch)

            indices = [set() for i in range(10)]  # indices of elements marked as cls
            pred_indices = [set() for c in range(10)]
            for i in range(N):
                indices[y1[i]].add(i)

                pred = int(np.argmax(logits[epoch, i, :]))
                # pred = y0[i]
                pred_indices[pred].add(i)

            means = []
            S = []
            for c in range(10):
                ind_c_c = list(pred_indices[c].intersection(indices[c]))  # elements marked as c and predicted the same
                # ind_c_c = list(pred_indices[c])
                S.append(features[epoch, ind_c_c, :])
                mu_c_c = np.mean(S[c], 0)
                means.append(mu_c_c)
            means = np.array(means)

            for i in range(N):
                # dist = calc_cosine_distance_to_mean(means, features[epoch, i, :])
                # dist = calc_dist_to_mean(means, features[epoch, i, :])
                dist = np.array([calc_lid(features[epoch, i, :], S[c]) for c in range(10)])
                distances_to_mean[i, epoch, :] = dist
        np.save('distances_to_mean/' + dataset_name + '/' + model_name + '/lids', distances_to_mean)
    else:
        from mpl_toolkits.mplot3d import Axes3D
        # distances_to_mean = np.load('distances_to_mean/distances_clean.npy')
        distances_to_mean = np.load('distances_to_mean/' + dataset_name + '/' + model_name + '/cosine_distances.npy')
        # distances_to_mean = np.load('distances_to_mean/euclid_distances_clean.npy')
        # distances_to_mean = np.load('distances_to_mean/euclid_distances.npy')
        # distances_to_mean = np.load('distances_to_mean/lids_clean.npy')
        # distances_to_mean = np.load('distances_to_mean/' + dataset_name + '/' + model_name + '/lids.npy')

        c0 = 0
        c1 = 1

        # epochs = range(n_epochs)
        epochs = list(range(3, n_epochs))

        sample = [np.empty((0, 10)) for epoch in epochs]
        for epoch in epochs:
            for i in range(N):
                # pred = int(np.argmax(logits[epoch, i, :]))
                tr = y0[i]
                label = y1[i]

                # if pred == c0 and label == c1:
                if tr == c0 and label == c1:
                # if pred == c0:
                # if y0[i] == c0:
                # if label == c1:
                    # sample[epoch, i] = distances_to_mean[i, epoch, c1]  # distance to label class centroid
                    # sample[epoch, i] = distances_to_mean[i, epoch, 1]  # distance to predicted class centroid
                    d = distances_to_mean[i, epoch, :]

                    sample[epoch - epochs[0]] = np.vstack((sample[epoch - epochs[0]], d))

        st_devs = [np.std(sample[epoch - epochs[0]], 0) for epoch in epochs]
        st_devs = np.array(st_devs)

        means = [np.mean(sample[epoch - epochs[0]], 0) for epoch in epochs]
        means = np.array(means)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        for c in range(10):
            color = 'C0'
            if c == c0:
                color = 'C1'
            elif c == c1:
                color = 'C2'

            plt.plot([c] * len(epochs), epochs, means[:, c], color=color)
            plt.plot([c] * len(epochs), epochs, means[:, c] - st_devs[:, c], color=color, alpha=0.25)
            plt.plot([c] * len(epochs), epochs, means[:, c] + st_devs[:, c], color=color, alpha=0.25)

        plt.show()

        # while True:
        #     i = np.random.randint(N)
        #     print(y0[i], y1[i])
        #     means = distances_to_mean[i, :, :]
        #
        #     fig = plt.figure()
        #     ax = fig.gca(projection='3d')
        #
        #     for c in range(10):
        #         plt.plot([c] * n_epochs, np.arange(n_epochs), means[:, c], color='C0')
        #
        #     plt.show()
        #     plt.cla()


def test_distance_to_first():
    X, Y = read_dataset('train')
    X1, Y1 = read_dataset('train25')

    model_name = 'model25_relu'
    features = np.load('lid_features/' + model_name + '.npy')
    # features = np.load('feature_matrices_old/model_none.npy')
    logits = np.load('logits/' + model_name + '.npy')
    n_epochs = features.shape[0]
    N = features.shape[1]
    d = features.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    # distances_to_first = np.empty((N, n_epochs))
    # for epoch in range(n_epochs):
    #     print(epoch)
    #
    #     for i in range(N):
    #         # dist = calc_cosine_distance_to_mean(features[epoch, i, :].reshape(1, -1), features[0, i, :])
    #         dist = calc_dist_to_mean(features[epoch, i, :].reshape(1, -1), features[0, i, :])
    #         distances_to_first[i, epoch] = dist[0]
    # np.save('distances_to_first/euclid_distances', distances_to_first)

    distances_to_first = np.load('distances_to_first/cos_distances_clean.npy')
    # distances_to_first = np.load('distances_to_first/cos_distances.npy')
    # distances_to_first = np.load('distances_to_first/euclid_distances_clean.npy')
    # distances_to_first = np.load('distances_to_first/euclid_distances.npy')
    # distances_to_first = np.load('distances_to_first/lids_clean.npy')
    # distances_to_first = np.load('distances_to_first/lids.npy')

    c0 = 2
    c1 = 2

    sample = [[] for epoch in range(n_epochs)]
    for epoch in range(n_epochs):
        for i in range(N):
            pred = int(np.argmax(logits[epoch, i, :]))
            label = y1[i]

            # if pred == c0 and label == c1:
            # if pred == c0:
            if y0[i] == c0:
            # if label == c1:
                # sample[epoch, i] = distances_to_first[i, epoch, c1]  # distance to label class centroid
                # sample[epoch, i] = distances_to_first[i, epoch, 1]  # distance to predicted class centroid
                d = distances_to_first[i, epoch]

                sample[epoch].append(d)

    means = [np.mean(sample[epoch]) for epoch in range(n_epochs)]
    means = np.array(means)

    plt.plot(means)

    plt.tight_layout()
    plt.grid()
    plt.show()


def test_logits_accuracy():
    dataset_name = 'cifar-10'
    # dataset_name = 'mnist'
    X, Y = read_dataset(name=dataset_name, type='train')
    X1, Y1 = read_dataset(name=dataset_name, type='train25')

    model_name = '25_lam_1e-3_lr_times_1e-1_augmented_try2'
    # model_name = 'model25_relu'
    logits = np.load('logits/' + dataset_name + '/' + model_name + '.npy')
    distances_to_mean = np.load('distances_to_mean/' + dataset_name + '/' + model_name + '/cosine_distances.npy')
    # distances_to_mean = np.load('distances_to_mean/' + dataset_name + '/lids.npy')

    n_epochs = logits.shape[0]
    N = logits.shape[1]
    d = logits.shape[2]

    y0 = [int(np.argmax(Y[it])) for it in range(N)]
    y1 = [int(np.argmax(Y1[it])) for it in range(N)]

    epochs = list(range(n_epochs))
    for epoch in epochs:
        acc_sm = np.zeros(1)
        cnt_sm = np.zeros(len(acc_sm))
        for i in range(N):
            lgts = logits[epoch, i, :]
            preds = [[c, lgts[c]] for c in range(10)]
            preds = list(sorted(preds, key=lambda it: -it[1]))
            preds_ind = [it[0] for it in preds]
            # preds_val = [it[1] for it in preds]

            dists = [[c, distances_to_mean[i, epoch, c]] for c in range(10)]
            dists = list(sorted(dists, key=lambda it: it[1]))
            dists_ind = [it[0] for it in dists]
            # dists_val = [it[1] for it in dists]

            # if preds_ind[0] == y1[i]:
            #     acc_sm[0] += int(preds_ind[0] == y0[i])
            #     cnt_sm[0] += 1
            # else:
            #     for j in range(1, 4):
            #         acc_sm[j] += int(y0[i] in preds_ind[:j])
            #         # acc_sm[j] += int(y0[i] in preds_ind[:j] or y0[i] in dists_ind[:j])
            #         cnt_sm[j] += 1

            if y0[i] != y1[i]:
                for j in range(0, 1):
                    acc_sm[j] += int(y0[i] in preds_ind[:j + 1])
                    # acc_sm[j] += int(y0[i] in dists_ind[:j + 1])
                    # acc_sm[j] += int(y0[i] in preds_ind[:j + 1] or y0[i] in dists_ind[:j + 1])
                    cnt_sm[j] += 1

            # pr = np.argmax(logits[epoch, i, :])
            #
            # if pr == y1[i]:
            #     acc_sm[0] += int(pr == y0[i])
            #     cnt_sm[0] += 1

        accs = acc_sm / cnt_sm

        # print(epoch + 1, list(np.round(accs, 3)))
        print('[', epoch + 1, ',', list(np.round(accs, 3))[0], '],')


if __name__ == '__main__':
    # test_single_epoch(20)
    # test_LID_with_epoch_progress()
    # test_weights_with_epoch_progress()
    # test_weights_for_single_element_with_epoch_progress()
    # test_distr()
    # test_distr_per_class()
    # test_mahalanobis()
    # test_lid_em()
    # test_relative_lid_calc_clean()
    # test_relative_lid_calc_noise()
    # test_distances()
    # test_distances_with_predictions()
    # test_distances_based_weights()
    # test_search_for_scale( )
    test_mean_distances_change()
    # test_distance_to_first()
    # test_logits_accuracy()
