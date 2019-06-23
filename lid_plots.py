import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.datasets
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA


K = 50


def two_d_rotate(X, origin, angle):
    np_origin = np.array(origin).reshape((1, 2))
    return np_origin + np.matmul(X - np_origin, np.array([[np.cos(angle), - np.sin(angle)],
                                                          [np.sin(angle), np.cos(angle)]]))


def calc_ref_lids(X, ref_X, k=K):
    lids = []

    for x in X:
        x = np.copy(x).reshape((1, -1))

        distances = np.sqrt(np.sum(np.power(x - ref_X, 2), -1))

        sorted_indices = np.argsort(distances, -1)

        k_nearest = distances[sorted_indices[1: k + 1]]

        distance_ratios = k_nearest / k_nearest[-1]
        lid = - k / np.sum(np.log(distance_ratios + 1e-12), -1)

        lids.append(lid)

    return np.array(lids)


def calc_lids(x, k=K):
    n = len(x)

    norm_squared = np.reshape(np.sum(x * x, -1), (-1, 1))
    norm_squared_t = np.transpose(norm_squared)

    dot_products = np.matmul(x, np.transpose(x))

    distances_squared = np.maximum(norm_squared - 2 * dot_products + norm_squared_t, 0)
    distances = np.sqrt(distances_squared)
    sorted_indices = np.argsort(distances, -1)

    k_nearest = distances[np.repeat(np.arange(n), k), sorted_indices[:, 1: k + 1].reshape((-1,))].reshape((n, -1))

    distance_ratios = np.transpose(np.multiply(np.transpose(k_nearest), 1 / k_nearest[:, -1]))
    lids = - k / np.sum(np.log(distance_ratios + 1e-12), -1)

    return lids


def plot_points(X, marker='.', markersize=20, hollow_markers=False, color='C0', fix_wnd_size=True, show=True):
    dim = X.shape[1]

    if dim == 2:
        if fix_wnd_size:
            diff = (X[:, 0].max() - X[:, 0].min(), X[:, 1].max() - X[:, 1].min())
            fig_size = np.array(plt.figaspect(diff[1] / diff[0]))
            fig_size *= 9 / fig_size.max()
            plt.subplots(figsize=fig_size)

        if hollow_markers:
            plt.scatter(X[:, 0], X[:, 1], marker=marker, color=color, s=markersize, facecolors='none', edgecolors=color)
        else:
            plt.scatter(X[:, 0], X[:, 1], marker=marker, color=color, s=markersize)
    elif dim == 3:
        fig = plt.figure('')
        ax = fig.add_subplot(111, projection='3d')
        # plt.subplots(subplot_kw={'projection': '3d'})

        # for x in X:
        #     ax.scatter(x[0], x[1], x[2], marker='o')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, color=color, s=markersize)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    if show:
        plt.grid()
        plt.tight_layout()
        plt.show()


def presentation_demo():
    N = 1000
    std = 0.1
    noise_ratio = 0.4
    M = int(noise_ratio * N)

    mu_x1 = (0.35, 1.25)
    X1 = np.random.normal(mu_x1, std, size=(N - M, 2))
    X1 = two_d_rotate(X1, mu_x1, -35)

    mu_x2 = (1.4, 0.7)
    X2 = np.random.normal(mu_x2, std, size=(N - M, 2))
    X2 = two_d_rotate(X2, mu_x2, 50)

    X12 = np.append(X1, X2, 0)

    X3 = np.random.normal(mu_x2, std, size=(M, 2))
    X4 = np.random.normal(mu_x1, std, size=(M, 2))

    print(calc_lids(X12).mean())

    move_speed = 0.1
    for i in range(10):
        if i > 0:
            X3 = (1 - move_speed) * X3 + move_speed * np.array(mu_x1).reshape((1, 2))
            X4 = (1 - move_speed) * X4 + move_speed * np.array(mu_x2).reshape((1, 2))

        plt.xlim(-0.25, 1.75)
        plt.ylim(-0.25, 1.75)

        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')

        plot_points(X1, marker='o', hollow_markers=True, color='C1', fix_wnd_size=False, show=False)
        plot_points(X2, marker='s', hollow_markers=True, color='C0', fix_wnd_size=False, show=False)
        plot_points(X3, marker='.', hollow_markers=True, color='C1', fix_wnd_size=False, show=False)
        plot_points(X4, marker='.', hollow_markers=True, color='C0', fix_wnd_size=False, show=True)

        lids = calc_lids(np.vstack((X12, X3, X4)))
        lids12 = lids[:2 * N - 2 * M]
        print(lids.mean(), lids12.mean())


def presentation_demo2():
    Y = np.load('logs/ftr_mov/Y.npy')
    Y0 = np.load('logs/ftr_mov/Y_clean.npy')
    Y_cls_pair = np.load('logs/ftr_mov/Y_cls_pair.npy')

    is_sample_clean = np.equal(np.argmax(Y, -1), np.argmax(Y0, -1))
    clean_samples_ind = np.nonzero(is_sample_clean)[0]
    clean_samples_ind_set = set(clean_samples_ind)

    c1 = 0
    c2 = 1

    c1_ind = np.nonzero(Y_cls_pair == c1)[0]
    c2_ind = np.nonzero(Y_cls_pair == c2)[0]

    c11_ind = np.array(list(set(c1_ind).intersection(clean_samples_ind_set)), dtype=np.int)
    c12_ind = np.array(list(set(c1_ind).difference(clean_samples_ind_set)), dtype=np.int)
    c22_ind = np.array(list(set(c2_ind).intersection(clean_samples_ind_set)), dtype=np.int)
    c21_ind = np.array(list(set(c2_ind).difference(clean_samples_ind_set)), dtype=np.int)

    for i in range(0, 51):
        X = np.load('logs/ftr_mov/X_{}.npy'.format(i))

        X11 = X[c11_ind]
        X12 = X[c12_ind]
        X21 = X[c21_ind]
        X22 = X[c22_ind]

        pca = PCA(n_components=2)
        pca.fit(X)
        X11_pca = pca.transform(X11)
        X12_pca = pca.transform(X12)
        X21_pca = pca.transform(X21)
        X22_pca = pca.transform(X22)

        plot_points(X11_pca, marker='^', hollow_markers=True, color='C0', fix_wnd_size=False, show=False)
        plot_points(X22_pca, marker='o', hollow_markers=True, color='C1', fix_wnd_size=False, show=False)
        plot_points(X12_pca, marker='.', hollow_markers=True, color='C1', fix_wnd_size=False, show=False)
        plot_points(X21_pca, marker='.', hollow_markers=True, color='C0', fix_wnd_size=False, show=False)

        plt.show()


if __name__ == '__main__':
    N = 1000
    noise = 1e-2

    #
    # Random uniform
    #

    # X = np.random.random((N, 3))

    #
    # Circle
    #

    # angles = np.random.random((N, 1)) * 2 * np.pi
    # X = np.hstack((np.cos(angles), np.sin(angles)))

    #
    # Sphere
    #

    # radius = 1
    # angles1 = np.arccos(1 - 2 * np.random.random((N, 1)))
    # angles2 = np.random.random((N, 1)) * 2 * np.pi
    # X = np.hstack((np.sin(angles1) * np.cos(angles2),
    #                np.sin(angles1) * np.sin(angles2),
    #                np.cos(angles1))) * radius

    #
    # Swiss roll
    #
    # X = sklearn.datasets.make_swiss_roll(N)[0]
    # rot = Rotation.from_euler('xyz', angles=[90, 45, 45], degrees=True)
    # X = rot.apply(X)

    #
    # Add noise
    #

    # X += np.random.normal(scale=noise, size=X.shape)

    # print(calc_lids(X).mean())
    # print(calc_lid(np.array((0, 0)), x))
    # plot_points(X)


    # presentation_demo()
    presentation_demo2()