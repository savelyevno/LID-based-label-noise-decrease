import os
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf


input_folder_prefix = '../mnist/'
output_folder_prefix = 'datasets/'


def read_image(image_path):
    return Image.open(image_path).convert('L')


def mnist_image_to_data(img):
    return np.reshape(np.float32(img) / 256, (1, 784))


def mnist_read_images(foler):
    X = np.empty((0, 784))
    Y = np.empty((0, 10))
    for subdir, dirs, files in os.walk(input_folder_prefix + foler):
        print(subdir)
        for file in files:
            label = subdir[-1]

            try:
                label = int(label)
            except:
                continue

            image_path = os.path.join(subdir, file)

            img = read_image(image_path)
            x = mnist_image_to_data(img)
            # X = np.vstack((X, x))
            X = np.append(X, x, 0)

            y = np.zeros((1, 10))
            y[0, label] = 1
            # Y = np.vstack((Y, y))
            Y = np.append(Y, y, 0)

    return X, Y


def write_dataset(name, type, X, Y):
    with open(output_folder_prefix + name + '/' + type + '.pkl', 'wb') as file:
        pickle.dump((X, Y), file)


def read_dataset(name, type):
    with open(output_folder_prefix + name + '/' + type + '.pkl', 'rb') as file:
        (X, Y) = pickle.load(file)
    return X, Y


def mnist_read_images_write_dataset(folder, output_name):
    X, Y = mnist_read_images(folder)
    print('done reading')
    write_dataset(output_name, X, Y)
    print('done writing')


def cifar10_convert():
    dataset_name = 'cifar-10'

    def append_data(filename, X, Y):
        with open('datasets/{}/raw/{}'.format(dataset_name, filename), 'rb') as f:
            dict_data = pickle.load(f, encoding='bytes')
            X = np.array(np.vstack((X, dict_data[b'data'])), dtype=np.float32)
            y = dict_data[b'labels']

        T = np.zeros((len(y), 10))
        T[np.arange(len(y)), y] = 1
        Y = np.vstack((Y, T))

        return X, Y

    X_train = np.empty((0, 3072))
    Y_train = np.empty((0, 10), dtype=int)
    for i in range(1, 5 + 1):
        X_train, Y_train = append_data('data_batch_' + str(i), X_train, Y_train)

    X_test = np.empty((0, 3072))
    Y_test = np.empty((0, 10), dtype=int)
    X_test, Y_test = append_data('test_batch', X_test, Y_test)

    X_train = np.transpose(X_train.reshape((-1, 3, 32, 32)), [0, 2, 3, 1]) / 255
    mean = np.mean(X_train, axis=0)
    X_train -= mean

    X_test = np.transpose(X_test.reshape((-1, 3, 32, 32)), [0, 2, 3, 1]) / 255
    X_test -= mean

    with open('datasets/{}/train.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump((X_train, Y_train), f)
    with open('datasets/{}/test.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump((X_test, Y_test), f)


def cifar100_convert():
    dataset_name = 'cifar-100'

    def read_data(filename):
        with open('datasets/{}/raw/{}'.format(dataset_name, filename), 'rb') as f:
            dict_data = pickle.load(f, encoding='bytes')
            X = np.array(dict_data[b'data'], dtype=np.float32)
            y = dict_data[b'fine_labels']

        Y = np.zeros((len(y), 100))
        Y[np.arange(len(y)), y] = 1

        return X, Y

    X_train, Y_train = read_data('train')
    X_test, Y_test = read_data('test')

    X_train = np.transpose(X_train.reshape((-1, 3, 32, 32)), [0, 2, 3, 1]) / 255
    mean = np.mean(X_train, axis=0)
    X_train -= mean

    X_test = np.transpose(X_test.reshape((-1, 3, 32, 32)), [0, 2, 3, 1]) / 255
    X_test -= mean

    with open('datasets/{}/train.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump((X_train, Y_train), f)
    with open('datasets/{}/test.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump((X_test, Y_test), f)


def create_validation_set(dataset_name, n_samples_per_class, augmentation_multiplier, random_seed=0):
    X, Y = read_dataset(name=dataset_name, type='train')

    np.random.seed(random_seed)

    Y_ind = np.argmax(Y, 1)
    n_classes = Y_ind.max() + 1
    N = X.shape[0]

    Y_ind_per_class = [[] for c in range(n_classes)]
    for i in np.random.permutation(N):
        Y_ind_per_class[Y_ind[i]].append(i)

    val_Y_ind_per_class = [Y_ind_per_class[c][:n_samples_per_class] for c in range(n_classes)]
    val_ind = []
    for lst in val_Y_ind_per_class:
        val_ind.extend(lst)

    all_except_val_ind = []
    val_ind_set = set(val_ind)
    for i in range(N):
        if i not in val_ind_set:
            all_except_val_ind.append(i)

    X_new_train = X[all_except_val_ind]
    Y_new_train = Y[all_except_val_ind]

    X_val = X[val_ind]
    Y_val = Y[val_ind]

    X_val_res = np.copy(X_val)
    Y_val_res = np.copy(Y_val)

    if augmentation_multiplier > 0:
        data_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        data_augmenter.fit(X_val)
        X_augmented_iter = data_augmenter.flow(X_val, batch_size=X_val.shape[0], shuffle=False)

        for i in range(augmentation_multiplier):
            X_augmented = X_augmented_iter.next()

            X_val_res = np.append(X_val_res, X_augmented, axis=0)
            Y_val_res = np.append(Y_val_res, Y_val, axis=0)

    write_dataset(dataset_name, 'train_without_val', X_new_train, Y_new_train)
    write_dataset(dataset_name, 'validation', X_val, Y_val)


if __name__ == '__main__':
    # mnist_read_images_write_dataset('trainingSet', 'train')
    # mnist_read_images_write_dataset('testSet', 'test')
    # mnist_read_images_write_dataset('trainingSample', 'train_sample')

    # name = 'train'
    # X = np.load(name + '_data.npy')
    # Y = np.load(name + '_labels.npy')
    #
    # write_dataset(name, X, Y)

    # cifar10_convert()
    # cifar100_convert()

    create_validation_set('cifar-10', 100, 1)
