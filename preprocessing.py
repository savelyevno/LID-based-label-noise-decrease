import os
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt


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


def cifar_10_convert():
    def append_data(filename, X, Y):
        with open('datasets/cifar-10/raw/' + filename, 'rb') as f:
            dict_data = pickle.load(f, encoding='bytes')
            X = np.vstack((X, dict_data[b'data']))
            y = dict_data[b'labels']

        T = np.zeros((len(y), 10))
        T[np.arange(len(y)), y] = 1
        Y = np.vstack((Y, T))

        return X, Y

    X_train = np.empty((0, 3072))
    Y_train = np.empty((0, 10), dtype=int)
    for i in range(1, 6):
        X_train, Y_train = append_data('data_batch_' + str(i), X_train, Y_train)

    X_test = np.empty((0, 3072))
    Y_test = np.empty((0, 10), dtype=int)
    X_test, Y_test = append_data('test_batch', X_test, Y_test)

    def proccess_X(X):
        X = np.transpose(X.reshape((-1, 3, 32, 32)), [0, 2, 3, 1]) / 255.0
        mean = np.mean(X, axis=0)
        # std = np.std(X)
        X -= mean

        return X

    X_train = proccess_X(X_train)
    X_test = proccess_X(X_test)

    with open('datasets/cifar-10/train.pkl', 'wb') as f:
        pickle.dump((X_train, Y_train), f)
    with open('datasets/cifar-10/test.pkl', 'wb') as f:
        pickle.dump((X_test, Y_test), f)


if __name__ == '__main__':
    # mnist_read_images_write_dataset('trainingSet', 'train')
    # mnist_read_images_write_dataset('testSet', 'test')
    # mnist_read_images_write_dataset('trainingSample', 'train_sample')

    # name = 'train'
    # X = np.load(name + '_data.npy')
    # Y = np.load(name + '_labels.npy')
    #
    # write_dataset(name, X, Y)

    cifar_10_convert()
