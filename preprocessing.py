import os
import numpy as np
from PIL import Image
import pickle


input_folder_prefix = '../mnist/'
output_folder_prefix = 'datasets/'


def read_image(image_path):
    return Image.open(image_path).convert('L')


def image_to_data(img):
    return np.reshape(np.float32(img) / 256, (1, 784))


def read_images(foler):
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
            x = image_to_data(img)
            # X = np.vstack((X, x))
            X = np.append(X, x, 0)

            y = np.zeros((1, 10))
            y[0, label] = 1
            # Y = np.vstack((Y, y))
            Y = np.append(Y, y, 0)

    return X, Y


def write_dataset(name, X, Y):
    with open(output_folder_prefix + name + '.pkl', 'wb') as file:
        pickle.dump((X, Y), file)


def read_images_write_dataset(folder, output_name):
    X, Y = read_images(folder)
    print('done reading')
    write_dataset(output_name, X, Y)
    print('done writing')


def read_dataset(name):
    with open(output_folder_prefix + name + '.pkl', 'rb') as file:
        (X, Y) = pickle.load(file)
    return X, Y


if __name__ == '__main__':
    read_images_write_dataset('trainingSet', 'train')
    # read_images_write_dataset('testSet', 'test')
    # read_images_write_dataset('trainingSample', 'train_sample')

    # name = 'train'
    # X = np.load(name + '_data.npy')
    # Y = np.load(name + '_labels.npy')
    #
    # write_dataset(name, X, Y)
