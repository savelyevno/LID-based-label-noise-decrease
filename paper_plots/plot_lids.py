import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import csv


figure(figsize=(8, 6), dpi=160)


def plot():
    # dataset_type = 'mnist'
    # dataset_type = 'cifar-10'
    dataset_type = 'cifar-100'

    # path = 'paper_plots/data/{}/lids/'.format(dataset_type)
    path = 'paper_plots/data/{}/lids_improved/'.format(dataset_type)

    data = []

    for sub_dir in ['0', '20', '40', '60']:
        sub_dir_path = os.path.join(path, sub_dir)
        for filename in os.listdir(sub_dir_path):
            if not filename.endswith('.csv'):
                continue

            full_path = os.path.join(sub_dir_path, filename)

            with open(full_path) as f:
                reader = csv.DictReader(f)

                new_data = []
                for row in reader:
                    new_data.append((float(row['Step']), float(row['Value'])))
                data.append(new_data)
            break

    data = np.array(data)
    n = 59000 if dataset_type == 'mnist' else 49000
    data[:, :, 0] /= (n / 128)

    plt.plot(data[0, :, 0], data[0, :, 1], label='0% noise')
    plt.plot(data[1, :, 0], data[1, :, 1], label='20% noise')
    plt.plot(data[2, :, 0], data[2, :, 1], label='40% noise')
    plt.plot(data[3, :, 0], data[3, :, 1], label='60% noise')

    plt.xlabel('Epoch')
    plt.ylabel('LID', )
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
