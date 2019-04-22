import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import csv


figure(figsize=(8, 6), dpi=160)


def plot():
    path = 'paper_plots/data/cifar-10'
    # path = 'paper_plots/data/cifar-100'

    data = []

    for sub_dir in ['20', '40', '60']:
    # for sub_dir in ['20', '40']:
        sub_dir_data = []

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
                sub_dir_data.append(new_data)

        # if sub_dir == '40':
        #     sub_dir_data.append(sub_dir_data[-1])

        data.append(sub_dir_data)

    data = np.array(data)

    meaned_data = np.mean(data, 1)
    meaned_data[:, :, 0] /= (49000 / 128)

    plt.plot(meaned_data[0, 20:, 0], meaned_data[0, 20:, 1], label='20% noise')
    plt.plot(meaned_data[1, 20:, 0], meaned_data[1, 20:, 1], label='40% noise')
    plt.plot(meaned_data[2, 20:, 0], meaned_data[2, 20:, 1], label='60% noise')
    plt.axvline(x=40, linestyle='--', color='r', linewidth=1)
    plt.axvline(x=80, linestyle='--', color='r', linewidth=1)

    # plt.plot(meaned_data[0, 60:, 0], meaned_data[0, 60:, 1], label='20% noise')
    # plt.plot(meaned_data[1, 60:, 0], meaned_data[1, 60:, 1], label='40% noise')
    # plt.plot(meaned_data[2, 20:, 0], meaned_data[2, 20:, 1], label='60% noise')
    # plt.axvline(x=80, linestyle='--', color='r', linewidth=1)
    # plt.axvline(x=120, linestyle='--', color='r', linewidth=1)
    # plt.axvline(x=160, linestyle='--', color='r', linewidth=1)

    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy', )
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
