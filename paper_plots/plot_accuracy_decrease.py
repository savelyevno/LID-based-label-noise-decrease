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

    # path = 'paper_plots/data/{}/accuracy_decrease/'.format(dataset_type)
    path = 'paper_plots/data/{}/accuracy_decrease_improved/'.format(dataset_type)

    data = []

    for sub_dir in ['0', '20', '40', '60']:
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

        if sub_dir == '60':
            # sub_dir_data.append(sub_dir_data[-1])
            lst = [(0, 0)]*(len(data[-1][0]) - len(sub_dir_data[0]))
            sub_dir_data[0].extend(lst)
        data.append(sub_dir_data)

    data = np.array(data)

    meaned_data = np.mean(data, 1)
    meaned_data[:, :, 0] /= (49000 / 128)

    if dataset_type == 'mnist':
        plt.plot(meaned_data[0, :, 0], meaned_data[0, :, 1], label='0% noise')
        plt.plot(meaned_data[1, :, 0], meaned_data[1, :, 1], label='20% noise')
        plt.plot(meaned_data[2, :, 0], meaned_data[2, :, 1], label='40% noise')
        plt.plot(meaned_data[3, :, 0], meaned_data[3, :, 1], label='60% noise')
        plt.axvline(x=20, linestyle='--', color='r', linewidth=1)
        plt.axvline(x=40, linestyle='--', color='r', linewidth=1)
    elif dataset_type == 'cifar-10':
        st = 0
        plt.plot(meaned_data[0, st:, 0], meaned_data[0, st:, 1], label='0% noise')
        plt.plot(meaned_data[1, st:, 0], meaned_data[1, st:, 1], label='20% noise')
        plt.plot(meaned_data[2, st:, 0], meaned_data[2, st:, 1], label='40% noise')
        plt.plot(meaned_data[3, st:, 0], meaned_data[3, st:, 1], label='60% noise')
        plt.axvline(x=30, linestyle='--', color='r', linewidth=1)
        plt.axvline(x=40, linestyle='--', color='r', linewidth=1)
    else:
        plt.plot(meaned_data[0, 40:, 0], meaned_data[0, 40:, 1], label='0% noise')
        plt.plot(meaned_data[1, 40:, 0], meaned_data[1, 40:, 1], label='20% noise')
        plt.plot(meaned_data[2, 40:, 0], meaned_data[2, 40:, 1], label='40% noise')
        plt.plot(meaned_data[3, 40:-65, 0], meaned_data[3, 40:-65, 1], label='60% noise')
        plt.axvline(x=80, linestyle='--', color='r', linewidth=1)
        plt.axvline(x=85, linestyle='--', color='r', linewidth=1)
        plt.axvline(x=125, linestyle='--', color='r', linewidth=1)

    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy', )
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
