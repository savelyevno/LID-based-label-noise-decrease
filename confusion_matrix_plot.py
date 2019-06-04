import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def get_confusion_matrix_image(cm, class_count, normalize):
    if normalize:
        cm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

    class_names = np.arange(class_count) + 1

    plot_numbers = class_count <= 10

    fig, ax = plt.subplots(dpi=100 if plot_numbers else 150)
    canvas = FigureCanvas(fig)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    if plot_numbers:
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion matrix',
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    else:
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=[], yticklabels=[],
               title='Confusion matrix',
               ylabel='True label',
               xlabel='Predicted label')
    # fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    image = np.fromstring(s, dtype=np.uint8).reshape((height, width, 4))

    plt.close()

    return image[:, :, :3]


if __name__ == '__main__':
    class_count = 100
    cm = np.random.randint(0, 1000, (class_count, class_count))
    image = get_confusion_matrix_image(cm, class_count, True)
    print(image.shape)
    # plt.figure().set_size_inches((7.1, 6))
    plt.tight_layout()
    plt.imshow(image)
    plt.show()


