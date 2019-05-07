import numpy as np


# noinspection PyAttributeOutsideInit
class DatasetMetricCalculator:
    def __init__(self, keep_values=False, class_count=None):
        self.class_count = class_count
        self.keep_values = keep_values
        self.clear()

    @property
    def mean(self):
        return self.sum / max(1, self.element_count)

    @property
    def median(self):
        return np.median(self.values)

    @property
    def mean_per_class(self):
        return self.sum_per_class / np.maximum(1, self.element_count_per_class)

    @property
    def std(self):
        return np.std(self._values)

    @property
    def std_per_class(self):
        stds = [np.std(lst) for lst in self._values_per_class]
        return np.array(stds)

    @property
    def mean_weighted(self):
        return self.mean_per_class.mean()

    @property
    def values(self):
        return np.array(self._values)

    @property
    def values_per_class(self):
        return [np.array(it) for it in self._values_per_class]

    def add_batch_values(self, batch_values):
        if not isinstance(batch_values, np.ndarray):
            batch_values = np.array([batch_values])
        self.sum += batch_values.sum()
        self.element_count += len(batch_values)

        self._values.extend(batch_values)

    def add_batch_values_with_labels(self, batch_values, batch_labels):
        labels_argmaxed = np.argmax(batch_labels, -1)
        self.add_batch_values_with_labels_argmaxed(batch_values, labels_argmaxed)

    def add_batch_values_with_labels_argmaxed(self, batch_values, batch_labels_argmaxed):
        labels_raveled = batch_labels_argmaxed.ravel()
        values_raveled = batch_values.ravel()
        np.add.at(self.sum_per_class, labels_raveled, values_raveled)
        np.add.at(self.element_count_per_class, labels_raveled, 1)

        if self.keep_values:
            for lbl, val in zip(labels_raveled, values_raveled):
                self._values_per_class[lbl].append(val)

        self.add_batch_values(values_raveled)

    def clear(self):
        self.sum = 0
        self.element_count = 0

        self.sum_per_class = np.zeros(self.class_count)
        self.element_count_per_class = np.zeros(self.class_count)

        self._values_per_class = [[] for _ in range(self.class_count)]
        self._values = []
