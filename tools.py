import numpy as np


def bitmask_contains(bit_mask, bit_pos):
    res = bit_mask & (1 << bit_pos)
    return res != 0


def softmax(x, w=None):
    if w is None:
        w = np.ones(len(x))
    sm = np.sum(np.exp(x) * w)
    return (np.exp(x) * w) / sm
