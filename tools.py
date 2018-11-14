def bitmask_contains(bit_mask, bit_pos):
    res = bit_mask & (1 << bit_pos)
    return res != 0
