import numpy as np

from upolygon import rle_encode


def test_rle_encode_matches_numpy2_default_float_mask():
    """Regression: counts buffer must use a dtype that matches the memoryview on all platforms.

    NumPy 2 on Windows + Python 3.13 failed when counts used C long + np.int_ (LP32 vs int64 mismatch).
    """
    mask = np.zeros((4, 4))
    mask[1:3, 1:3] = 1.0
    counts = rle_encode(mask)
    assert isinstance(counts, list)
    assert sum(counts) == mask.size


def test_rle_encode_accepts_int8_mask():
    mask = np.zeros((3, 3), dtype=np.int8)
    mask[1, 1] = 1
    assert sum(rle_encode(mask)) == mask.size
