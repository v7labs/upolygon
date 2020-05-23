import numpy as np

from upolygon import draw_polygon

triangle = [5, 5, 8, 1, 0, 0]
triangle_sum = 25


def test_does_nothing_for_empty_an_empty_polygon():
    mask = np.zeros((10, 10), dtype=np.int32)
    draw_polygon(mask, [], 1)
    assert np.all(mask == 0)


def test_writes_the_given_value():
    mask_1 = np.zeros((100, 100), dtype=np.int32)
    mask_2 = np.zeros((100, 100), dtype=np.int32)
    draw_polygon(mask_1, [triangle], 1)
    draw_polygon(mask_2, [triangle], 2)
    print(mask_1[0:10, 0:10])
    assert np.sum(mask_1) * 2 == np.sum(mask_2)


def test_square():
    # straight lines can be tricky
    square = [0, 0, 0, 10, 10, 10, 10, 0]
    mask = np.zeros((100, 100), dtype=np.int32)
    draw_polygon(mask, [square], 1)
    assert np.sum(mask) == 11 * 11


def test_decimals_in_path():
    square = [0.5, 0.5, 0.5, 10.5, 10.5, 10.5, 10.5, 0.5]
    mask = np.zeros((100, 100), dtype=np.int32)
    draw_polygon(mask, [square], 1)
    print(np.sum(mask))
    assert np.sum(mask) == 11 * 11


def test_out_of_bound():
    square = [0, 0, 0, 10, 10, 10, 10, 0]
    mask = np.zeros((1, 1), dtype=np.int32)
    draw_polygon(mask, [square], 1)
    assert np.sum(mask) == 1


def test_supports_uint8():
    mask = np.zeros((100, 100), dtype=np.uint8)
    draw_polygon(mask, [triangle], 1)
    assert np.sum(mask) == triangle_sum


def test_supports_int8():
    mask = np.zeros((100, 100), dtype=np.int8)
    draw_polygon(mask, [triangle], 1)
    assert np.sum(mask) == triangle_sum


def test_supports_int32():
    mask = np.zeros((100, 100), dtype=np.int32)
    draw_polygon(mask, [triangle], 1)
    assert np.sum(mask) == triangle_sum


def test_supports_float():
    mask = np.zeros((100, 100), dtype=np.float)
    draw_polygon(mask, [triangle], 1)
    assert np.sum(mask) == triangle_sum
