import numpy as np

from upolygon import draw_polygon

triangle = [5, 5, 8, 1, 0, 0]
triangle_sum = 270
triangle_result = np.array(
    [
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int32,
)
triangle_mask_size = triangle_result.shape


def test_does_nothing_for_empty_an_empty_polygon():
    mask = np.zeros((10, 10), dtype=np.int32)
    draw_polygon(mask, [], 1)
    assert np.all(mask == 0)


def test_writes_the_given_value():
    mask_1 = np.zeros((100, 100), dtype=np.int32)
    mask_2 = np.zeros((100, 100), dtype=np.int32)
    draw_polygon(mask_1, [triangle], 1)
    draw_polygon(mask_2, [triangle], 2)
    assert np.sum(mask_1) * 2 == np.sum(mask_2)


def test_crops_negative_coordinates():
    mask = np.zeros((100, 100), dtype=np.int32)
    draw_polygon(mask, [[-50, 0, 50, 50, 200, 200]], 1)


def test_crop_out_of_bound_horizontal_line():
    mask = np.zeros((100, 100), dtype=np.int32)
    draw_polygon(mask, [[-50, 0, 200, 0]], 1)


def test_crop_out_of_bound_vertical_line():
    mask = np.zeros((100, 100), dtype=np.int32)
    draw_polygon(mask, [[0, -50, 0, -200]], 1)


def test_1px_tall_polygons():
    polygon = [[0, 0, 8, 0, 8, 1, 0, 1]]
    mask = np.zeros((10, 10), dtype=np.int32)
    draw_polygon(mask, polygon, 1)
    expected = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        dtype=np.uint8,
    )
    assert np.all(mask == expected)


def test_holes_in_polygons():
    polygon = [[8, 4, 0, 4, 0, 0, 8, 0], [7, 1, 1, 1, 1, 3, 7, 3]]
    mask = np.zeros((10, 10), dtype=np.int32)
    draw_polygon(mask, polygon, 1)
    expected = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        dtype=np.uint8,
    )
    assert np.all(mask == expected)

def test_rectangle_large_segments():
    square = [1, 1, 5, 1, 5, 5, 1, 5]
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    mask = np.zeros((7, 7), dtype=np.int32)
    draw_polygon(mask, [square], 1)
    assert np.all(mask == expected)


def test_rectangle_tiny_segments():
    square = [
        1,
        1,
        2,
        1,
        3,
        1,
        4,
        1,
        5,
        1,
        5,
        2,
        5,
        3,
        5,
        4,
        5,
        5,
        4,
        5,
        3,
        5,
        2,
        5,
        1,
        5,
        1,
        4,
        1,
        3,
        1,
        2,
    ]

    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    mask = np.zeros((7, 7), dtype=np.int32)
    draw_polygon(mask, [square], 1)
    assert np.all(mask == expected)


def test_decimals_in_path():
    square = [0.5, 0.5, 0.5, 10.5, 10.5, 10.5, 10.5, 0.5]
    mask = np.zeros((100, 100), dtype=np.int32)
    draw_polygon(mask, [square], 1)
    assert np.sum(mask) == 11 * 11


def test_out_of_bound():
    square = [0, 0, 0, 10, 10, 10, 10, 0]
    mask = np.zeros((1, 1), dtype=np.int32)
    draw_polygon(mask, [square], 1)
    assert np.sum(mask) == 1


def test_supports_uint8():
    mask = np.zeros(triangle_mask_size, dtype=np.uint8)
    draw_polygon(mask, [triangle], 1)
    assert np.all(mask == triangle_result)


def test_supports_int8():
    mask = np.zeros(triangle_mask_size, dtype=np.int8)
    draw_polygon(mask, [triangle], 1)
    assert np.all(mask == triangle_result)


def test_supports_int32():
    mask = np.zeros(triangle_mask_size, dtype=np.int32)
    draw_polygon(mask, [triangle], 1)
    assert np.all(mask == triangle_result)


def test_supports_float():
    mask = np.zeros(triangle_mask_size, dtype=np.float)
    draw_polygon(mask, [triangle], 1)
    assert np.all(mask == triangle_result)
