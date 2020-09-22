import numpy as np

from upolygon import draw_polygon, find_contours, simplify_polygon


def test_single_pixel():
    mask = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)
    _labels, external_paths, internal_paths = find_contours(mask)
    assert len(external_paths) == 1
    assert len(internal_paths) == 0


def test_finds_singular_outer_path():
    mask = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=np.uint8)
    _labels, external_paths, internal_paths = find_contours(mask)
    assert len(external_paths) == 1
    assert len(internal_paths) == 0
    print(mask)
    print(np.array(draw_polygon(mask.copy() * 0, external_paths, 1)))
    print(external_paths)
    assert np.all(mask == draw_polygon(mask.copy() * 0, external_paths, 1))


def test_finds_two_outer_path():
    mask = np.array([[0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0]], dtype=np.uint8,)
    _labels, external_paths, internal_paths = find_contours(mask)
    assert len(external_paths) == 2
    assert len(internal_paths) == 0
    assert np.all(mask == draw_polygon(mask.copy() * 0, external_paths, 1))
