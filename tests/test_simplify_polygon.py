import numpy as np

from upolygon import simplify_polygon


def test_empty_path():
    assert len(simplify_polygon([], 1)) == 0


def test_empty_sub_path():
    assert len(simplify_polygon([[]], 1)) == 1


def test_removes_linear_points():
    path = [0, 0, 0, 5, 0, 10, 0, 15]
    assert simplify_polygon([path], 1) == [[0, 0, 0, 15]]


def test_keeps_non_linear_points():
    path = [0, 0, 0, 5, 0, 7, 10, 10, 0, 15]
    assert simplify_polygon([path], 1) == [[0, 0, 0, 7, 10, 10, 0, 15]]


def test_respects_epsilon():
    path = [0, 0, 1, 1, 0, 10]
    assert simplify_polygon([path], 1) == [[0, 0, 0, 10]]
    assert simplify_polygon([path], 0.9) == [[0, 0, 1, 1, 0, 10]]
