#cython: language_level=3

cimport cython
import numpy as np
from libc.math cimport abs, sqrt


cdef perpendicular_distance(float px, float py, float ax, float ay, float bx, float by):
  return abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax) / sqrt((by - ay) * (by - ay) + (bx - ax) * (bx - ax))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False) 
def simplify_single_polygon(list path, float epsilon):
    if len(path) <= 1:
        return path
    cdef int max_distance = 0
    cdef int index = 0
    cdef int end = len(path) // 2 - 1
    for i in range(1, end):
        distance = perpendicular_distance(path[2*i], path[2*i+1], path[0], path[1], path[2*end], path[2*end+1])
        if distance >= max_distance:
            max_distance = distance 
            index = i
    
    if max_distance > epsilon:
        res1 = simplify_single_polygon(path[:2*index+2], epsilon) 
        res2 = simplify_single_polygon(path[2*index:], epsilon)
        return res1[0:len(res1)-2] + res2
    else:
        return [path[0], path[1], path[end*2], path[end*2+1]]

# Basic Ramer–Douglas–Peucker algorithm
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)            
def simplify_polygon(list paths, float epsilon):
    return [simplify_single_polygon(path, epsilon) for path in paths]
