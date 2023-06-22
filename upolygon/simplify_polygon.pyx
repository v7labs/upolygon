#cython: language_level=3

cimport cython
from libc.math cimport abs, sqrt


cdef perpendicular_distance(float px, float py, float ax, float ay, float bx, float by):
    cdef float dist = sqrt((by - ay) * (by - ay) + (bx - ax) * (bx - ax))
    if dist < 0.0001:
        return sqrt((py - ay) * (py - ay) + (px - ax) * (px - ax))
    return abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax) / dist

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False) 
def simplify_single_polygon(list path, float epsilon):
    print("Test it's compiling")
    # Note that we are using an iterative version of this algorithm
    # instead of the classical recursive to prevent reaching python's
    # max recursion.
    # Uses a stack to avoid recursion and iterates over the path where 
    # path takes the form of [x1,y1,x2,y2,...,xn,yn], therefore the x,y
    # tuple is at index 2*i and 2*i+1 respectively and the length is half of the array
    # Iterative algorithm comparison found here: https://namekdev.net/2014/06/iterative-version-of-ramer-douglas-peucker-line-simplification-algorithm/
    cdef int length = len(path) // 2
    cdef int startIndex = 0
    cdef int endIndex = length
    cdef float max_distance = 0
    cdef int index = 0
    cdef int i
    deleted = [False] * length
    stack = [(startIndex,endIndex)]
    while stack:
        startIndex, endIndex = stack.pop()
        if startIndex == endIndex:
            continue
        max_distance = 0
        for i in range(startIndex+1,endIndex-1):
            if deleted[i]:
                continue
            distance = perpendicular_distance(path[2*i], path[2*i+1], path[startIndex*2], path[startIndex*2+1], path[2*(endIndex-1)], path[2*(endIndex-1)+1])
            if distance > max_distance:
                max_distance = distance 
                index = i
        if max_distance > epsilon:
            stack.append((startIndex,index))
            stack.append((index, endIndex))
        else:
            for i in range(startIndex+1,endIndex-1):
                deleted[i] = True
    result = []
    for i in range(0, length):
        if not deleted[i]:
            result.append(path[2*i])
            result.append(path[2*i+1])
    
    return result

# Basic Ramer–Douglas–Peucker algorithm
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)            
def simplify_polygon(list paths, float epsilon):
    return [simplify_single_polygon(path, epsilon) for path in paths]
