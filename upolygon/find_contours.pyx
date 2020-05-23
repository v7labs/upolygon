#cython: language_level=3

cimport cython
import numpy as np

# This implementation is based on https://www.iis.sinica.edu.tw/papers/fchang/1362-F.pdf

# When tracing there is a clockwise index around the current point P
# 5 6 7
# 4 P 0
# 3 2 1
cdef int* directions_x = [1, 1, 0, -1, -1, -1,  0,  1] 
cdef int* directions_y = [0, 1, 1,  1,  0, -1, -1, -1] 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int tracer(int px, int py, int old_index, int *nx, int *ny, unsigned char[:, :] image, char[:, :] labels) nogil:
    # move two steps clockwise from the previous entry in the trace
    cdef int start_index = (old_index + 2) % 8
    cdef int i
    cdef int tmpx, tmpy
    for i in range(start_index, start_index + 8):
        i = i % 8
        tmpx = directions_x[i] + px 
        tmpy = directions_y[i] + py
        if image[tmpy][tmpx] == 1:
            nx[0] = tmpx
            ny[0] = tmpy
            # adding four to the index gives us the relative position of px,py to nx,ny in the next call
            return i + 4
        else:
            labels[tmpy][tmpx] = -1
            
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef contour_trace(int px, int py, int c, unsigned char[:, :]image, char[:,:] labels, int inner):
        cdef int sx = px 
        cdef int sy = py
        cdef int nx, ny, tx, ty
        cdef int index = 1 if inner else 5
        cdef int last_point_was_s = False
        path = [px-1, py-1]
        index = tracer(px, py, index, &nx, &ny, image, labels)
        tx = nx 
        ty = ny
        
        # S was a single point
        if tx == sx and ty == sy:
            return path
        
        path.append(tx-1)
        path.append(ty-1)
        
        labels[ny][nx] = c
        while True:
            index = tracer(nx, ny, index, &nx, &ny, image, labels)
            if last_point_was_s and nx == tx and ny == ty:
                return path
            path.append(nx-1)
            path.append(ny-1)
            labels[ny][nx] = c
            last_point_was_s = nx == sx and ny == sy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)            
def find_contours(unsigned char[:,:] image):
    cdef int px = 1
    cdef int py = 1
    cdef int c  = 1 
    cdef int width = image.shape[1] - 1
    cdef int height = image.shape[0] - 1
    image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    cdef char[:,:] labels = np.zeros((image.shape[0], image.shape[1]), dtype=np.int8)
    inner_paths = []
    outer_paths = []

    while py < height:
        while image[py][px] == 0 and px < width:
            px += 1
        if image[py][px] == 1:
            handled = False
            # STEP 1
            if labels[py][px] == 0 and image[py-1][px] == 0:
                labels[py][px] = c
                path = contour_trace(px, py, c, image, labels, 0)
                outer_paths.append(path)
                c += 1
                handled = True
            # STEP 2
            if labels[py+1][px] != -1 and image[py+1][px] == 0:
                handled = True
                # unlabeled
                if labels[py][px] == 0:
                    path = contour_trace(px, py, labels[py][px-1], image, labels, 1)
                else:
                    path = contour_trace(px, py, labels[py][px], image, labels, 1)
                inner_paths.append(path)
            # STEP 3
            if not handled and labels[py][px] == 0:
                labels[py][px] = labels[py][px-1]
        px += 1
        if px > width-1:
            px = 1
            py = py + 1
    return labels, outer_paths, inner_paths