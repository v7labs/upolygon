#cython: language_level=3

cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def rle_encode(binary_mask):
    # at most there can be len(binary_mask) different values, therefor we prealloace an array of that size
    # unused counts will be stripped at the end
    cdef long[:] counts = np.zeros(binary_mask.shape[0] * binary_mask.shape[1], dtype=np.int)
    cdef char[:] mask_view = binary_mask.ravel(order="F").astype(np.int8)

    cdef char last_elem = 0
    cdef char elem
    cdef long running_length = 0
    cdef int i = 0
    cdef int j = 0
     
    for j in range(mask_view.shape[0]):
         if mask_view[j] != last_elem:
             counts[i] = running_length
             i += 1
             running_length = 0
             last_elem = mask_view[j]
         running_length += 1
    counts[i] = running_length
    i += 1
    return counts.base[0:i].tolist()
    return np.array(counts[0:i], dtype=np.int)
    # return counts[0:i] # np.array(counts[0:i])

def rle_decode(counts, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    cdef int val = 1
    cdef int n = 0
    cdef int pos
    for pos in range(len(counts)):
        val = not val
        img[n : n + counts[pos]] = val
        n += counts[pos]
    return img.reshape(shape).T
