#cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, free, qsort
from libc.math cimport ceil

cdef struct s_edge:
    float    y_min
    float    y_max
    float    x_val
    float    m_inv

cdef struct s_active_edge:
    float    y_max
    float    x_val
    float    m_inv

ctypedef fused data_type:
    char
    unsigned char
    int
    unsigned int
    double
    long long

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef draw_straight_line(float x1, float x2, int y, data_type[:, :] img, data_type value):
    cdef int x = max(<int>ceil(x1),0)
    cdef int max_x = min(<int>ceil(x2), img.shape[1]-1)
    while x <= max_x:
        img[y][x] = value
        x = x + 1

# Sort edges first by y_min and then by x_val
cdef int cmp_edges(const void* a, const void* b) nogil:
    cdef s_edge a_v = (<s_edge*>a)[0]
    cdef s_edge b_v = (<s_edge*>b)[0]
    if a_v.y_min < b_v.y_min:
        return -1
    elif a_v.y_min == b_v.y_min:
        if a_v.x_val < b_v.x_val:
            return -1
        elif a_v.x_val == b_v.x_val:
            return 0
        else:
            return 1
    else:
        return 1
    
# Sort active edges by x_val
cdef int cmp_active_edges(const void* a, const void* b) nogil:
    cdef s_active_edge a_v = (<s_active_edge*>a)[0]
    cdef s_active_edge b_v = (<s_active_edge*>b)[0]
    if a_v.x_val < b_v.x_val:
        return -1
    elif a_v.x_val == b_v.x_val:
        return 0
    else:
        return 1
    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int find_edges(s_edge *edges, list path):
    cdef int length = len(path)
    cdef float x1
    cdef float y1
    cdef float x2
    cdef float y2
    cdef int i
    cdef int idx = 0
    x1, y1 = path[length-2:]
    for i in range(0, length, 2):
        x2, y2 = path[i], path[i+1]
        if y1 == y2:
            x1, y1 = x2, y2
            continue
        elif y1 < y2:
            edges[idx].y_min = y1 
            edges[idx].y_max = y2
            edges[idx].x_val = x1
        else:
            edges[idx].y_min = y2 
            edges[idx].y_max = y1
            edges[idx].x_val = x2
        
        edges[idx].m_inv = (x1 - x2) / (y1 -y2)
        idx += 1
        x1, y1 = x2, y2
    return idx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void move_down(s_edge* edges, int i, int length):
    while i+1 < length:
        edges[i].y_min = edges[i+1].y_min
        edges[i].y_max = edges[i+1].y_max
        edges[i].x_val = edges[i+1].x_val
        edges[i].m_inv = edges[i+1].m_inv
        i += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void move_active_down(s_active_edge* edges, int i, int length):
    while i+1 < length:
        edges[i].y_max = edges[i+1].y_max
        edges[i].x_val = edges[i+1].x_val
        edges[i].m_inv = edges[i+1].m_inv
        i += 1



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def draw_polygon(data_type[:, :] img, list paths, data_type value):
    cdef int edges_length = sum(len(path) // 2 for path in paths)
    cdef s_edge* edges = <s_edge*>malloc(sizeof(s_edge) * edges_length)
    cdef int edges_so_far = 0
    for path in paths:
        edges_so_far += find_edges(edges + edges_so_far, path)
    # no point in continuing if there are no edges
    if edges_so_far == 0:
        free(edges)
        return img
    # edges_so_far can be smaller than edges_length if there are straight lines
    edges_length = edges_so_far
    qsort(edges, edges_so_far, sizeof(s_edge), &cmp_edges)

    cdef int active_edge_length = 0
    cdef s_active_edge* active_edges = <s_active_edge*>malloc(sizeof(s_active_edge) * edges_length)
    cdef int scanline_y = <int>round(edges[0].y_min)
    cdef int i
    cdef int ymin
    while edges_length > 0 or active_edge_length > 0:
        for i in range(edges_length):
            if edges[i].y_min == scanline_y:
                active_edges[active_edge_length].y_max = edges[i].y_max
                active_edges[active_edge_length].x_val = edges[i].x_val
                active_edges[active_edge_length].m_inv = edges[i].m_inv
                active_edge_length += 1
            elif edges[i].y_min > scanline_y:
                break
        
        qsort(active_edges, active_edge_length, sizeof(s_active_edge), &cmp_active_edges)

        for i in range(0,active_edge_length, 2):
            draw_straight_line(active_edges[i].x_val, active_edges[i+1].x_val, scanline_y, img, value)
            active_edges[i].x_val += active_edges[i].m_inv
            active_edges[i+1].x_val += active_edges[i+1].m_inv
            
        for i in reversed(range(edges_length)):
            if edges[i].y_min == scanline_y:
                move_down(edges, i, edges_length)
                edges_length -= 1
        for i in reversed(range(active_edge_length)):
            if active_edges[i].y_max == scanline_y:
                move_active_down(active_edges, i, active_edge_length)
                active_edge_length -= 1
        scanline_y += 1

    free(edges)
    free(active_edges)
    return img
