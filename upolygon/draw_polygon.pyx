#cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, free, qsort
from libc.math cimport ceil, floor, round
from array import array

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
cdef inline void draw_straight_line(float x1, float x2, int y, data_type[:, :] img, data_type value) nogil:
    cdef int x = max(<int>ceil(x1),0)
    cdef int max_x = min(<int>floor(x2), img.shape[1]-1)
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
    
# draw Bresenham 8-connected line 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
cdef void draw_edge_line(data_type [:,:] img, int x1, int y1, int x2, int y2, data_type value):
    cdef int dx = x2 - x1
    cdef int dy = y2 - y1
    cdef int x, y
    
    # special case for vertical lines
    if dx == 0:
        y1, y2 = min(y1, y2), max(y1, y2)+1
        for y in range(y1, y2):
            img[y][x1] = value
        return
    
    # special case for horizontal lines
    if dy == 0:
        x1, x2 = min(x1, x2), max(x1, x2)+1
        for x in range(x1, x2):
            img[y1][x] = value
        return
    
    if x1 > img.shape[1] or x2 < 0 or y1 > img.shape[0] or y2 < 0:
        return

    cdef int delta_x = 1
    cdef int delta_y = 1
    
    if dx < 0:
        dx = -dx
        dy = -dy
        x1, y1, x2, y2 = x2, y2, x1, y1
    
    if dy < 0:
        delta_y = -1
        dy = -dy

        
    cdef int flip = dy/dx > 1
    cdef int count = abs(max(0,min(img.shape[1], x2)) - max(0, x1)) +1
    if flip:
        dx, dy = dy, dx
        count = abs(max(0,min(img.shape[0], y2)) - max(0, y1)) +1
           
    cdef int minus_err = 2 * dy
    cdef int plus_err = 2 * (dy - dx)

    cdef int err = (dy + dy) - dx
    # cdef int count = dx + 1
    
    if flip:
        y = max(0, y1)
        x = max(0, x1)
        for _i in range(count):
            img[y][x] = value
            if err <= 0:
                err  = err + minus_err
            else:
                x = x + delta_x
                err = err + plus_err
            y = y + delta_y 
    else:
        y = max(0, y1)
        x = max(0, x1)
        for _i in range(count):
            img[y][x] = value
            if err <= 0:
                err  = err + minus_err
            else:
                y = y + delta_y
                err = err + plus_err
            x = x + delta_x
            


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int find_edges(s_edge *edges, list path, data_type [:,:] img, data_type value):
    cdef int length = len(path)
    cdef float[:] path_mv = memoryview(array('f', path))
    cdef float x1
    cdef float y1
    cdef float x2
    cdef float y2
    cdef int i
    cdef int idx = 0
    x1, y1 = path_mv[length-2], path_mv[length-1]
    y1 = round(y1)
    for i in range(0, length, 2):
        x2, y2 = path_mv[i], path_mv[i+1]
        y2 = round(y2)
        draw_edge_line(img, <int>x1, <int>y1, <int>x2, <int>y2, value)

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
        edges_so_far += find_edges(edges + edges_so_far, path, img, value)
    # no point in continuing if there are no edges
    if edges_so_far == 0:
        free(edges)
        return img
    # edges_so_far can be smaller than edges_length if there are straight lines
    edges_length = edges_so_far
    qsort(edges, edges_so_far, sizeof(s_edge), &cmp_edges)

    cdef int active_edge_length = 0
    cdef s_active_edge* active_edges = <s_active_edge*>malloc(sizeof(s_active_edge) * edges_length)
    cdef s_active_edge edge
    cdef int scanline_y = <int>round(edges[0].y_min)
    cdef int max_scanline_y = img.shape[0]
    cdef int i, j, a, b, x
    cdef int ymin
    while (edges_length > 0 or active_edge_length > 0) and scanline_y < max_scanline_y:
        for i in range(edges_length):
            if edges[i].y_min == scanline_y:
                active_edges[active_edge_length].y_max = edges[i].y_max
                active_edges[active_edge_length].x_val = edges[i].x_val
                active_edges[active_edge_length].m_inv = edges[i].m_inv
                active_edge_length += 1
            elif edges[i].y_min > scanline_y:
                break
        
        for i in reversed(range(active_edge_length)):
            if active_edges[i].y_max == scanline_y:
                move_active_down(active_edges, i, active_edge_length)
                active_edge_length -= 1

        #qsort(active_edges, active_edge_length, sizeof(s_active_edge), &cmp_active_edges)
        for i in range(1, active_edge_length):
            edge = active_edges[i]
            j = i - 1
            while j >= 0 and edge.x_val < active_edges[j].x_val:
                active_edges[j+1] = active_edges[j]
                j -= 1
            active_edges[j+1] = edge

        
        for i in range(0, active_edge_length, 2):
            draw_straight_line(active_edges[i].x_val, active_edges[i+1].x_val, scanline_y, img, value)

        for i in range(0,active_edge_length):
            active_edges[i].x_val += active_edges[i].m_inv
            
        for i in reversed(range(edges_length)):
            if edges[i].y_min == scanline_y:
                move_down(edges, i, edges_length)
                edges_length -= 1
        scanline_y += 1

    free(edges)
    free(active_edges)
    return img.base
