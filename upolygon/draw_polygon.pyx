#cython: language_level=3, warn.undeclared=True, warn.unused=True, 
cimport cython
from libc.stdlib cimport malloc, free, qsort
from libc.math cimport ceil, floor, round
from array import array

# An edge beteween two adjacent points.
# x_val is the x position at y_min.
cdef struct s_edge:
    float    y_min
    float    y_max
    float    x_val
    float    m_inv

# An active edge intersects the scanline
# x_val is updated at every iteration
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

cdef int clip(int value, int min_value, int max_value) nogil:
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value

# Clip the lines inside a rectangle (0,0) (w,h)
# For details see https://arxiv.org/pdf/1908.01350.pdf
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline int clip_line(int w, int h, int* x1, int* y1, int* x2, int* y2) nogil:
    cdef double _x1 = x1[0]
    cdef double _x2 = x2[0]
    cdef double _y1 = y1[0]
    cdef double _y2 = y2[0]

    # first check if both point are outside the viewpoint on the same side
    # if so skip them
    if _x1 < 0 and _x2 < 0:
        return 0
    if _x1 > w and _x2 >= w:
        return 0
    if _y1 < 0 and _y2 < 0:
        return 0
    if _y1 > h and _y2 >= h:
        return 0
    
    if _x1 < 0:
        _y1 = (_y2-_y1) / (_x2 - _x1)  * (0-_x1) + _y1
        _x1 = 0
    elif _x1 >= w:
        _y1 = (_y2-_y1) / (_x2 - _x1)  * (w-_x1) + _y1
        _x1 = w
        
    if _y1 < 0:
        _x1 = (_x2-_x1) / (_y2 - _y1)  * (0-_y1) + _x1
        _y1 = 0 
    elif _y1 >= h:
        _x1 = (_x2-_x1) / (_y2 - _y1)  * (h-_y1) + _x1
        _y1 = h 

    if _x2 < 0:
        _y2 = (_y2-_y1) / (_x2 - _x1)  * (0-_x1) + _y1
        _x2 = 0
    elif _x2 >= w:
        _y2 = (_y2-_y1) / (_x2 - _x1)  * (w-_x1) + _y1
        _x2 = w

    if _y2 < 0:
        _x2 = (_x2-_x1) / (_y2 - _y1)  * (0-_y1) + _x1
        _y2 = 0 
    elif _y2 >= h:
        _x2 = (_x2-_x1) / (_y2 - _y1)  * (h-_y1) + _x1
        _y2 = h 

    if (_x1 < 0 and _x2 < 0) or (_x1 >= w and _x2 >= w):
        return 0
    
    x1[0] = <int>_x1
    x2[0] = <int>_x2
    y1[0] = <int>_y1
    y2[0] = <int>_y2
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void draw_straight_line(float x1, float x2, int y, data_type[:, :] mask, data_type value) nogil:
    cdef int x = max(<int>ceil(x1),0)
    cdef int max_x = min(<int>floor(x2), mask.shape[1]-1) + 1
    cdef int i
    for i in range(x, max_x):
        mask[y][i] = value

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
    
# draw Bresenham 8-connected line 
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void draw_edge_line(data_type [:,:] mask, int x1, int y1, int x2, int y2, data_type value):
    cdef int dx = x2 - x1
    cdef int dy = y2 - y1
    cdef int x, y
    
    # special case for vertical lines
    if dx == 0:
        if x1 < 0 or x1 >= mask.shape[1]:
            return
        y1, y2 = clip(y1, 0, mask.shape[0]-1), clip(y2, 0, mask.shape[0]-1)
        y1, y2 = min(y1, y2), max(y1, y2)
        for y in range(y1, y2+1):
            mask[y][x1] = value
        return
    
    # special case for horizontal lines
    if dy == 0:
        if y1 < 0 or y1 >= mask.shape[0]:
            return
        x1, x2 = clip(x1, 0, mask.shape[1]-1), clip(x2, 0, mask.shape[1]-1)
        x1, x2 = min(x1, x2), max(x1, x2)
        for x in range(x1, x2+1):
            mask[y1][x] = value
        return

    if clip_line(mask.shape[1], mask.shape[0], &x1, &y1, &x2, &y2) == 0:
        return

    dx = x2 - x1
    dy = y2 - y1

    cdef int delta_x = 1
    cdef int delta_y = 1
    
    if dx < 0:
        dx = -dx
        dy = -dy
        x1, y1, x2, y2 = x2, y2, x1, y1
    
    if dy < 0:
        delta_y = -1
        dy = -dy

        
    cdef int flip = dy > dx
    cdef int count = abs(x1 - x2)
    if flip:
        dx, dy = dy, dx
        count = abs(y1 - y2)
           
    cdef int minus_err = 2 * dy
    cdef int plus_err = 2 * (dy - dx)

    cdef int err = (dy + dy) - dx
    
    if flip:
        y = max(0, y1)
        x = max(0, x1)
        for _i in range(count):
            mask[y][x] = value
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
            mask[y][x] = value
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
cdef int find_edges(s_edge *edges, list path, data_type [:,:] mask, data_type value):
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
        draw_edge_line(mask, <int>x1, <int>y1, <int>x2, <int>y2, value)

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
        
        if edges[idx].y_max < 0:
            x1, y1 = x2, y2
            continue 

        if edges[idx].y_min < 0:
            edges[idx].x_val = (x2-x1) / (y2 - y1)  * (0-y1) + x1
            edges[idx].y_min = 0

        if edges[idx].y_max >= mask.shape[0]:
            edges[idx].y_max = mask.shape[0]

        edges[idx].m_inv = (x1 - x2) / (y1 -y2)
        idx += 1
        x1, y1 = x2, y2
    return idx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void move_active_down(s_active_edge* edges, int i, int length):
    cdef int j
    for j in range(i, length-1):
        edges[j] = edges[j+1]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def draw_polygon(data_type[:, :] mask, list paths, data_type value):
    """Draws a polygon with value
    Args:
        mask: 2d mask on which the polygon will be drawn, note that the mask will be modified
        paths: a list of paths ([[x1,y1,x2,y2,...],[x1,y1,x2,y2,...]])
        value: value used to draw the polygon on the mask

    Returns:
        The input mask, this is purely for convenience since the input mask is modified. 
 
    Example:
        triangle_mask = draw_polygon(np.zeroes(100,100), [[0,50, 50, 0, 50, 50]], 255)
    """
    cdef int edges_length = sum(len(path) // 2 for path in paths)
    cdef s_edge* edges = <s_edge*>malloc(sizeof(s_edge) * edges_length)
    cdef int edges_so_far = 0
    for path in paths:
        edges_so_far += find_edges(edges + edges_so_far, path, mask, value)
    # no point in continuing if there are no edges
    if edges_so_far == 0:
        free(edges)
        return mask.base
    # edges_so_far can be smaller than edges_length if there are straight lines
    edges_length = edges_so_far
    qsort(edges, edges_so_far, sizeof(s_edge), &cmp_edges)

    cdef int active_edge_length = 0
    # keep an offset of which edges we have already processed, this way we can skip them.
    cdef int edge_dead_offset = 0
    cdef s_active_edge* active_edges = <s_active_edge*>malloc(sizeof(s_active_edge) * edges_length)
    cdef s_active_edge edge
    cdef int scanline_y = <int>round(edges[0].y_min)
    cdef int max_scanline_y = mask.shape[0]
    cdef int i, j

    while (edge_dead_offset < edges_length or active_edge_length > 0) and scanline_y < max_scanline_y:
        for i in range(edge_dead_offset, edges_length):
            if edges[i].y_min == scanline_y:
                active_edges[active_edge_length].y_max = edges[i].y_max
                active_edges[active_edge_length].x_val = edges[i].x_val
                active_edges[active_edge_length].m_inv = edges[i].m_inv
                active_edge_length += 1
                edge_dead_offset += 1
            elif edges[i].y_min > scanline_y:
                break
        
        # When an active edge is outside the scanline it can be retired
        # TODO: this could probably fused with the insert sort. 
        for i in reversed(range(active_edge_length)):
            if active_edges[i].y_max == scanline_y:
                move_active_down(active_edges, i, active_edge_length)
                active_edge_length -= 1

        # Sort the active edges by x_val.
        # This is implemented as insertion sort since the list is mostly sorted
        # only edges that cross at this specific scanline will swap places. 
        for i in range(1, active_edge_length):
            edge = active_edges[i]
            j = i - 1
            while j >= 0 and edge.x_val < active_edges[j].x_val:
                active_edges[j+1] = active_edges[j]
                j -= 1
            active_edges[j+1] = edge

        
        for i in range(0, active_edge_length, 2):
            draw_straight_line(active_edges[i].x_val, active_edges[i+1].x_val, scanline_y, mask, value)

        for i in range(0,active_edge_length):
            active_edges[i].x_val += active_edges[i].m_inv
       
        scanline_y += 1
    
    free(edges)
    free(active_edges)
    return mask.base
