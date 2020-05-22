# uPolygon (WIP)
Library of handy polygon related functions to speed up machine learning projects.

It was born as a replacement for `cv2.drawContours` when generating masks for instance segmentation, without having to bring in all of opencv.

## TODO
- [x] draw_polygon
- [ ] polygon_area
- [ ] point_in_polygon
- [ ] find_contours


## Usage
This library expects all polygons to be model as a list of paths, each path is a list of alternating x and y coordinates (`[x1,y1,x2,y2,...]`). 

A simple triangle would be declared as: 
```python
triangle = [[50,50, 100,0, 0,0]]
```

Complex polygons (holes and/or disjoints) follow the even-odd rule. 


## draw_polygon
```python
from upolygon import draw_polygon 
import numpy as np

mask = np.zeros((100,100), dtype=np.int32)
draw_polygon(mask, [[50,50, 100,0, 0,0]], 1)
```

Equivalent of calling `cv2.drawContours(mask, [np.array([[50,50], [100,0], [0,0]])], -1, 1, cv2.FILLED)` when using opencv. 

uPolygon is ~ 6 times faster than opencv for larger polygons. (TODO: add benchmarks)