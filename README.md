# uPolygon
Library of handy polygon related functions to speed up machine learning projects.

It was born as a replacement for `cv2.fillPoly` when generating masks for instance segmentation, without having to bring in all of opencv.

## TODO
- [x] draw_polygon
- [x] find_contours
- [ ] polygon_area
- [ ] point_in_polygon


## Usage
This library expects all polygons to be model as a list of paths, each path is a list of alternating x and y coordinates (`[x1,y1,x2,y2,...]`). 

A simple triangle would be declared as: 
```python
triangle = [[50,50, 100,0, 0,0]]
```

Complex polygons (holes and/or disjoints) follow the even-odd rule. 


## draw_polygon

`draw_polygon(mask: array[:, :], paths: path[]) -> array[:, :]`

```python
from upolygon import draw_polygon 
import numpy as np

mask = np.zeros((100,100), dtype=np.int32)
draw_polygon(mask, [[50,50, 100,0, 0,0]], 1)
```

Equivalent of calling `cv2.fillPoly(mask, [np.array([[50,50], [100,0], [0,0]])], 1)` or `cv2.drawContours(mask, [np.array([[50,50], [100,0], [0,0]])], -1, 1, cv2.FILLED)` when using opencv. 

uPolygon is ~ 6 times faster than opencv for large random polygons with many intersecting lines.
For smaller polygons or few intersections, uPolygon is half as fast as opencv. 

## find_contours
`find_contours(mask: array[:, :]) -> (array[:, :], path[:], path[:])`

0 is treated as background, 1 is treated as foreground. 
```python
from upolygon import find_contours
import numpy as np

mask = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0]
    ], dtype=np.uint8)

_labels, external_paths, internal_paths = find_contours(mask)
```

Similar to OpenCV's `cv2.findContours` but lacking hierarchies. Also similar to BoofCV's `LinearContourLabelChang2004` which is based on the same [algorithm](https://www.iis.sinica.edu.tw/papers/fchang/1362-F.pdf).


Note that currently the input mask to find_contour needs to be uint8.

## rle_encode
`rle_encode(mask: array[:,:]) -> list`
Takes a 2-dim binary mask and generates a run length encoding according to the coco specs

~ 15 times faster than written in plain python

