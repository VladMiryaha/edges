# Introduction.
Structured Edge Detection Toolbox V3.0, by Piotr Dollar (pdollar-at-gmail.com)
FOR MORE DETAILS SEE https://github.com/pdollar/edges

# License.

This code is published under the MSR-LA Full Rights License.
Please read license.txt for more info.

# This fork (C++ and Python wrappers)

This fork adds a C++ and Python wrapper for structured edges and edge-boxes object proposals (see papers above), removing the Matlab dependency. To keep the code clean and reduce effort I used the C++ implementation of structured random forest edges from Philipp Krähenbühl (http://www.philkr.net/home/gop), included in ./cpp/external.

The Python wrapper requires Boost::python and my fork of Sudeep Pillai's numpy-opencv-converter (see https://github.com/samarth-robo/numpy-opencv-converter).

## Usage:
1. For structured random forest edges use the function `edge_detect(const Mat &im, Mat &E, Mat &O, string st_path)` in `./cpp/src/edge_detect.cpp`. `st_path` is the full path to the trained structured random forest, which can be obtained at http://googledrive.com/host/0B6qziMs8hVGieFg0UzE0WmZaOW8/code/gop_data.zip (link taken from http://www.philkr.net/home/gop)

2. For the C++ wrapper of edge-boxes see the file `.cpp/src/edge_boxes_demo.cpp`

3. For the Python wrapper, build the `cpp` directory using `.cpp/CMakeLists.txt` to get `edge_boxes_python.so` in the `build` folder. Add the `build` folder to your `PYTHONPATH` and then:
```python
import cv2, os
from edge_boxes_python import edge_boxes_python
eb = edge_boxes_python(os.path.expanduser('~') + '/Documents/MATLAB/edges/cpp/external/gop_1.3/data/sf.dat') # string is path to the trained structured random forest, see 1.
im = cv2.imread('test.jpg')
bbs = eb.get_edge_boxes(im)
```
