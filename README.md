# Introduction.
Structured Edge Detection Toolbox V3.0, by Piotr Dollar (pdollar-at-gmail.com)
FOR MORE DETAILS SEE https://github.com/pdollar/edges

# License.

This code is published under the MSR-LA Full Rights License.
Please read license.txt for more info.

# This fork (C++ and Python wrappers)

This fork adds a C++ and Python wrapper for structured edges and edge-boxes object proposals (see papers above), removing the Matlab dependency. To keep the code clean and reduce effort I used the C++ implementation of structured random forest edges from Philipp Krähenbühl (http://www.philkr.net/home/gop), included in ./cpp/external.

The Python wrapper requires
1. Boost::python
2. My fork of Sudeep Pillai's numpy-opencv-converter (see https://github.com/samarth-robo/numpy-opencv-converter)
3. My fork of Hilton Bristow's cvmatio (see https://github.com/samarth-robo/cvmatio)

## Installation
1. Clone and compile numpy-opencv-converter:
```
git clone https://github.com/samarth-robo/numpy-opencv-converter.git
cd numpy-opencv-converter/build
cmake ..
make
```
2. Clone, make and install cvmatio:
```
git clone https://github.com/samarth-robo/cvmatio.git
cd cvmatio/build
cmake ..
make
make install
```
3. Clone edges and compile edges
```
git clone https://github.com/samarth-robo/edges.git
cd edges
```
Now edit `CMakeLists.txt` and set the values of `CVMATIO_PATH` and `NUMPY_OPENCV_CONVERTER_PATH`
Then,
```
cmake ..
make -j6
```

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
