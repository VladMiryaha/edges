/*******************************************************************************
* Structured Edge Detection Toolbox      Version 3.01
* Code written by Piotr Dollar and Larry Zitnick, 2014.
* Licensed under the MSR-LA Full Rights License [see license.txt]
*******************************************************************************/

// Modified by Samarth Brahmbhatt to remove MEX parts

#include "math.h"
#include <algorithm>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "contour/structuredforest.h"
#include "imgproc/image.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if(argc == 1) {
        cout << "Usage: ./edge_boxes image_file" << endl;
        return -1;
    }

    Mat im = cv::imread(argv[1]);
    if(im.data == NULL) {
        cout << "Error reading image" << endl;
        return -1;
    }

    // use GOP library to get Structured Forest Edges
    Mat im_rgb;
    cvtColor(im, im_rgb, CV_BGR2RGB);
    Image8u im_8u(im_rgb.data, im_rgb.cols, im_rgb.rows, im_rgb.channels());
	MultiScaleStructuredForest detector;
	detector.load("/home/samarth/research/gop_1.3/data/sf.dat");
    cout << "GOP execution started..." << endl;  
    RMatrixXf im_gop_e = detector.detectAndFilter(im_8u);
    cout << "GOP execution finished" << endl;  

    return 0;
}
