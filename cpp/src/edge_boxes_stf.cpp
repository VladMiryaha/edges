/*******************************************************************************
* Structured Edge Detection Toolbox      Version 3.01
* Code written by Piotr Dollar and Larry Zitnick, 2014.
* Licensed under the MSR-LA Full Rights License [see license.txt]
*******************************************************************************/

// Modified by Samarth Brahmbhatt to remove MEX parts

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "contour/structuredforest.h"
#include "imgproc/image.h"

using namespace std;
using namespace cv;

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv(Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, Mat& dst) {
    Mat _src(src.rows(), src.cols(), DataType<_Tp>::type, (void*)src.data(), src.stride()*sizeof(_Tp));
    _src.copyTo(dst);
}

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

    cout << "Edge matrix has size " << im_gop_e.rows() << " x " << im_gop_e.cols() << " and size is " << im_gop_e.size() << endl;

    Eigen::MatrixXf::Index r, c;
    float max = im_gop_e.maxCoeff(&r, &c);
    cout << "Max = " << max << " at (" << r << ", " << c << ")" << endl;
    float min = im_gop_e.minCoeff(&r, &c);
    cout << "Min = " << min << " at (" << r << ", " << c << ")" << endl;

    Mat im_e;
    eigen2cv(im_gop_e, im_e);
    cout << im_e.size() << endl;

    Mat im_e_show;
    im_e.convertTo(im_e_show, CV_8U, 255);
    imshow("Edges", im_e_show);
    waitKey(-1);

    return 0;
}
