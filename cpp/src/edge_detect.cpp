/*******************************************************************************
* Structured Edge Detection Toolbox      Version 3.01
* Code written by Piotr Dollar and Larry Zitnick, 2014.
* Licensed under the MSR-LA Full Rights License [see license.txt]
*******************************************************************************/

// Modified by Samarth Brahmbhatt to remove MEX parts

#include <opencv2/highgui/highgui.hpp>

#include "contour/structuredforest.h"
#include "imgproc/image.h"
#include "edge_detect.h"

using namespace std;
using namespace cv;

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv(Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, Mat& dst) {
    Mat _src(src.rows(), src.cols(), DataType<_Tp>::type, (void*)src.data(), src.stride()*sizeof(_Tp));
    _src.copyTo(dst);
}

Mat sf_edges(Mat& im) {
    // use GOP library to get Structured Forest Edges
    Mat im_rgb;
    cvtColor(im, im_rgb, CV_BGR2RGB);
    Image8u im_8u(im_rgb.data, im_rgb.cols, im_rgb.rows, im_rgb.channels());

    MultiScaleStructuredForest detector;
    detector.load("/home/samarth/research/gop_1.3/data/sf.dat");
    RMatrixXf im_gop_e = detector.detectAndFilter(im_8u);

    Mat im_e;
    eigen2cv(im_gop_e, im_e);

    return im_e;
}

// function to calculate the numerical gradient of a matrix in X and  Y directions
void gradient(Mat &I, Mat &gx, Mat &gy) {
    gx = Mat(I.rows, I.cols, CV_32FC1);
    gy = Mat(I.rows, I.cols, CV_32FC1);
    
    // compute rows of Gy 
    for(int i = 1; i < I.rows-1; i++) gy.row(i) = (I.row(i+1) - I.row(i-1)) * 0.5;
    gy.row(0) = I.row(1) - I.row(0);
    gy.row(gy.rows-1) = I.row(I.rows-1) - I.row(I.rows-2);

    // compute the columns of Gx
    for(int i = 1; i < I.cols-1; i++) gx.col(i) = (I.col(i+1) - I.col(i-1)) * 0.5;
    gx.col(0) = I.col(1) - I.col(0);
    gx.col(gx.cols-1) = I.col(I.cols-1) - I.col(I.cols-2);
}

void vis_matrix(Mat &m, char *window_name) {
    double min_val, max_val;
    minMaxLoc(m, &min_val, &max_val);
    Mat m_show;
    convertScaleAbs(m-min_val, m_show, 255.0/(max_val - min_val));
    namedWindow(window_name, WINDOW_NORMAL);
    imshow(window_name, m_show);
}

Mat signum(Mat &src) {
    Mat dst = Mat::zeros(src.size(), CV_32FC1);
    add(dst, 1, dst, src>0);
    add(dst, -1, dst, src<0);
    //dst += ((src > 0) & 1);
    //dst -= ((src < 0) & 1);

    return dst;
} 

Mat coarse_ori(Mat E) { // get coarse orientation, see Piotr Dollar's edgesDetect.m
    int r = 4;
    copyMakeBorder(E, E, r, r, r, r, BORDER_REFLECT);

    //f = [1:r r+1 r:-1:1]/(r+1)^2;
    Mat f = (r+1) * Mat::ones(1, 2*r+1, CV_32F);
    for(int i = 0; i < r; i++) {
        f.at<float>(0, i) = i+1;
        f.at<float>(0, f.cols-i-1) = i+1;
    }
    sepFilter2D(E, E, CV_32F, f, f.t());
    //vis_matrix(E, "E_conv");

    Mat Ox, Oy, Oxx, Oxy, Oyy;
    gradient(E, Ox, Oy);
    gradient(Ox, Oxx, Oxy);
    gradient(Oy, Oxy, Oyy);

    //vis_matrix(Ox, "Ox");
    //vis_matrix(Oy, "Oy");

    //O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
    Mat M, O;
    cartToPolar(Oxx + 1e-5, Oyy.mul(-signum(Oxy)), M, O);
    // convert from atan2 to atan
    add(O, -PI, O, Oxx<0);
    //O -= PI*((Oxx < 0) & 1);
    // mod pi
    add(O, PI, O, O<0);
    //O += PI*((O < 0) & 1);
    return O;
}

void edge_detect(Mat &im, Mat &E, Mat &O) {
    // get structured forest edges
    E = sf_edges(im);
    //vis_matrix(E, "E");
    // get edge orientation
    O = coarse_ori(E);
    //vis_matrix(O, "O");
    // NMS on edges
    //E = edge_nms(E, O, 2, 0, 1, 4); 
    //vis_matrix(E, "E_nms");
}

/*
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

    Mat im_e = sf_edges(im);
    vis_matrix(im_e, "E");

    Mat O = coarse_ori(im_e);

    Mat im_e_nms = edge_nms(im_e, O, 2, 0, 1, 4);
    vis_matrix(im_e_nms, "E_nms");

    waitKey(-1);
    return 0;
}
*/
