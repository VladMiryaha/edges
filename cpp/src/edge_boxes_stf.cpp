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

// compute x and y gradients for just one column (uses sse)
void grad1( float *I, float *Gx, float *Gy, int h, int w, int x ) {
    int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
    // compute column of Gx
    Ip=I-h; In=I+h; r=.5f;
    if(x==0) { r=1; Ip+=h; } else if(x==w-1) { r=1; In-=h; }
    if( h<4 || h%4>0 || (size_t(I)&15) || (size_t(Gx)&15) ) {
        for( y=0; y<h; y++ ) *Gx++=(*In++-*Ip++)*r;
    } else {
        _G=(__m128*) Gx; _Ip=(__m128*) Ip; _In=(__m128*) In; _r = SET(r);
        for(y=0; y<h; y+=4) *_G++=MUL(SUB(*_In++,*_Ip++),_r);
    }
    // compute column of Gy
#define GRADY(r) *Gy++=(*In++-*Ip++)*r;
    Ip=I; In=Ip+1;
    // GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
    y1=((~((size_t) Gy) + 1) & 15)/4; if(y1==0) y1=4; if(y1>h-1) y1=h-1;
    GRADY(1); Ip--; for(y=1; y<y1; y++) GRADY(.5f);
    _r = SET(.5f); _G=(__m128*) Gy;
    for(; y+4<h-1; y+=4, Ip+=4, In+=4, Gy+=4)
        *_G++=MUL(SUB(LDu(*In),LDu(*Ip)),_r);
    for(; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
#undef GRADY
}

void grad2(float *I, float *Gx, float *Gy, int h, int w, int d) {
    int o, x, c, a=w*h; for(c=0; c<d; c++) for(x=0; x<w; x++) {
        o=c*a+x*h; grad1( I+o, Gx+o, Gy+o, h, w, x );
    }
}

void gradient(Mat I, Mat &gx, Mat &gy) {
    if(I.type() != CV_32FC1) {
        cout << "Input matrix is not CV_32FC1" << endl;
        return;
    }
    gx = Mat(I.rows, I.cols, CV_32FC1);
    gy = Mat(I.rows, I.cols, CV_32FC1);
    grad2(I.data, gx.data, gy.data, I.rows, I.cols, 1);
}

Mat coarse_ori(Mat E) { // get coarse orientation, see Piotr Dollar's edgesDetect.m
    int r = 4;
    cout << E.size() << endl;
    copyMakeBorder(E, E, r, r, r, r, BORDER_REFLECT);
    cout << E.size() << endl;

    //f = [1:r r+1 r:-1:1]/(r+1)^2;
    Mat f = (r+1) * Mat::ones(1, 2*r+1, CV_32F);
    for(int i = 0; i < r; i++) {
        f.at<float>(0, i) = i+1;
        f.at<float>(0, f.cols-i-1) = i+1;
    }
    sepFilter2D(E, E, CV_32F, f, f.T());

    Mat Ox, Oy, Oxx, Oxy, Oyy;
    gradient(E, Ox, Oy);
    gradient(Ox, Oxx, Oxy);
    gradient(Oy, Oxy, Oyy);

    Mat O;
    //O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);

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

    Mat im_e_show;
    im_e.convertTo(im_e_show, CV_8U, 255);
    imshow("Edges", im_e_show);
    waitKey(-1);

    return 0;
}
