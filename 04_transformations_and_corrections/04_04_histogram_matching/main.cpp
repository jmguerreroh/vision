/**
 * Pixel to pixel transformation sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// Compute histogram and CDF for an image with mask
void do1ChnHist(const Mat& _i, const Mat& _mask, double* h, double* cdf) {
  Mat _t = _i.reshape(1,1);
  Mat _tm;
  _mask.copyTo(_tm);
  _tm = _tm.reshape(1,1);
  for(int p=0;p<_t.cols;p++) {
    if(_tm.at<int>(0,p) > 0) {
      uchar c = _t.at<int>(0,p);
      h += 1;
    }
  }
 
  //normalize hist
  Mat _tmp(1,256,CV_64FC1,h);
  double minVal,maxVal;
  minMaxLoc(_tmp,&minVal,&maxVal);
  _tmp = _tmp / maxVal;
  
  cdf[0] = h[0];
  for(int j=1;j<256;j++) {
    cdf[j] = cdf[j-1]+h[j];
  }
 
  //normalize CDF
  _tmp.data = (uchar*)cdf;
  minMaxLoc(_tmp,&minVal,&maxVal);
  _tmp = _tmp / maxVal;
}
 
// match histograms of 'src' to that of 'dst', according to both masks
void histMatchRGB(Mat& src, const Mat& src_mask, const Mat& dst, const Mat& dst_mask) {
#ifdef BTM_DEBUG
    namedWindow("original source",CV_WINDOW_AUTOSIZE);
    imshow("original source",src);
    namedWindow("original query",CV_WINDOW_AUTOSIZE);
    imshow("original query",dst);
#endif
 
    vector<Mat> chns;
    split(src,chns);
    vector<Mat> chns1;
    split(dst,chns1);
    Mat src_hist = Mat::zeros(1,256,CV_64FC1);
    Mat dst_hist = Mat::zeros(1,256,CV_64FC1);
    Mat src_cdf = Mat::zeros(1,256,CV_64FC1);
    Mat dst_cdf = Mat::zeros(1,256,CV_64FC1);
    Mat Mv(1,256,CV_8UC1);
    uchar* M = Mv.ptr<uchar>();
 
    for(int i=0;i<3;i++) {
        src_hist.setTo(Scalar(0));
        dst_hist.setTo(Scalar(0));
        src_cdf.setTo(Scalar(0));
        src_cdf.setTo(Scalar(0));
 
        do1ChnHist(chns[i],src_mask,src_hist,src_cdf);
        do1ChnHist(chns1[i],dst_mask,dst_hist,dst_cdf);
 
        uchar last = 0;
        double* _src_cdf = src_cdf.ptr<double>();
        double* _dst_cdf = dst_cdf.ptr<double>();
 
        for(int j=0;j<src_cdf.cols;j++) {
            double F1j = _src_cdf[j];
 
            for(uchar k = last; k<dst_cdf.cols; k++) {
                double F2k = _dst_cdf[k];
                if(abs(F2k - F1j) < HISTMATCH_EPSILON || F2k > F1j) {
                    M[j] = k;
                    last = k;
                    break;
                }
            }
        }
 
        Mat lut(1,256,CV_8UC1,M);
        LUT(chns[i],lut,chns[i]);
    }
 
    Mat res;
    merge(chns,res);
 
#ifdef BTM_DEBUG
    namedWindow("matched",CV_WINDOW_AUTOSIZE);
    imshow("matched",res);
 
    waitKey(BTM_WAIT_TIME);
#endif
 
    res.copyTo(src);
}

int main( int argc, char** argv ) {
  // Load an image
  Mat src = imread("../../images/lenna.jpg", 0); 
  if ( src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }
  
  resize(src, src, Size(512, 512));

  // 1. Method inverse
  Mat dst1(src.rows, src.cols, src.type());
  // Read pixel values
  for ( int i=0; i<src.rows; i++ ) {
    for ( int j=0; j<src.cols; j++ ) {
      // You can now access the pixel value and calculate the new value
      dst1.at<uchar>(i,j) = (uint)(255 - (uint)src.at<uchar>(i,j));
    }
  }

  // 2. Method threshold
  Mat dst2(src.rows, src.cols, src.type());
  uint threshold_p = 150;
  // Read pixel values
  for ( int i=0; i<src.rows; i++ ) {
    for ( int j=0; j<src.cols; j++ ) {
      // You can now access the pixel value and calculate the new value
      uint value = (uint)(255 - (uint)src.at<uchar>(i,j));
      if (value > threshold_p) 
        dst2.at<uchar>(i,j) = (uint)255;
      else 
        dst2.at<uchar>(i,j) = (uint)0;
    }
  }

  // Show images
  imshow("Original", src);
  imshow("Pixel to pixel inverse", dst1);
  imshow("Pixel to pixel threshold", dst2);


  waitKey(0);
  return 0;
}