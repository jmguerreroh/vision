/**
 * Cycle through pixels sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <math.h> 

using namespace cv;
using namespace std;




int main( int argc, char** argv ) {
  // Load an image
  Mat src = imread( "../../images/RGB.jpg", IMREAD_COLOR );
  if ( src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }


  // Split RGB channels
  vector<Mat> BGR_channels;
  split( src, BGR_channels );

  vector<Mat> CMY_channels;
  for (int i = 0; i < 3; i++) {
      CMY_channels.push_back(cv::Mat(src.size(), CV_8UC1));
  }

  // Now I can access each channel separately
  for( int i=0; i<src.rows; i++ ){
    for( int j=0; j<src.cols; j++ ){
      CMY_channels[0].at<uchar>(i,j) = (uchar)(255 - (uint)BGR_channels[0].at<uchar>(i,j));
      CMY_channels[1].at<uchar>(i,j) = (uchar)(255 - (uint)BGR_channels[1].at<uchar>(i,j));
      CMY_channels[2].at<uchar>(i,j) = (uchar)(255 - (uint)BGR_channels[2].at<uchar>(i,j));
    }
  }



  // Create CMY image
  vector<Mat> channels;
  channels.push_back(CMY_channels[0]);
  channels.push_back(CMY_channels[1]);
  channels.push_back(CMY_channels[2]);

  Mat CMY_image;
  merge(channels, CMY_image);

  


  // Resize images
  int tam = 4;
  int h = src.size().height/tam, w = src.size().width/tam;

  cv::resize(src,src,Size(h,w),0,0,INTER_LANCZOS4);
  cv::resize(BGR_channels[0],BGR_channels[0],Size(h,w),0,0,INTER_LANCZOS4);
  cv::resize(BGR_channels[1],BGR_channels[1],Size(h,w),0,0,INTER_LANCZOS4);
  cv::resize(BGR_channels[2],BGR_channels[2],Size(h,w),0,0,INTER_LANCZOS4);

  cv::Mat win_mat1_rgb(cv::Size(BGR_channels[0].size().height, BGR_channels[0].size().width*2), CV_8UC3);
  cv::Mat win_mat2_rgb(cv::Size(win_mat1_rgb.size().height, (win_mat1_rgb.size().width+BGR_channels[1].size().width)), CV_8UC3);
  cv::hconcat(BGR_channels[0], BGR_channels[1], win_mat1_rgb);
  cv::hconcat(win_mat1_rgb, BGR_channels[2], win_mat2_rgb);

  // Show image
  namedWindow( "BGR Original", WINDOW_AUTOSIZE );
  imshow("BGR Original", src);
  cv::imshow("BGR Channels", win_mat2_rgb);
 
 
  cv::resize(CMY_image,CMY_image,Size(h,w),0,0,INTER_LANCZOS4);
  cv::resize(CMY_channels[0],CMY_channels[0],Size(h,w),0,0,INTER_LANCZOS4);
  cv::resize(CMY_channels[1],CMY_channels[1],Size(h,w),0,0,INTER_LANCZOS4);
  cv::resize(CMY_channels[2],CMY_channels[2],Size(h,w),0,0,INTER_LANCZOS4);

  cv::Mat win_mat1_cmy(cv::Size(CMY_channels[0].size().height, CMY_channels[0].size().width*2), CV_8UC3);
  cv::Mat win_mat2_cmy(cv::Size(win_mat1_cmy.size().height, (win_mat1_cmy.size().width+CMY_channels[0].size().width)), CV_8UC3);
  cv::hconcat(CMY_channels[0], CMY_channels[1], win_mat1_cmy);
  cv::hconcat(win_mat1_cmy, CMY_channels[2], win_mat2_cmy);

  imshow("CMY Original", CMY_image);
  cv::imshow("CMY Channels", win_mat2_cmy);


  


  Mat HSV_opencv;
  cvtColor(src, HSV_opencv, COLOR_RGB2HSV);
  imshow("HSV OpenCV", HSV_opencv);




  vector<Mat> HSI_channels;
  for (int i = 0; i < 3; i++) {
      HSI_channels.push_back(cv::Mat(src.size(), CV_8UC1));
  }

  // Now I can access each channel separately
  for( int i=0; i<src.rows; i++ ){
    for( int j=0; j<src.cols; j++ ){
      double B = (double)BGR_channels[0].at<uchar>(i,j) / 255;
      double G = (double)BGR_channels[1].at<uchar>(i,j) / 255;
      double R = (double)BGR_channels[2].at<uchar>(i,j) /255;
      cout << R << " " << G << " " << B << endl;
      double H = ( acos((((R-G)+(R-B))/2) / sqrt( pow((R-B),2) + (R-B)*(G-B))) );
      if (B > G) H = 360 - H;
      double S = ( 1 - ((3/(R+G+B))*std::min(min(R,G),B)) );
      double I = ((R+G+B)/3);
      //cout << H << " " << S << " " << I << endl;
      HSI_channels[0].at<uchar>(i,j) = H * 255;
      //HSI_channels[0].at<uchar>(i,j) = (uchar)(255 - (uint)RGB_channels[0].at<uchar>(i,j));
      HSI_channels[1].at<uchar>(i,j) = S * 255;
      HSI_channels[2].at<uchar>(i,j) = I * 255;
    }
  }


  // Create CMY image
  vector<Mat> channels_hsi;
  channels_hsi.push_back(HSI_channels[0]);
  channels_hsi.push_back(HSI_channels[1]);
  channels_hsi.push_back(HSI_channels[2]);

  Mat HSI_image;
  merge(channels_hsi, HSI_image);


  cv::resize(HSI_image,HSI_image,Size(h,w),0,0,INTER_LANCZOS4);
  cv::resize(HSI_channels[0],HSI_channels[0],Size(h,w),0,0,INTER_LANCZOS4);
  cv::resize(HSI_channels[1],HSI_channels[1],Size(h,w),0,0,INTER_LANCZOS4);
  cv::resize(HSI_channels[2],HSI_channels[2],Size(h,w),0,0,INTER_LANCZOS4);

  cv::Mat win_mat1_hsi(cv::Size(HSI_channels[0].size().height, HSI_channels[0].size().width*2), CV_8UC3);
  cv::Mat win_mat2_hsi(cv::Size(win_mat1_hsi.size().height, (win_mat1_hsi.size().width+HSI_channels[0].size().width)), CV_8UC3);
  cv::hconcat(HSI_channels[0], HSI_channels[1], win_mat1_hsi);
  cv::hconcat(win_mat1_hsi, HSI_channels[2], win_mat2_hsi);

  imshow("HSI Original", HSI_image);
  cv::imshow("HSI Channels", win_mat2_hsi);


  
  waitKey(0);
  return 0;
}