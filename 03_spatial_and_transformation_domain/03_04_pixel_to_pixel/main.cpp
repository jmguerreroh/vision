/**
 * Pixel to pixel transformation - sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  // Load an image
  Mat src = imread("../../data/lena.jpg", 0);
  if (src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  resize(src, src, Size(512, 512));

  // 1. Method inverse
  Mat dst1(src.rows, src.cols, src.type());
  // Read pixel values
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      // You can now access the pixel value and calculate the new value
      dst1.at<uchar>(i, j) = (uint)(255 - (uint)src.at<uchar>(i, j));
    }
  }

  // 2. Method threshold
  Mat dst2(src.rows, src.cols, src.type());
  uint threshold_p = 150;
  // Read pixel values
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      // You can now access the pixel value and calculate the new value
      uint value = (uint)(255 - (uint)src.at<uchar>(i, j));
      if (value > threshold_p) {
        dst2.at<uchar>(i, j) = (uint)255;
      } else {
        dst2.at<uchar>(i, j) = (uint)0;
      }
    }
  }

  // Show images
  imshow("Original", src);
  imshow("Pixel to pixel inverse", dst1);
  imshow("Pixel to pixel threshold", dst2);

  waitKey(0);
  return 0;
}
