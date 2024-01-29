/**
 * Discrete Cosine Transform - sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
  // Read image
  Mat src = imread("../../data/lena.jpg", 0);
  if (src.empty()) {
    cout << "the image is not exist" << endl;
    return -1;
  }
  resize(src, src, Size(512, 512));
  src.convertTo(src, CV_32F, 1.0 / 255);

  // Discrete Cosine Transform
  Mat srcDCT;
  dct(src, srcDCT);

  // Inverse Discrete Cosine Transform
  Mat InvDCT;
  idct(srcDCT, InvDCT);

  // Show images
  imshow("src", src);
  imshow("dct", srcDCT);
  imshow("idct", InvDCT);
  waitKey();

  return 0;
}
