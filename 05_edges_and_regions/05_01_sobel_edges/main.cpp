/**
 * Sobel edges - sample code
 * @author José Miguel Guerrero
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  // Read image
  Mat src = imread("../../data/lena.jpg", 0);
  if (src.empty()) {
    cout << "the image is not exist" << endl;
    return -1;
  }

  resize(src, src, Size(512, 512));
  src.convertTo(src, CV_32F, 1.0 / 255);

  // Masks
  Mat SobelGx = (Mat_<char>(3, 3) <<   -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);

  Mat SobelGy = (Mat_<char>(3, 3) <<   -1, -2, -1,
    0, 0, 0,
    1, 2, 1);

  // Applying masks
  Mat SobelHorizontal, SobelVertical;
  filter2D(src, SobelVertical, src.depth(), SobelGx);
  filter2D(src, SobelHorizontal, src.depth(), SobelGy);

  // Show images
  imshow("Original", src);
  imshow("Horizontal edges", SobelHorizontal);
  imshow("Vertical edges", SobelVertical);
  waitKey();

  return 0;
}
