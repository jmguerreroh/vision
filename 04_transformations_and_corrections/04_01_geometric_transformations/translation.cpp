/**
 * Translation - sample code
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
  Mat src = imread("../../data/lena.jpg", IMREAD_COLOR);
  if (src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  // Create windows
  namedWindow("Original image", WINDOW_AUTOSIZE);
  namedWindow("Translation", WINDOW_AUTOSIZE);

  imshow("Original image", src);

  // Translation
  Mat translation_dst;
  float displacement_x = 100, displacement_y = 100;
  float data[6] = {1, 0, displacement_x, 0, 1, displacement_y};
  Mat trans_mat(2, 3, CV_32F, data);
  warpAffine(src, translation_dst, trans_mat, src.size());
  imshow("Translation", translation_dst);

  waitKey();
  return 0;
}
