/**
 * Rotation - sample code
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
  namedWindow("Rotation", WINDOW_AUTOSIZE);

  imshow("Original image", src);

  // Rotation
  Mat rotation_dst;
  Point center = Point(src.cols / 2, src.rows / 2);
  double angle = -50.0;
  double scale = 0.6;
  Mat rot_mat = getRotationMatrix2D(center, angle, scale);
  warpAffine(src, rotation_dst, rot_mat, src.size());
  imshow("Rotation", rotation_dst);

  waitKey();
  return 0;
}
