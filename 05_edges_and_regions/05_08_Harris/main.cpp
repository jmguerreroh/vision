/**
 * Harris corners detector - sample code
 * @author José Miguel Guerrero
 * https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

Mat src, src_gray;
int max_thresh = 255;
const char * source_window = "Source image";
const char * corners_window = "Corners detected";
const char * trackbar = "Threshold:";
void cornerHarris_demo(int, void *);

int main(int argc, char ** argv)
{
  // Read image
  CommandLineParser parser(argc, argv, "{@input | building.jpg | input image}");
  src = imread(samples::findFile(parser.get<String>("@input") ) );
  if (src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  // Change image to grayscale
  cvtColor(src, src_gray, COLOR_BGR2GRAY);
  namedWindow(source_window);

  // Create trackbar
  createTrackbar(trackbar, source_window, nullptr, max_thresh, cornerHarris_demo);
  setTrackbarPos(trackbar, source_window, 200);
  imshow(source_window, src);

  // Callback
  cornerHarris_demo(0, 0);
  waitKey();
  return 0;
}

void cornerHarris_demo(int, void *)
{
  int blockSize = 2;
  int apertureSize = 3;
  double k = 0.04;
  int thresh = getTrackbarPos(trackbar, source_window);

  // Corner detector using default config
  Mat dst = Mat::zeros(src.size(), CV_32FC1);
  cornerHarris(src_gray, dst, blockSize, apertureSize, k);

  // Normalize and scale image from 0 to 255
  Mat dst_norm, dst_norm_scaled;
  normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs(dst_norm, dst_norm_scaled);

  // Drawing circles in detected corners
  for (int i = 0; i < dst_norm.rows; i++) {
    for (int j = 0; j < dst_norm.cols; j++) {
      if ( (int) dst_norm.at<float>(i, j) > thresh) {
        circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
      }
    }
  }

  // Show images
  namedWindow(corners_window);
  imshow(corners_window, dst_norm_scaled);
}
