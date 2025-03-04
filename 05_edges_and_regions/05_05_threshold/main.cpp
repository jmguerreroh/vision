/**
 * Thresholding using Binary al Otsu - sample code
 * @author José Miguel Guerrero
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main()
{
  // Create image variables
  Mat image, gray, bw_otsu, bw_otsu_bin, bw_bin;

  // Read image
  image = imread("../../data/RGB.jpg", IMREAD_COLOR);

  // Convert image to grayscale
  cvtColor(image, gray, COLOR_BGR2GRAY);

  // Convert image to binary using Otsu
  double otsu_val = cv::threshold(gray, bw_otsu, 0, 255, THRESH_OTSU);
  putText(
    bw_otsu, to_string(otsu_val), Point(
      bw_otsu.cols - 100,
      15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

  // Convert image to binary using Fix value
  double bin_val = cv::threshold(gray, bw_bin, 100, 255, THRESH_BINARY);
  putText(
    bw_bin, to_string(bin_val), Point(
      bw_bin.cols - 100,
      15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

  // Convert image to binary combining Otsu and Fix value
  double otsu_comb_val = cv::threshold(gray, bw_otsu_bin, 100, 255, THRESH_BINARY + THRESH_OTSU);
  putText(
    bw_otsu_bin, to_string(otsu_comb_val), Point(
      bw_otsu_bin.cols - 100,
      15), FONT_HERSHEY_SIMPLEX, 0.5,
    Scalar(0, 0, 255));

  // Show images
  imshow("Original", image);
  imshow("Otsu", bw_otsu);
  imshow("Binary", bw_bin);
  imshow("Otsu + Binary", bw_otsu_bin);

  // Wait to press a key
  waitKey(0);

  return 0;
}
