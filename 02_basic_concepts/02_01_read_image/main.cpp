/**
 * Read image - sample code
 * @author José Miguel Guerrero
 */

#include <opencv2/highgui.hpp>

int main()
{
  // Create image variable
  cv::Mat image;

  // Read image
  image = cv::imread("../../data/lena.jpg", cv::IMREAD_COLOR);

  // Show image
  cv::imshow("TEST IMAGE", image);

  // Wait to press a key
  cv::waitKey(0);

  return 0;
}
