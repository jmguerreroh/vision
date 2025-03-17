/**
 * Morphological contours - sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

// Global variables
cv::Mat src, dst;
int const max_operator = 1;
int const max_elem = 2;
int const max_kernel_size = 21;
const char * window_name = "Erode and Dilate Demo";
const char * trackbar1 = "Operator: 0: In - 1: Out";
const char * trackbar2 = "Element: 0: Rect - 1: Cross - 2: Ellipse";
const char * trackbar3 = "Kernel size: 2n +1";

void morphological_contours(int, void *)
{
  int morph_operator = cv::getTrackbarPos(trackbar1, window_name);
  int morph_elem = cv::getTrackbarPos(trackbar2, window_name);
  int morph_size = cv::getTrackbarPos(trackbar3, window_name);
  cv::Mat element = cv::getStructuringElement(
    morph_elem, cv::Size(
      2 * morph_size + 1,
      2 * morph_size + 1),
    cv::Point(morph_size, morph_size) );
  if (morph_operator == 0) {
    cv::erode(src, dst, element);
    // Internal contours
    dst = src - dst;
  } else {
    cv::dilate(src, dst, element);
    // External contours
    dst = dst - src;
  }
  cv::imshow(window_name, dst);
}

int main(int argc, char ** argv)
{
  // Load an image
  cv::CommandLineParser parser(argc, argv,
    "{@input | horse.png | input image}");
  src = cv::imread(cv::samples::findFile(parser.get<std::string>("@input") ), cv::IMREAD_COLOR);
  if (src.empty() ) {
    std::cout << "Could not open or find the image!\n" << std::endl;
    std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return -1;
  }

  // Create windows
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);   // Create window

  cv::createTrackbar(
    trackbar1, window_name, nullptr, max_operator,
    morphological_contours);
  cv::createTrackbar(
    trackbar2, window_name, nullptr, max_elem,
    morphological_contours);
  cv::createTrackbar(
    trackbar3, window_name, nullptr, max_kernel_size,
    morphological_contours);
  cv::setTrackbarPos(trackbar3, window_name, 1);

  // Default start
  morphological_contours(0, 0);

  cv::waitKey(0);

  return 0;
}
