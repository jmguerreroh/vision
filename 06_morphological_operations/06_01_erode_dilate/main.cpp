/**
 * Erosion and Dilation - sample code
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
const char * trackbar1 = "Operator: 0: Erode - 1: Dilate ";
const char * trackbar2 = "Element: 0: Rect - 1: Cross - 2: Ellipse";
const char * trackbar3 = "Kernel size: 2n +1";

void erode_dilate(int, void *)
{
  int morph_operator = cv::getTrackbarPos(trackbar1, window_name);
  int morph_elem = cv::getTrackbarPos(trackbar2, window_name);
  int morph_size = cv::getTrackbarPos(trackbar3, window_name);

  // morph_elem: The shape of the structuring element, which can be:
  //    cv::MORPH_RECT (rectangular)
  //    cv::MORPH_ELLIPSE (elliptical)
  //    cv::MORPH_CROSS (cross-shaped)
  // cv::Size(2 * morph_size + 1, 2 * morph_size + 1):
  //    Defines the size of the kernel.
  //    The formula 2 * morph_size + 1 ensures an odd size, so there is a well-defined center.
  // cv::Point(morph_size, morph_size):
  //    Specifies the anchor point (center) of the kernel, typically at the middle.
  cv::Mat element = cv::getStructuringElement(
    morph_elem, cv::Size(
      2 * morph_size + 1,
      2 * morph_size + 1),
    cv::Point(morph_size, morph_size) );

  // src: input image
  // dst: output image
  // element: structuring element
  if (morph_operator == 0) {
    cv::erode(src, dst, element);
  } else {
    cv::dilate(src, dst, element);
  }
  cv::imshow(window_name, dst);
}

int main(int argc, char ** argv)
{
  // Load an image
  cv::CommandLineParser parser(argc, argv, "{@input | crop.png | input image}");
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
    erode_dilate);
  cv::createTrackbar(
    trackbar2, window_name, nullptr, max_elem,
    erode_dilate);
  cv::createTrackbar(
    trackbar3, window_name, nullptr, max_kernel_size,
    erode_dilate);
  cv::setTrackbarPos(trackbar3, window_name, 1);

  // Default start
  erode_dilate(0, 0);

  cv::waitKey(0);
  return 0;
}
