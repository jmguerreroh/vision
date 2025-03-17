/**
 * Morphological operations: Opening and Closing - sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

cv::Mat src, dst;
int const max_operator = 1;
int const max_elem = 2;
int const max_kernel_size = 21;
const char * window_name = "Opening and Closing Demo";
const char * trackbar1 = "Operator: 0: Opening - 1: Closing";
const char * trackbar2 = "Element: 0: Rect - 1: Cross - 2: Ellipse";
const char * trackbar3 = "Kernel size: 2n +1";

void morphological_operations(int, void *)
{
  int morph_operator = cv::getTrackbarPos(trackbar1, window_name);
  int morph_elem = cv::getTrackbarPos(trackbar2, window_name);
  int morph_size = cv::getTrackbarPos(trackbar3, window_name);

  cv::Mat element = cv::getStructuringElement(
    morph_elem, cv::Size(
      2 * morph_size + 1,
      2 * morph_size + 1),
    cv::Point(morph_size, morph_size) );

  // morphological transformations:
  //    cv::MORPH_ERODE - Performs erosion, removing small objects and shrinking bright areas.
  //    cv::MORPH_DILATE - Performs dilation, expanding bright areas.
  //    cv::MORPH_OPEN - Opening: erosion followed by dilation (removes noise).
  //    cv::MORPH_CLOSE - Closing: dilation followed by erosion (fills small holes).
  //    cv::MORPH_GRADIENT - Morphological gradient: difference between dilation and erosion (useful for edge detection).
  //    cv::MORPH_TOPHAT - Top-hat transform: difference between the original image and the opened image (extracts small bright regions).
  //    cv::MORPH_BLACKHAT - Black-hat transform: difference between the closed image and the original image (extracts small dark regions).
  int operation = morph_operator + 2;

  // src: input image
  // dst: output image
  // operation: morphological operation
  // element: structuring element
  cv::morphologyEx(src, dst, operation, element);
  cv::imshow(window_name, dst);
}

int main(int argc, char ** argv)
{
  cv::CommandLineParser parser(argc, argv, "{@input | crop.png | input image}");
  src = cv::imread(cv::samples::findFile(parser.get<std::string>("@input") ), cv::IMREAD_COLOR);
  if (src.empty()) {
    std::cout << "Could not open or find the image!\n" << std::endl;
    std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);   // Create window

  cv::createTrackbar(
    trackbar1, window_name, nullptr, max_operator,
    morphological_operations);
  cv::createTrackbar(
    trackbar2, window_name, nullptr, max_elem,
    morphological_operations);
  cv::createTrackbar(
    trackbar3, window_name, nullptr, max_kernel_size,
    morphological_operations);
  cv::setTrackbarPos(trackbar3, window_name, 1);

  // Default start
  morphological_operations(0, 0);

  cv::waitKey(0);

  return 0;
}
