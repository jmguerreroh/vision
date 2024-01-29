/**
 * Opening and Closing - sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;

Mat src, dst;
int const max_operator = 1;
int const max_elem = 2;
int const max_kernel_size = 21;
const char * window_name = "Opening and Closing Demo";
const char * trackbar1 = "Operator:\n 0: Opening - 1: Closing";
const char * trackbar2 = "Element:\n 0: Rect - 1: Cross - 2: Ellipse";
const char * trackbar3 = "Kernel size:\n 2n +1";

void Morphology_Operations(int, void *)
{
  int morph_operator = getTrackbarPos(trackbar1, window_name);
  int morph_elem = getTrackbarPos(trackbar2, window_name);
  int morph_size = getTrackbarPos(trackbar3, window_name);
  // Since MORPH_X : 2,3,4,5 and 6
  int operation = morph_operator + 2;
  Mat element = getStructuringElement(
    morph_elem, Size(
      2 * morph_size + 1,
      2 * morph_size + 1),
    Point(morph_size, morph_size) );
  morphologyEx(src, dst, operation, element);
  imshow(window_name, dst);
}

int main(int argc, char ** argv)
{
  CommandLineParser parser(argc, argv, "{@input | crop.png | input image}");
  src = imread(samples::findFile(parser.get<String>("@input") ), IMREAD_COLOR);
  if (src.empty()) {
    std::cout << "Could not open or find the image!\n" << std::endl;
    std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  namedWindow(window_name, WINDOW_AUTOSIZE);   // Create window
  createTrackbar(
    trackbar1, window_name, nullptr, max_operator,
    Morphology_Operations);
  createTrackbar(
    trackbar2, window_name, nullptr, max_elem,
    Morphology_Operations);
  createTrackbar(
    trackbar3, window_name, nullptr, max_kernel_size,
    Morphology_Operations);
  setTrackbarPos(trackbar3, window_name, 1);

  Morphology_Operations(0, 0);
  waitKey(0);
  return 0;
}
