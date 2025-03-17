/**
 * Flood Fill Demo - Sample Code
 * This program demonstrates the floodFill() function in OpenCV.
 *
 * @author José Miguel Guerrero
 * Based on OpenCV documentation: https://docs.opencv.org/3.4/d1/d17/samples_2cpp_2ffilldemo_8cpp-example.html
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

/**
 * Displays help information about the program.
 */
static void help(char ** argv)
{
  std::cout << "\nThis program demonstrates the floodFill() function\n"
            << "Call:\n" << argv[0] << " [image_name -- Default: fruits.jpg]\n" << std::endl;
  std::cout << "Hot keys: \n"
    "\tESC - quit the program\n"
    "\tc - switch color/grayscale mode\n"
    "\tm - switch mask mode\n"
    "\tr - restore the original image\n"
    "\ts - use null-range floodfill\n"
    "\tf - use gradient floodfill with fixed(absolute) range\n"
    "\tg - use gradient floodfill with floating(relative) range\n"
    "\t4 - use 4-connectivity mode\n"
    "\t8 - use 8-connectivity mode\n" << std::endl;
}

// Global variables for image processing
cv::Mat image0, image, gray, mask;
int ffillMode = 1;
int loDiff = 20, upDiff = 20;
int connectivity = 4;
bool isColor = true;
bool useMask = false;
int newMaskVal = 255;

const char * window_name = "image";
const char * trackbar1 = "lo_diff";
const char * trackbar2 = "up_diff";

/**
 * Mouse callback function to apply flood fill at the clicked position.
 */
static void onMouse(int event, int x, int y, int, void *)
{
  if (event != cv::EVENT_LBUTTONDOWN) {
    return;
  }

  int loDiff = cv::getTrackbarPos(trackbar1, window_name);
  int upDiff = cv::getTrackbarPos(trackbar2, window_name);
  cv::Point seed = cv::Point(x, y);
  int lo = ffillMode == 0 ? 0 : loDiff; // 0 or loDiff
  int up = ffillMode == 0 ? 0 : upDiff; // 0 or upDiff
  int flags = connectivity + (newMaskVal << 8) + (ffillMode == 1 ? cv::FLOODFILL_FIXED_RANGE : 0);
  int b = (unsigned)cv::theRNG() & 255;
  int g = (unsigned)cv::theRNG() & 255;
  int r = (unsigned)cv::theRNG() & 255;
  cv::Rect ccomp;
  cv::Scalar newVal = isColor ? cv::Scalar(b, g, r) : cv::Scalar(r * 0.299 + g * 0.587 + b * 0.114);
  cv::Mat dst = isColor ? image : gray;
  int area;

  // Flood-fill function to fill connected regions in the image starting from the 'seed' point.
  // It fills all connected regions where pixel intensities are within the specified range [lo, up].
  // The filled pixels are assigned the new value 'newVal'. The operation modifies the 'dst' image and can optionally update the connected components information in 'ccomp'.
  // The 'mask' defines where filling is allowed, and the 'flags' control the behavior of the fill operation.
  // This function is useful for segmenting connected regions in an image based on pixel intensity similarity.
  //    dst: The input/output image where the fill will occur. The image is modified in place.
  //    mask: A binary mask defining the area where filling is allowed. It must be a single-channel image with values 0 (not allowed) and 1 (allowed).
  //    seed: The starting point (x, y) from where the flood fill will begin.
  //    newVal: The new color or intensity value to assign to the filled pixels.
  //    ccomp: A pointer to a cv::ConnectedComponents object that will store information about connected components (optional).
  //    lo: The lower bound for pixel intensity to be considered part of the region to fill (scalar value for each channel if in color).
  //    up: The upper bound for pixel intensity to be considered part of the region to fill (scalar value for each channel if in color).
  //    flags: Flags that modify the flood fill behavior. Common flags include:
  //      4: Fill pixels up to the connected components' borders.
  //      8: Handle smaller connected regions.
  //      FLOODFILL_MASK_ONLY: Modify the mask only, not the image.
  if (useMask) {
    cv::threshold(mask, mask, 1, 128, cv::THRESH_BINARY);
    area = cv::floodFill(dst, mask, seed, newVal, &ccomp, cv::Scalar(lo, lo, lo),
      cv::Scalar(up, up, up), flags);
    cv::imshow("mask", mask);
  } else {
    area = cv::floodFill(dst, seed, newVal, &ccomp, cv::Scalar(lo, lo, lo), cv::Scalar(up, up, up),
      flags);
  }

  cv::imshow(window_name, dst);
  std::cout << area << " pixels were repainted\n";
}

/**
 * Main function to initialize the program and handle user input.
 */
int main(int argc, char ** argv)
{
  cv::CommandLineParser parser(argc, argv,
    "{help h | | show help message}{@image|fruits.jpg| input image}");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  std::string filename = parser.get<std::string>("@image");
  image0 = cv::imread(cv::samples::findFile(filename), 1);
  if (image0.empty()) {
    std::cout << "Image empty\n";
    parser.printMessage();
    return 0;
  }

  help(argv);
  image0.copyTo(image);
  cv::cvtColor(image0, gray, cv::COLOR_BGR2GRAY);
  mask.create(image0.rows + 2, image0.cols + 2, CV_8UC1);
  cv::namedWindow(window_name, 0);

  // Create trackbars for user control
  cv::createTrackbar(trackbar1, window_name, nullptr, 255, 0);
  cv::setTrackbarPos(trackbar1, window_name, 20);
  cv::createTrackbar(trackbar2, window_name, nullptr, 255, 0);
  cv::setTrackbarPos(trackbar2, window_name, 20);
  cv::setMouseCallback(window_name, onMouse, 0);

  for (;; ) {
    cv::imshow(window_name, isColor ? image : gray);
    char c = (char)cv::waitKey(0);
    if (c == 27) {
      std::cout << "Exiting ...\n";
      break;
    }

    switch (c) {
      case 'c':
        isColor = !isColor;
        std::cout << (isColor ? "Color mode is set\n" : "Grayscale mode is set\n");
        image0.copyTo(image);
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        mask = cv::Scalar::all(0);
        break;
      case 'm':
        useMask = !useMask;
        if (useMask) {
          cv::namedWindow("mask", 0);
          mask = cv::Scalar::all(0);
          cv::imshow("mask", mask);
        } else {
          cv::destroyWindow("mask");
        }
        break;
      case 'r':
        std::cout << "Original image is restored\n";
        image0.copyTo(image);
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        mask = cv::Scalar::all(0);
        break;
      case 's':
        ffillMode = 0;
        std::cout << "Simple floodfill mode is set\n";
        break;
      case 'f':
        ffillMode = 1;
        std::cout << "Fixed Range floodfill mode is set\n";
        break;
      case 'g':
        ffillMode = 2;
        std::cout << "Gradient (floating range) floodfill mode is set\n";
        break;
      case '4':
        connectivity = 4;
        std::cout << "4-connectivity mode is set\n";
        break;
      case '8':
        connectivity = 8;
        std::cout << "8-connectivity mode is set\n";
        break;
    }
  }

  return 0;
}
