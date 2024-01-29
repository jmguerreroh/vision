/**
 * Fill demo - sample code
 * @author José Miguel Guerrero
 *
 * https://docs.opencv.org/3.4/d1/d17/samples_2cpp_2ffilldemo_8cpp-example.html
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

static void help(char ** argv)
{
  cout << "\nThis program demonstrated the floodFill() function\n"
    "Call:\n"
       << argv[0]
       << " [image_name -- Default: fruits.jpg]\n" << endl;
  cout << "Hot keys: \n"
    "\tESC - quit the program\n"
    "\tc - switch color/grayscale mode\n"
    "\tm - switch mask mode\n"
    "\tr - restore the original image\n"
    "\ts - use null-range floodfill\n"
    "\tf - use gradient floodfill with fixed(absolute) range\n"
    "\tg - use gradient floodfill with floating(relative) range\n"
    "\t4 - use 4-connectivity mode\n"
    "\t8 - use 8-connectivity mode\n" << endl;
}

// Global variables
Mat image0, image, gray, mask;
int ffillMode = 1;
int loDiff = 20, upDiff = 20;
int connectivity = 4;
int isColor = true;
bool useMask = false;
int newMaskVal = 255;

const char * window_name = "image";
const char * trackbar1 = "lo_diff";
const char * trackbar2 = "up_diff";

// Get seed using the Mouse
static void onMouse(int event, int x, int y, int, void *)
{
  // Use only left button
  if (event != EVENT_LBUTTONDOWN) {
    return;
  }

  int loDiff = getTrackbarPos(trackbar1, window_name);
  int upDiff = getTrackbarPos(trackbar2, window_name);
  // Get seed
  Point seed = Point(x, y);
  // Get lower and upper values using sliders
  int lo = ffillMode == 0 ? 0 : loDiff;
  int up = ffillMode == 0 ? 0 : upDiff;
  // Flags, connectivity and mask used
  int flags = connectivity + (newMaskVal << 8) +
    (ffillMode == 1 ? FLOODFILL_FIXED_RANGE : 0);
  // Create a random color
  int b = (unsigned)theRNG() & 255;
  int g = (unsigned)theRNG() & 255;
  int r = (unsigned)theRNG() & 255;
  Rect ccomp;
  Scalar newVal = isColor ? Scalar(b, g, r) : Scalar(r * 0.299 + g * 0.587 + b * 0.114);
  Mat dst = isColor ? image : gray;
  int area;

  if (useMask) {    // Show selected areas in a binary image
    threshold(mask, mask, 1, 128, THRESH_BINARY);
    area = floodFill(
      dst, mask, seed, newVal, &ccomp, Scalar(lo, lo, lo),
      Scalar(up, up, up), flags);
    imshow("mask", mask);
  } else { // Fill areas in original image
    area = floodFill(
      dst, seed, newVal, &ccomp, Scalar(lo, lo, lo),
      Scalar(up, up, up), flags);
  }
  imshow(window_name, dst);
  cout << area << " pixels were repainted\n";
}

int main(int argc, char ** argv)
{
  cv::CommandLineParser parser(argc, argv,
    "{help h | | show help message}{@image|fruits.jpg| input image}");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  string filename = parser.get<string>("@image");
  image0 = imread(samples::findFile(filename), 1);
  if (image0.empty() ) {
    cout << "Image empty\n";
    parser.printMessage();
    return 0;
  }

  // Show help and create images and windows
  help(argv);
  image0.copyTo(image);
  cvtColor(image0, gray, COLOR_BGR2GRAY);
  mask.create(image0.rows + 2, image0.cols + 2, CV_8UC1);
  namedWindow(window_name, 0);

  // Trackbars
  createTrackbar(trackbar1, window_name, nullptr, 255, 0);
  setTrackbarPos(trackbar1, window_name, 20);
  createTrackbar(trackbar2, window_name, nullptr, 255, 0);
  setTrackbarPos(trackbar2, window_name, 20);
  setMouseCallback(window_name, onMouse, 0);     // Mouse callback

  for (;; ) {
    imshow(window_name, isColor ? image : gray);
    char c = (char)waitKey(0);
    if (c == 27) {
      cout << "Exiting ...\n";
      break;
    }
    switch (c) {
      case 'c':       // Change between color and grayscale
        if (isColor) {
          cout << "Grayscale mode is set\n";
          cvtColor(image0, gray, COLOR_BGR2GRAY);
          mask = Scalar::all(0);
          isColor = false;
        } else {
          cout << "Color mode is set\n";
          image0.copyTo(image);
          mask = Scalar::all(0);
          isColor = true;
        }
        break;
      case 'm':       // Show or not the mask
        if (useMask) {
          destroyWindow("mask");
          useMask = false;
        } else {
          namedWindow("mask", 0);
          mask = Scalar::all(0);
          imshow("mask", mask);
          useMask = true;
        }
        break;
      case 'r':       // Restore image
        cout << "Original image is restored\n";
        image0.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);
        mask = Scalar::all(0);
        break;
      case 's':
        cout << "Simple floodfill mode is set\n";
        ffillMode = 0;
        break;
      case 'f':
        cout << "Fixed Range floodfill mode is set\n";
        ffillMode = 1;
        break;
      case 'g':
        cout << "Gradient (floating range) floodfill mode is set\n";
        ffillMode = 2;
        break;
      case '4':
        cout << "4-connectivity mode is set\n";
        connectivity = 4;
        break;
      case '8':
        cout << "8-connectivity mode is set\n";
        connectivity = 8;
        break;
    }
  }
  return 0;
}
