/**
 * @file main.cpp
 * @brief cv::Mat copy semantics and Regions of Interest (ROI)
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - That cv::Mat is a HEADER pointing to shared pixel data
 * - The difference between operator= (shares data) and clone()/copyTo() (duplicates data)
 * - How to extract a Region of Interest (ROI) with cv::Rect
 * - That an ROI shares memory with its parent image
 *
 * @note This is the #1 source of bugs for OpenCV beginners: assigning a Mat
 *       with '=' does NOT copy the image. Two Mat objects end up sharing the
 *       same pixels, and modifying one silently modifies the other.
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace
{
// The images this example works on are about 1400 px on the long side, and
// several windows at that size do not fit on a normal screen. The processing
// always runs at full resolution: only the copy sent to the screen is reduced,
// with INTER_AREA, which is the interpolation meant for shrinking
constexpr int MAX_DISPLAY_SIDE = 800;

// Returns the copy that goes to the screen, already reduced. Whatever is drawn
// on the result keeps its size in screen pixels, so labels are written here and
// not on the full-resolution frame: drawn before the reduction they shrink with
// it and stop being readable
cv::Mat fitToScreen(const cv::Mat & image)
{
  const int side = std::max(image.cols, image.rows);
  if (side <= MAX_DISPLAY_SIDE || image.empty()) {
    return image.clone();
  }
  const double factor = static_cast<double>(MAX_DISPLAY_SIDE) / side;
  cv::Mat reduced;
  cv::resize(image, reduced, cv::Size(), factor, factor, cv::INTER_AREA);
  return reduced;
}

void showFit(const std::string & window, const cv::Mat & image)
{
  cv::imshow(window, fitToScreen(image));
}
}  // namespace

int main(int argc, char ** argv)
{
  // Load the image (same pattern as 02_01)
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/starry_night.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  cv::Mat original = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);

  if (original.empty()) {
    std::cerr << "Error: Could not load image from: " << image_path << std::endl;
    return EXIT_FAILURE;
  }

  // ========================================
  // Part 1 - Assignment (=) shares the pixel data
  // ========================================
  //
  // A cv::Mat is a small HEADER (size, type, a pointer to the pixels and a
  // reference counter). operator= copies only the header: after this line
  // both objects point to the SAME pixel buffer. This makes passing images
  // around cheap, but it is not a copy.
  cv::Mat shared = original;

  // Proof: painting on 'shared' also changes 'original'
  cv::circle(shared, cv::Point(80, 80), 40, cv::Scalar(0, 0, 255), cv::FILLED);

  std::cout << "Part 1: 'shared = original' copies the header, not the pixels."
            << std::endl;
  std::cout << "        Drawing on 'shared' ALSO modified 'original' (see window)."
            << std::endl;
  showFit("1. Original after drawing on 'shared'", original);

  // ========================================
  // Part 2 - clone() / copyTo() duplicate the pixel data
  // ========================================
  //
  // clone() always allocates a new buffer and copies every pixel.
  // copyTo() does the same, reusing the destination buffer when it already
  // has the right size and type (useful inside loops to avoid reallocations).
  cv::Mat independent = original.clone();

  cv::Mat independent2;
  original.copyTo(independent2);

  // Proof: painting on the clone leaves 'original' untouched
  cv::circle(independent, cv::Point(200, 80), 40, cv::Scalar(0, 255, 0), cv::FILLED);

  std::cout << "\nPart 2: clone()/copyTo() duplicate the pixels." << std::endl;
  std::cout << "        The green circle exists only in the clone." << std::endl;
  showFit("2. Clone with green circle", independent);
  showFit("3. Original (no green circle)", original);

  // ========================================
  // Part 3 - Region of Interest (ROI)
  // ========================================
  //
  // Indexing a Mat with a cv::Rect creates a new HEADER that views a
  // rectangular window of the SAME pixel buffer -- no pixels are copied.
  // This is the idiomatic way to process a subregion: any OpenCV function
  // applied to the ROI writes directly into the parent image.
  const cv::Rect roi_rect(original.cols / 4, original.rows / 4,
                          original.cols / 2, original.rows / 2);
  cv::Mat roi = original(roi_rect);

  // Proof: converting the ROI to grayscale-looking colors modifies the
  // center of 'original' in place (cvtColor is the conversion of 02_02;
  // here it is only the excuse to write into the ROI)
  cv::Mat gray_roi;
  cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);
  cv::cvtColor(gray_roi, roi, cv::COLOR_GRAY2BGR);  // Write back INTO the ROI

  std::cout << "\nPart 3: an ROI is a window into the parent's pixels." << std::endl;
  std::cout << "        Desaturating the ROI modified the center of the original."
            << std::endl;
  showFit("4. Original with desaturated ROI", original);

  // A detached copy of a region needs an explicit clone:
  //   cv::Mat safe_crop = original(roi_rect).clone();
  cv::Mat safe_crop = original(roi_rect).clone();
  showFit("5. Independent crop (clone of the ROI)", safe_crop);

  // isContinuous() reveals the difference: full images store their rows
  // back-to-back in one block; an ROI skips memory between rows, so many
  // "process the whole buffer at once" tricks do not apply to it
  std::cout << "\nMemory layout:" << std::endl;
  std::cout << "  original.isContinuous(): " << std::boolalpha
            << original.isContinuous() << std::endl;
  std::cout << "  roi.isContinuous():      " << roi.isContinuous() << std::endl;

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
