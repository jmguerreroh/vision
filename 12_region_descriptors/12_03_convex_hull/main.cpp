/**
 * @file main.cpp
 * @brief Convex hull, solidity and convexity defects of a region
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates how to obtain the convex hull of a contour
 *       with cv::convexHull, how to derive the solidity from it, and how to
 *       read the valleys between region and hull with cv::convexityDefects
 * @see https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
 *
 * Solidity is the area of the region divided by the area of its convex hull.
 * It is 1 for a convex region and falls as inlets appear, so it measures how
 * far the region is from being convex. Being a ratio of two areas it is
 * invariant to translation, rotation and scale.
 *
 * The convexity defects go one step further: instead of summarising the shape
 * in a single number they say how many inlets there are and where. Each one is
 * returned as four values (start, end, farthest point, depth), and the depth
 * is what separates a real valley from the one-pixel noise that any digital
 * contour carries.
 *
 * Usage: ./12_03_convex_hull
 *        ./12_03_convex_hull ../../data/horse.png
 *        ./12_03_convex_hull --depth=12
 *        ./12_03_convex_hull --help
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
// The images this example works on are about 1400 px on the long side, and
// several windows at that size do not fit on a normal screen. The processing
// always runs at full resolution: only the copy sent to the screen is reduced,
// with INTER_AREA, which is the interpolation meant for shrinking
constexpr int MAX_DISPLAY_SIDE = 800;

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
  cv::CommandLineParser parser(argc, argv,
    "{help h  |                     | Show this help message}"
    "{@input  | ../../data/horse.png| Input file}"
    "{depth   | 10.0                | Minimum defect depth, in pixels}");

  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");
  const double min_depth = parser.get<double>("depth");

  // Without this, a malformed value is reported by the parser but the example
  // carries on with the default, which is the hardest kind of failure to find
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  const cv::Mat src = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_GRAYSCALE);
  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    return EXIT_FAILURE;
  }

  // Otsu picks the threshold from the histogram, so the example works the same
  // on a clean silhouette and on a photograph that is not perfectly bilevel
  cv::Mat binary;
  cv::threshold(src, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  if (contours.empty()) {
    std::cerr << "Error: no region found in the image." << std::endl;
    return EXIT_FAILURE;
  }

  // The largest external contour is the region of interest. Anything else is
  // specks of noise, and taking the largest is the cheapest way to skip them
  const auto & contour = *std::max_element(
    contours.begin(), contours.end(),
    [](const std::vector<cv::Point> & a, const std::vector<cv::Point> & b) {
      return cv::contourArea(a) < cv::contourArea(b);
    });

  // cv::convexHull has two modes, and both are needed here:
  //   returnPoints = true  gives the polygon, to measure and to draw
  //   returnPoints = false gives the INDICES into the contour, which is the
  //                        form cv::convexityDefects requires
  std::vector<cv::Point> hull_points;
  cv::convexHull(contour, hull_points, false, true);

  std::vector<int> hull_indices;
  cv::convexHull(contour, hull_indices, false, false);

  const double area = cv::contourArea(contour);
  const double hull_area = cv::contourArea(hull_points);
  const double solidity = (hull_area > 0.0) ? area / hull_area : 0.0;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Region" << std::endl;
  std::cout << "  contour points        " << contour.size() << std::endl;
  std::cout << "  area                  " << area << " px" << std::endl;
  std::cout << "  convex hull area      " << hull_area << " px" << std::endl;
  std::cout << "  hull vertices         " << hull_points.size() << std::endl;
  std::cout << "  perimeter             " << cv::arcLength(contour, true) << " px" << std::endl;
  std::cout << std::setprecision(4);
  std::cout << "  solidity              " << solidity
            << "   (area / hull area)" << std::endl;

  // Colour copy to draw on: the region in grey, the hull in red, the defects
  // in orange with their farthest point in green, as in the book figure
  cv::Mat canvas;
  cv::cvtColor(binary, canvas, cv::COLOR_GRAY2BGR);
  canvas.setTo(cv::Scalar(60, 60, 60), binary > 0);
  cv::drawContours(canvas, std::vector<std::vector<cv::Point>>{hull_points}, 0,
    cv::Scalar(40, 40, 200), 2);

  int deep = 0, shallow = 0;
  // convexityDefects needs at least a triangle, and it refuses a hull given as
  // points: it wants the indices computed above
  if (hull_indices.size() > 3) {
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hull_indices, defects);

    std::cout << "\nConvexity defects deeper than " << std::setprecision(1)
              << min_depth << " px" << std::endl;
    std::cout << std::setprecision(2);
    for (const cv::Vec4i & d : defects) {
      // The fourth value is the depth in fixed point, 8 fractional bits:
      // dividing by 256 turns it back into pixels. Forgetting this is the
      // classic mistake with this function, and it makes every defect look
      // 256 times deeper than it is
      const double depth = d[3] / 256.0;
      if (depth < min_depth) {
        ++shallow;
        continue;
      }
      ++deep;
      const cv::Point start = contour[d[0]];
      const cv::Point end = contour[d[1]];
      const cv::Point far = contour[d[2]];

      cv::line(canvas, start, far, cv::Scalar(30, 140, 230), 1);
      cv::line(canvas, far, end, cv::Scalar(30, 140, 230), 1);
      cv::circle(canvas, far, 5, cv::Scalar(60, 180, 75), -1);

      std::cout << "  depth " << std::setw(7) << depth << " px"
                << "   farthest point (" << far.x << ", " << far.y << ")" << std::endl;
    }
    std::cout << "\n  " << deep << " defect(s) kept, " << shallow
              << " discarded as contour noise." << std::endl;
    std::cout << "  The discarded ones are one or two pixels deep: every digital"
              << std::endl;
    std::cout << "  contour has them, and they are not features of the shape."
              << std::endl;
  } else {
    std::cout << "\nThe hull has " << hull_indices.size()
              << " vertices: too few for convexityDefects, which needs a polygon."
              << std::endl;
  }

  showFit("Convex hull and convexity defects", canvas);
  std::cout << "\nPress any key to close the window..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
