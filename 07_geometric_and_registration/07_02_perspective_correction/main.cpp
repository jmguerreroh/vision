/**
 * @file main.cpp
 * @brief Perspective correction with a homography
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Homography: the projective transformation that maps two views of a plane
 * - Perspective correction: warping a slanted plane back to a frontal view
 * - Why an affine transformation is not enough when there is perspective
 *
 * @note A homography is a 3x3 matrix with 8 degrees of freedom (it is defined
 *       up to scale), so it needs 4 point correspondences, no three of them
 *       collinear:
 *         | h11  h12  tx |
 *         | h21  h22  ty |
 *         | vx   vy   1  |
 *       The last row is what the affine transform of 07_01 does not have: it
 *       makes w' depend on the position, and that division is what allows
 *       parallel lines to converge. With vx = vy = 0 the matrix is affine.
 *
 * @note A chessboard is used on purpose: its cells are square, so the result
 *       can be checked by eye. If the rectification is right, the cells come
 *       out square and aligned with the image axes.
 *
 * @see https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
 */

#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace Config
{
// Corners of the inner grid of the chessboard in data/left12.jpg, in this
// order: top-left, top-right, bottom-right, bottom-left. They were measured
// once with cv::findChessboardCorners (see 14_01_camera_calibration) and are
// hardcoded here so that the example stays about the homography
const std::vector<cv::Point2f> GRID_CORNERS = {
  {227.4f, 82.0f}, {423.5f, 70.9f}, {449.5f, 408.0f}, {198.6f, 408.8f}
};

// The inner grid is 6 x 9 corners, that is, 5 x 8 cells. Keeping that ratio in
// the output is what makes the rectified cells square
constexpr int CELL_SIZE = 58;
constexpr int OUTPUT_WIDTH = 5 * CELL_SIZE;
constexpr int OUTPUT_HEIGHT = 8 * CELL_SIZE;

// Drawing parameters
const cv::Scalar MARK_COLOR(0, 0, 255);       // BGR: red
constexpr int MARK_RADIUS = 6;
constexpr int LINE_THICKNESS = 2;
}

/**
 * @brief Destination corners: the four corners of the output image
 * @param width Width of the output image, in pixels
 * @param height Height of the output image, in pixels
 * @return Corners in the same order as Config::GRID_CORNERS
 */
std::vector<cv::Point2f> outputCorners(int width, int height)
{
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  return {{0.0f, 0.0f}, {w, 0.0f}, {w, h}, {0.0f, h}};
}

/**
 * @brief Draws the four selected corners over a copy of the image
 * @param src Input image
 * @param corners The four correspondences chosen in the input image
 * @return Copy of src with the quadrilateral and its corners marked
 */
cv::Mat drawSelection(const cv::Mat & src, const std::vector<cv::Point2f> & corners)
{
  cv::Mat marked = src.clone();

  std::vector<cv::Point> polygon(corners.begin(), corners.end());
  cv::polylines(marked, polygon, true, Config::MARK_COLOR, Config::LINE_THICKNESS);

  for (size_t i = 0; i < corners.size(); ++i) {
    cv::circle(marked, corners[i], Config::MARK_RADIUS, Config::MARK_COLOR, cv::FILLED);
    cv::putText(marked, std::to_string(i + 1), corners[i] + cv::Point2f(10, -10),
      cv::FONT_HERSHEY_SIMPLEX, 0.7, Config::MARK_COLOR, 2);
  }
  return marked;
}

/**
 * @brief Rectifies the selected quadrilateral into a frontal view
 * @param src Input image
 * @param corners The four corners of the plane in the input image
 * @return Rectified image of size OUTPUT_WIDTH x OUTPUT_HEIGHT
 */
cv::Mat correctPerspective(const cv::Mat & src, const std::vector<cv::Point2f> & corners)
{
  const std::vector<cv::Point2f> destination =
    outputCorners(Config::OUTPUT_WIDTH, Config::OUTPUT_HEIGHT);

  // Exactly 4 correspondences: the homography is determined, there is nothing
  // to fit. With more than 4, and possible mismatches, the right call is
  // findHomography with RANSAC, used in 13_04_ransac_matching
  cv::Mat h = cv::getPerspectiveTransform(corners, destination);

  std::cout << "\n=== Homography ===" << std::endl;
  std::cout << h << std::endl;
  std::cout << "Note the last row: it is not (0, 0, 1), so this transformation "
            << "is not affine" << std::endl;

  // warpPerspective walks the OUTPUT pixels and looks up where each one comes
  // from in the input (inverse mapping), interpolating the fractional positions
  cv::Mat corrected;
  cv::warpPerspective(src, corrected, h,
    cv::Size(Config::OUTPUT_WIDTH, Config::OUTPUT_HEIGHT), cv::INTER_LINEAR);
  return corrected;
}

/**
 * @brief Same rectification attempted with an affine transform
 * @param src Input image
 * @param corners The four corners of the plane in the input image
 * @return Result of using only three of the four correspondences
 *
 * An affine transform has 6 degrees of freedom and is fixed by 3 points, so the
 * fourth corner cannot be chosen: it lands wherever the other three send it.
 * The board therefore stays skewed and the cells are not square.
 */
cv::Mat tryWithAffine(const cv::Mat & src, const std::vector<cv::Point2f> & corners)
{
  const std::vector<cv::Point2f> destination =
    outputCorners(Config::OUTPUT_WIDTH, Config::OUTPUT_HEIGHT);

  const cv::Point2f source_tri[3] = {corners[0], corners[1], corners[3]};
  const cv::Point2f destination_tri[3] = {destination[0], destination[1], destination[3]};

  cv::Mat m = cv::getAffineTransform(source_tri, destination_tri);

  // Where does the corner that was left out end up? It should land on the
  // opposite corner of the output, and it does not: that gap is the perspective
  // that an affine transformation cannot represent
  std::vector<cv::Point2f> left_out = {corners[2]}, mapped;
  cv::transform(left_out, mapped, m);

  std::cout << "\n=== Affine attempt (3 points) ===" << std::endl;
  std::cout << "Fourth corner lands at " << mapped[0] << " instead of " << destination[2]
            << ", an error of " << cv::norm(mapped[0] - destination[2]) << " px" << std::endl;

  cv::Mat affine_result;
  cv::warpAffine(src, affine_result, m,
    cv::Size(Config::OUTPUT_WIDTH, Config::OUTPUT_HEIGHT), cv::INTER_LINEAR);
  return affine_result;
}

int main(int argc, char ** argv)
{
  // Load input image
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/left12.jpg | Input file}");
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
  cv::Mat src = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << image_path << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Perspective Correction Demo ===" << std::endl;
  std::cout << "Image loaded: " << src.cols << "x" << src.rows << " pixels" << std::endl;
  std::cout << "The four corners are hardcoded for data/left12.jpg; with another "
            << "image they must be updated" << std::endl;

  cv::imshow("Original with selected corners", drawSelection(src, Config::GRID_CORNERS));
  cv::imshow("Perspective corrected (4 points)",
    correctPerspective(src, Config::GRID_CORNERS));

  // The same problem solved with the wrong model, for comparison
  cv::imshow("Affine attempt (3 points)", tryWithAffine(src, Config::GRID_CORNERS));

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
