/**
 * @file main.cpp
 * @brief Line fitting by least squares and by RANSAC on the same edge pixels
 * @author José Miguel Guerrero Hernández
 * @note A cornice of the facade gives a horizontal row of edge pixels, and a
 *       bird flying above it adds a small blob of edge pixels far from it.
 *       Least squares fits the line with a closed formula, but it weighs every
 *       point by its squared distance, so the bird drags the line towards it.
 *       RANSAC fits lines through random pairs of points, keeps the one that
 *       most points agree with, and only then applies least squares to those
 *       points: the bird is left out without being told which pixels are its.
 */

#include <string>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

namespace
{
// Where the cornice is: a row of the image, and a band around it in which the
// edge pixels are searched for. A coarse detection would give it, the same way
// a Hough transform gives where a line is and then least squares refines it
constexpr int CORNICE_ROW = 530;
constexpr int CORNICE_FIRST_COL = 240;
constexpr int CORNICE_LAST_COL = 500;
constexpr int BAND = 5;

// The bird flying above the cornice: every edge pixel inside this rectangle is
// one of its outlines, and nothing else is
const cv::Rect BIRD(305, 355, 40, 45);

// The part of the image that is shown, and how much it is enlarged on screen
const cv::Rect VIEW(225, 345, 290, 215);
constexpr int ZOOM = 3;

// A line y = m x + b together with how well it fits the points it was fitted to
struct Line
{
  double m = 0.0;
  double b = 0.0;
  double rms = 0.0;
};

// Least squares with the five sums of the closed formula: N, sum x, sum y,
// sum x^2 and sum x y. The residual minimized is vertical, so the error
// reported is the root mean square of y - (m x + b)
Line fitLeastSquares(const std::vector<cv::Point2f> & points)
{
  double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (size_t i = 0; i < points.size(); i++) {
    const double x = points[i].x, y = points[i].y;
    n += 1;
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  Line line;
  const double denominator = n * sxx - sx * sx;   // zero for a vertical line
  line.m = (n * sxy - sx * sy) / denominator;
  line.b = (sxx * sy - sx * sxy) / denominator;

  double squares = 0;
  for (size_t i = 0; i < points.size(); i++) {
    const double residual = points[i].y - (line.m * points[i].x + line.b);
    squares += residual * residual;
  }
  line.rms = std::sqrt(squares / n);
  return line;
}

// Inclination of the line in degrees, the angle the text of the book reports
double angleDegrees(const Line & line)
{
  return std::atan(line.m) * 180.0 / CV_PI;
}

// Perpendicular distance from r to the line through p and q: the cross product
// of the direction p->q and the vector p->r is the area of the parallelogram
// they span, and dividing by the length of the base leaves its height
double distanceToLine(const cv::Point2f & p, const cv::Point2f & q, const cv::Point2f & r)
{
  const cv::Point2f d = q - p;
  const cv::Point2f v = r - p;
  return std::abs(d.x * v.y - d.y * v.x) / cv::norm(d);
}

// Draws the points, the cornice in green and the bird in red, and the line,
// on the enlarged view
cv::Mat drawFit(
  const cv::Mat & image, const std::vector<cv::Point2f> & cornice,
  const std::vector<cv::Point2f> & bird, const Line & line, const std::string & label)
{
  cv::Mat view;
  cv::resize(image(VIEW), view, cv::Size(), ZOOM, ZOOM, cv::INTER_NEAREST);
  cv::cvtColor(view, view, cv::COLOR_GRAY2BGR);

  // The point (x, y) of the image is the pixel (x - VIEW.x) * ZOOM of the view
  for (size_t i = 0; i < cornice.size(); i++) {
    const cv::Point2f p = (cornice[i] - cv::Point2f(VIEW.x, VIEW.y)) * ZOOM;
    cv::circle(view, p, 2, cv::Scalar(0, 200, 0), cv::FILLED);
  }
  for (size_t i = 0; i < bird.size(); i++) {
    const cv::Point2f p = (bird[i] - cv::Point2f(VIEW.x, VIEW.y)) * ZOOM;
    cv::circle(view, p, 2, cv::Scalar(0, 0, 220), cv::FILLED);
  }

  // The line is drawn from the left edge of the view to the right one
  const double x0 = VIEW.x, x1 = VIEW.x + VIEW.width;
  const cv::Point2f a((x0 - VIEW.x) * ZOOM, (line.m * x0 + line.b - VIEW.y) * ZOOM);
  const cv::Point2f b((x1 - VIEW.x) * ZOOM, (line.m * x1 + line.b - VIEW.y) * ZOOM);
  cv::line(view, a, b, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);

  cv::putText(view, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  return view;
}
}  // namespace

int main(int argc, char ** argv)
{
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/building_facade.png | Input file}"
    "{threshold | 4.0 | RANSAC: distance in pixels to join the consensus}"
    "{outliers | 0.5 | RANSAC: expected fraction of outliers}"
    "{confidence | 0.99 | RANSAC: probability of drawing one clean sample}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");
  const double threshold = parser.get<double>("threshold");
  const double outliers = parser.get<double>("outliers");
  const double confidence = parser.get<double>("confidence");

  // Without this, a malformed value (--threshold=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  cv::Mat gray = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_GRAYSCALE);

  if (gray.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << image_path << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }
  if (gray.cols <= CORNICE_LAST_COL || gray.rows <= CORNICE_ROW + BAND) {
    std::cerr << "Error: the cornice and the bird are located for building_facade.png;"
              << " this image is too small to contain them." << std::endl;
    return EXIT_FAILURE;
  }

  // ========================================
  // The edge pixels
  // ========================================

  // Canny after a Gaussian blur, as in the edge detection examples
  cv::Mat blurred, edges;
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);
  cv::Canny(blurred, edges, 60, 140);

  // The cornice: in every column, the first edge pixel inside the band
  std::vector<cv::Point2f> cornice;
  for (int x = CORNICE_FIRST_COL; x <= CORNICE_LAST_COL; x++) {
    for (int y = CORNICE_ROW - BAND; y < CORNICE_ROW + BAND; y++) {
      if (edges.at<uchar>(y, x) > 0) {
        cornice.push_back(cv::Point2f(x, y));
        break;
      }
    }
  }

  // The bird: every edge pixel inside its rectangle
  std::vector<cv::Point2f> bird;
  for (int y = BIRD.y; y < BIRD.y + BIRD.height; y++) {
    for (int x = BIRD.x; x < BIRD.x + BIRD.width; x++) {
      if (edges.at<uchar>(y, x) > 0) {
        bird.push_back(cv::Point2f(x, y));
      }
    }
  }

  // All of them together: this is what a fitting method gets, with nothing that
  // tells which point comes from which structure
  std::vector<cv::Point2f> all = cornice;
  all.insert(all.end(), bird.begin(), bird.end());

  // ========================================
  // Method 1: least squares
  // ========================================

  const Line clean = fitLeastSquares(cornice);
  const Line dragged = fitLeastSquares(all);

  // ========================================
  // Method 2: RANSAC
  // ========================================

  // How many samples: a sample of two points is clean with probability
  // (1 - e)^2, so N = log(1 - p) / log(1 - (1 - e)^2) of them give at least
  // one clean sample with probability p. N counts attempts, so the value is
  // rounded up
  const double clean_sample = (1.0 - outliers) * (1.0 - outliers);
  const int iterations =
    static_cast<int>(std::log(1.0 - confidence) / std::log(1.0 - clean_sample)) + 1;

  // Fixed seed: the same samples, and the same result, on every run
  cv::RNG rng(12345);
  std::vector<bool> best_consensus(all.size(), false);
  int best_size = 0;
  for (int k = 0; k < iterations; k++) {
    // 1. Sample: the minimal sample of a line, two different points
    const int i = rng.uniform(0, static_cast<int>(all.size()));
    const int j = rng.uniform(0, static_cast<int>(all.size()));
    if (i == j || cv::norm(all[j] - all[i]) == 0) {
      continue;
    }
    // 2. Estimate: the line through them. 3. Count: the points near it
    std::vector<bool> consensus(all.size(), false);
    int size = 0;
    for (size_t r = 0; r < all.size(); r++) {
      if (distanceToLine(all[i], all[j], all[r]) < threshold) {
        consensus[r] = true;
        size++;
      }
    }
    // 4. Keep the best: the line most points agree with
    if (size > best_size) {
      best_size = size;
      best_consensus = consensus;
    }
  }

  // The final model is not the line through the two sampled points: least
  // squares is applied to the whole consensus, now that it is known
  std::vector<cv::Point2f> inliers;
  int bird_inside = 0;
  for (size_t r = 0; r < all.size(); r++) {
    if (best_consensus[r]) {
      inliers.push_back(all[r]);
      if (r >= cornice.size()) {
        bird_inside++;
      }
    }
  }
  const Line robust = fitLeastSquares(inliers);

  // ========================================
  // Results
  // ========================================

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Edge pixels: " << cornice.size() << " of the cornice and "
            << bird.size() << " of the bird" << std::endl;
  std::cout << "Least squares, cornice only:  angle " << angleDegrees(clean)
            << " deg, error " << clean.rms << " px" << std::endl;
  std::cout << "Least squares, with the bird: angle " << angleDegrees(dragged)
            << " deg, error " << dragged.rms << " px" << std::endl;
  std::cout << "RANSAC (" << iterations << " iterations, threshold " << threshold
            << " px): angle " << angleDegrees(robust) << " deg, error "
            << robust.rms << " px" << std::endl;
  std::cout << "  consensus of " << inliers.size() << " points, " << bird_inside
            << " of them from the bird" << std::endl;

  cv::imshow("1. Edges (Canny)", edges(VIEW));
  cv::imshow("2. Least squares", drawFit(gray, cornice, bird, dragged, "Least squares"));
  cv::imshow("3. RANSAC", drawFit(gray, cornice, bird, robust, "RANSAC"));

  // Wait for user input and exit
  cv::waitKey(0);
  return EXIT_SUCCESS;
}
