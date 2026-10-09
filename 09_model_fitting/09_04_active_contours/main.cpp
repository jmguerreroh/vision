/**
 * @file main.cpp
 * @brief Active contour (snake) fitted with the greedy algorithm of Williams and Shah
 * @author José Miguel Guerrero Hernández
 * @note A closed curve starts as a circle around an object and moves until it
 *       sticks to its outline. Each point minimizes an energy with three terms:
 *       continuity (keeps the points evenly spaced), curvature (keeps the
 *       curve smooth, like a rod) and image (pulls the curve to strong
 *       gradients). The greedy algorithm moves one point at a time to the
 *       position of lowest energy in its m x m neighbourhood, and stops when
 *       almost no point moves.
 *       The curvature term rounds the corners of the object. The second run
 *       keeps them: where the curve has a sharp turn over a strong edge, the
 *       curvature weight of that point is set to zero.
 */

#include <string>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

namespace
{
constexpr int POINTS = 80;            // points of the curve
constexpr int NEIGHBOURHOOD = 5;      // side m of the window each point examines
constexpr int MAX_ITERATIONS = 400;
constexpr double STOP_FRACTION = 0.02;    // T1: stop when fewer points move
constexpr double CORNER_CURVATURE = 0.25; // T2: minimum turn to be a corner
constexpr double CORNER_EDGE = 0.35;      // T3: minimum edge strength under it
constexpr double IMAGE_MIN_RANGE = 0.02;  // see normalizeTerm

// Smoothing before the gradient. It sets how far away the curve notices an
// edge: with a small sigma the gradient is zero a few pixels from the outline,
// and a curve that starts farther away has nothing that pulls it there.
// The kernel covers about three sigmas on each side
constexpr double SIGMA = 10.0;
constexpr int KERNEL = 61;

// The three terms of a neighbourhood are measured in different units, so each
// one is brought to [0, 1] before its weight is applied. With min_range, a
// neighbourhood whose values hardly change is not stretched to the whole
// interval: it is the precaution of Williams and Shah with the image term,
// because far from the edges the gradient only varies with the noise, and
// stretched to [0, 1] it would weigh as much as a real edge, so the curve
// would wander following the noise
void normalizeTerm(std::vector<double> & term, double min_range = 0.0)
{
  const double low = *std::min_element(term.begin(), term.end());
  const double high = *std::max_element(term.begin(), term.end());
  const double range = std::max(high - low, min_range);
  for (size_t k = 0; k < term.size(); k++) {
    term[k] = range > 0 ? (term[k] - low) / range : 0.0;
  }
}

// How sharply the curve turns at point i: the difference between the unit
// vectors of the segment that arrives and the segment that leaves. It is 0 on
// a straight stretch and about 1.41 at a right angle
double turn(const std::vector<cv::Point2f> & curve, int i)
{
  const int n = static_cast<int>(curve.size());
  const cv::Point2f in = curve[i] - curve[(i - 1 + n) % n];
  const cv::Point2f out = curve[(i + 1) % n] - curve[i];
  const double in_length = cv::norm(in), out_length = cv::norm(out);
  if (in_length == 0 || out_length == 0) {
    return 0.0;
  }
  return cv::norm(in * (1.0 / in_length) - out * (1.0 / out_length));
}

// One iteration of the greedy algorithm: every point, in order, moves to the
// position of lowest energy in its neighbourhood. Returns how many moved
int greedyIteration(
  std::vector<cv::Point2f> & curve, const cv::Mat & edge_strength,
  double alpha, const std::vector<double> & beta, double gamma)
{
  const int n = static_cast<int>(curve.size());
  const int half = NEIGHBOURHOOD / 2;

  // Mean spacing between consecutive points, which continuity asks every
  // point to keep with the previous one
  double spacing = 0;
  for (int i = 0; i < n; i++) {
    spacing += cv::norm(curve[(i + 1) % n] - curve[i]);
  }
  spacing /= n;

  int moved = 0;
  for (int i = 0; i < n; i++) {
    // The previous point has already moved in this iteration, the next one not yet
    const cv::Point2f previous = curve[(i - 1 + n) % n];
    const cv::Point2f next = curve[(i + 1) % n];

    std::vector<cv::Point2f> candidates;
    std::vector<double> continuity, curvature, image;
    for (int dy = -half; dy <= half; dy++) {
      for (int dx = -half; dx <= half; dx++) {
        // Candidates stay one pixel inside the image
        const float x = std::min(std::max(curve[i].x + dx, 1.0f),
                                 static_cast<float>(edge_strength.cols - 2));
        const float y = std::min(std::max(curve[i].y + dy, 1.0f),
                                 static_cast<float>(edge_strength.rows - 2));
        const cv::Point2f c(x, y);
        candidates.push_back(c);
        // Continuity, as Williams and Shah write it: how far the distance to
        // the previous point is from the mean spacing. It pulls the points
        // nowhere, it only keeps them evenly spaced. The squared distance to
        // the previous point, the direct discretization of |c'|^2, would make
        // every point pull backwards: the curve would keep sliding along the
        // outline and never stop moving
        const double gap = spacing - cv::norm(c - previous);
        continuity.push_back(gap * gap);
        // Curvature: squared second difference, small when c is halfway
        // between its two neighbours
        const cv::Point2f s = previous - 2 * c + next;
        curvature.push_back(s.x * s.x + s.y * s.y);
        // Image: minus the edge strength, so that strong edges have low energy
        image.push_back(-edge_strength.at<float>(cvRound(y), cvRound(x)));
      }
    }
    normalizeTerm(continuity);
    normalizeTerm(curvature);
    normalizeTerm(image, IMAGE_MIN_RANGE);

    size_t best = 0;
    double best_energy = 0.0;
    for (size_t k = 0; k < candidates.size(); k++) {
      const double energy = alpha * continuity[k] + beta[i] * curvature[k] + gamma * image[k];
      if (k == 0 || energy < best_energy) {
        best_energy = energy;
        best = k;
      }
    }
    if (candidates[best] != curve[i]) {
      moved++;
    }
    curve[i] = candidates[best];
  }
  return moved;
}

// Marks the points that look like a real corner: a turn that is a local
// maximum, larger than T2 and over an edge stronger than T3
std::vector<bool> findCorners(
  const std::vector<cv::Point2f> & curve, const cv::Mat & edge_strength)
{
  const int n = static_cast<int>(curve.size());
  std::vector<bool> corners(n, false);
  for (int i = 0; i < n; i++) {
    const double k = turn(curve, i);
    const bool local_max = k >= turn(curve, (i - 1 + n) % n) &&
      k >= turn(curve, (i + 1) % n);
    const float edge = edge_strength.at<float>(cvRound(curve[i].y), cvRound(curve[i].x));
    corners[i] = local_max && k > CORNER_CURVATURE && edge > CORNER_EDGE;
  }
  return corners;
}

// Runs greedy iterations until fewer than T1 N points move, or the maximum
// is reached. Returns how many were run.
// The greedy algorithm can also end in a cycle: a few points jump between two
// neighbouring cells, back and forth, and the count of moved points never goes
// below T1 N. When the curve is back where it was two iterations ago it has
// reached that cycle, and it stops as well
int evolve(
  std::vector<cv::Point2f> & curve, const cv::Mat & edge_strength,
  double alpha, const std::vector<double> & beta, double gamma)
{
  const int n = static_cast<int>(curve.size());
  std::vector<cv::Point2f> two_ago, one_ago = curve;
  for (int iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
    const int moved = greedyIteration(curve, edge_strength, alpha, beta, gamma);
    if (moved < STOP_FRACTION * n || curve == two_ago) {
      return iteration;
    }
    two_ago = one_ago;
    one_ago = curve;
  }
  return MAX_ITERATIONS;
}

// The curve drawn over the image, with the starting circle and the corners
cv::Mat drawCurve(
  const cv::Mat & gray, const cv::Point2f & center, double radius,
  const std::vector<cv::Point2f> & curve, const std::vector<bool> & corners,
  const std::string & label)
{
  cv::Mat view;
  cv::cvtColor(gray, view, cv::COLOR_GRAY2BGR);
  cv::circle(view, center, cvRound(radius), cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
  std::vector<cv::Point> polygon;
  for (size_t i = 0; i < curve.size(); i++) {
    polygon.push_back(cv::Point(cvRound(curve[i].x), cvRound(curve[i].y)));
  }
  cv::polylines(view, polygon, true, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
  for (size_t i = 0; i < curve.size(); i++) {
    if (corners[i]) {
      cv::circle(view, curve[i], 6, cv::Scalar(0, 200, 0), 2, cv::LINE_AA);
    }
  }
  cv::putText(view, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
  return view;
}
}  // namespace

int main(int argc, char ** argv)
{
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/shapes.png | Input file}"
    "{cx | 550 | Center of the starting circle, x}"
    "{cy | 550 | Center of the starting circle, y}"
    "{radius | 100 | Radius of the starting circle}"
    "{alpha | 1.0 | Weight of continuity}"
    "{beta | 1.0 | Weight of curvature}"
    "{gamma | 1.2 | Weight of the image}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");
  const cv::Point2f center(static_cast<float>(parser.get<double>("cx")),
                           static_cast<float>(parser.get<double>("cy")));
  const double radius = parser.get<double>("radius");
  const double alpha = parser.get<double>("alpha");
  const double beta = parser.get<double>("beta");
  const double gamma = parser.get<double>("gamma");

  // Without this, a malformed value (--radius=xyz) is reported by the parser
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

  // ========================================
  // Edge strength: what pulls the curve
  // ========================================

  cv::Mat smooth, gx, gy, edge_strength;
  gray.convertTo(smooth, CV_32F);
  cv::GaussianBlur(smooth, smooth, cv::Size(KERNEL, KERNEL), SIGMA);
  cv::Sobel(smooth, gx, CV_32F, 1, 0, 3);
  cv::Sobel(smooth, gy, CV_32F, 0, 1, 3);
  cv::magnitude(gx, gy, edge_strength);
  cv::normalize(edge_strength, edge_strength, 0, 1, cv::NORM_MINMAX);

  // ========================================
  // The starting curve: a circle around the object
  // ========================================

  std::vector<cv::Point2f> start;
  for (int i = 0; i < POINTS; i++) {
    const double t = 2 * CV_PI * i / POINTS;
    start.push_back(cv::Point2f(center.x + radius * std::cos(t),
                                center.y + radius * std::sin(t)));
  }

  // ========================================
  // Greedy evolution, without and with corners
  // ========================================

  // First run: the same curvature weight for every point
  std::vector<cv::Point2f> plain = start;
  const int plain_iterations =
    evolve(plain, edge_strength, alpha, std::vector<double>(POINTS, beta), gamma);
  const std::vector<bool> no_corners(POINTS, false);

  // Second run: it starts from where the first one stopped, because corners
  // are only looked for once the curve lies on the outline. Earlier, any bend
  // of the moving curve could be taken for a corner and lose its stiffness.
  // They are looked for once: recomputed at every iteration, a point that
  // gains or loses its corner mark changes the energy of its neighbours, and
  // the curve can keep oscillating there without ever stopping
  std::vector<cv::Point2f> cornered = plain;
  const std::vector<bool> corners = findCorners(cornered, edge_strength);
  std::vector<double> corner_beta(POINTS, beta);
  for (int i = 0; i < POINTS; i++) {
    if (corners[i]) {
      corner_beta[i] = 0.0;
    }
  }
  const int corner_iterations = evolve(cornered, edge_strength, alpha, corner_beta, gamma);

  int corner_count = 0;
  for (size_t i = 0; i < corners.size(); i++) {
    if (corners[i]) {
      corner_count++;
    }
  }
  std::cout << "Without corners: converged in " << plain_iterations << " iterations" << std::endl;
  std::cout << "With corners: " << corner_iterations << " more iterations, "
            << corner_count << " points with zero curvature weight" << std::endl;

  // ========================================
  // Display
  // ========================================

  cv::Mat strength_view;
  edge_strength.convertTo(strength_view, CV_8U, 255);
  cv::imshow("1. Edge strength", strength_view);
  cv::imshow("2. Without corners",
    drawCurve(gray, center, radius, plain, no_corners, "Same beta everywhere"));
  cv::imshow("3. With corners",
    drawCurve(gray, center, radius, cornered, corners, "Beta = 0 at corners"));

  // Wait for user input and exit
  cv::waitKey(0);
  return EXIT_SUCCESS;
}
