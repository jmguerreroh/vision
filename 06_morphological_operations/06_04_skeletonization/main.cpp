/**
 * @file main.cpp
 * @brief Skeleton extraction using Zhang-Suen thinning algorithm
 * @author José Miguel Guerrero Hernández
 *
 * @note The Zhang-Suen algorithm extracts the skeleton (medial axis) of binary
 *          shapes by iteratively removing boundary pixels while preserving topology.
 *
 *          8-neighborhood labeling (used in conditions):
 *            P9  P2  P3
 *            P8  P1  P4
 *            P7  P6  P5
 *
 *          Algorithm conditions for pixel removal:
 *          1. 2 <= N(P1) <= 6  (N = number of white neighbors)
 *          2. S(P1) = 1        (S = number of 0->1 transitions in P2..P9..P2)
 *          3. Step 1: P2*P4*P6 = 0 AND P4*P6*P8 = 0
 *             Step 2: P2*P4*P8 = 0 AND P2*P6*P8 = 0
 *
 *          This example compares manual implementation with OpenCV's
 *          cv::ximgproc::thinning() (requires opencv_contrib).
 */

#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>  // Requires opencv_contrib
#include <iostream>
#include <vector>
#include <chrono>

// 8-neighborhood offsets in clockwise order starting from P2 (top)
// Index: 0=P2, 1=P3, 2=P4, 3=P5, 4=P6, 5=P7, 6=P8, 7=P9
const int DX[] = {-1, -1, 0, 1, 1, 1, 0, -1};
const int DY[] = {0, 1, 1, 1, 0, -1, -1, -1};

/**
 * @brief Get neighbor pixel value (0 or 1)
 * @param img Binary image (0 or 255 values)
 * @param x Row of center pixel
 * @param y Column of center pixel
 * @param idx Neighbor index (0-7, clockwise from top)
 * @return 1 if neighbor is white, 0 otherwise
 */
inline int getNeighbor(const cv::Mat & img, int x, int y, int idx)
{
  return img.at<uchar>(x + DX[idx], y + DY[idx]) == 255 ? 1 : 0;
}

/**
 * @brief Count white neighbors in the 8-neighborhood
 * @param img Binary image
 * @param x Row coordinate
 * @param y Column coordinate
 * @return Number of white neighbors (0-8)
 */
int countNeighbors(const cv::Mat & img, int x, int y)
{
  int count = 0;
  for (int i = 0; i < 8; i++) {
    count += getNeighbor(img, x, y, i);
  }
  return count;
}

/**
 * @brief Count 0->1 transitions in circular scan of neighbors
 * @param img Binary image
 * @param x Row coordinate
 * @param y Column coordinate
 * @return Number of transitions (1 = simple point, >1 = junction)
 */
int countTransitions(const cv::Mat & img, int x, int y)
{
  int transitions = 0;
  for (int i = 0; i < 8; i++) {
    // Check transition from current neighbor to next (wrapping to 0)
    if (getNeighbor(img, x, y, i) == 0 && getNeighbor(img, x, y, (i + 1) % 8) == 1) {
      transitions++;
    }
  }
  return transitions;
}

/**
 * @brief Check if pixel can be removed in the given step
 * @param img Binary image
 * @param x Row coordinate
 * @param y Column coordinate
 * @param step Thinning step (1 or 2)
 * @return true if pixel should be removed
 */
bool canRemove(const cv::Mat & img, int x, int y, int step)
{
  // Get neighbor values (P2=idx0, P3=idx1, ..., P9=idx7)
  int p2 = getNeighbor(img, x, y, 0);
  int p4 = getNeighbor(img, x, y, 2);
  int p6 = getNeighbor(img, x, y, 4);
  int p8 = getNeighbor(img, x, y, 6);

  int neighbors = countNeighbors(img, x, y);
  int transitions = countTransitions(img, x, y);

  // Conditions 1 and 2 (common to both steps)
  if (neighbors < 2 || neighbors > 6 || transitions != 1) {
    return false;
  }

  // Condition 3 (different for each step)
  if (step == 1) {
    // Step 1: P2*P4*P6 = 0 AND P4*P6*P8 = 0
    return (p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0);
  } else {
    // Step 2: P2*P4*P8 = 0 AND P2*P6*P8 = 0
    return (p2 * p4 * p8 == 0) && (p2 * p6 * p8 == 0);
  }
}

/**
 * @brief Perform one sub-iteration of Zhang-Suen thinning
 * @param img Binary image (modified in place)
 * @param step Thinning step (1 or 2)
 * @return Number of pixels removed
 */
int thinningStep(cv::Mat & img, int step)
{
  std::vector<cv::Point> toRemove;

  // Find pixels to remove (skip border pixels)
  for (int x = 1; x < img.rows - 1; x++) {
    for (int y = 1; y < img.cols - 1; y++) {
      if (img.at<uchar>(x, y) == 255 && canRemove(img, x, y, step)) {
        toRemove.push_back(cv::Point(y, x));
      }
    }
  }

  // Remove marked pixels
  for (const auto & p : toRemove) {
    img.at<uchar>(p.y, p.x) = 0;
  }

  return static_cast<int>(toRemove.size());
}

/**
 * @brief Apply Zhang-Suen thinning algorithm (manual implementation)
 * @param img Binary image (modified in place)
 * @param show_progress If true, display intermediate results
 * @return Number of iterations performed
 */
int zhangSuenThinning(cv::Mat & img, bool show_progress = false)
{
  int iteration = 0;
  int removed;

  do {
    removed = 0;
    removed += thinningStep(img, 1);
    removed += thinningStep(img, 2);
    iteration++;

    if (show_progress) {
      cv::imshow("Thinning Progress", img);
      cv::waitKey(10);
    }
  } while (removed > 0);

  return iteration;
}

/**
 * @brief Color skeleton pixels in red on a BGR image
 * @param skeleton Binary skeleton image
 * @return BGR image with red skeleton
 */
cv::Mat colorSkeleton(const cv::Mat & skeleton)
{
  cv::Mat colored;
  cv::cvtColor(skeleton, colored, cv::COLOR_GRAY2BGR);

  for (int i = 0; i < skeleton.rows; i++) {
    for (int j = 0; j < skeleton.cols; j++) {
      if (skeleton.at<uchar>(i, j) == 255) {
        colored.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
      }
    }
  }
  return colored;
}

/**
 * @brief Compute difference between two binary images
 * @param img1 First binary image
 * @param img2 Second binary image
 * @return Number of different pixels
 */
int countDifferences(const cv::Mat & img1, const cv::Mat & img2)
{
  cv::Mat diff;
  cv::absdiff(img1, img2, diff);
  return cv::countNonZero(diff);
}

int main(int argc, char ** argv)
{
  // Load image from argument or default
  const char * filename = argc >= 2 ? argv[1] : "../../data/star.jpg";
  cv::Mat original = cv::imread(filename, cv::IMREAD_GRAYSCALE);

  if (original.empty()) {
    std::cerr << "Error: Cannot load image: " << filename << std::endl;
    return EXIT_FAILURE;
  }

  // Binarize the image
  cv::Mat binary;
  cv::threshold(original, binary, 127, 255, cv::THRESH_BINARY);

  // Manual Zhang-Suen implementation
  cv::Mat skeleton_manual = binary.clone();
  auto t1 = std::chrono::high_resolution_clock::now();
  int iterations = zhangSuenThinning(skeleton_manual, false);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration_manual = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  // OpenCV ximgproc::thinning (Zhang-Suen algorithm)
  // cv::ximgproc::thinning(src, dst, thinningType)
  // thinningType:
  //   THINNING_ZHANGSUEN (0): Zhang-Suen algorithm (same as manual)
  //   THINNING_GUOHALL (1): Guo-Hall algorithm (alternative)
  cv::Mat skeleton_opencv;
  t1 = std::chrono::high_resolution_clock::now();
  cv::ximgproc::thinning(binary, skeleton_opencv, cv::ximgproc::THINNING_ZHANGSUEN);
  t2 = std::chrono::high_resolution_clock::now();
  auto duration_opencv = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  // Also try Guo-Hall algorithm for comparison
  cv::Mat skeleton_guo_hall;
  cv::ximgproc::thinning(binary, skeleton_guo_hall, cv::ximgproc::THINNING_GUOHALL);

  // Compare results
  int diff_manual_vs_opencv = countDifferences(skeleton_manual, skeleton_opencv);
  int diff_zhang_vs_guo = countDifferences(skeleton_opencv, skeleton_guo_hall);

  std::cout << "=== Thinning Comparison ===" << std::endl;
  std::cout << "Manual Zhang-Suen: " << iterations << " iterations, "
            << duration_manual << " ms" << std::endl;
  std::cout << "OpenCV Zhang-Suen: " << duration_opencv << " ms" << std::endl;

  // Calculate speedup (which one is faster)
  if (duration_manual < duration_opencv && duration_manual > 0) {
    std::cout << "Manual is " << (static_cast<double>(duration_opencv) / duration_manual)
              << "x faster than OpenCV" << std::endl;
  } else if (duration_opencv < duration_manual && duration_opencv > 0) {
    std::cout << "OpenCV is " << (static_cast<double>(duration_manual) / duration_opencv)
              << "x faster than Manual" << std::endl;
  } else {
    std::cout << "Both implementations have similar performance" << std::endl;
  }

  std::cout << "Difference (Manual vs OpenCV): " << diff_manual_vs_opencv << " pixels" << std::endl;
  std::cout << "Difference (Zhang-Suen vs Guo-Hall): " << diff_zhang_vs_guo << " pixels" <<
    std::endl;

  // Create visualization grid
  // Row 1: Original | Manual Zhang-Suen
  // Row 2: OpenCV Guo-Hall | OpenCV Zhang-Suen
  cv::Mat binary_bgr;
  cv::cvtColor(binary, binary_bgr, cv::COLOR_GRAY2BGR);

  cv::Mat manual_viz = colorSkeleton(skeleton_manual);
  cv::Mat opencv_viz = colorSkeleton(skeleton_opencv);
  cv::Mat guohall_viz = colorSkeleton(skeleton_guo_hall);

  // Add labels
  cv::putText(binary_bgr, "Original", cv::Point(10, 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
  cv::putText(manual_viz, "Manual Zhang-Suen", cv::Point(10, 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
  cv::putText(opencv_viz, "OpenCV Zhang-Suen", cv::Point(10, 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
  cv::putText(guohall_viz, "OpenCV Guo-Hall", cv::Point(10, 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

  cv::Mat row1, row2, comparison;
  cv::hconcat(binary_bgr, manual_viz, row1);
  cv::hconcat(guohall_viz, opencv_viz, row2);
  cv::vconcat(row1, row2, comparison);

  // Resize if too large
  if (comparison.cols > 1200) {
    double scale = 1200.0 / comparison.cols;
    cv::resize(comparison, comparison, cv::Size(), scale, scale);
  }

  cv::imshow("Thinning Comparison", comparison);

  // Show difference image (pixels that differ between manual and OpenCV)
  cv::Mat diff_img;
  cv::absdiff(skeleton_manual, skeleton_opencv, diff_img);
  if (cv::countNonZero(diff_img) > 0) {
    cv::imshow("Difference (Manual vs OpenCV)", diff_img * 255);
  }

  cv::waitKey(0);
  return EXIT_SUCCESS;
}
