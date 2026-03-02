/**
 * @file main.cpp
 * @brief RANSAC-based image alignment using ORB features and homography
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates feature-based image alignment using
 *       ORB feature detection, brute-force matching, and RANSAC homography
 * @see https://github.com/spmallick/learnopencv/tree/master/ImageAlignment-FeatureBased
 */

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace Config
{
// ORB feature detection parameters
constexpr int MAX_FEATURES = 500;

// Matching parameters
constexpr float GOOD_MATCH_PERCENT = 0.15f;
constexpr int MIN_MATCH_COUNT = 10;

// RANSAC parameters
constexpr double RANSAC_THRESHOLD = 3.0;

// Display parameters
constexpr double RESIZE_SCALE = 0.5;
constexpr int RESIZE_THRESHOLD = 800;
constexpr double TEXT_FONT_SCALE = 1.0;
constexpr int TEXT_THICKNESS = 2;
constexpr int TEXT_MARGIN_X = 10;
constexpr int TEXT_Y_POSITION = 30;
}

/**
 * @brief Aligns im_1 to im_2 using feature-based registration with RANSAC
 * @param im_1 Input image to be aligned
 * @param im_2 Reference image (target alignment)
 * @param im_1_registered Output aligned image
 * @param homography Output homography matrix
 * @param matches_img Output matches visualization
 * @return true if alignment successful, false otherwise
 *
 * Algorithm steps:
 * 1. Detect ORB features in both images
 * 2. Match features using Brute-Force matcher with Hamming distance
 * 3. Keep only the best matches (sorted by distance)
 * 4. Compute homography using RANSAC (rejects outliers)
 * 5. Warp input image using the computed homography
 */
bool alignImages(
  const cv::Mat & im_1,
  const cv::Mat & im_2,
  cv::Mat & im_1_registered,
  cv::Mat & homography,
  cv::Mat & matches_img)
{
  // ========================================
  // Preprocessing: Convert to Grayscale
  // ========================================
  cv::Mat im_1_gray, im_2_gray;
  cv::cvtColor(im_1, im_1_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(im_2, im_2_gray, cv::COLOR_BGR2GRAY);

  // ========================================
  // Feature Detection and Description (ORB)
  // ========================================
  // ORB (Oriented FAST and Rotated BRIEF):
  // - Fast, rotation-invariant binary descriptor
  // - Patent-free alternative to SIFT/SURF
  // - Uses FAST keypoint detector + BRIEF descriptor
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  cv::Mat descriptors_1, descriptors_2;

  cv::Ptr<cv::Feature2D> orb = cv::ORB::create(Config::MAX_FEATURES);
  orb->detectAndCompute(im_1_gray, cv::Mat(), keypoints_1, descriptors_1);
  orb->detectAndCompute(im_2_gray, cv::Mat(), keypoints_2, descriptors_2);

  std::cout << "Detected " << keypoints_1.size() << " features in image 1" << std::endl;
  std::cout << "Detected " << keypoints_2.size() << " features in image 2" << std::endl;

  // ========================================
  // Feature Matching
  // ========================================
  // Brute-Force matcher with Hamming distance (for binary descriptors)
  // Hamming distance: counts differing bits between binary descriptors
  std::vector<cv::DMatch> matches;
  cv::Ptr<cv::DescriptorMatcher> matcher =
    cv::DescriptorMatcher::create("BruteForce-Hamming");
  matcher->match(descriptors_1, descriptors_2, matches, cv::Mat());

  std::cout << "Found " << matches.size() << " initial matches" << std::endl;

  // Sort matches by distance (lower distance = better match)
  std::sort(matches.begin(), matches.end());

  // Keep only the top GOOD_MATCH_PERCENT of matches
  const int num_good_matches = static_cast<int>(matches.size() * Config::GOOD_MATCH_PERCENT);
  matches.erase(matches.begin() + num_good_matches, matches.end());

  std::cout << "Keeping top " << matches.size() << " matches ("
            << (Config::GOOD_MATCH_PERCENT * 100) << "%)" << std::endl;

  // Check if we have enough matches for homography estimation
  if (static_cast<int>(matches.size()) < Config::MIN_MATCH_COUNT) {
    std::cerr << "Error: Not enough good matches (" << matches.size()
              << " < " << Config::MIN_MATCH_COUNT << ")" << std::endl;
    return false;
  }

  // Create matches visualization
  cv::drawMatches(im_1, keypoints_1, im_2, keypoints_2, matches, matches_img);

  // ========================================
  // Homography Estimation with RANSAC
  // ========================================
  // Extract point correspondences from good matches
  std::vector<cv::Point2f> points1, points2;
  for (size_t i = 0; i < matches.size(); ++i) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  // cv::findHomography: Computes perspective transformation
  // RANSAC: Random Sample Consensus - iteratively fits model to inliers
  //   - Randomly selects minimal subset (4 points for homography)
  //   - Computes model and counts inliers
  //   - Keeps best model after iterations
  // Output: 3x3 homography matrix H where p_2 = H·p_1
  //
  // NOTE: RANSAC is stochastic (random), so each run produces slightly
  //       different homographies (but visually similar alignments).
  //       Use cv::setRNGSeed(0) for reproducible results.
  //
  // Threshold: Maximum reprojection error (in pixels) to consider a point as inlier
  //   - Lower values (e.g., 3.0) = stricter, more accurate but fewer inliers
  //   - Higher values (e.g., 5.0) = more permissive, more inliers but less accurate
  cv::Mat inlier_mask;
  homography = cv::findHomography(points1, points2, cv::RANSAC,
                                  Config::RANSAC_THRESHOLD, inlier_mask);

  // Count inliers vs outliers
  const int inliers = cv::countNonZero(inlier_mask);
  const int outliers = static_cast<int>(matches.size()) - inliers;
  const double inlier_percent = 100.0 * inliers / matches.size();
  const double outlier_percent = 100.0 * outliers / matches.size();

  std::cout << "\nRANSAC Results:" << std::endl;
  std::cout << "  Inliers:  " << inliers << " (" << inlier_percent << "%)" << std::endl;
  std::cout << "  Outliers: " << outliers << " (" << outlier_percent << "%)" << std::endl;

  if (homography.empty()) {
    std::cerr << "Error: Could not compute homography" << std::endl;
    return false;
  }

  // ========================================
  // Warp Image using Homography
  // ========================================
  // cv::warpPerspective: Applies perspective transformation
  // Transforms im1 to align with im2's coordinate system
  cv::warpPerspective(im_1, im_1_registered, homography, im_2.size());

  return true;
}

/**
 * @brief Resize image for display if it exceeds threshold
 */
cv::Mat resizeForDisplay(const cv::Mat & img)
{
  cv::Mat result;
  if (img.cols > Config::RESIZE_THRESHOLD) {
    cv::resize(img, result, cv::Size(), Config::RESIZE_SCALE,
              Config::RESIZE_SCALE, cv::INTER_LANCZOS4);
    return result;
  }
  return img.clone();
}

int main(int argc, char ** argv)
{
  // ========================================
  // Load Input Images
  // ========================================
  // Default image paths (can be overridden by command line arguments)
  const std::string ref_filename = argc >= 2 ? argv[1] : "../../data/form.jpg";
  const std::string img_filename = argc >= 3 ? argv[2] : "../../data/scanned-form.jpg";

  std::cout << "========================================" << std::endl;
  std::cout << "RANSAC-based Image Alignment" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Reading reference image: " << ref_filename << std::endl;

  const cv::Mat im_reference = cv::imread(ref_filename, cv::IMREAD_COLOR);
  if (im_reference.empty()) {
    std::cerr << "Error: Could not load reference image!" << std::endl;
    std::cerr << "Path: " << ref_filename << std::endl;
    std::cerr << "Usage: " << argv[0] << " [reference_image] [image_to_align]" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Reading image to align: " << img_filename << std::endl;

  const cv::Mat im = cv::imread(img_filename, cv::IMREAD_COLOR);
  if (im.empty()) {
    std::cerr << "Error: Could not load image to align!" << std::endl;
    std::cerr << "Path: " << img_filename << std::endl;
    std::cerr << "Usage: " << argv[0] << " [reference_image] [image_to_align]" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "\nReference image size: " << im_reference.size() << std::endl;
  std::cout << "Input image size: " << im.size() << std::endl;

  // ========================================
  // Perform Image Alignment
  // ========================================
  cv::Mat im_aligned, homography, matches_visualization;

  std::cout << "\nPerforming feature-based alignment..." << std::endl;
  std::cout << "========================================" << std::endl;

  const bool success = alignImages(im, im_reference, im_aligned,
                                   homography, matches_visualization);

  if (!success) {
    std::cerr << "\nAlignment failed!" << std::endl;
    return EXIT_FAILURE;
  }

  // ========================================
  // Display Results
  // ========================================
  std::cout << "\nEstimated Homography Matrix:" << std::endl;
  std::cout << homography << std::endl;

  // Resize images for display if needed
  const cv::Mat ref_display = resizeForDisplay(im_reference);
  const cv::Mat input_display = resizeForDisplay(im);
  const cv::Mat aligned_display = resizeForDisplay(im_aligned);
  const cv::Mat matches_display = resizeForDisplay(matches_visualization);

  // Create side-by-side comparison
  cv::Mat comparison;
  cv::hconcat(ref_display, aligned_display, comparison);

  const cv::Scalar text_color(10, 10, 255);
  cv::putText(comparison, "Reference",
             cv::Point(Config::TEXT_MARGIN_X, Config::TEXT_Y_POSITION),
             cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE,
             text_color, Config::TEXT_THICKNESS);
  cv::putText(comparison, "Aligned",
             cv::Point(ref_display.cols + Config::TEXT_MARGIN_X, Config::TEXT_Y_POSITION),
             cv::FONT_HERSHEY_SIMPLEX, Config::TEXT_FONT_SCALE,
             text_color, Config::TEXT_THICKNESS);

  // ========================================
  // Display All Windows
  // ========================================
  cv::imshow("1. Reference Image", ref_display);
  cv::imshow("2. Input Image (to align)", input_display);
  cv::imshow("3. Feature Matches", matches_display);
  cv::imshow("4. Aligned Result", aligned_display);
  cv::imshow("5. Comparison (Reference | Aligned)", comparison);

  std::cout << "\n========================================" << std::endl;
  std::cout << "Press any key to exit..." << std::endl;
  std::cout << "========================================" << std::endl;

  cv::waitKey();

  // ========================================
  // Cleanup and Exit
  // ========================================
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
