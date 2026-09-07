/**
 * @file main.cpp
 * @brief Disparity filtering demo sample demonstrating stereo matching and WLS filtering
 * @author José Miguel Guerrero Hernández
 *
 * === How Disparity Works in OpenCV ===
 *
 * Disparity is the horizontal pixel difference between the position of the same 3D point
 * projected onto the left and right images of a rectified stereo pair:
 *
 *   disparity = x_left - x_right
 *
 * Once disparity is known, depth (Z) can be recovered using:
 *
 *   Z = (f * b) / disparity
 *
 * where f is the focal length (pixels) and b is the baseline (distance between cameras).
 *
 * --- Fixed-point representation (the 16x multiplier) ---
 *
 * OpenCV stereo matchers (StereoBM and StereoSGBM) return disparity values stored as
 * CV_16S (16-bit signed integers) scaled by a factor of 16:
 *
 *   stored_value = real_disparity * 16
 *
 * So a stored value of 256 means the actual disparity is 256 / 16 = 16 pixels.
 * This fixed-point encoding preserves sub-pixel precision without using floats.
 * To convert to a true float disparity map:
 *
 *   cv::Mat disp_float;
 *   disparity.convertTo(disp_float, CV_32F, 1.0 / 16.0);
 *
 * --- num_disparities and the 16-multiple requirement ---
 *
 * The parameter max_disparity (num_disparities) sets the disparity search range [0, max_disparity).
 * It MUST be positive and a multiple of 16 because the internal SIMD implementation
 * processes blocks of 16 disparities at a time. If the value is not a multiple of 16 it
 * is rounded up:
 *
 *   if (max_disp % 16 != 0) max_disp += 16 - (max_disp % 16);
 *
 * A typical value is 160 (= 16 × 10), meaning the algorithm searches up to 160 pixels
 * of horizontal shift between the two views.
 *
 * --- WLS post-filtering ---
 *
 * Raw disparity maps from BM / SGBM are noisy near depth discontinuities and in
 * textureless regions. The Weighted Least Squares (WLS) filter refines them by
 * jointly using the raw left disparity, the right disparity (for a confidence map),
 * and the original left color image as a guide. Key parameters:
 *   - lambda (wls_lambda):  smoothness strength (larger = smoother, typical 8000)
 *   - sigma  (wls_sigma):   sensitivity to color edges (larger = less edge-aware, typical 1.5)
 *
 * Reference: https://docs.opencv.org/master/d3/d14/tutorial_ximgproc_disparity_filtering.html
 */

#include <cstdlib>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <string>
#include <cmath>

// Function Prototypes
cv::Rect computeROI(cv::Size2i src_sz, cv::Ptr<cv::StereoMatcher> matcher_instance);

// Command Line Arguments
const std::string keys =
  "{help h usage ? |                  | print this message                                                }"
  "{@left          |../../data/aloeL.jpg    | left view of the stereopair                                       }"
  "{@right         |../../data/aloeR.jpg    | right view of the stereopair                                      }"
  "{GT             |../../data/aloeGT.png   | optional ground-truth disparity (MPI-Sintel or Middlebury format) }"
  "{dst_path       |None              | optional path to save the resulting filtered disparity map        }"
  "{dst_raw_path   |None              | optional path to save raw disparity map before filtering          }"
  "{algorithm      |bm                | stereo matching method (bm or sgbm)                               }"
  "{filter         |wls_conf          | used post-filtering (wls_conf or wls_no_conf or fbs_conf)         }"
  "{no-display     |                  | don't display results                                             }"
  "{no-downscale   |                  | force stereo matching on full-sized views to improve quality      }"
  "{dst_conf_path  |None              | optional path to save the confidence map used in filtering        }"
  "{vis_mult       |1.0               | coefficient used to scale disparity map visualizations            }"
  "{max_disparity  |160               | parameter of stereo matching                                      }"
  "{window_size    |-1                | parameter of stereo matching                                      }"
  "{wls_lambda     |8000.0            | parameter of wls post-filtering                                   }"
  "{wls_sigma      |1.5               | parameter of wls post-filtering                                   }"
  "{fbs_spatial    |16.0              | parameter of fbs post-filtering                                   }"
  "{fbs_luma       |8.0               | parameter of fbs post-filtering                                   }"
  "{fbs_chroma     |8.0               | parameter of fbs post-filtering                                   }"
  "{fbs_lambda     |128.0             | parameter of fbs post-filtering                                   }"
;

int main(int argc, char ** argv)
{
  // ----------------------------------------
  // Parse command line arguments
  // ----------------------------------------
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Disparity Filtering Demo");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  // ----------------------------------------
  // Read command line arguments
  // ----------------------------------------
  std::string left_im = parser.get<std::string>(0);
  std::string right_im = parser.get<std::string>(1);
  std::string GT_path = parser.get<std::string>("GT");
  std::string dst_path = parser.get<std::string>("dst_path");
  std::string dst_raw_path = parser.get<std::string>("dst_raw_path");
  std::string dst_conf_path = parser.get<std::string>("dst_conf_path");
  std::string algo = parser.get<std::string>("algorithm");
  std::string filter = parser.get<std::string>("filter");
  bool no_display = parser.has("no-display");
  bool no_downscale = parser.has("no-downscale");
  int max_disp = parser.get<int>("max_disparity");
  double lambda = parser.get<double>("wls_lambda");
  double sigma = parser.get<double>("wls_sigma");
  double fbs_spatial = parser.get<double>("fbs_spatial");
  double fbs_luma = parser.get<double>("fbs_luma");
  double fbs_chroma = parser.get<double>("fbs_chroma");
  double fbs_lambda = parser.get<double>("fbs_lambda");
  double vis_mult = parser.get<double>("vis_mult");

  // ----------------------------------------
  // Set window size for stereo matching
  // ----------------------------------------
  int wsize;
  int window_size_arg = parser.get<int>("window_size");
  if (window_size_arg >= 0) {
    // User provided window_size value
    wsize = window_size_arg;
  } else {
    if (algo == "sgbm") {
      // Default window size for SGBM
      wsize = 3;
    } else if (!no_downscale && algo == "bm" && filter == "wls_conf") {
      // Default window size for BM on downscaled views (downscaling is performed only for wls_conf)
      wsize = 7;
    } else {
      // Default window size for BM on full-sized views
      wsize = 15;
    }
  }

  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  // ----------------------------------------
  // Load stereo images
  // ----------------------------------------
  // Left image
  cv::Mat left = cv::imread(left_im, cv::IMREAD_COLOR);
  if (left.empty()) {
    std::cout << "Cannot read image file: " << left_im;
    return EXIT_FAILURE;
  }

  // Right image
  cv::Mat right = cv::imread(right_im, cv::IMREAD_COLOR);
  if (right.empty()) {
    std::cout << "Cannot read image file: " << right_im;
    return EXIT_FAILURE;
  }

  // Ground truth disparity map
  bool no_gt;
  cv::Mat GT_disp;
  if (GT_path == "../../data/aloeGT.png" && left_im != "../../data/aloeL.jpg") {
    no_gt = true;
  } else {
    no_gt = false;
    if (cv::ximgproc::readGT(GT_path, GT_disp) != 0) {
      std::cout << "Cannot read ground truth image file: " << GT_path << std::endl;
      return EXIT_FAILURE;
    }
  }

  // ----------------------------------------
  // Initialize matrices and variables
  // ----------------------------------------
  // Matrices to store the left and right images formatted for stereo matching
  cv::Mat left_for_matcher, right_for_matcher;
  // Matrices to store the disparity maps computed for the left and right images
  cv::Mat left_disp, right_disp;
  // Matrices for post-processed disparity maps, including filtering and solving steps
  cv::Mat filtered_disp, solved_disp, solved_filtered_disp;
  // Confidence map used for disparity filtering, initialized to full confidence (255)
  cv::Mat conf_map = cv::Mat(left.rows, left.cols, CV_8U);
  conf_map = cv::Scalar(255);
  // Region of Interest (ROI) for disparity processing
  cv::Rect ROI;
  // Pointer to the WLS (Weighted Least Squares) filter for disparity refinement
  cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
  // Timing variables
  double matching_time = 0, filtering_time = 0;
  double solving_time = 0;

  // ----------------------------------------
  // Validate parameters
  // ----------------------------------------
  if (max_disp <= 0 || max_disp % 16 != 0) {
    std::cout << "Incorrect max_disparity value: it should be positive and divisible by 16";
    return EXIT_FAILURE;
  }
  if (wsize <= 0 || wsize % 2 != 1) {
    std::cout << "Incorrect window_size value: it should be positive and odd";
    return EXIT_FAILURE;
  }

  // ----------------------------------------
  // Stereo matching and disparity filtering
  // ----------------------------------------
  // Filtering with confidence (WLS with confidence)
  if (filter == "wls_conf") {
    if (!no_downscale) {
      // Downscale the views to speed-up the matching stage, as we will need to compute both left
      // and right disparity maps for confidence map computation
      max_disp /= 2;
      if (max_disp % 16 != 0) {
        max_disp += 16 - (max_disp % 16); // Ensure max_disp is a multiple of 16
      }
      cv::resize(left, left_for_matcher, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR_EXACT);
      cv::resize(right, right_for_matcher, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR_EXACT);
    } else {
      // If no downscaling is applied, simply clone the original images
      left_for_matcher = left.clone();
      right_for_matcher = right.clone();
    }

    if (algo == "bm") {
      // --- Matcher: StereoBM ---
      // Using StereoBM for faster processing. If speed is not critical, StereoSGBM provides better quality
      // The WLS filter is created using the StereoMatcher instance
      // Another matcher instance is created for computing the right disparity map

      cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(max_disp, wsize);
      wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
      cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

      cv::cvtColor(left_for_matcher, left_for_matcher, cv::COLOR_BGR2GRAY);
      cv::cvtColor(right_for_matcher, right_for_matcher, cv::COLOR_BGR2GRAY);

      matching_time = static_cast<double>(cv::getTickCount());
      // Compute disparity maps for both left and right views
      left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
      matching_time = (static_cast<double>(cv::getTickCount()) - matching_time) /
        cv::getTickFrequency();
    } else if (algo == "sgbm") {
      // --- Matcher: StereoSGBM ---
      // Using StereoSGBM, which provides better quality than StereoBM
      cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(0, max_disp, wsize);
      // Set the P1 and P2 parameters for the SGBM algorithm
      left_matcher->setP1(24 * wsize * wsize);
      left_matcher->setP2(96 * wsize * wsize);
      left_matcher->setPreFilterCap(63);
      left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
      wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
      cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

      matching_time = static_cast<double>(cv::getTickCount());
      // Compute disparity maps for both left and right views
      left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
      matching_time = (static_cast<double>(cv::getTickCount()) - matching_time) /
        cv::getTickFrequency();

    } else {
      std::cout << "Unsupported algorithm";
      return EXIT_FAILURE;
    }

    // --- WLS Filtering ---
    // Disparity maps computed by the respective matcher instances, as well as the source left view are passed to the filter.
    // Note that we are using the original non-downscaled view to guide the filtering process. The disparity map is automatically
    // upscaled in an edge-aware fashion to match the original view resolution. The result is stored in filtered_disp.
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    filtering_time = static_cast<double>(cv::getTickCount());
    wls_filter->filter(left_disp, left, filtered_disp, right_disp);
    filtering_time = (static_cast<double>(cv::getTickCount()) - filtering_time) /
      cv::getTickFrequency();

    // Retrieve the confidence map
    conf_map = wls_filter->getConfidenceMap();
    // Get the ROI that was used in the last filter call
    ROI = wls_filter->getROI();

    if (!no_downscale) {
      // If downscaling was applied, upscale disparity and ROI for proper comparison
      cv::resize(left_disp, left_disp, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR_EXACT);
      left_disp = left_disp * 2.0;
      ROI = cv::Rect(ROI.x * 2, ROI.y * 2, ROI.width * 2, ROI.height * 2);
    }
  }
  // Filtering with FBS and confidence using WLS pre-processing
  else if (filter == "fbs_conf") {
    if (!no_downscale) {
      // Downscale the views to speed-up the matching stage, as we will need to compute both left
      // and right disparity maps for confidence map computation
      max_disp /= 2;
      if (max_disp % 16 != 0) {
        max_disp += 16 - (max_disp % 16);
      }
      cv::resize(left, left_for_matcher, cv::Size(), 0.5, 0.5);
      cv::resize(right, right_for_matcher, cv::Size(), 0.5, 0.5);
    } else {
      // Keep the original views
      left_for_matcher = left.clone();
      right_for_matcher = right.clone();
    }

    if (algo == "bm") {
      // --- Matcher: StereoBM ---
      // Compute disparity maps for both left and right views
      cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(max_disp, wsize);
      wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
      cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

      cv::cvtColor(left_for_matcher, left_for_matcher, cv::COLOR_BGR2GRAY);
      cv::cvtColor(right_for_matcher, right_for_matcher, cv::COLOR_BGR2GRAY);

      matching_time = static_cast<double>(cv::getTickCount());
      left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
      matching_time = (static_cast<double>(cv::getTickCount()) - matching_time) /
        cv::getTickFrequency();
    } else if (algo == "sgbm") {
      // --- Matcher: StereoSGBM ---
      cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(0, max_disp, wsize);
      left_matcher->setP1(24 * wsize * wsize);
      left_matcher->setP2(96 * wsize * wsize);
      left_matcher->setPreFilterCap(63);
      left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
      wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
      cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

      matching_time = static_cast<double>(cv::getTickCount());
      // Compute disparity maps for both left and right views
      left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
      matching_time = (static_cast<double>(cv::getTickCount()) - matching_time) /
        cv::getTickFrequency();

    } else {
      std::cout << "Unsupported algorithm";
      return EXIT_FAILURE;
    }

    // --- WLS Filtering (pre-step for FBS) ---
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    filtering_time = static_cast<double>(cv::getTickCount());
    wls_filter->filter(left_disp, left, filtered_disp, right_disp);
    filtering_time = (static_cast<double>(cv::getTickCount()) - filtering_time) /
      cv::getTickFrequency();
    conf_map = wls_filter->getConfidenceMap();

    cv::Mat left_disp_resized;
    cv::resize(left_disp, left_disp_resized, left.size());

    // Get the ROI that was used in the last filter call
    ROI = wls_filter->getROI();
    if (!no_downscale) {
      // Upscale raw disparity and ROI back for a proper comparison
      cv::resize(left_disp, left_disp, cv::Size(), 2.0, 2.0);
      left_disp = left_disp * 2.0;
      left_disp_resized = left_disp_resized * 2.0;
      ROI = cv::Rect(ROI.x * 2, ROI.y * 2, ROI.width * 2, ROI.height * 2);
    }

#ifdef HAVE_EIGEN
    // --- FBS Filtering ---
    solving_time = static_cast<double>(cv::getTickCount());
    cv::ximgproc::fastBilateralSolverFilter(
          left, left_disp_resized, conf_map / 255.0f, solved_disp, fbs_spatial,
          fbs_luma, fbs_chroma, fbs_lambda);
    solving_time = (static_cast<double>(cv::getTickCount()) - solving_time) /
      cv::getTickFrequency();

    cv::ximgproc::fastBilateralSolverFilter(
          left, filtered_disp, conf_map / 255.0f, solved_filtered_disp,
          fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda);
#else
    (void)fbs_spatial;
    (void)fbs_luma;
    (void)fbs_chroma;
    (void)fbs_lambda;
#endif

  }
  // Filtering without confidence (WLS without confidence)
  else if (filter == "wls_no_conf") {
    // There is no convenience function for the case of filtering with no confidence,
    // so we will need to set the ROI and matcher parameters manually
    left_for_matcher = left.clone();
    right_for_matcher = right.clone();

    if (algo == "bm") {
      // --- Matcher: StereoBM (no confidence) ---
      cv::Ptr<cv::StereoBM> matcher = cv::StereoBM::create(max_disp, wsize);
      matcher->setTextureThreshold(0);
      matcher->setUniquenessRatio(0);
      cv::cvtColor(left_for_matcher, left_for_matcher, cv::COLOR_BGR2GRAY);
      cv::cvtColor(right_for_matcher, right_for_matcher, cv::COLOR_BGR2GRAY);
      ROI = computeROI(left_for_matcher.size(), matcher);
      wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
      wls_filter->setDepthDiscontinuityRadius(static_cast<int>(std::ceil(0.33 * wsize)));

      matching_time = static_cast<double>(cv::getTickCount());
      matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      matching_time = (static_cast<double>(cv::getTickCount()) - matching_time) /
        cv::getTickFrequency();

    } else if (algo == "sgbm") {
      // --- Matcher: StereoSGBM (no confidence) ---
      cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(0, max_disp, wsize);
      matcher->setUniquenessRatio(0);
      matcher->setDisp12MaxDiff(1000000);
      matcher->setSpeckleWindowSize(0);
      matcher->setP1(24 * wsize * wsize);
      matcher->setP2(96 * wsize * wsize);
      matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
      ROI = computeROI(left_for_matcher.size(), matcher);
      wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
      wls_filter->setDepthDiscontinuityRadius(static_cast<int>(std::ceil(0.5 * wsize)));

      matching_time = static_cast<double>(cv::getTickCount());
      matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      matching_time = (static_cast<double>(cv::getTickCount()) - matching_time) /
        cv::getTickFrequency();

    } else {
      std::cout << "Unsupported algorithm";
      return EXIT_FAILURE;
    }

    // --- WLS Filtering (no confidence) ---
    wls_filter->setLambda(lambda);

  } else {
    std::cout << "Unsupported filter";
    return EXIT_FAILURE;
  }

  // ----------------------------------------
  // Print statistics
  // ----------------------------------------
  std::cout.precision(2);
  std::cout << "Matching time:  " << matching_time << "s" << std::endl;
  std::cout << "Filtering time: " << filtering_time << "s" << std::endl;
  std::cout << "Solving time: " << solving_time << "s" << std::endl;
  std::cout << std::endl;

  double MSE_before, percent_bad_before, MSE_after, percent_bad_after;
  if (!no_gt) {
    MSE_before = cv::ximgproc::computeMSE(GT_disp, left_disp, ROI);
    percent_bad_before = cv::ximgproc::computeBadPixelPercent(GT_disp, left_disp, ROI);
    MSE_after = cv::ximgproc::computeMSE(GT_disp, filtered_disp, ROI);
    percent_bad_after = cv::ximgproc::computeBadPixelPercent(GT_disp, filtered_disp, ROI);

    std::cout.precision(5);
    std::cout << "MSE before filtering: " << MSE_before << std::endl;
    std::cout << "MSE after filtering:  " << MSE_after << std::endl;
    std::cout << std::endl;
    std::cout.precision(3);
    std::cout << "Percent of bad pixels before filtering: " << percent_bad_before << std::endl;
    std::cout << "Percent of bad pixels after filtering:  " << percent_bad_after << std::endl;
  }

  // ----------------------------------------
  // Save output images
  // ----------------------------------------
  if (dst_path != "None") {
    cv::Mat filtered_disp_vis;
    cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
    cv::imwrite(dst_path, filtered_disp_vis);
  }
  if (dst_raw_path != "None") {
    cv::Mat raw_disp_vis;
    cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, vis_mult);
    cv::imwrite(dst_raw_path, raw_disp_vis);
  }
  if (dst_conf_path != "None") {
    cv::imwrite(dst_conf_path, conf_map);
  }

  // ----------------------------------------
  // Display results
  // ----------------------------------------
  if (!no_display) {
    cv::namedWindow("left", cv::WINDOW_AUTOSIZE);
    cv::resize(left, left, left.size() / 2);
    cv::imshow("left", left);
    cv::namedWindow("right", cv::WINDOW_AUTOSIZE);
    cv::resize(right, right, right.size() / 2);
    cv::imshow("right", right);

    if (!no_gt) {
      cv::Mat GT_disp_vis;
      cv::ximgproc::getDisparityVis(GT_disp, GT_disp_vis, vis_mult);
      cv::namedWindow("ground-truth disparity", cv::WINDOW_AUTOSIZE);
      cv::resize(GT_disp_vis, GT_disp_vis, GT_disp_vis.size() / 2);
      cv::imshow("ground-truth disparity", GT_disp_vis);
    }

    cv::Mat raw_disp_vis;
    cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, vis_mult);
    cv::namedWindow("raw disparity", cv::WINDOW_AUTOSIZE);
    cv::resize(raw_disp_vis, raw_disp_vis, raw_disp_vis.size() / 2);
    cv::imshow("raw disparity", raw_disp_vis);
    cv::Mat filtered_disp_vis;
    cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
    cv::namedWindow("filtered disparity", cv::WINDOW_AUTOSIZE);
    cv::resize(filtered_disp_vis, filtered_disp_vis, filtered_disp_vis.size() / 2);
    cv::imshow("filtered disparity", filtered_disp_vis);

    if (!solved_disp.empty()) {
      cv::Mat solved_disp_vis;
      cv::ximgproc::getDisparityVis(solved_disp, solved_disp_vis, vis_mult);
      cv::namedWindow("solved disparity", cv::WINDOW_AUTOSIZE);
      cv::resize(solved_disp_vis, solved_disp_vis, solved_disp_vis.size() / 2);
      cv::imshow("solved disparity", solved_disp_vis);

      cv::Mat solved_filtered_disp_vis;
      cv::ximgproc::getDisparityVis(solved_filtered_disp, solved_filtered_disp_vis, vis_mult);
      cv::namedWindow("solved wls disparity", cv::WINDOW_AUTOSIZE);
      cv::resize(solved_filtered_disp_vis, solved_filtered_disp_vis,
        solved_filtered_disp_vis.size() / 2);
      cv::imshow("solved wls disparity", solved_filtered_disp_vis);
    }

    cv::waitKey(0);
  }

  return EXIT_SUCCESS;
}

/**
 * @brief Computes the Region of Interest (ROI) for disparity processing
 * @param src_sz Source image size
 * @param matcher_instance Pointer to the stereo matcher instance
 * @return cv::Rect The computed ROI
 */
cv::Rect computeROI(cv::Size2i src_sz, cv::Ptr<cv::StereoMatcher> matcher_instance)
{
  int min_disparity = matcher_instance->getMinDisparity();
  int num_disparities = matcher_instance->getNumDisparities();
  int block_size = matcher_instance->getBlockSize();

  int bs2 = block_size / 2;
  int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

  int xmin = maxD + bs2;
  int xmax = src_sz.width + minD - bs2;
  int ymin = bs2;
  int ymax = src_sz.height - bs2;

  cv::Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
  return r;
}
