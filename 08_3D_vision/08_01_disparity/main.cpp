/**
 * Disparity filtering demo sample
 * @author José Miguel Guerrero
 *
 * Reference: https://docs.opencv.org/master/d3/d14/tutorial_ximgproc_disparity_filtering.html
 */

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <string>

cv::Rect computeROI(cv::Size2i src_sz, cv::Ptr<cv::StereoMatcher> matcher_instance);

const std::string keys =
  "{help h usage ? |                  | print this message                                                }"
  "{@left          |data/aloeL.jpg    | left view of the stereopair                                       }"
  "{@right         |data/aloeR.jpg    | right view of the stereopair                                      }"
  "{GT             |data/aloeGT.png   | optional ground-truth disparity (MPI-Sintel or Middlebury format) }"
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
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Disparity Filtering Demo");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  // Read command line arguments
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

  // Set window size for stereo matching
  int wsize;
  if (parser.get<int>("window_size") >= 0) {
    // User provided window_size value
    wsize = parser.get<int>("window_size");
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
    return -1;
  }

  //! [load_views]
  // Left image
  cv::Mat left = cv::imread(left_im, cv::IMREAD_COLOR);
  if (left.empty() ) {
    std::cout << "Cannot read image file: " << left_im;
    return -1;
  }

  // Right image
  cv::Mat right = cv::imread(right_im, cv::IMREAD_COLOR);
  if (right.empty() ) {
    std::cout << "Cannot read image file: " << right_im;
    return -1;
  }

  // Ground truth disparity map
  bool noGT;
  cv::Mat GT_disp;
  if (GT_path == "data/aloeGT.png" && left_im != "data/aloeL.jpg") {
    noGT = true;
  } else {
    noGT = false;
    if (cv::ximgproc::readGT(GT_path, GT_disp) != 0) {
      std::cout << "Cannot read ground truth image file: " << GT_path << std::endl;
      return -1;
    }
  }
  //! [load_views]

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
  double matching_time, filtering_time;
  double solving_time = 0;
  if (max_disp <= 0 || max_disp % 16 != 0) {
    std::cout << "Incorrect max_disparity value: it should be positive and divisible by 16";
    return -1;
  }
  if (wsize <= 0 || wsize % 2 != 1) {
    std::cout << "Incorrect window_size value: it should be positive and odd";
    return -1;
  }

  // Filtering with confidence (significantly better quality than WLS without confidence)
  if (filter == "wls_conf") {
    if (!no_downscale) {
      // Downscale the views to speed-up the matching stage, as we will need to compute both left
      // and right disparity maps for confidence map computation
      //! [downscale]
      max_disp /= 2;
      if (max_disp % 16 != 0) {
        max_disp += 16 - (max_disp % 16); // Ensure max_disp is a multiple of 16
      }
      cv::resize(left, left_for_matcher, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR_EXACT);
      cv::resize(right, right_for_matcher, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR_EXACT);
      //! [downscale]

    } else {
      // If no downscaling is applied, simply clone the original images
      left_for_matcher = left.clone();
      right_for_matcher = right.clone();
    }

    if (algo == "bm") {
      // Using StereoBM for faster processing. If speed is not critical, StereoSGBM provides better quality
      // The WLS filter is created using the StereoMatcher instance
      // Another matcher instance is created for computing the right disparity map

      //! [matching]
      cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(max_disp, wsize);
      wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
      cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

      cv::cvtColor(left_for_matcher, left_for_matcher, cv::COLOR_BGR2GRAY);
      cv::cvtColor(right_for_matcher, right_for_matcher, cv::COLOR_BGR2GRAY);

      matching_time = (double)cv::getTickCount();
      // Compute disparity maps for both left and right views
      left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
      matching_time = ((double)cv::getTickCount() - matching_time) / cv::getTickFrequency();
      //! [matching]

    } else if (algo == "sgbm") {
      // Using StereoSGBM, which provides better quality than StereoBM
      cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(0, max_disp, wsize);
      // Set the P1 and P2 parameters for the SGBM algorithm
      left_matcher->setP1(24 * wsize * wsize);
      left_matcher->setP2(96 * wsize * wsize);
      left_matcher->setPreFilterCap(63);
      left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
      wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
      cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

      matching_time = (double)cv::getTickCount();
      // Compute disparity maps for both left and right views
      left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
      matching_time = ((double)cv::getTickCount() - matching_time) / cv::getTickFrequency();

    } else {
      std::cout << "Unsupported algorithm";
      return -1;
    }
    // Disparity maps computed by the respective matcher instances, as well as the source left view are passed to the filter.
    // Note that we are using the original non-downscaled view to guide the filtering process. The disparity map is automatically
    // upscaled in an edge-aware fashion to match the original view resolution. The result is stored in filtered_disp.
    //! [filtering]
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    filtering_time = (double)cv::getTickCount();
    wls_filter->filter(left_disp, left, filtered_disp, right_disp);
    filtering_time = ((double)cv::getTickCount() - filtering_time) / cv::getTickFrequency();
    //! [filtering]

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
  // Filtering with fbs and confidence using also wls pre-processing
  else if (filter == "fbs_conf") {
    if (!no_downscale) {
      // Downscale the views to speed-up the matching stage, as we will need to compute both left
      // and right disparity maps for confidence map computation
      //! [downscale_wls]
      max_disp /= 2;
      if (max_disp % 16 != 0) {
        max_disp += 16 - (max_disp % 16);
      }
      cv::resize(left, left_for_matcher, cv::Size(), 0.5, 0.5);
      cv::resize(right, right_for_matcher, cv::Size(), 0.5, 0.5);
      //! [downscale_wls]

    } else {
      // Keep the original views
      left_for_matcher = left.clone();
      right_for_matcher = right.clone();
    }

    if (algo == "bm") {
      // Compute disparity maps for both left and right views
      //! [matching_wls]
      cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(max_disp, wsize);
      wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
      cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

      cv::cvtColor(left_for_matcher, left_for_matcher, cv::COLOR_BGR2GRAY);
      cv::cvtColor(right_for_matcher, right_for_matcher, cv::COLOR_BGR2GRAY);

      matching_time = (double)cv::getTickCount();
      left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
      matching_time = ((double)cv::getTickCount() - matching_time) / cv::getTickFrequency();
      //! [matching_wls]

    } else if (algo == "sgbm") {
      cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(0, max_disp, wsize);
      left_matcher->setP1(24 * wsize * wsize);
      left_matcher->setP2(96 * wsize * wsize);
      left_matcher->setPreFilterCap(63);
      left_matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
      wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
      cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

      matching_time = (double)cv::getTickCount();
      // Compute disparity maps for both left and right views
      left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
      matching_time = ((double)cv::getTickCount() - matching_time) / cv::getTickFrequency();

    } else {
      std::cout << "Unsupported algorithm";
      return -1;
    }

    //! [filtering_wls]
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    filtering_time = (double)cv::getTickCount();
    wls_filter->filter(left_disp, left, filtered_disp, right_disp);
    filtering_time = ((double)cv::getTickCount() - filtering_time) / cv::getTickFrequency();
    //! [filtering_wls]

    conf_map = wls_filter->getConfidenceMap();

    cv::Mat left_disp_resized;
    resize(left_disp, left_disp_resized, left.size());

    // Get the ROI that was used in the last filter call:
    ROI = wls_filter->getROI();
    if (!no_downscale) {
      // upscale raw disparity and ROI back for a proper comparison:
      resize(left_disp, left_disp, cv::Size(), 2.0, 2.0);
      left_disp = left_disp * 2.0;
      left_disp_resized = left_disp_resized * 2.0;
      ROI = cv::Rect(ROI.x * 2, ROI.y * 2, ROI.width * 2, ROI.height * 2);
    }

    #ifdef HAVE_EIGEN
        //! [filtering_fbs]
    solving_time = (double)getTickCount();
    fastBilateralSolverFilter(
          left, left_disp_resized, conf_map / 255.0f, solved_disp, fbs_spatial,
          fbs_luma, fbs_chroma, fbs_lambda);
    solving_time = ((double)getTickCount() - solving_time) / getTickFrequency();
        //! [filtering_fbs]

        //! [filtering_wls2fbs]
    fastBilateralSolverFilter(
          left, filtered_disp, conf_map / 255.0f, solved_filtered_disp,
          fbs_spatial, fbs_luma, fbs_chroma, fbs_lambda);
        //! [filtering_wls2fbs]
    #else
    (void)fbs_spatial;
    (void)fbs_luma;
    (void)fbs_chroma;
    (void)fbs_lambda;
    #endif

  } else if (filter == "wls_no_conf") {
    // There is no convenience function for the case of filtering with no confidence,
    // so we will need to set the ROI and matcher parameters manually
    left_for_matcher = left.clone();
    right_for_matcher = right.clone();

    if (algo == "bm") {
      cv::Ptr<cv::StereoBM> matcher = cv::StereoBM::create(max_disp, wsize);
      matcher->setTextureThreshold(0);
      matcher->setUniquenessRatio(0);
      cv::cvtColor(left_for_matcher, left_for_matcher, cv::COLOR_BGR2GRAY);
      cv::cvtColor(right_for_matcher, right_for_matcher, cv::COLOR_BGR2GRAY);
      ROI = computeROI(left_for_matcher.size(), matcher);
      wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
      wls_filter->setDepthDiscontinuityRadius((int)ceil(0.33 * wsize));

      matching_time = (double)cv::getTickCount();
      matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      matching_time = ((double)cv::getTickCount() - matching_time) / cv::getTickFrequency();

    } else if (algo == "sgbm") {
      cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(0, max_disp, wsize);
      matcher->setUniquenessRatio(0);
      matcher->setDisp12MaxDiff(1000000);
      matcher->setSpeckleWindowSize(0);
      matcher->setP1(24 * wsize * wsize);
      matcher->setP2(96 * wsize * wsize);
      matcher->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
      ROI = computeROI(left_for_matcher.size(), matcher);
      wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(false);
      wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5 * wsize));

      matching_time = (double)cv::getTickCount();
      matcher->compute(left_for_matcher, right_for_matcher, left_disp);
      matching_time = ((double)cv::getTickCount() - matching_time) / cv::getTickFrequency();

    } else {
      std::cout << "Unsupported algorithm";
      return -1;
    }

    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    filtering_time = (double)cv::getTickCount();
    wls_filter->filter(left_disp, left, filtered_disp, cv::Mat(), ROI);
    filtering_time = ((double)cv::getTickCount() - filtering_time) / cv::getTickFrequency();

  } else {
    std::cout << "Unsupported filter";
    return -1;
  }

  // Collect and print all the stats
  std::cout.precision(2);
  std::cout << "Matching time:  " << matching_time << "s" << std::endl;
  std::cout << "Filtering time: " << filtering_time << "s" << std::endl;
  std::cout << "Solving time: " << solving_time << "s" << std::endl;
  std::cout << std::endl;

  double MSE_before, percent_bad_before, MSE_after, percent_bad_after;
  if (!noGT) {
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

  // We use a convenience function getDisparityVis to visualize the disparity maps. The second parameter defines the contrast
  // (all disparity values are scaled by this value in the visualization).
  if (dst_path != "None") {
    cv::Mat filtered_disp_vis;
    cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, vis_mult);
    imwrite(dst_path, filtered_disp_vis);
  }
  if (dst_raw_path != "None") {
    cv::Mat raw_disp_vis;
    cv::ximgproc::getDisparityVis(left_disp, raw_disp_vis, vis_mult);
    imwrite(dst_raw_path, raw_disp_vis);
  }
  if (dst_conf_path != "None") {
    imwrite(dst_conf_path, conf_map);
  }

  if (!no_display) {
    cv::namedWindow("left", cv::WINDOW_AUTOSIZE);
    cv::resize(left, left, left.size() / 2);
    cv::imshow("left", left);
    cv::namedWindow("right", cv::WINDOW_AUTOSIZE);
    cv::resize(right, right, right.size() / 2);
    cv::imshow("right", right);

    if (!noGT) {
      cv::Mat GT_disp_vis;
      cv::ximgproc::getDisparityVis(GT_disp, GT_disp_vis, vis_mult);
      cv::namedWindow("ground-truth disparity", cv::WINDOW_AUTOSIZE);
      cv::resize(GT_disp_vis, GT_disp_vis, GT_disp_vis.size() / 2);
      cv::imshow("ground-truth disparity", GT_disp_vis);
    }

    //! [visualization]
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
    //! [visualization]
  }

  return 0;
}

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
