/**
 * @file main.cpp
 * @brief Haar Wavelet Transform for real-time video denoising
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - 2D Haar Wavelet Transform (DWT) implementation
 * - Multi-resolution analysis with configurable iterations
 * - Wavelet coefficient shrinkage methods for denoising:
 *   - Hard shrinkage: threshold small coefficients to zero
 *   - Soft shrinkage: threshold and reduce magnitude
 *   - Garrot shrinkage: non-linear shrinkage function
 * - Real-time video processing with denoising
 *
 * Haar Wavelet Decomposition:
 * - Approximation (LL): low-pass in both directions (top-left)
 * - Horizontal detail (HL): vertical edges (top-right)
 * - Vertical detail (LH): horizontal edges (bottom-left)
 * The first letter is the filter along x and the second the filter along y,
 * as in the book (HL = PA/PB, LH = PB/PA).
 * - Diagonal detail (HH): diagonal edges (bottom-right)
 *
 * Mathematical basis (unnormalized Haar, block p00..p11):
 * - Forward:  LL = (p00 + p01 + p10 + p11) / 2   (approximation)
 *             details = signed half-sums of the same four pixels
 * - Inverse:  reconstructs the original exactly from the coefficients
 *
 * Note on normalization: LL is TWICE the block average. The orthonormal
 * Haar transform uses a 1/2 factor in both directions; here the extra x2
 * of the forward pass is compensated by the 0.5 factor of the inverse,
 * so forward+inverse is still an identity (a common practical convention).
 *
 * @note Requires a connected camera for real-time processing
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cassert>

/**
 * @brief Enumeration of wavelet coefficient shrinkage methods
 *
 * Shrinkage is used to denoise images by reducing small coefficients
 * that are likely to be noise rather than signal.
 */
enum ShrinkageType
{
  SHRINK_NONE   = 0,    ///< No filtering applied
  SHRINK_HARD   = 1,    ///< Hard shrinkage: zero if |d| <= T, else keep d
  SHRINK_SOFT   = 2,    ///< Soft shrinkage: zero if |d| <= T, else sgn(d)*(|d|-T)
  SHRINK_GARROT = 3     ///< Garrot shrinkage: zero if |d| <= T, else d - T²/d
};

/**
 * @brief Displays usage information
 * @param argv Command line arguments
 */
void printHelp(char ** argv)
{
  std::cout << "\n"
            << "Haar Wavelet Transform Demo\n"
            << "===========================\n"
            << "This program applies Haar wavelet transform for image denoising.\n\n"
            << "Usage: " << argv[0] << " [options]\n"
            << "  No arguments: Use camera (real-time video processing)\n"
            << "  <image_path>: Process static image\n\n"
            << "Controls (video mode):\n"
            << "  SPACE - Save current frame\n"
            << "  Q/ESC - Quit\n\n";
}

/**
 * @brief Signum function
 * @param x Input value
 * @return -1 if x < 0, 0 if x == 0, 1 if x > 0
 */
float sgn(float x)
{
  if (x > 0.0f) {return 1.0f;}
  if (x < 0.0f) {return -1.0f;}
  return 0.0f;
}

/**
 * @brief Soft shrinkage function
 *
 * Reduces magnitude by threshold T, zeros values below T.
 * Provides smoother results than hard shrinkage.
 *
 * @param d Wavelet coefficient value
 * @param T Threshold value
 * @return Shrunk coefficient: sgn(d) * max(|d| - T, 0)
 */
float softShrink(float d, float T)
{
  float abs_d = std::fabs(d);
  if (abs_d > T) {
    return sgn(d) * (abs_d - T);
  }
  return 0.0f;
}

/**
 * @brief Hard shrinkage function
 *
 * Keeps coefficients above threshold unchanged, zeros the rest.
 * Simple but can introduce artifacts at threshold boundary.
 *
 * @param d Wavelet coefficient value
 * @param T Threshold value
 * @return Original value if |d| > T, else 0
 */
float hardShrink(float d, float T)
{
  if (std::fabs(d) > T) {
    return d;
  }
  return 0.0f;
}

/**
 * @brief Garrot (non-negative garrote) shrinkage function
 *
 * Non-linear shrinkage that provides a compromise between
 * hard and soft shrinkage. Better preserves large coefficients.
 *
 * @param d Wavelet coefficient value
 * @param T Threshold value
 * @return Shrunk coefficient: d - T²/d if |d| > T, else 0
 */
float garrotShrink(float d, float T)
{
  if (std::fabs(d) > T) {
    return d - (T * T) / d;
  }
  return 0.0f;
}

/**
 * @brief Pad image to make dimensions divisible by 2^nIterations
 * @param src Source image
 * @param nIterations Number of wavelet decomposition levels
 * @return Padded image with appropriate dimensions
 */
static cv::Mat padForWavelet(const cv::Mat & src, int nIterations)
{
  int divisor = 1 << nIterations;  // 2^nIterations

  // Calculate required dimensions (must be divisible by divisor)
  int new_cols = ((src.cols + divisor - 1) / divisor) * divisor;
  int new_rows = ((src.rows + divisor - 1) / divisor) * divisor;

  if (new_cols == src.cols && new_rows == src.rows) {
    return src.clone();
  }

  // Pad image to required dimensions
  cv::Mat padded;
  cv::copyMakeBorder(src, padded, 0, new_rows - src.rows,
                     0, new_cols - src.cols,
                     cv::BORDER_REPLICATE);

  std::cout << "Padded from " << src.cols << "x" << src.rows
            << " to " << padded.cols << "x" << padded.rows
            << " (divisible by " << divisor << ")" << std::endl;

  return padded;
}

/**
 * @brief Perform 2D Haar Wavelet Transform (forward DWT)
 *
 * Decomposes the image into four sub-bands at each level:
 * - LL (Approximation): Average of 2x2 block - top-left quadrant
 * - HL (Horizontal detail): Vertical edge info - top-right quadrant
 * - LH (Vertical detail): Horizontal edge info - bottom-left quadrant
 * - HH (Diagonal detail): Diagonal edge info - bottom-right quadrant
 *
 * Layout after transform (NIter=1):
 * +-------+-------+
 * |  LL   |  HL   |
 * | (c)   | (dh)  |
 * +-------+-------+
 * |  LH   |  HH   |
 * | (dv)  | (dd)  |
 * +-------+-------+
 *
 * @param src Source matrix (CV_32FC1), modified during processing
 * @param dst Destination matrix (CV_32FC1) for wavelet coefficients
 * @param nIterations Number of decomposition levels
 */
static void haarWaveletTransform(cv::Mat & src, cv::Mat & dst, int nIterations)
{
  // CV_Assert stays active in release builds (plain assert() is compiled
  // out with -O2/NDEBUG, silently removing the type check)
  CV_Assert(src.type() == CV_32FC1);
  CV_Assert(dst.type() == CV_32FC1);

  int width = src.cols;
  int height = src.rows;

  for (int k = 0; k < nIterations; k++) {
    // Current sub-band dimensions (halved at each level)
    int half_width = width >> (k + 1);
    int half_height = height >> (k + 1);

    for (int y = 0; y < half_height; y++) {
      for (int x = 0; x < half_width; x++) {
        // Get 2x2 block of pixels
        float p00 = src.at<float>(2 * y, 2 * x);                  // top-left
        float p01 = src.at<float>(2 * y, 2 * x + 1);              // top-right
        float p10 = src.at<float>(2 * y + 1, 2 * x);              // bottom-left
        float p11 = src.at<float>(2 * y + 1, 2 * x + 1);          // bottom-right

        // Approximation (LL): 2x the average of the four pixels
        // (see the normalization note in the file header)
        float c = (p00 + p01 + p10 + p11) * 0.5f;
        dst.at<float>(y, x) = c;

        // Horizontal detail (HL): difference between columns
        float dh = (p00 + p10 - p01 - p11) * 0.5f;
        dst.at<float>(y, x + half_width) = dh;

        // Vertical detail (LH): difference between rows
        float dv = (p00 + p01 - p10 - p11) * 0.5f;
        dst.at<float>(y + half_height, x) = dv;

        // Diagonal detail (HH): diagonal difference
        float dd = (p00 - p01 - p10 + p11) * 0.5f;
        dst.at<float>(y + half_height, x + half_width) = dd;
      }
    }
    // Copy result for next iteration (operates on LL quadrant)
    dst.copyTo(src);
  }
}

/**
 * @brief Perform Inverse 2D Haar Wavelet Transform (IDWT) with shrinkage
 *
 * Reconstructs the image from wavelet coefficients while optionally
 * applying shrinkage to detail coefficients for denoising.
 *
 * The shrinkage is applied only to detail coefficients (HL, LH, HH),
 * not to the approximation (LL), as noise primarily affects high-frequency
 * components.
 *
 * @param src Source matrix with wavelet coefficients (modified)
 * @param dst Destination matrix for reconstructed image
 * @param nIterations Number of decomposition levels to reconstruct
 * @param shrinkageType Type of shrinkage filter (SHRINK_NONE, SHRINK_HARD, etc.)
 * @param threshold Shrinkage threshold value
 */
static void inverseHaarWavelet(
  cv::Mat & src, cv::Mat & dst, int nIterations,
  ShrinkageType shrinkageType = SHRINK_NONE,
  float threshold = 50.0f)
{
  CV_Assert(src.type() == CV_32FC1);
  CV_Assert(dst.type() == CV_32FC1);

  int width = src.cols;
  int height = src.rows;

  // Reconstruct from coarsest to finest level
  for (int k = nIterations; k > 0; k--) {
    int half_width = width >> k;
    int half_height = height >> k;

    for (int y = 0; y < half_height; y++) {
      for (int x = 0; x < half_width; x++) {
        // Extract coefficients from the four quadrants
        float c = src.at<float>(y, x);                                 // LL (approximation)
        float dh = src.at<float>(y, x + half_width);                   // HL (horizontal detail)
        float dv = src.at<float>(y + half_height, x);                  // LH (vertical detail)
        float dd = src.at<float>(y + half_height, x + half_width);     // HH (diagonal detail)

        // Apply shrinkage to detail coefficients for denoising
        switch (shrinkageType) {
          case SHRINK_HARD:
            dh = hardShrink(dh, threshold);
            dv = hardShrink(dv, threshold);
            dd = hardShrink(dd, threshold);
            break;
          case SHRINK_SOFT:
            dh = softShrink(dh, threshold);
            dv = softShrink(dv, threshold);
            dd = softShrink(dd, threshold);
            break;
          case SHRINK_GARROT:
            dh = garrotShrink(dh, threshold);
            dv = garrotShrink(dv, threshold);
            dd = garrotShrink(dd, threshold);
            break;
          case SHRINK_NONE:
          default:
            // No shrinkage applied
            break;
        }

        // Reconstruct 2x2 block from coefficients
        dst.at<float>(y * 2, x * 2) = 0.5f * (c + dh + dv + dd);
        dst.at<float>(y * 2, x * 2 + 1) = 0.5f * (c - dh + dv - dd);
        dst.at<float>(y * 2 + 1, x * 2) = 0.5f * (c + dh - dv - dd);
        dst.at<float>(y * 2 + 1, x * 2 + 1) = 0.5f * (c - dh - dv + dd);
      }
    }

    // Copy reconstructed region for next level
    int reconstructed_width = width >> (k - 1);
    int reconstructed_height = height >> (k - 1);
    cv::Mat src_region = src(cv::Rect(0, 0, reconstructed_width, reconstructed_height));
    cv::Mat dst_region = dst(cv::Rect(0, 0, reconstructed_width, reconstructed_height));
    dst_region.copyTo(src_region);
  }
}

/**
 * @brief Normalize matrix values to [0, 1] range for visualization
 * @param mat Input/output matrix to normalize
 */
void normalizeForDisplay(cv::Mat & mat)
{
  double min_val, max_val;
  cv::minMaxLoc(mat, &min_val, &max_val);

  if ((max_val - min_val) > 0) {
    mat = (mat - min_val) / (max_val - min_val);
  }
}

/**
 * @brief Process a static image with Haar wavelet denoising
 * @param image_path Path to the input image
 * @param NUM_ITERATIONS Number of Haar decomposition levels
 * @return 0 on success, negative on error
 */
int processImage(const std::string & image_path, const int NUM_ITERATIONS)
{
  // Configuration parameters
  const ShrinkageType FILTER_TYPE = SHRINK_GARROT;  // Denoising method
  const float THRESHOLD = 30.0f;                    // Shrinkage threshold

  std::cout << "=== Haar Wavelet Image Denoising ===" << std::endl;
  std::cout << "Image: " << image_path << std::endl;
  std::cout << "Decomposition levels: " << NUM_ITERATIONS << std::endl;
  std::cout << "Shrinkage type: Garrot" << std::endl;
  std::cout << "Threshold: " << THRESHOLD << std::endl;

  // Load image
  cv::Mat image = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cerr << "Error: Could not load image '" << image_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Original size: " << image.cols << "x" << image.rows << " pixels" << std::endl;

  // Pad image if necessary
  cv::Mat padded_image = padForWavelet(image, NUM_ITERATIONS);

  // Convert to float
  cv::Mat src_float;
  padded_image.convertTo(src_float, CV_32FC1);

  // Allocate working matrices
  cv::Mat wavelet_coeffs(src_float.rows, src_float.cols, CV_32FC1);
  cv::Mat temp_coeffs(src_float.rows, src_float.cols, CV_32FC1);
  cv::Mat filtered_result(src_float.rows, src_float.cols, CV_32FC1);

  // Perform forward wavelet transform
  wavelet_coeffs = 0;
  haarWaveletTransform(src_float, wavelet_coeffs, NUM_ITERATIONS);

  // Copy coefficients for inverse transform
  wavelet_coeffs.copyTo(temp_coeffs);

  // Perform inverse transform with denoising
  inverseHaarWavelet(temp_coeffs, filtered_result, NUM_ITERATIONS,
                     FILTER_TYPE, THRESHOLD);

  // Display results
  cv::Mat original_display;
  padded_image.convertTo(original_display, CV_32FC1);
  normalizeForDisplay(original_display);

  cv::Mat coeff_display = wavelet_coeffs.clone();
  normalizeForDisplay(coeff_display);

  cv::Mat filtered_display = filtered_result.clone();
  normalizeForDisplay(filtered_display);

  cv::imshow("Original Image", original_display);
  cv::imshow("Wavelet Coefficients", coeff_display);
  cv::imshow("Denoised Image", filtered_display);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}

/**
 * @brief Process video stream with Haar wavelet denoising
 *
 * Displays three windows:
 * 1. Original video frame
 * 2. Wavelet coefficients (multi-resolution decomposition)
 * 3. Filtered/denoised result
 *
 * @param capture OpenCV video capture object
 * @param NUM_ITERATIONS Number of Haar decomposition levels
 * @return 0 on success
 */
int processVideo(cv::VideoCapture & capture, const int NUM_ITERATIONS)
{
  // Configuration parameters
  const ShrinkageType FILTER_TYPE = SHRINK_GARROT;  // Denoising method
  const float THRESHOLD = 30.0f;                    // Shrinkage threshold

  int frame_count = 0;

  std::cout << "=== Haar Wavelet Video Denoising ===" << std::endl;
  std::cout << "Decomposition levels: " << NUM_ITERATIONS << std::endl;
  std::cout << "Shrinkage type: Garrot" << std::endl;
  std::cout << "Threshold: " << THRESHOLD << std::endl;
  std::cout << "\nControls:" << std::endl;
  std::cout << "  SPACE - Save current frame" << std::endl;
  std::cout << "  Q/ESC - Quit" << std::endl;

  // Create resizable windows
  cv::namedWindow("Original", cv::WINDOW_KEEPRATIO);
  cv::namedWindow("Wavelet Coefficients", cv::WINDOW_KEEPRATIO);
  cv::namedWindow("Denoised", cv::WINDOW_KEEPRATIO);

  // Read first frame to get dimensions
  cv::Mat frame;
  capture >> frame;
  if (frame.empty()) {
    std::cerr << "Error: Could not read frame from camera" << std::endl;
    return EXIT_FAILURE;
  }

  // Determine required dimensions for wavelet transform
  int divisor = 1 << NUM_ITERATIONS;  // 2^NUM_ITERATIONS
  int required_cols = ((frame.cols + divisor - 1) / divisor) * divisor;
  int required_rows = ((frame.rows + divisor - 1) / divisor) * divisor;

  if (required_cols != frame.cols || required_rows != frame.rows) {
    std::cout << "Note: Frames will be padded from " << frame.cols << "x" << frame.rows
              << " to " << required_cols << "x" << required_rows << std::endl;
  }

  // Allocate working matrices with required dimensions
  cv::Mat gray_frame(required_rows, required_cols, CV_8UC1);
  cv::Mat src_float(required_rows, required_cols, CV_32FC1);
  cv::Mat wavelet_coeffs(required_rows, required_cols, CV_32FC1);
  cv::Mat temp_coeffs(required_rows, required_cols, CV_32FC1);
  cv::Mat filtered_result(required_rows, required_cols, CV_32FC1);

  // Main processing loop
  while (true) {
    // Capture frame
    capture >> frame;
    if (frame.empty()) {
      continue;
    }

    // Convert to grayscale
    cv::Mat gray_temp;
    cv::cvtColor(frame, gray_temp, cv::COLOR_BGR2GRAY);

    // Pad if necessary
    if (gray_temp.cols != gray_frame.cols || gray_temp.rows != gray_frame.rows) {
      cv::copyMakeBorder(gray_temp, gray_frame, 0, gray_frame.rows - gray_temp.rows,
                         0, gray_frame.cols - gray_temp.cols,
                         cv::BORDER_REPLICATE);
    } else {
      gray_frame = gray_temp;
    }

    // Convert to float format
    gray_frame.convertTo(src_float, CV_32FC1);

    // Initialize output and perform forward wavelet transform
    wavelet_coeffs = 0;
    haarWaveletTransform(src_float, wavelet_coeffs, NUM_ITERATIONS);

    // Copy coefficients for inverse transform
    wavelet_coeffs.copyTo(temp_coeffs);

    // Perform inverse transform with denoising
    inverseHaarWavelet(temp_coeffs, filtered_result, NUM_ITERATIONS,
                           FILTER_TYPE, THRESHOLD);

    // Display original frame
    cv::imshow("Original", frame);

    // Display normalized wavelet coefficients
    cv::Mat coeff_display = wavelet_coeffs.clone();
    normalizeForDisplay(coeff_display);
    cv::imshow("Wavelet Coefficients", coeff_display);

    // Display normalized filtered result
    cv::Mat filtered_display = filtered_result.clone();
    normalizeForDisplay(filtered_display);
    cv::imshow("Denoised", filtered_display);

    // Handle keyboard input
    char key = static_cast<char>(cv::waitKey(5));
    switch (key) {
      case 'q':
      case 'Q':
      case 27:  // ESC key
        return EXIT_SUCCESS;

      case ' ': // Save frame
        {
          std::ostringstream oss;
          oss << "frame_" << std::setfill('0') << std::setw(3) << frame_count++ << ".jpg";
          cv::imwrite(oss.str(), frame);
          std::cout << "Saved: " << oss.str() << std::endl;
        }
        break;

      default:
        break;
    }
  }

  return EXIT_SUCCESS;
}

int main(int argc, char ** argv)
{
  const int NUM_ITERATIONS = 1;  // Decomposition levels

  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | | Image file; if omitted, the default camera is used}");
  if (parser.has("help")) {
    parser.printMessage();
    printHelp(argv);
    return EXIT_SUCCESS;
  }

  // If image path provided, process static image
  const std::string image_path = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  if (!image_path.empty()) {
    return processImage(image_path, NUM_ITERATIONS);
  }

  // Otherwise, use camera for real-time processing
  std::cout << "No image specified, using camera..." << std::endl;
  cv::VideoCapture capture(0);

  if (!capture.isOpened()) {
    std::cerr << "Error: Could not open camera" << std::endl;
    std::cerr << "Make sure a camera is connected and accessible" << std::endl;
    std::cerr << "\nAlternatively, provide an image path as argument." << std::endl;
    return EXIT_FAILURE;
  }

  return processVideo(capture, NUM_ITERATIONS);
}
