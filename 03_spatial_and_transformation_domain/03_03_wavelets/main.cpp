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
 * - Horizontal detail (LH): vertical edges (top-right)
 * - Vertical detail (HL): horizontal edges (bottom-left)
 * - Diagonal detail (HH): diagonal edges (bottom-right)
 *
 * Mathematical basis:
 * - Forward: c = (a + b + c + d) / 2 (approximation)
 *           d = (a - b - c + d) / 2 (details)
 * - Inverse: reconstructs original from coefficients
 *
 * @note Requires a connected camera for real-time processing
 */

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <cstdio>
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
  float absD = std::fabs(d);
  if (absD > T) {
    return sgn(d) * (absD - T);
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
  int newCols = ((src.cols + divisor - 1) / divisor) * divisor;
  int newRows = ((src.rows + divisor - 1) / divisor) * divisor;

  if (newCols == src.cols && newRows == src.rows) {
    return src.clone();
  }

  // Pad image to required dimensions
  cv::Mat padded;
  cv::copyMakeBorder(src, padded, 0, newRows - src.rows,
                     0, newCols - src.cols,
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
 * - LH (Horizontal detail): Vertical edge info - top-right quadrant
 * - HL (Vertical detail): Horizontal edge info - bottom-left quadrant
 * - HH (Diagonal detail): Diagonal edge info - bottom-right quadrant
 *
 * Layout after transform (NIter=1):
 * +-------+-------+
 * |  LL   |  LH   |
 * | (c)   | (dh)  |
 * +-------+-------+
 * |  HL   |  HH   |
 * | (dv)  | (dd)  |
 * +-------+-------+
 *
 * @param src Source matrix (CV_32FC1), modified during processing
 * @param dst Destination matrix (CV_32FC1) for wavelet coefficients
 * @param nIterations Number of decomposition levels
 */
static void haarWaveletTransform(cv::Mat & src, cv::Mat & dst, int nIterations)
{
  assert(src.type() == CV_32FC1);
  assert(dst.type() == CV_32FC1);

  int width = src.cols;
  int height = src.rows;

  for (int k = 0; k < nIterations; k++) {
    // Current sub-band dimensions (halved at each level)
    int halfWidth = width >> (k + 1);
    int halfHeight = height >> (k + 1);

    for (int y = 0; y < halfHeight; y++) {
      for (int x = 0; x < halfWidth; x++) {
        // Get 2x2 block of pixels
        float p00 = src.at<float>(2 * y, 2 * x);                  // top-left
        float p01 = src.at<float>(2 * y, 2 * x + 1);              // top-right
        float p10 = src.at<float>(2 * y + 1, 2 * x);              // bottom-left
        float p11 = src.at<float>(2 * y + 1, 2 * x + 1);          // bottom-right

        // Approximation (LL): average of all four pixels
        float c = (p00 + p01 + p10 + p11) * 0.5f;
        dst.at<float>(y, x) = c;

        // Horizontal detail (LH): difference between columns
        float dh = (p00 + p10 - p01 - p11) * 0.5f;
        dst.at<float>(y, x + halfWidth) = dh;

        // Vertical detail (HL): difference between rows
        float dv = (p00 + p01 - p10 - p11) * 0.5f;
        dst.at<float>(y + halfHeight, x) = dv;

        // Diagonal detail (HH): diagonal difference
        float dd = (p00 - p01 - p10 + p11) * 0.5f;
        dst.at<float>(y + halfHeight, x + halfWidth) = dd;
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
 * The shrinkage is applied only to detail coefficients (LH, HL, HH),
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
  assert(src.type() == CV_32FC1);
  assert(dst.type() == CV_32FC1);

  int width = src.cols;
  int height = src.rows;

  // Reconstruct from coarsest to finest level
  for (int k = nIterations; k > 0; k--) {
    int halfWidth = width >> k;
    int halfHeight = height >> k;

    for (int y = 0; y < halfHeight; y++) {
      for (int x = 0; x < halfWidth; x++) {
        // Extract coefficients from the four quadrants
        float c = src.at<float>(y, x);                                 // LL (approximation)
        float dh = src.at<float>(y, x + halfWidth);                    // LH (horizontal detail)
        float dv = src.at<float>(y + halfHeight, x);                   // HL (vertical detail)
        float dd = src.at<float>(y + halfHeight, x + halfWidth);       // HH (diagonal detail)

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
    int reconstructedWidth = width >> (k - 1);
    int reconstructedHeight = height >> (k - 1);
    cv::Mat srcRegion = src(cv::Rect(0, 0, reconstructedWidth, reconstructedHeight));
    cv::Mat dstRegion = dst(cv::Rect(0, 0, reconstructedWidth, reconstructedHeight));
    dstRegion.copyTo(srcRegion);
  }
}

/**
 * @brief Normalize matrix values to [0, 1] range for visualization
 * @param mat Input/output matrix to normalize
 */
void normalizeForDisplay(cv::Mat & mat)
{
  double minVal, maxVal;
  cv::minMaxLoc(mat, &minVal, &maxVal);

  if ((maxVal - minVal) > 0) {
    mat = (mat - minVal) / (maxVal - minVal);
  }
}

/**
 * @brief Process a static image with Haar wavelet denoising
 * @param imagePath Path to the input image
 * @return 0 on success, negative on error
 */
int processImage(const std::string & imagePath, const int NUM_ITERATIONS)
{
  // Configuration parameters
  const ShrinkageType FILTER_TYPE = SHRINK_GARROT;  // Denoising method
  const float THRESHOLD = 30.0f;                    // Shrinkage threshold

  std::cout << "=== Haar Wavelet Image Denoising ===" << std::endl;
  std::cout << "Image: " << imagePath << std::endl;
  std::cout << "Decomposition levels: " << NUM_ITERATIONS << std::endl;
  std::cout << "Shrinkage type: Garrot" << std::endl;
  std::cout << "Threshold: " << THRESHOLD << std::endl;

  // Load image
  cv::Mat image = cv::imread(cv::samples::findFile(imagePath), cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cerr << "Error: Could not load image '" << imagePath << "'" << std::endl;
    return -1;
  }

  std::cout << "Original size: " << image.cols << "x" << image.rows << " pixels" << std::endl;

  // Pad image if necessary
  cv::Mat paddedImage = padForWavelet(image, NUM_ITERATIONS);

  // Convert to float
  cv::Mat srcFloat;
  paddedImage.convertTo(srcFloat, CV_32FC1);

  // Allocate working matrices
  cv::Mat waveletCoeffs(srcFloat.rows, srcFloat.cols, CV_32FC1);
  cv::Mat tempCoeffs(srcFloat.rows, srcFloat.cols, CV_32FC1);
  cv::Mat filteredResult(srcFloat.rows, srcFloat.cols, CV_32FC1);

  // Perform forward wavelet transform
  waveletCoeffs = 0;
  haarWaveletTransform(srcFloat, waveletCoeffs, NUM_ITERATIONS);

  // Copy coefficients for inverse transform
  waveletCoeffs.copyTo(tempCoeffs);

  // Perform inverse transform with denoising
  inverseHaarWavelet(tempCoeffs, filteredResult, NUM_ITERATIONS,
                     FILTER_TYPE, THRESHOLD);

  // Display results
  cv::Mat originalDisplay;
  paddedImage.convertTo(originalDisplay, CV_32FC1);
  normalizeForDisplay(originalDisplay);

  cv::Mat coeffDisplay = waveletCoeffs.clone();
  normalizeForDisplay(coeffDisplay);

  cv::Mat filteredDisplay = filteredResult.clone();
  normalizeForDisplay(filteredDisplay);

  cv::imshow("Original Image", originalDisplay);
  cv::imshow("Wavelet Coefficients", coeffDisplay);
  cv::imshow("Denoised Image", filteredDisplay);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return 0;
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
 * @return 0 on success
 */
int processVideo(cv::VideoCapture & capture, const int NUM_ITERATIONS)
{
  // Configuration parameters
  const ShrinkageType FILTER_TYPE = SHRINK_GARROT;  // Denoising method
  const float THRESHOLD = 30.0f;                    // Shrinkage threshold

  int frameCount = 0;
  char filename[256];

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
    return -1;
  }

  // Determine required dimensions for wavelet transform
  int divisor = 1 << NUM_ITERATIONS;  // 2^NUM_ITERATIONS
  int requiredCols = ((frame.cols + divisor - 1) / divisor) * divisor;
  int requiredRows = ((frame.rows + divisor - 1) / divisor) * divisor;

  if (requiredCols != frame.cols || requiredRows != frame.rows) {
    std::cout << "Note: Frames will be padded from " << frame.cols << "x" << frame.rows
              << " to " << requiredCols << "x" << requiredRows << std::endl;
  }

  // Allocate working matrices with required dimensions
  cv::Mat grayFrame(requiredRows, requiredCols, CV_8UC1);
  cv::Mat srcFloat(requiredRows, requiredCols, CV_32FC1);
  cv::Mat waveletCoeffs(requiredRows, requiredCols, CV_32FC1);
  cv::Mat tempCoeffs(requiredRows, requiredCols, CV_32FC1);
  cv::Mat filteredResult(requiredRows, requiredCols, CV_32FC1);

  // Main processing loop
  while (true) {
    // Capture frame
    capture >> frame;
    if (frame.empty()) {
      continue;
    }

    // Convert to grayscale
    cv::Mat grayTemp;
    cv::cvtColor(frame, grayTemp, cv::COLOR_BGR2GRAY);

    // Pad if necessary
    if (grayTemp.cols != grayFrame.cols || grayTemp.rows != grayFrame.rows) {
      cv::copyMakeBorder(grayTemp, grayFrame, 0, grayFrame.rows - grayTemp.rows,
                         0, grayFrame.cols - grayTemp.cols,
                         cv::BORDER_REPLICATE);
    } else {
      grayFrame = grayTemp;
    }

    // Convert to float format
    grayFrame.convertTo(srcFloat, CV_32FC1);

    // Initialize output and perform forward wavelet transform
    waveletCoeffs = 0;
    haarWaveletTransform(srcFloat, waveletCoeffs, NUM_ITERATIONS);

    // Copy coefficients for inverse transform
    waveletCoeffs.copyTo(tempCoeffs);

    // Perform inverse transform with denoising
    inverseHaarWavelet(tempCoeffs, filteredResult, NUM_ITERATIONS,
                           FILTER_TYPE, THRESHOLD);

    // Display original frame
    cv::imshow("Original", frame);

    // Display normalized wavelet coefficients
    cv::Mat coeffDisplay = waveletCoeffs.clone();
    normalizeForDisplay(coeffDisplay);
    cv::imshow("Wavelet Coefficients", coeffDisplay);

    // Display normalized filtered result
    cv::Mat filteredDisplay = filteredResult.clone();
    normalizeForDisplay(filteredDisplay);
    cv::imshow("Denoised", filteredDisplay);

    // Handle keyboard input
    char key = static_cast<char>(cv::waitKey(5));
    switch (key) {
      case 'q':
      case 'Q':
      case 27:  // ESC key
        return 0;

      case ' ': // Save frame
        std::snprintf(filename, sizeof(filename), "frame_%03d.jpg", frameCount++);
        cv::imwrite(filename, frame);
        std::cout << "Saved: " << filename << std::endl;
        break;

      default:
        break;
    }
  }

  return 0;
}

int main(int argc, char ** argv)
{
  const int NUM_ITERATIONS = 1;  // Decomposition levels
  printHelp(argv);

  // If image path provided, process static image
  if (argc >= 2) {
    return processImage(argv[1], NUM_ITERATIONS);
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
