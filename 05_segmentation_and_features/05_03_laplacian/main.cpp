/**
 * @file main.cpp
 * @brief Laplacian edge detection using manual masks and OpenCV's Laplacian function
 * @author José Miguel Guerrero Hernández
 * @note The Laplacian is a second-order derivative operator that detects edges
 *       by finding areas of rapid intensity change. It's isotropic (rotation invariant)
 *       and detects edges in all directions simultaneously.
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main(int argc, char ** argv)
{
  // Load input image
  const std::string imagePath = argc >= 2 ? argv[1] : "lena.jpg";
  cv::Mat src = cv::imread(cv::samples::findFile(imagePath), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << imagePath << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return -1;
  }

  // Resize to standard dimensions
  cv::resize(src, src, cv::Size(512, 512));

  // Convert to grayscale for better edge detection
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // ========================================
  // Method 1: Manual Laplacian masks with filter2D
  // ========================================

  // Define Laplacian kernels (second-order derivative approximation)
  // Note: Kernels with center +4/-4 (or +8/-8) differ only in sign. After applying
  //       abs(), both produce identical results, so we show different connectivity instead.

  // 4-connected Laplacian kernel (considers only horizontal/vertical neighbors)
  cv::Mat kernel_4connected = (cv::Mat_<char>(3, 3) <<
    0, -1, 0,
    -1, 4, -1,
    0, -1, 0);

  // 8-connected Laplacian kernel (considers all 8 neighbors) - more sensitive
  cv::Mat kernel_8connected = (cv::Mat_<char>(3, 3) <<
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1);

  // Apply convolution with Laplacian masks
  // filter2D(src, dst, ddepth, kernel) - convolves image with kernel
  // Note: CV_16S (16-bit signed) is used to properly handle negative values
  //       from the second derivative. Using CV_8U would clip negatives to 0.
  cv::Mat laplacian_4conn, laplacian_8conn;
  cv::filter2D(gray, laplacian_4conn, CV_16S, kernel_4connected);
  cv::filter2D(gray, laplacian_8conn, CV_16S, kernel_8connected);

  // Convert to absolute values for display (8-bit unsigned)
  cv::Mat abs_laplacian_4conn, abs_laplacian_8conn;
  cv::convertScaleAbs(laplacian_4conn, abs_laplacian_4conn);
  cv::convertScaleAbs(laplacian_8conn, abs_laplacian_8conn);

  // Display manual mask results
  cv::imshow("Original (Color)", src);
  cv::imshow("Grayscale", gray);
  cv::imshow("Manual: Laplacian 4-connected", abs_laplacian_4conn);
  cv::imshow("Manual: Laplacian 8-connected", abs_laplacian_8conn);

  // ========================================
  // Method 2: OpenCV Laplacian function
  // ========================================

  // cv::Laplacian(src, dst, ddepth, ksize, scale, delta, borderType)
  // Parameters:
  //   src: input image (single-channel preferred)
  //   dst: output image (same size as input)
  //   ddepth: output image depth (CV_16S recommended to avoid overflow)
  //   ksize: aperture size for Sobel operator (1, 3, 5, or 7)
  //          ksize=1 uses a basic 3x3 kernel without Gaussian smoothing
  //          ksize>1 uses a more sophisticated calculation via Sobel operators
  cv::Mat laplacian_opencv;
  cv::Laplacian(gray, laplacian_opencv, CV_16S, 1);

  // Convert to absolute values for display
  cv::Mat abs_laplacian_opencv;
  cv::convertScaleAbs(laplacian_opencv, abs_laplacian_opencv);

  // Display OpenCV Laplacian result
  cv::imshow("OpenCV: Laplacian (ksize=1)", abs_laplacian_opencv);

  // Wait for user input and exit
  cv::waitKey();

  return 0;
}
