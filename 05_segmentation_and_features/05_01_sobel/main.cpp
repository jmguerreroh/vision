/**
 * @file main.cpp
 * @brief Sobel edge detection using manual convolution masks and OpenCV's Sobel function
 * @author José Miguel Guerrero Hernández
 * @note This example demonstrates two methods for computing image gradients:
 *       1) Manual convolution with Sobel kernels using filter2D
 *       2) OpenCV's optimized Sobel function
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main(int argc, char ** argv)
{
  // Load input image
  const std::string imagePath = argc >= 2 ? argv[1] : "lena.jpg";
  cv::Mat src = cv::imread(cv::samples::findFile(imagePath), cv::IMREAD_GRAYSCALE);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << imagePath << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return -1;
  }

  // Resize image to standard size for consistent processing
  cv::resize(src, src, cv::Size(512, 512));

  // ========================================
  // Method 1: Manual Sobel masks with filter2D
  // ========================================

  // Define Sobel kernels for gradient computation
  // Kernel for X-direction gradient (detects vertical edges)
  cv::Mat kernel_gx = (cv::Mat_<char>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);

  // Kernel for Y-direction gradient (detects horizontal edges)
  cv::Mat kernel_gy = (cv::Mat_<char>(3, 3) <<
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1);

  // Apply convolution with Sobel kernels using filter2D
  // filter2D(src, dst, ddepth, kernel) - convolves image with kernel
  // Note: CV_16S (16-bit signed) is used instead of src.depth() to properly handle
  //       negative gradient values. Using CV_8U would clip negatives to 0, losing information.
  cv::Mat manual_grad_x, manual_grad_y;
  cv::Mat manual_grad_combined;
  cv::filter2D(src, manual_grad_x, CV_16S, kernel_gx);
  cv::filter2D(src, manual_grad_y, CV_16S, kernel_gy);

  // Convert to absolute values for display
  // convertScaleAbs computes: dst = saturate_cast<uchar>(|src|)
  // Takes absolute value and converts to CV_8U (0-255) with saturation
  // Values > 255 are truncated to 255
  cv::Mat manual_abs_grad_x, manual_abs_grad_y;
  cv::convertScaleAbs(manual_grad_x, manual_abs_grad_x);
  cv::convertScaleAbs(manual_grad_y, manual_abs_grad_y);

  // Combine gradients using weighted sum: G = 0.5*|Gx| + 0.5*|Gy|
  // Note: 0.5 weights prevent saturation (max = 0.5*255 + 0.5*255 = 255, not 510)
  cv::addWeighted(manual_abs_grad_x, 0.5, manual_abs_grad_y, 0.5, 0, manual_grad_combined);

  // Display results from manual mask method
  cv::imshow("Original", src);
  cv::imshow("Manual: Gradient X (vertical edges)", manual_abs_grad_x);
  cv::imshow("Manual: Gradient Y (horizontal edges)", manual_abs_grad_y);
  cv::imshow("Manual: Combined Edges (horizontal + vertical)", manual_grad_combined);

  // ========================================
  // Method 2: OpenCV Sobel function
  // ========================================

  // Sobel(src, dst, ddepth, dx, dy, ksize)
  //   dx, dy: order of derivative in x and y direction
  //   ksize: size of Sobel kernel (1, 3, 5, or 7)
  //   ddepth: output image depth (CV_16S recommended to avoid overflow)
  cv::Mat sobel_grad_x, sobel_grad_y, sobel_grad_xy;
  cv::Mat sobel_abs_grad_x, sobel_abs_grad_y, sobel_abs_grad_xy;
  cv::Mat sobel_grad_combined;

  // Compute gradients in X and Y directions using CV_16S to handle negative values
  cv::Sobel(src, sobel_grad_x, CV_16S, 1, 0, 3);    // Gradient in X (detects vertical edges)
  cv::Sobel(src, sobel_grad_y, CV_16S, 0, 1, 3);    // Gradient in Y (detects horizontal edges)
  cv::Sobel(src, sobel_grad_xy, CV_16S, 1, 1, 3);   // Combined gradient (diagonal edges)

  // Convert gradients to absolute values (8-bit unsigned)
  // This handles negative gradient values and scales to displayable range
  cv::convertScaleAbs(sobel_grad_x, sobel_abs_grad_x);
  cv::convertScaleAbs(sobel_grad_y, sobel_abs_grad_y);
  cv::convertScaleAbs(sobel_grad_xy, sobel_abs_grad_xy);

  // Approximate total gradient magnitude using weighted sum
  cv::addWeighted(sobel_abs_grad_x, 0.5, sobel_abs_grad_y, 0.5, 0, sobel_grad_combined);

  // Display results from OpenCV Sobel method
  cv::imshow("Sobel: Gradient X (vertical edges)", sobel_abs_grad_x);
  cv::imshow("Sobel: Gradient Y (horizontal edges)", sobel_abs_grad_y);
  cv::imshow("Sobel: Gradient XY (diagonal edges)", sobel_abs_grad_xy);
  cv::imshow("Sobel: Combined Edges (horizontal + vertical)", sobel_grad_combined);

  // Wait for user input and exit
  cv::waitKey();

  return 0;
}
