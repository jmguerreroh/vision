/**
 * @file main.cpp
 * @brief Discrete Cosine Transform (DCT) in OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - How to compute the Discrete Cosine Transform (DCT)
 * - How to visualize the DCT coefficients
 * - How to reconstruct the image using Inverse DCT (IDCT)
 *
 * @note The DCT is similar to DFT but uses only real numbers (cosines).
 *       It's widely used in image/video compression (JPEG, MPEG).
 *
 *       Key properties:
 *       - Concentrates energy in low-frequency coefficients (top-left)
 *       - No complex numbers (only real values)
 *       - Slightly better energy compaction than DFT
 *
 *       DCT basis functions are cosines of varying frequencies.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, char ** argv)
{
  // cv::Mat is OpenCV's main structure for storing images
  // Mat = Matrix, represents an image as a matrix of pixels
  cv::Mat image;

  // cv::imread() loads an image from a file
  // Parameters:
  //   - File path (string)
  //   - Read mode:
  //     * cv::IMREAD_COLOR (1): Load color image in BGR format (default)
  //     * cv::IMREAD_GRAYSCALE (0): Load image in grayscale
  //     * cv::IMREAD_UNCHANGED (-1): Load image with alpha channel if present
  const char * filename = argc >= 2 ? argv[1] : "../../data/lena.jpg";
  image = cv::imread(cv::samples::findFile(filename), cv::IMREAD_GRAYSCALE);

  // Verify that the image was loaded successfully
  // An empty image indicates an error (file not found, invalid format, etc.)
  if (image.empty()) {
    std::cerr << "Error: Could not load image '" << filename << "'" << std::endl;
    std::cerr << "Please verify the file exists and the path is correct." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Image loaded: " << image.cols << "x" << image.rows << " pixels" << std::endl;

  // DCT requires even dimensions - pad if necessary
  cv::Mat padded_image = image;
  if (image.cols % 2 != 0 || image.rows % 2 != 0) {
    int new_cols = (image.cols % 2 == 0) ? image.cols : image.cols + 1;
    int new_rows = (image.rows % 2 == 0) ? image.rows : image.rows + 1;
    cv::copyMakeBorder(image, padded_image, 0, new_rows - image.rows,
                       0, new_cols - image.cols, cv::BORDER_REPLICATE);
    std::cout << "Padded to even dimensions: " << padded_image.cols << "x"
              << padded_image.rows << std::endl;
  }

  // Convert to float and normalize to [0, 1] range
  // DCT requires floating-point input
  cv::Mat src_float;
  padded_image.convertTo(src_float, CV_32F, 1.0 / 255.0);

  // Compute the Discrete Cosine Transform
  //
  // The 2D DCT transforms spatial data into frequency coefficients.
  //
  // DCT formula:
  //   F(u,v) = C(u)C(v) * sum_{x,y} f(x,y) * cos((2x+1)uπ/2N) * cos((2y+1)vπ/2M)
  //
  // Where C(0) = 1/sqrt(N), C(k) = sqrt(2/N) for k > 0
  //
  // The result is a matrix of DCT coefficients:
  //   - Top-left corner (0,0): DC coefficient (average brightness)
  //   - Moving right/down: Increasing horizontal/vertical frequencies
  cv::Mat dct_result;
  cv::dct(src_float, dct_result);

  std::cout << "DCT computed successfully" << std::endl;

  // Visualize the DCT coefficients
  //
  // DCT coefficients have a large dynamic range.
  // We use logarithmic scale for better visualization.
  cv::Mat dct_visualization;

  // Take absolute value (coefficients can be negative)
  cv::Mat dct_abs = cv::abs(dct_result);

  // Apply log scale: log(1 + |DCT|)
  dct_abs += 1.0;
  cv::log(dct_abs, dct_visualization);

  // Normalize for display
  cv::normalize(dct_visualization, dct_visualization, 0, 1, cv::NORM_MINMAX);

  // Reconstruct image using Inverse DCT
  //
  // IDCT converts frequency coefficients back to spatial domain.
  // With all coefficients, we get perfect reconstruction.
  cv::Mat idct_result;
  cv::idct(dct_result, idct_result);

  std::cout << "IDCT computed - image reconstructed" << std::endl;

  // Demonstrate compression by zeroing high frequencies
  //
  // DCT enables compression by keeping only low-frequency coefficients.
  // High frequencies (bottom-right) contain fine details/noise.
  cv::Mat dct_compressed = dct_result.clone();

  // Keep only top-left portion (low frequencies)
  int keep_size = 64;  // Keep 64x64 coefficients out of 512x512

  // Validate keepSize
  keep_size = std::min(keep_size, std::min(dct_compressed.cols, dct_compressed.rows));

  // Zero out high-frequency coefficients
  // Right region (high horizontal frequencies)
  cv::Mat high_freq_region = dct_compressed(cv::Rect(keep_size, 0,
                                                   dct_compressed.cols - keep_size,
                                                   dct_compressed.rows));
  high_freq_region.setTo(0);

  // Bottom region (high vertical frequencies)
  high_freq_region = dct_compressed(cv::Rect(0, keep_size,
                                           dct_compressed.cols,
                                           dct_compressed.rows - keep_size));
  high_freq_region.setTo(0);

  // Reconstruct from compressed DCT
  cv::Mat compressed_reconstruction;
  cv::idct(dct_compressed, compressed_reconstruction);

  // Visualize compressed DCT
  cv::Mat compressed_dct_vis;
  cv::Mat compressed_abs = cv::abs(dct_compressed) + 1.0;
  cv::log(compressed_abs, compressed_dct_vis);
  cv::normalize(compressed_dct_vis, compressed_dct_vis, 0, 1, cv::NORM_MINMAX);

  // Calculate compression ratio and quality metrics
  int total_coeffs = image.cols * image.rows;
  int kept_coeffs = keep_size * keep_size;
  double compression_ratio = static_cast<double>(total_coeffs) / kept_coeffs;

  // Calculate PSNR (Peak Signal-to-Noise Ratio) to measure reconstruction quality
  // Higher PSNR = better quality (typically > 30 dB is good)
  double psnr = cv::PSNR(src_float, compressed_reconstruction);

  std::cout << "\nCompression Statistics:" << std::endl;
  std::cout << "  Keeping " << keep_size << "x" << keep_size << " of "
            << image.cols << "x" << image.rows << " coefficients" << std::endl;
  std::cout << "  Data retained: " << (100.0 / compression_ratio) << "%" << std::endl;
  std::cout << "  Compression ratio: " << compression_ratio << ":1" << std::endl;
  std::cout << "  Reconstruction PSNR: " << psnr << " dB" << std::endl;

  // Display results
  cv::imshow("Original Image", src_float);
  cv::imshow("DCT Coefficients (log scale)", dct_visualization);
  cv::imshow("IDCT Reconstruction", idct_result);
  cv::imshow("Compressed DCT Coefficients", compressed_dct_vis);
  cv::imshow("Compressed Reconstruction", compressed_reconstruction);

  std::cout << "\nWindows displayed:" << std::endl;
  std::cout << "  - Original grayscale image" << std::endl;
  std::cout << "  - DCT coefficients (log scale for visibility)" << std::endl;
  std::cout << "  - Perfect reconstruction via IDCT" << std::endl;
  std::cout << "  - Compressed DCT (only low frequencies)" << std::endl;
  std::cout << "  - Lossy reconstruction from compressed DCT" << std::endl;
  std::cout << "\nPress any key to exit..." << std::endl;

  cv::waitKey(0);

  return EXIT_SUCCESS;
}
