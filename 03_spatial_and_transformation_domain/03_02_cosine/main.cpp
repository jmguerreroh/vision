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

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
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
  const char * filename = argc >= 2 ? argv[1] : "lena.jpg";
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
  cv::Mat paddedImage = image;
  if (image.cols % 2 != 0 || image.rows % 2 != 0) {
    int newCols = (image.cols % 2 == 0) ? image.cols : image.cols + 1;
    int newRows = (image.rows % 2 == 0) ? image.rows : image.rows + 1;
    cv::copyMakeBorder(image, paddedImage, 0, newRows - image.rows,
                       0, newCols - image.cols, cv::BORDER_REPLICATE);
    std::cout << "Padded to even dimensions: " << paddedImage.cols << "x"
              << paddedImage.rows << std::endl;
  }

  // Convert to float and normalize to [0, 1] range
  // DCT requires floating-point input
  cv::Mat srcFloat;
  paddedImage.convertTo(srcFloat, CV_32F, 1.0 / 255.0);

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
  cv::Mat dctResult;
  cv::dct(srcFloat, dctResult);

  std::cout << "DCT computed successfully" << std::endl;

  // Visualize the DCT coefficients
  //
  // DCT coefficients have a large dynamic range.
  // We use logarithmic scale for better visualization.
  cv::Mat dctVisualization;

  // Take absolute value (coefficients can be negative)
  cv::Mat dctAbs = cv::abs(dctResult);

  // Apply log scale: log(1 + |DCT|)
  dctAbs += 1.0;
  cv::log(dctAbs, dctVisualization);

  // Normalize for display
  cv::normalize(dctVisualization, dctVisualization, 0, 1, cv::NORM_MINMAX);

  // Reconstruct image using Inverse DCT
  //
  // IDCT converts frequency coefficients back to spatial domain.
  // With all coefficients, we get perfect reconstruction.
  cv::Mat idctResult;
  cv::idct(dctResult, idctResult);

  std::cout << "IDCT computed - image reconstructed" << std::endl;

  // Demonstrate compression by zeroing high frequencies
  //
  // DCT enables compression by keeping only low-frequency coefficients.
  // High frequencies (bottom-right) contain fine details/noise.
  cv::Mat dctCompressed = dctResult.clone();

  // Keep only top-left portion (low frequencies)
  int keepSize = 64;  // Keep 64x64 coefficients out of 512x512

  // Validate keepSize
  keepSize = std::min(keepSize, std::min(dctCompressed.cols, dctCompressed.rows));

  // Zero out high-frequency coefficients
  // Right region (high horizontal frequencies)
  cv::Mat highFreqRegion = dctCompressed(cv::Rect(keepSize, 0,
                                                   dctCompressed.cols - keepSize,
                                                   dctCompressed.rows));
  highFreqRegion.setTo(0);

  // Bottom region (high vertical frequencies)
  highFreqRegion = dctCompressed(cv::Rect(0, keepSize,
                                           dctCompressed.cols,
                                           dctCompressed.rows - keepSize));
  highFreqRegion.setTo(0);

  // Reconstruct from compressed DCT
  cv::Mat compressedReconstruction;
  cv::idct(dctCompressed, compressedReconstruction);

  // Visualize compressed DCT
  cv::Mat compressedDctVis;
  cv::Mat compressedAbs = cv::abs(dctCompressed) + 1.0;
  cv::log(compressedAbs, compressedDctVis);
  cv::normalize(compressedDctVis, compressedDctVis, 0, 1, cv::NORM_MINMAX);

  // Calculate compression ratio and quality metrics
  int totalCoeffs = image.cols * image.rows;
  int keptCoeffs = keepSize * keepSize;
  double compressionRatio = (double)totalCoeffs / keptCoeffs;

  // Calculate PSNR (Peak Signal-to-Noise Ratio) to measure reconstruction quality
  // Higher PSNR = better quality (typically > 30 dB is good)
  double psnr = cv::PSNR(srcFloat, compressedReconstruction);

  std::cout << "\nCompression Statistics:" << std::endl;
  std::cout << "  Keeping " << keepSize << "x" << keepSize << " of "
            << image.cols << "x" << image.rows << " coefficients" << std::endl;
  std::cout << "  Data retained: " << (100.0 / compressionRatio) << "%" << std::endl;
  std::cout << "  Compression ratio: " << compressionRatio << ":1" << std::endl;
  std::cout << "  Reconstruction PSNR: " << psnr << " dB" << std::endl;

  // Display results
  cv::imshow("Original Image", srcFloat);
  cv::imshow("DCT Coefficients (log scale)", dctVisualization);
  cv::imshow("IDCT Reconstruction", idctResult);
  cv::imshow("Compressed DCT Coefficients", compressedDctVis);
  cv::imshow("Compressed Reconstruction", compressedReconstruction);

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
