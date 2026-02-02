/**
 * @file main.cpp
 * @brief Fourier Basis Wave Visualization (equivalent to freq.py)
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Generation of Fourier basis functions (sinusoidal waves)
 * - Visualization of frequency components
 *
 * The 2D Fourier basis function is:
 *   Z(x,y) = cos(2π(ux/M + vy/N))
 *
 * Where:
 * - u: horizontal oscillations (frequency in x direction)
 * - v: vertical oscillations (frequency in y direction)
 * - M, N: image dimensions
 */

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <cmath>

/**
 * @brief Generates a 2D Fourier basis wave for frequency (u, v)
 *
 * Creates a cosine wave pattern that represents one frequency component
 * of the 2D Fourier transform. This basis function oscillates:
 *   - 'u' times horizontally (across columns)
 *   - 'v' times vertically (across rows)
 *
 * Formula: Z(x,y) = cos(2π(ux/M + vy/N))
 *
 * @param u Horizontal frequency (number of complete cycles in x direction)
 * @param v Vertical frequency (number of complete cycles in y direction)
 * @param M Number of columns (width of the generated pattern)
 * @param N Number of rows (height of the generated pattern)
 * @return CV_32F matrix containing the basis wave with values in [-1, 1]
 */
cv::Mat ondaBase(int u, int v, int M = 500, int N = 500)
{
  cv::Mat Z(N, M, CV_32F);

  // Generate 2D cosine wave pattern
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < M; x++) {
      // Calculate phase: 2π(ux/M + vy/N)
      // This creates u horizontal cycles and v vertical cycles
      double angle = 2.0 * CV_PI * ((double)u * x / M + (double)v * y / N);
      Z.at<float>(y, x) = static_cast<float>(std::cos(angle));
    }
  }

  return Z;
}

/**
 * @brief Computes the Discrete Fourier Transform of a grayscale image
 *
 * Performs the following steps:
 * 1. Pads the image to optimal size for FFT performance
 * 2. Converts to complex format (real + imaginary planes)
 * 3. Applies DFT to obtain frequency domain representation
 *
 * @param image Input grayscale image (CV_8U or CV_32F)
 * @return Complex DFT result (CV_32FC2) with real and imaginary components
 *         Size may be larger than input due to optimal padding
 */
cv::Mat computeDFT(const cv::Mat & image)
{
  // Pad to optimal size for DFT performance (powers of 2, 3, 5)
  cv::Mat padded;
  int optimalRows = cv::getOptimalDFTSize(image.rows);
  int optimalCols = cv::getOptimalDFTSize(image.cols);

  cv::copyMakeBorder(image, padded,
                     0, optimalRows - image.rows,
                     0, optimalCols - image.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  // Create complex image with real and imaginary parts
  cv::Mat realPart;
  padded.convertTo(realPart, CV_32F);  // Real part = image data
  cv::Mat imaginaryPart = cv::Mat::zeros(padded.size(), CV_32F);  // Imaginary part = 0

  cv::Mat planes[] = {realPart, imaginaryPart};
  cv::Mat complexImage;
  cv::merge(planes, 2, complexImage);  // Merge into 2-channel complex matrix

  // Compute DFT
  cv::dft(complexImage, complexImage, cv::DFT_COMPLEX_OUTPUT);

  return complexImage;
}

/**
 * @brief Displays usage information
 */
void printHelp(char ** argv)
{
  std::cout << "\n"
            << "Fourier Basis Wave Visualization\n"
            << "=================================\n"
            << "Generates and displays Fourier basis functions.\n"
            << "Shows progressive image reconstruction from frequency components.\n\n"
            << "Usage modes:\n"
            << "  1) Single basis wave:  " << argv[0] << " <u> <v> <size>\n"
            << "     - Displays only the basis wave for frequency (u, v)\n"
            << "     - size: dimension of the square wave pattern\n\n"
            << "  2) Image reconstruction: " << argv[0] << " <image_path> [max_freq] [size]\n"
            << "     - Progressive reconstruction from frequency components\n"
            << "     - image_path: Image to decompose and reconstruct (required)\n"
            << "     - max_freq: Maximum frequency (default: image width/2)\n"
            << "     - size: Basis wave size (default: max(width, height))\n\n"
            << "Formula: Z(x,y) = cos(2π(ux/M + vy/N))\n\n"
            << "Display (reconstruction mode):\n"
            << "  Left: Original image\n"
            << "  Center: Current basis wave\n"
            << "  Right: Progressive reconstruction (sum of basis * coefficients)\n\n"
            << "Controls:\n"
            << "  SPACE - Pause/Resume\n"
            << "  Q/ESC - Quit\n\n";
}

int main(int argc, char ** argv)
{
  printHelp(argv);

  // ============================================================================
  // SECTION 1: Parse command-line arguments and load input image
  // ============================================================================
  cv::Mat referenceImage;  // Resized image for reconstruction
  cv::Mat originalImage;   // Original image before resizing
  int maxFreq = -1;        // Maximum frequency to process (-1 = auto)
  int size = -1;           // Size of basis waves and reconstruction (-1 = auto)
  bool singleBasisMode = false;  // Display only one basis wave
  int singleU = 0, singleV = 0;  // Frequency for single basis mode

  // Check if all three arguments are numeric (single basis wave mode)
  if (argc == 4) {
    std::string arg1 = argv[1];
    std::string arg2 = argv[2];
    std::string arg3 = argv[3];

    if (arg1.find_first_not_of("0123456789") == std::string::npos &&
      arg2.find_first_not_of("0123456789") == std::string::npos &&
      arg3.find_first_not_of("0123456789") == std::string::npos)
    {
      // All three are numbers: single basis wave mode
      singleBasisMode = true;
      singleU = std::atoi(argv[1]);
      singleV = std::atoi(argv[2]);
      size = std::atoi(argv[3]);

      std::cout << "Single basis wave mode: u=" << singleU
                << ", v=" << singleV << ", size=" << size << "x" << size << std::endl;
    }
  }

  // Parse first argument: can be either image path or max frequency
  if (!singleBasisMode && argc >= 2) {
    std::string filename = argv[1];

    // Check if argument is purely numeric (old-style max_freq parameter)
    if (filename.find_first_not_of("0123456789") == std::string::npos) {
      maxFreq = std::atoi(argv[1]);
    } else {
      // Argument is a file path - attempt to load image
      originalImage = cv::imread(cv::samples::findFile(filename), cv::IMREAD_GRAYSCALE);
      if (originalImage.empty()) {
        std::cerr << "Error: Could not load image '" << filename << "'" << std::endl;
        return EXIT_FAILURE;
      }
      std::cout << "Reference image loaded: " << originalImage.cols << "x"
                << originalImage.rows << std::endl;

      // Set intelligent defaults based on image dimensions
      if (size == -1) {
        size = std::max(originalImage.cols, originalImage.rows);
      }
      if (maxFreq == -1) {
        maxFreq = std::max(originalImage.cols, originalImage.rows) / 2;
      }
    }
  }

  if (argc >= 3) {
    maxFreq = std::atoi(argv[2]);
  }
  if (argc >= 4) {
    size = std::atoi(argv[3]);
  }

  // Set final defaults if still not set
  if (size == -1) {
    size = std::max(originalImage.cols, originalImage.rows);
  }
  if (maxFreq == -1) {
    maxFreq = size - 1;  // All frequencies for perfect reconstruction
  }

  // ============================================================================
  // MODE 1: Display single basis wave (no image required)
  // ============================================================================
  if (singleBasisMode) {
    std::cout << "\nGenerating basis wave Z(x,y) = cos(2π(" << singleU << "*x/" << size
              << " + " << singleV << "*y/" << size << "))" << std::endl;

    cv::Mat basisWave = ondaBase(singleU, singleV, size, size);

    // Normalize for display: [-1,1] -> [0,255]
    cv::Mat display;
    cv::normalize(basisWave, display, 0, 1, cv::NORM_MINMAX);
    cv::Mat displayU8;
    display.convertTo(displayU8, CV_8U, 255);

    std::string windowName = "Fourier Basis: u=" + std::to_string(singleU) +
      ", v=" + std::to_string(singleV);
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, displayU8);

    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
  }

  // Resize image to working size
  if (!originalImage.empty()) {
    cv::resize(originalImage, referenceImage, cv::Size(size, size));
  }

  if (referenceImage.empty()) {
    std::cerr << "Error: Image is required for reconstruction demo." << std::endl;
    std::cerr << "Usage: " << argv[0] << " <image_path> [max_freq] [size]" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Max frequency: " << maxFreq << std::endl;
  std::cout << "Basis wave size: " << size << "x" << size << std::endl;

  // ============================================================================
  // SECTION 2: Compute DFT of reference image
  // ============================================================================
  std::cout << "Computing DFT..." << std::endl;
  cv::Mat complexDFT = computeDFT(referenceImage);
  std::cout << "DFT size: " << complexDFT.cols << "x" << complexDFT.rows << std::endl;

  // Auto-adjust maxFreq to include all DFT frequencies if user didn't specify
  if (argc < 3) {
    maxFreq = std::max(complexDFT.rows, complexDFT.cols) - 1;
    std::cout << "Using full DFT spectrum, maxFreq updated to: " << maxFreq << std::endl;
  }

  // ============================================================================
  // SECTION 3: Progressive reconstruction animation loop
  // ============================================================================
  std::cout << "\nStarting animation loop..." << std::endl;
  std::cout << "Press SPACE to pause/resume, Q or ESC to quit" << std::endl;

  const std::string windowName = "Fourier Basis Reconstruction";
  cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

  bool paused = false;
  int u = 0;
  int v = 0;

  // Partial DFT spectrum: accumulates frequencies progressively
  // Starts empty (all zeros), one frequency added per iteration
  cv::Mat partialDFT = cv::Mat::zeros(complexDFT.size(), CV_32FC2);

  while (true) {
    if (!paused) {
      // --- Step 1: Generate basis wave for current frequency (u,v) ---
      cv::Mat basisWave = ondaBase(u, v, size, size);

      // --- Step 2: Get DFT coefficient at this frequency ---
      int dftU = u;
      int dftV = v;

      // Validate frequency is within DFT bounds (may differ from maxFreq)
      if (dftU >= complexDFT.cols || dftV >= complexDFT.rows) {
        // Out of bounds - skip to next frequency
        v++;
        if (v > maxFreq) {
          v = 0;
          u++;
        }
        continue;
      }

      // Extract complex coefficient: F(u,v) = real + i*imag
      cv::Vec2f coefficient = complexDFT.at<cv::Vec2f>(dftV, dftU);
      float realPart = coefficient[0];
      float imagPart = coefficient[1];
      float magnitude = std::sqrt(realPart * realPart + imagPart * imagPart);

      // --- Step 3: Add frequency to partial spectrum ---
      partialDFT.at<cv::Vec2f>(dftV, dftU) = coefficient;

      // For real-valued images, DFT has conjugate symmetry:
      // F(M-u, N-v) = conjugate(F(u,v))
      // We must add both the frequency and its conjugate pair for correct IDFT
      if (u > 0 || v > 0) {  // Skip DC component (u=0, v=0) - it has no pair
        int conjU = (u == 0) ? 0 : (complexDFT.cols - u);
        int conjV = (v == 0) ? 0 : (complexDFT.rows - v);
        if (conjU < complexDFT.cols && conjV < complexDFT.rows) {
          // Conjugate: flip sign of imaginary part
          partialDFT.at<cv::Vec2f>(conjV, conjU) = cv::Vec2f(realPart, -imagPart);
        }
      }

      // --- Step 4: Inverse DFT to reconstruct image from partial spectrum ---
      // DFT_SCALE: normalize by 1/N (required for proper amplitude)
      // DFT_REAL_OUTPUT: output only real part (imaginary should be ~0)
      cv::Mat reconstructedComplex;
      cv::idft(partialDFT, reconstructedComplex, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

      // Extract real channel (imaginary part is negligible for real images)
      cv::Mat reconstruction;
      if (reconstructedComplex.channels() == 2) {
        cv::Mat planes[2];
        cv::split(reconstructedComplex, planes);
        reconstruction = planes[0];  // Real part only
      } else {
        reconstruction = reconstructedComplex;  // Already real
      }

      // Remove padding: crop back to working size
      reconstruction = reconstruction(cv::Rect(0, 0, size, size));

      // --- Step 5: Prepare images for display ---

      // Basis wave: normalize from [-1,1] to [0,255] for visualization
      cv::Mat basisDisplay;
      cv::normalize(basisWave, basisDisplay, 0, 1, cv::NORM_MINMAX);
      cv::Mat basisU8;
      basisDisplay.convertTo(basisU8, CV_8U, 255);

      // Reconstruction: convert to 8-bit and clamp to valid range
      cv::Mat reconstructionU8;
      reconstruction.convertTo(reconstructionU8, CV_8U);
      reconstructionU8 = cv::max(0, cv::min(255, reconstructionU8));

      // Create 3-panel display: [Original | Basis | Reconstruction]
      cv::Mat combined;
      cv::hconcat(referenceImage, basisU8, combined);
      cv::hconcat(combined, reconstructionU8, combined);

      // Build informative window title showing current state
      int totalFreqs = (u * (maxFreq + 1)) + v + 1;  // Number of frequencies processed
      int maxTotalFreqs = (maxFreq + 1) * (maxFreq + 1);  // Total frequencies to process
      std::string windowTitle = "Fourier: u=" + std::to_string(u) +
        ", v=" + std::to_string(v) +
        " (" + std::to_string(totalFreqs) + "/" + std::to_string(maxTotalFreqs) + ")" +
        " | Mag=" + std::to_string((int)magnitude) +
        " | Original - Basis - Reconstruction";

      cv::setWindowTitle(windowName, windowTitle);
      cv::imshow(windowName, combined);

      // --- Step 6: Move to next frequency ---
      // Scan order: increment v first (vertical), then u (horizontal)
      // Pattern: (0,0), (0,1), (0,2), ..., (0,maxFreq), (1,0), (1,1), ...
      v++;
      if (v > maxFreq) {
        v = 0;  // Reset v, move to next u
        u++;
        if (u > maxFreq) {
          // All frequencies processed - reconstruction complete!
          std::cout << "\nReached maximum frequency (" << maxFreq << "," << maxFreq << ")" <<
            std::endl;
          std::cout << "Reconstruction complete. Press any key to exit..." << std::endl;
          paused = true;  // Pause to show final result
        }
      }
    }

    // Keyboard controls
    int key = cv::waitKey(paused ? 0 : 10);  // 10ms between frames when running
    if (key == 27 || key == 'q' || key == 'Q') {  // ESC or Q
      break;
    } else if (key == ' ') {  // SPACE
      paused = !paused;
      std::cout << (paused ? "Paused" : "Resumed") << " at u=" << u << ", v=" << v << std::endl;
    } else if (paused && u > maxFreq) {
      // Any other key when paused at the end - exit
      break;
    }
  }

  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
