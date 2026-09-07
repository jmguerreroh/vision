/**
 * @file frequencies.cpp
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

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
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
cv::Mat basicWave(int u, int v, int M = 500, int N = 500)
{
  cv::Mat Z(N, M, CV_32F);

  // Generate 2D cosine wave pattern
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < M; x++) {
      // Calculate phase: 2π(ux/M + vy/N)
      // This creates u horizontal cycles and v vertical cycles
      double angle = 2.0 * CV_PI *
        (static_cast<double>(u) * x / M + static_cast<double>(v) * y / N);
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
  int optimal_rows = cv::getOptimalDFTSize(image.rows);
  int optimal_cols = cv::getOptimalDFTSize(image.cols);

  cv::copyMakeBorder(image, padded,
                     0, optimal_rows - image.rows,
                     0, optimal_cols - image.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  // Create complex image with real and imaginary parts
  cv::Mat real_part;
  padded.convertTo(real_part, CV_32F);  // Real part = image data
  cv::Mat imaginary_part = cv::Mat::zeros(padded.size(), CV_32F);  // Imaginary part = 0

  cv::Mat planes[] = {real_part, imaginary_part};
  cv::Mat complex_image;
  cv::merge(planes, 2, complex_image);  // Merge into 2-channel complex matrix

  // Compute DFT
  cv::dft(complex_image, complex_image, cv::DFT_COMPLEX_OUTPUT);

  return complex_image;
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
            << "  2) Image reconstruction: " << argv[0] << " [image_path] [max_freq] [size]\n"
            << "     - Progressive reconstruction from frequency components\n"
            << "     - image_path: Image to decompose and reconstruct (default: starry_night.jpg)\n"
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
  cv::Mat reference_image;  // Resized image for reconstruction
  cv::Mat original_image;   // Original image before resizing
  int max_freq = -1;        // Maximum frequency to process (-1 = auto)
  int size = -1;           // Size of basis waves and reconstruction (-1 = auto)
  bool single_basis_mode = false;  // Display only one basis wave
  int single_u = 0, single_v = 0;  // Frequency for single basis mode

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
      single_basis_mode = true;
      single_u = std::atoi(argv[1]);
      single_v = std::atoi(argv[2]);
      size = std::atoi(argv[3]);

      std::cout << "Single basis wave mode: u=" << single_u
                << ", v=" << single_v << ", size=" << size << "x" << size << std::endl;
    }
  }

  // Parse first argument: can be either image path or max frequency
  if (!single_basis_mode && argc >= 2) {
    std::string filename = argv[1];

    // Check if argument is purely numeric (old-style max_freq parameter)
    if (filename.find_first_not_of("0123456789") == std::string::npos) {
      max_freq = std::atoi(argv[1]);
    } else {
      // Argument is a file path - attempt to load image
      original_image = cv::imread(cv::samples::findFile(filename, false), cv::IMREAD_GRAYSCALE);

      if (original_image.empty()) {
        std::cerr << "Error: Could not load image '" << filename << "'" << std::endl;
        return EXIT_FAILURE;
      }
      std::cout << "Reference image loaded: " << original_image.cols << "x"
                << original_image.rows << std::endl;

      // Set intelligent defaults based on image dimensions
      if (size == -1) {
        size = std::max(original_image.cols, original_image.rows);
      }
      if (max_freq == -1) {
        max_freq = std::max(original_image.cols, original_image.rows) / 2;
      }
    }
  } else if (!single_basis_mode && argc == 1) {
    // No arguments provided - use default image
    std::string filename = "../../data/starry_night.jpg";

    original_image = cv::imread(cv::samples::findFile(filename, false), cv::IMREAD_GRAYSCALE);
    if (original_image.empty()) {
      std::cerr << "Error: Could not load default image '" << filename << "'" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Using default image: " << filename << std::endl;
    std::cout << "Reference image loaded: " << original_image.cols << "x"
              << original_image.rows << std::endl;

    // Set intelligent defaults based on image dimensions
    if (size == -1) {
      size = std::max(original_image.cols, original_image.rows);
    }
    if (max_freq == -1) {
      max_freq = std::max(original_image.cols, original_image.rows) / 2;
    }
  }

  if (argc >= 3) {
    max_freq = std::atoi(argv[2]);
  }
  if (argc >= 4) {
    size = std::atoi(argv[3]);
  }

  // Set final defaults if still not set
  if (size == -1) {
    size = std::max(original_image.cols, original_image.rows);
  }
  if (max_freq == -1) {
    max_freq = size - 1;  // All frequencies for perfect reconstruction
  }

  // ============================================================================
  // MODE 1: Display single basis wave (no image required)
  // ============================================================================
  if (single_basis_mode) {
    std::cout << "\nGenerating basis wave Z(x,y) = cos(2π(" << single_u << "*x/" << size
              << " + " << single_v << "*y/" << size << "))" << std::endl;

    cv::Mat basis_wave = basicWave(single_u, single_v, size, size);

    // Normalize for display: [-1,1] -> [0,255]
    cv::Mat display;
    cv::normalize(basis_wave, display, 0, 1, cv::NORM_MINMAX);
    cv::Mat displayU8;
    display.convertTo(displayU8, CV_8U, 255);

    std::string windowName = "Fourier Basis: u=" + std::to_string(single_u) +
      ", v=" + std::to_string(single_v);
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, displayU8);

    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
  }

  // Resize image to working size
  if (!original_image.empty()) {
    cv::resize(original_image, reference_image, cv::Size(size, size));
  }

  if (reference_image.empty()) {
    std::cerr << "Error: Image is required for reconstruction demo." << std::endl;
    std::cerr << "Usage: " << argv[0] << " <image_path> [max_freq] [size]" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Max frequency: " << max_freq << std::endl;
  std::cout << "Basis wave size: " << size << "x" << size << std::endl;

  // ============================================================================
  // SECTION 2: Compute DFT of reference image
  // ============================================================================
  std::cout << "Computing DFT..." << std::endl;
  cv::Mat complex_dft = computeDFT(reference_image);
  std::cout << "DFT size: " << complex_dft.cols << "x" << complex_dft.rows << std::endl;

  // Auto-adjust max_freq to include all DFT frequencies if user didn't specify
  if (argc < 3) {
    max_freq = std::max(complex_dft.rows, complex_dft.cols) - 1;
    std::cout << "Using full DFT spectrum, max_freq updated to: " << max_freq << std::endl;
  }

  // ============================================================================
  // SECTION 3: Progressive reconstruction animation loop
  // ============================================================================
  std::cout << "\nStarting animation loop..." << std::endl;
  std::cout << "Press SPACE to pause/resume, Q or ESC to quit" << std::endl;

  const std::string window_name = "Fourier Basis Reconstruction";
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  bool paused = false;
  int u = 0;
  int v = 0;

  // Partial DFT spectrum: accumulates frequencies progressively
  // Starts empty (all zeros), one frequency added per iteration
  cv::Mat partial_dft = cv::Mat::zeros(complex_dft.size(), CV_32FC2);

  while (true) {
    if (!paused) {
      // --- Step 1: Generate basis wave for current frequency (u,v) ---
      cv::Mat basis_wave = basicWave(u, v, size, size);

      // --- Step 2: Get DFT coefficient at this frequency ---
      int dft_u = u;
      int dft_v = v;

      // Validate frequency is within DFT bounds (may differ from max_freq)
      if (dft_u >= complex_dft.cols || dft_v >= complex_dft.rows) {
        // Out of bounds - skip to next frequency
        v++;
        if (v > max_freq) {
          v = 0;
          u++;
        }
        continue;
      }

      // Extract complex coefficient: F(u,v) = real + i*imag
      cv::Vec2f coefficient = complex_dft.at<cv::Vec2f>(dft_v, dft_u);
      float real_part = coefficient[0];
      float imag_part = coefficient[1];
      float magnitude = std::sqrt(real_part * real_part + imag_part * imag_part);

      // --- Step 3: Add frequency to partial spectrum ---
      partial_dft.at<cv::Vec2f>(dft_v, dft_u) = coefficient;

      // For real-valued images, DFT has conjugate symmetry:
      // F(M-u, N-v) = conjugate(F(u,v))
      // We must add both the frequency and its conjugate pair for correct IDFT
      if (u > 0 || v > 0) {  // Skip DC component (u=0, v=0) - it has no pair
        int conj_u = (u == 0) ? 0 : (complex_dft.cols - u);
        int conj_v = (v == 0) ? 0 : (complex_dft.rows - v);
        if (conj_u < complex_dft.cols && conj_v < complex_dft.rows) {
          // Conjugate: flip sign of imaginary part
          partial_dft.at<cv::Vec2f>(conj_v, conj_u) = cv::Vec2f(real_part, -imag_part);
        }
      }

      // --- Step 4: Inverse DFT to reconstruct image from partial spectrum ---
      // DFT_SCALE: normalize by 1/N (required for proper amplitude)
      // DFT_REAL_OUTPUT: output only real part (imaginary should be ~0)
      cv::Mat reconstructed_complex;
      cv::idft(partial_dft, reconstructed_complex, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

      // Extract real channel (imaginary part is negligible for real images)
      cv::Mat reconstruction;
      if (reconstructed_complex.channels() == 2) {
        cv::Mat planes[2];
        cv::split(reconstructed_complex, planes);
        reconstruction = planes[0];  // Real part only
      } else {
        reconstruction = reconstructed_complex;  // Already real
      }

      // Remove padding: crop back to working size
      reconstruction = reconstruction(cv::Rect(0, 0, size, size));

      // --- Step 5: Prepare images for display ---

      // Basis wave: normalize from [-1,1] to [0,255] for visualization
      cv::Mat basis_display;
      cv::normalize(basis_wave, basis_display, 0, 1, cv::NORM_MINMAX);
      cv::Mat basis_u8;
      basis_display.convertTo(basis_u8, CV_8U, 255);

      // Reconstruction: convert to 8-bit and clamp to valid range
      cv::Mat reconstruction_u8;
      reconstruction.convertTo(reconstruction_u8, CV_8U);
      reconstruction_u8 = cv::max(0, cv::min(255, reconstruction_u8));

      // Create 3-panel display: [Original | Basis | Reconstruction]
      cv::Mat combined;
      cv::hconcat(reference_image, basis_u8, combined);
      cv::hconcat(combined, reconstruction_u8, combined);

      // Build informative window title showing current state
      int total_freqs = (u * (max_freq + 1)) + v + 1;  // Number of frequencies processed
      int max_total_freqs = (max_freq + 1) * (max_freq + 1);  // Total frequencies to process
      std::string window_title = "Fourier: u=" + std::to_string(u) +
        ", v=" + std::to_string(v) +
        " (" + std::to_string(total_freqs) + "/" + std::to_string(max_total_freqs) + ")" +
        " | Mag=" + std::to_string(static_cast<int>(magnitude)) +
        " | Original - Basis - Reconstruction";

      cv::setWindowTitle(window_name, window_title);
      cv::imshow(window_name, combined);

      // --- Step 6: Move to next frequency ---
      // Scan order: increment v first (vertical), then u (horizontal)
      // Pattern: (0,0), (0,1), (0,2), ..., (0,max_freq), (1,0), (1,1), ...
      v++;
      if (v > max_freq) {
        v = 0;  // Reset v, move to next u
        u++;
        if (u > max_freq) {
          // All frequencies processed - reconstruction complete!
          std::cout << "\nReached maximum frequency (" << max_freq << "," << max_freq << ")" <<
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
    } else if (paused && u > max_freq) {
      // Any other key when paused at the end - exit
      break;
    }
  }

  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
