/**
 * @file main.cpp
 * @brief Discrete Fourier Transform (DFT) in OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - How to compute the Discrete Fourier Transform (DFT) of an image
 * - How to visualize the frequency spectrum (magnitude)
 * - How to shift quadrants for centered spectrum display
 * - How to reconstruct the image using Inverse DFT (IDFT)
 *
 * @note The Fourier Transform decomposes an image into its frequency components:
 *       - Low frequencies: Smooth regions, gradual changes
 *       - High frequencies: Edges, sharp transitions, noise
 *
 *       The magnitude spectrum shows how much of each frequency is present.
 *       The phase spectrum (not shown here) contains structural information.
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace
{
// The images this example works on are about 1400 px on the long side, and
// several windows at that size do not fit on a normal screen. The processing
// always runs at full resolution: only the copy sent to the screen is reduced,
// with INTER_AREA, which is the interpolation meant for shrinking
constexpr int MAX_DISPLAY_SIDE = 800;

// Returns the copy that goes to the screen, already reduced. Whatever is drawn
// on the result keeps its size in screen pixels, so labels are written here and
// not on the full-resolution frame: drawn before the reduction they shrink with
// it and stop being readable
cv::Mat fitToScreen(const cv::Mat & image)
{
  const int side = std::max(image.cols, image.rows);
  if (side <= MAX_DISPLAY_SIDE || image.empty()) {
    return image.clone();
  }
  const double factor = static_cast<double>(MAX_DISPLAY_SIDE) / side;
  cv::Mat reduced;
  cv::resize(image, reduced, cv::Size(), factor, factor, cv::INTER_AREA);
  return reduced;
}

void showFit(const std::string & window, const cv::Mat & image)
{
  cv::imshow(window, fitToScreen(image));
}
}  // namespace

/**
 * @brief Displays usage information
 * @param argv Command line arguments
 */
void printHelp(char ** argv)
{
  std::cout << "\n"
            << "Discrete Fourier Transform (DFT) Demo\n"
            << "=====================================\n"
            << "This program computes the DFT of an image and displays its power spectrum.\n\n"
            << "Usage: " << argv[0] << " [image_path]\n"
            << "  image_path: Path to input image (default: starry_night.png)\n\n";
}

/**
 * @brief Computes the Discrete Fourier Transform of an image
 * @param image Input grayscale image
 * @return Complex matrix containing DFT result (real and imaginary parts)
 *
 * The DFT converts a spatial domain image to frequency domain.
 * The result is a complex matrix where each element contains:
 *   - Real part: cosine component amplitude
 *   - Imaginary part: sine component amplitude
 */
cv::Mat computeDFT(const cv::Mat & image)
{
  // Step 1: Expand image to optimal size for faster DFT computation
  // DFT is fastest when array size is a power of 2, or factors of 2, 3, and 5
  cv::Mat padded;
  int optimal_rows = cv::getOptimalDFTSize(image.rows);
  int optimal_cols = cv::getOptimalDFTSize(image.cols);

  // Pad with zeros on the right and bottom borders
  cv::copyMakeBorder(image, padded,
                     0, optimal_rows - image.rows,
                     0, optimal_cols - image.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  // Step 2: Create complex matrix with real and imaginary parts
  // Real part: the padded image converted to float
  cv::Mat real_part;
  padded.convertTo(real_part, CV_32F);

  // Imaginary part: zeros (input image has no imaginary component)
  cv::Mat imaginary_part = cv::Mat::zeros(padded.size(), CV_32F);

  // Step 3: Merge into a 2-channel complex matrix
  cv::Mat planes[] = {real_part, imaginary_part};
  cv::Mat complex_image;
  cv::merge(planes, 2, complex_image);

  // Step 4: Compute the DFT
  // DFT_COMPLEX_OUTPUT ensures output has both real and imaginary parts
  cv::dft(complex_image, complex_image, cv::DFT_COMPLEX_OUTPUT);

  return complex_image;
}

/**
 * @brief Shifts the zero-frequency component to the center of the spectrum
 * @param magI Input magnitude/complex image
 * @return Shifted image with DC component at center
 *
 * After DFT, the zero-frequency (DC) component is at the corners.
 * This function rearranges quadrants so DC is at the center,
 * which is the conventional way to display frequency spectra.
 *
 * Before shift:          After shift:
 * +-------+-------+      +-------+-------+
 * | Q0    | Q1    |      | Q3    | Q2    |
 * | (DC)  |       |      |       |       |
 * +-------+-------+  =>  +-------+-------+
 * | Q2    | Q3    |      | Q1    | Q0    |
 * |       |       |      |       | (DC)  |
 * +-------+-------+      +-------+-------+
 */
cv::Mat fftShift(const cv::Mat & magI)
{
  cv::Mat result = magI.clone();

  // Crop if odd number of rows or columns
  result = result(cv::Rect(0, 0, result.cols & -2, result.rows & -2));

  // Calculate center point
  int cx = result.cols / 2;
  int cy = result.rows / 2;

  // Define the four quadrants as ROIs (Region of Interest)
  cv::Mat q0(result, cv::Rect(0, 0, cx, cy));    // Top-Left
  cv::Mat q1(result, cv::Rect(cx, 0, cx, cy));   // Top-Right
  cv::Mat q2(result, cv::Rect(0, cy, cx, cy));   // Bottom-Left
  cv::Mat q3(result, cv::Rect(cx, cy, cx, cy));  // Bottom-Right

  // Swap quadrants diagonally
  cv::Mat tmp;
  q0.copyTo(tmp);  // Q0 <-> Q3
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);  // Q1 <-> Q2
  q2.copyTo(q1);
  tmp.copyTo(q2);

  return result;
}

/**
 * @brief Computes the magnitude spectrum from a complex DFT result
 * @param complex_i Complex matrix from DFT
 * @return Normalized magnitude spectrum ready for display
 *
 * The magnitude spectrum shows the amplitude of each frequency component.
 * Formula: magnitude = sqrt(Re^2 + Im^2)
 *
 * Logarithmic scale is applied because the dynamic range is too large
 * for display: log(1 + magnitude)
 */
cv::Mat computeSpectrum(const cv::Mat & complex_i)
{
  cv::Mat complex_img = complex_i.clone();

  // Step 1: Split into real and imaginary parts
  cv::Mat planes[2];
  cv::split(complex_img, planes);
  // planes[0] = Real part, planes[1] = Imaginary part

  // Step 2: Compute magnitude: sqrt(Re^2 + Im^2)
  cv::Mat magnitude_image;
  cv::magnitude(planes[0], planes[1], magnitude_image);

  // Step 3: Apply logarithmic scale for better visualization
  // Without log, bright spots would dominate and details would be invisible
  magnitude_image += cv::Scalar::all(1);  // Avoid log(0)
  cv::log(magnitude_image, magnitude_image);

  // Step 4: Normalize to range [0, 1] for display
  cv::normalize(magnitude_image, magnitude_image, 0, 1, cv::NORM_MINMAX);

  return magnitude_image;
}

int main(int argc, char ** argv)
{

  // Load the image
  //
  // cv::samples::findFile helps locate the image file in OpenCV sample directories.
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/starry_night.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    printHelp(argv);
    return EXIT_SUCCESS;
  }
  const std::string filename = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  cv::Mat image = cv::imread(cv::samples::findFile(filename, false), cv::IMREAD_GRAYSCALE);

  if (image.empty()) {
    std::cerr << "Error: Could not open image '" << filename << "'" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Image loaded: " << image.cols << "x" << image.rows << " pixels" << std::endl;

  // Compute the Discrete Fourier Transform
  cv::Mat complex_image = computeDFT(image);
  std::cout << "DFT computed successfully" << std::endl;

  // Compute and display the magnitude spectrum
  cv::Mat spectrum_original = computeSpectrum(complex_image);

  // Demonstrate quadrant shifting
  //
  // First shift: Move DC to center (for processing/visualization)
  cv::Mat shifted_complex = fftShift(complex_image);  // DC at center

  // Compute and display the magnitude spectrum after first shift
  cv::Mat spectrum_shifted = computeSpectrum(shifted_complex);

  // Here you could apply frequency domain filters:
  //   Low-pass filter: Keep center, remove edges (blur)
  //   High-pass filter: Remove center, keep edges (sharpen)
  //   Band-pass filter: Keep specific frequency range

  // Second shift: Move DC back to corners (for inverse DFT)
  cv::Mat rearranged = fftShift(shifted_complex);    // DC back to corners

  // Compute and display the magnitude spectrum after rearrangement
  cv::Mat spectrum_after = computeSpectrum(rearranged);

  // Reconstruct image using Inverse DFT
  //
  // IDFT converts frequency domain back to spatial domain.
  // DFT_REAL_OUTPUT: output only the real part (the imaginary residue of a
  //                  real-valued image is negligible).
  // We do not pass DFT_SCALE (the 1/N normalization) because the result is
  // normalized to [0,1] right after for display anyway.
  cv::Mat reconstructed;
  cv::idft(rearranged, reconstructed, cv::DFT_REAL_OUTPUT);
  cv::normalize(reconstructed, reconstructed, 0, 1, cv::NORM_MINMAX);

  // Display results
  showFit("Original Image", image);
  showFit("Spectrum Before DC Shift", spectrum_original);
  showFit("Spectrum After DC Shift", spectrum_shifted);
  showFit("Spectrum After Rearrangement", spectrum_after);
  showFit("Reconstructed (IDFT)", reconstructed);

  std::cout << "\nWindows displayed:" << std::endl;
  std::cout << "  - Original grayscale image" << std::endl;
  std::cout << "  - Magnitude spectrum (centered)" << std::endl;
  std::cout << "  - Magnitude spectrum after rearrangement" << std::endl;
  std::cout << "  - Reconstructed image from IDFT" << std::endl;
  std::cout << "\nPress any key to exit..." << std::endl;

  cv::waitKey(0);

  return EXIT_SUCCESS;
}
