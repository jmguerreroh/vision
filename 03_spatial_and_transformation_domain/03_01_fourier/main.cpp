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

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

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
            << "  image_path: Path to input image (default: lena.jpg)\n\n";
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
  int optimalRows = cv::getOptimalDFTSize(image.rows);
  int optimalCols = cv::getOptimalDFTSize(image.cols);

  // Pad with zeros on the right and bottom borders
  cv::copyMakeBorder(image, padded,
                     0, optimalRows - image.rows,
                     0, optimalCols - image.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  // Step 2: Create complex matrix with real and imaginary parts
  // Real part: the padded image converted to float
  cv::Mat realPart;
  padded.convertTo(realPart, CV_32F);

  // Imaginary part: zeros (input image has no imaginary component)
  cv::Mat imaginaryPart = cv::Mat::zeros(padded.size(), CV_32F);

  // Step 3: Merge into a 2-channel complex matrix
  cv::Mat planes[] = {realPart, imaginaryPart};
  cv::Mat complexImage;
  cv::merge(planes, 2, complexImage);

  // Step 4: Compute the DFT
  // DFT_COMPLEX_OUTPUT ensures output has both real and imaginary parts
  cv::dft(complexImage, complexImage, cv::DFT_COMPLEX_OUTPUT);

  return complexImage;
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
 * @param complexI Complex matrix from DFT
 * @return Normalized magnitude spectrum ready for display
 *
 * The magnitude spectrum shows the amplitude of each frequency component.
 * Formula: magnitude = sqrt(Re^2 + Im^2)
 *
 * Logarithmic scale is applied because the dynamic range is too large
 * for display: log(1 + magnitude)
 */
cv::Mat computeSpectrum(const cv::Mat & complexI)
{
  cv::Mat complexImg = complexI.clone();

  // Step 1: Shift to center the DC component
  cv::Mat shiftedComplex = fftShift(complexImg);

  // Step 2: Split into real and imaginary parts
  cv::Mat planes[2];
  cv::split(shiftedComplex, planes);
  // planes[0] = Real part, planes[1] = Imaginary part

  // Step 3: Compute magnitude: sqrt(Re^2 + Im^2)
  cv::Mat magnitudeImage;
  cv::magnitude(planes[0], planes[1], magnitudeImage);

  // Step 4: Apply logarithmic scale for better visualization
  // Without log, bright spots would dominate and details would be invisible
  magnitudeImage += cv::Scalar::all(1);  // Avoid log(0)
  cv::log(magnitudeImage, magnitudeImage);

  // Step 5: Normalize to range [0, 1] for display
  cv::normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);

  return magnitudeImage;
}

int main(int argc, char ** argv)
{
  printHelp(argv);

  // Load the image
  //
  // cv::samples::findFile helps locate the image file in OpenCV sample directories.
  const char * filename = argc >= 2 ? argv[1] : "lena.jpg";
  cv::Mat image = cv::imread(cv::samples::findFile(filename), cv::IMREAD_GRAYSCALE);

  if (image.empty()) {
    std::cerr << "Error: Could not open image '" << filename << "'" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Image loaded: " << image.cols << "x" << image.rows << " pixels" << std::endl;

  // Compute the Discrete Fourier Transform
  cv::Mat complexImage = computeDFT(image);
  std::cout << "DFT computed successfully" << std::endl;

  // Compute and display the magnitude spectrum
  cv::Mat spectrumOriginal = computeSpectrum(complexImage);

  // Demonstrate quadrant shifting
  //
  // First shift: Move DC to center (for processing/visualization)
  cv::Mat shiftedComplex = fftShift(complexImage);  // DC at center

  // Here you could apply frequency domain filters:
  //   Low-pass filter: Keep center, remove edges (blur)
  //   High-pass filter: Remove center, keep edges (sharpen)
  //   Band-pass filter: Keep specific frequency range

  // Second shift: Move DC back to corners (for inverse DFT)
  cv::Mat rearranged = fftShift(shiftedComplex);    // DC back to corners

  // Compute and display the magnitude spectrum after rearrangement
  cv::Mat spectrumAfter = computeSpectrum(rearranged);

  // Reconstruct image using Inverse DFT
  //
  // IDFT converts frequency domain back to spatial domain.
  // DFT_INVERSE: Perform inverse transform
  // DFT_REAL_OUTPUT: Output only real part (discard small imaginary residue)
  cv::Mat reconstructed;
  cv::idft(rearranged, reconstructed, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
  cv::normalize(reconstructed, reconstructed, 0, 1, cv::NORM_MINMAX);

  // Display results
  cv::imshow("Original Image", image);
  cv::imshow("Magnitude Spectrum", spectrumOriginal);
  cv::imshow("Spectrum After Processing", spectrumAfter);
  cv::imshow("Reconstructed (IDFT)", reconstructed);

  std::cout << "\nWindows displayed:" << std::endl;
  std::cout << "  - Original grayscale image" << std::endl;
  std::cout << "  - Magnitude spectrum (centered)" << std::endl;
  std::cout << "  - Magnitude spectrum after rearrangement" << std::endl;
  std::cout << "  - Reconstructed image from IDFT" << std::endl;
  std::cout << "\nPress any key to exit..." << std::endl;

  cv::waitKey(0);

  return EXIT_SUCCESS;
}
