/**
 * @file main.cpp
 * @brief Homomorphic filtering: correcting non-uniform illumination
 * @author José Miguel Guerrero Hernández
 *
 * An image is the product of two components, f = i * r:
 *   i(x,y) illumination, slow variation  -> low frequencies
 *   r(x,y) reflectance,  fast variation  -> high frequencies
 *
 * Linear filtering distributes over sums, not over products, so a high-pass
 * filter applied to f cannot separate them. Taking logarithms turns the
 * product into a sum and makes the problem linear:
 *
 *   ln f = ln i + ln r
 *
 * The pipeline is therefore: log -> DFT -> H(u,v) -> inverse DFT -> exp.
 *
 * H is not an ideal high-pass filter: killing the low frequencies would also
 * kill the mean level of the image. It is a high-frequency emphasis filter,
 *
 *   H(u,v) = (gammaH - gammaL) * [1 - exp(-c * D^2(u,v) / D0^2)] + gammaL
 *
 * with gammaL < 1 (attenuates illumination) and gammaH > 1 (boosts detail).
 *
 * To make the effect measurable, the example degrades a well-exposed image
 * with a synthetic light spot, so the undegraded original is available as
 * ground truth.
 *
 * @see https://docs.opencv.org/4.x/d2/de8/group__core__array.html
 */

#include <algorithm>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>

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

namespace Config
{
// Filter parameters
constexpr float GAMMA_L = 0.25f;      // gain at the origin: attenuates illumination
constexpr float GAMMA_H = 1.40f;      // gain far from the origin: boosts reflectance
constexpr float D0 = 8.0f;            // cut-off radius, in frequency samples
constexpr float C = 1.0f;             // sharpness of the transition

// Synthetic illumination: a soft spot on the upper right corner
constexpr double AMBIENT = 0.22;      // light everywhere, even in the shade
constexpr double SPOT = 0.78;         // extra light under the spot

// Local contrast measurement
constexpr int WINDOW = 31;            // side of the window, in pixels
constexpr double CLIP_PERCENT = 0.01; // discarded at each end before rescaling
}

/**
 * @brief Multiplies an image by a smooth synthetic illumination field
 * @param src Well-exposed input image, used as the reflectance
 * @param illumination Output illumination field, in [AMBIENT, AMBIENT + SPOT]
 * @return The observed image, i * r
 *
 * The maximum of the field is 1, so no pixel saturates and the product is
 * exact: whatever the filter recovers, it recovers from a true product.
 */
cv::Mat applySyntheticIllumination(const cv::Mat & src, cv::Mat & illumination)
{
  illumination.create(src.size(), CV_32F);
  const double cx = src.cols * 0.78, cy = src.rows * 0.18;
  const double sx = src.cols * 0.55, sy = src.rows * 0.75;

  for (int y = 0; y < src.rows; ++y) {
    float * row = illumination.ptr<float>(y);
    for (int x = 0; x < src.cols; ++x) {
      const double dx = (x - cx) / sx, dy = (y - cy) / sy;
      row[x] = static_cast<float>(Config::AMBIENT +
        Config::SPOT * std::exp(-(dx * dx + dy * dy)));
    }
  }

  cv::Mat observed;
  src.convertTo(observed, CV_32F);
  observed = observed.mul(illumination);
  observed.convertTo(observed, CV_8U);
  return observed;
}

/**
 * @brief Builds the transfer function H(u,v), in the layout the DFT expects
 * @param size Size of the (already padded) spectrum
 * @return CV_32F matrix with the filter
 *
 * cv::dft leaves the DC term at (0,0) and the spectrum wrapped around, so the
 * distance to the origin is measured to the NEAREST corner, not to the centre
 * of the matrix. Building H this way avoids having to shift quadrants.
 */
cv::Mat buildTransferFunction(cv::Size size)
{
  cv::Mat H(size, CV_32F);
  for (int v = 0; v < size.height; ++v) {
    float * row = H.ptr<float>(v);
    const float dv = static_cast<float>(std::min(v, size.height - v));
    for (int u = 0; u < size.width; ++u) {
      const float du = static_cast<float>(std::min(u, size.width - u));
      const float d2 = du * du + dv * dv;
      row[u] = (Config::GAMMA_H - Config::GAMMA_L) *
        (1.0f - std::exp(-Config::C * d2 / (Config::D0 * Config::D0))) +
        Config::GAMMA_L;
    }
  }
  return H;
}

/**
 * @brief Rescales a float image to [0,255] discarding the extreme percentiles
 *
 * The exponential is unbounded, and a handful of extreme pixels would set the
 * whole scale if cv::normalize with NORM_MINMAX were used directly.
 */
cv::Mat rescaleRobust(const cv::Mat & src)
{
  cv::Mat flat = src.reshape(1, 1).clone();
  cv::sort(flat, flat, cv::SORT_ASCENDING + cv::SORT_EVERY_ROW);

  const int n = flat.cols;
  const int low_index = static_cast<int>(Config::CLIP_PERCENT * n);
  const int high_index = n - 1 - low_index;
  const float low = flat.at<float>(low_index);
  const float high = flat.at<float>(high_index);

  cv::Mat scaled;
  src.convertTo(scaled, CV_8U, 255.0 / (high - low), -255.0 * low / (high - low));
  return scaled;
}

/**
 * @brief Applies the homomorphic filter to a grey-scale image
 */
cv::Mat homomorphicFilter(const cv::Mat & gray)
{
  // The DFT is much faster on sizes that factor into 2, 3 and 5. The border
  // is replicated rather than zero-padded: zeros would become a dark frame
  // after the logarithm and leak into the result.
  const int rows = cv::getOptimalDFTSize(gray.rows);
  const int cols = cv::getOptimalDFTSize(gray.cols);
  cv::Mat padded;
  cv::copyMakeBorder(gray, padded, 0, rows - gray.rows, 0, cols - gray.cols,
    cv::BORDER_REPLICATE);

  // 1. Logarithm. The +1 keeps ln(0) out of the way
  cv::Mat logarithm;
  padded.convertTo(logarithm, CV_32F);
  logarithm += 1.0f;
  cv::log(logarithm, logarithm);

  // 2. DFT, asking for the two channels explicitly
  cv::Mat spectrum;
  cv::dft(logarithm, spectrum, cv::DFT_COMPLEX_OUTPUT);

  // 3. Multiply by H. The filter is real, so it multiplies both the real and
  //    the imaginary parts: it changes the amplitude of each frequency and
  //    leaves its phase, that is, the position of the structures, untouched
  const cv::Mat H = buildTransferFunction(spectrum.size());
  std::vector<cv::Mat> parts(2);
  cv::split(spectrum, parts);
  parts[0] = parts[0].mul(H);
  parts[1] = parts[1].mul(H);
  cv::merge(parts, spectrum);

  // 4. Inverse DFT. DFT_SCALE applies the 1/N factor, which OpenCV does not
  //    add by default, and DFT_REAL_OUTPUT drops the residual imaginary part
  cv::Mat filtered;
  cv::idft(spectrum, filtered, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

  // 5. Undo the logarithm and come back to 8 bits
  cv::exp(filtered, filtered);
  filtered -= 1.0f;
  return rescaleRobust(filtered)(cv::Rect(0, 0, gray.cols, gray.rows)).clone();
}

/**
 * @brief Mean local contrast inside a region
 * @param gray Image to measure
 * @param region Rectangle to average over
 * @return Mean standard deviation of a WINDOW x WINDOW neighbourhood
 *
 * This is what the filter is expected to restore: in the shaded area the
 * differences between neighbouring pixels were multiplied by a small factor,
 * so the detail is still there but with almost no contrast.
 */
double localContrast(const cv::Mat & gray, const cv::Rect & region)
{
  cv::Mat f, mean, mean_of_squares, variance;
  gray.convertTo(f, CV_32F);
  cv::blur(f, mean, cv::Size(Config::WINDOW, Config::WINDOW));
  cv::blur(f.mul(f), mean_of_squares, cv::Size(Config::WINDOW, Config::WINDOW));
  cv::sqrt(cv::max(mean_of_squares - mean.mul(mean), 0.0f), variance);
  return cv::mean(variance(region))[0];
}

int main(int argc, char ** argv)
{
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/building_facade.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string path = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  cv::Mat original = cv::imread(cv::samples::findFile(path, false), cv::IMREAD_GRAYSCALE);

  if (original.empty()) {
    std::cerr << "Error: Could not load image" << std::endl;
    std::cerr << "Usage: " << argv[0] << " [image_path]" << std::endl;
    return EXIT_FAILURE;
  }
  cv::resize(original, original, cv::Size(480, 320));

  std::cout << "=== Homomorphic Filtering Demo ===" << std::endl;
  std::cout << "gammaL = " << Config::GAMMA_L << ", gammaH = " << Config::GAMMA_H
            << ", D0 = " << Config::D0 << std::endl;

  // Degrade the image with a known illumination field
  cv::Mat illumination;
  const cv::Mat observed = applySyntheticIllumination(original, illumination);
  const cv::Mat filtered = homomorphicFilter(observed);

  showFit("Reflectance (undegraded)", original);
  showFit("Illumination", illumination);
  showFit("Observed = illumination x reflectance", observed);
  showFit("After the homomorphic filter", filtered);

  // Measure the shaded area: bottom left, away from the spot
  const cv::Rect shade(20, 200, 140, 100);
  std::cout << "\nLocal contrast in the shaded area (" << Config::WINDOW << "x"
            << Config::WINDOW << " window):" << std::endl;
  std::cout << "  observed:   " << localContrast(observed, shade) << std::endl;
  std::cout << "  filtered:   " << localContrast(filtered, shade) << std::endl;
  std::cout << "  undegraded: " << localContrast(original, shade)
            << "   <- the ceiling, what the filter is trying to recover"
            << std::endl;

  // And how uniform the illumination has become
  const cv::Rect lit(300, 30, 140, 100);
  const double before = cv::mean(observed(lit))[0] / cv::mean(observed(shade))[0];
  const double after = cv::mean(filtered(lit))[0] / cv::mean(filtered(shade))[0];
  std::cout << "\nBrightness of the lit area divided by the shaded one:"
            << std::endl;
  std::cout << "  observed: " << before << "   filtered: " << after << std::endl;

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
