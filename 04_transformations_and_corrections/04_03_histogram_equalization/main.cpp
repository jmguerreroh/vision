/**
 * @file main.cpp
 * @brief Histogram calculation and equalization demonstration using OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Histogram calculation for BGR color channels
 * - Histogram visualization with line plots
 * - Histogram equalization per channel
 *
 * @see https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <array>

// Histogram configuration
namespace Config
{
constexpr int HIST_SIZE = 256;              // Number of bins (intensity levels)
constexpr int HIST_WIDTH = 512;             // Display width in pixels
constexpr int HIST_HEIGHT = 400;            // Display height in pixels
constexpr float HIST_MIN_RANGE = 0.0f;      // Minimum intensity value
constexpr float HIST_MAX_RANGE = 256.0f;    // Maximum intensity value
constexpr int NUM_BGR_CHANNELS = 3;         // Number of BGR channels
}

/**
 * @brief Get BGR color for channel visualization
 * @param channel Channel index (0=Blue, 1=Green, 2=Red)
 * @return Scalar color for drawing
 */
cv::Scalar getChannelColor(int channel)
{
  static const std::array<cv::Scalar, Config::NUM_BGR_CHANNELS> COLORS = {
    cv::Scalar(255, 0, 0),   // Blue
    cv::Scalar(0, 255, 0),   // Green
    cv::Scalar(0, 0, 255)    // Red
  };
  return COLORS[channel];
}

/**
 * @brief Calculate histograms for all channels of an image
 * @param channels Vector of single-channel images (BGR planes)
 * @return Vector of histogram matrices
 */
std::vector<cv::Mat> calculateHistograms(const std::vector<cv::Mat> & channels)
{
  std::vector<cv::Mat> histograms(channels.size());

  // Define histogram range [min, max)
  const float histRange[] = {Config::HIST_MIN_RANGE, Config::HIST_MAX_RANGE};
  const float * histRangePtr = histRange;
  const int channels_idx = 0;  // Process channel 0 of each image

  for (size_t i = 0; i < channels.size(); i++) {
    // cv::calcHist(images, nimages, channels, mask, hist, dims, histSize, ranges, uniform, accumulate)
    //   images:     Pointer to source images (here: single channel)
    //   nimages:    Number of source images (1)
    //   channels:   Channel indices to process (0 = first channel)
    //   mask:       Optional mask (cv::Mat() = no mask, use all pixels)
    //   hist:       Output histogram matrix
    //   dims:       Histogram dimensionality (1 = 1D histogram)
    //   histSize:   Number of bins per dimension (256 for 8-bit images)
    //   ranges:     Array of value ranges per dimension ([0, 256))
    //   uniform:    true = bins are equally spaced
    //   accumulate: false = clear histogram before computing
    cv::calcHist(&channels[i], 1, &channels_idx, cv::Mat(), histograms[i],
                 1, &Config::HIST_SIZE, &histRangePtr, true, false);
  }

  return histograms;
}

/**
 * @brief Draw histogram visualization for all channels
 * @param histograms Vector of histogram matrices (one per channel)
 * @return Image with drawn histogram lines
 */
cv::Mat drawHistogram(const std::vector<cv::Mat> & histograms)
{
  cv::Mat histImage(Config::HIST_HEIGHT, Config::HIST_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
  const int binWidth = cvRound(static_cast<double>(Config::HIST_WIDTH) / Config::HIST_SIZE);

  // Normalize histograms to fit display height
  std::vector<cv::Mat> normalized(histograms.size());
  for (size_t i = 0; i < histograms.size(); i++) {
    cv::normalize(histograms[i], normalized[i], 0, Config::HIST_HEIGHT, cv::NORM_MINMAX);
  }

  // Draw lines for each channel
  for (int x = 1; x < Config::HIST_SIZE; x++) {
    for (size_t ch = 0; ch < normalized.size(); ch++) {
      const int y1 = Config::HIST_HEIGHT - cvRound(normalized[ch].at<float>(x - 1));
      const int y2 = Config::HIST_HEIGHT - cvRound(normalized[ch].at<float>(x));

      cv::line(histImage,
               cv::Point(binWidth * (x - 1), y1),
               cv::Point(binWidth * x, y2),
               getChannelColor(ch), 2);
    }
  }

  return histImage;
}

/**
 * @brief Apply histogram equalization to all channels
 * @param channels Vector of single-channel images
 * @return Vector of equalized channels
 *
 * @note Histogram equalization is applied independently to each channel.
 *       For color images, this may cause color shifts. Consider using
 *       equalizeHist on the V channel of HSV or L channel of Lab for
 *       better color preservation.
 */
std::vector<cv::Mat> equalizeChannels(const std::vector<cv::Mat> & channels)
{
  std::vector<cv::Mat> equalized(channels.size());

  for (size_t i = 0; i < channels.size(); i++) {
    // cv::equalizeHist(src, dst)
    //   src: Input 8-bit single-channel image
    //   dst: Output image with equalized histogram (same size and type)
    //
    // Algorithm:
    //   1. Calculate histogram of input image
    //   2. Normalize histogram so sum = number of pixels
    //   3. Compute cumulative distribution function (CDF)
    //   4. Map original pixel values using normalized CDF as lookup table
    //
    // Result: Spreads intensity values across full range [0, 255]
    //         improving contrast in images with narrow intensity range
    cv::equalizeHist(channels[i], equalized[i]);
  }

  return equalized;
}

int main(int argc, char ** argv)
{
  // Load image
  cv::CommandLineParser parser(argc, argv, "{@input | lena.jpg | input image}");
  cv::Mat src = cv::imread(cv::samples::findFile(parser.get<cv::String>("@input")),
    cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not load input image" << std::endl;
    return -1;
  }

  std::cout << "=== Histogram Equalization Demo ===" << std::endl;
  std::cout << "Image: " << src.cols << "x" << src.rows << " pixels" << std::endl;

  // Split into BGR channels
  std::vector<cv::Mat> bgrChannels;
  cv::split(src, bgrChannels);

  // Calculate and draw original histogram
  std::vector<cv::Mat> histograms = calculateHistograms(bgrChannels);
  cv::Mat histImage = drawHistogram(histograms);

  // Display original
  cv::imshow("Original Image", src);
  cv::imshow("Original Histogram", histImage);

  // Equalize each channel
  std::vector<cv::Mat> equalizedChannels = equalizeChannels(bgrChannels);

  // Calculate and draw equalized histogram
  std::vector<cv::Mat> histogramsEq = calculateHistograms(equalizedChannels);
  cv::Mat histImageEq = drawHistogram(histogramsEq);

  // Merge equalized channels back
  cv::Mat equalizedImage;
  cv::merge(equalizedChannels, equalizedImage);

  // Display equalized
  cv::imshow("Equalized Image", equalizedImage);
  cv::imshow("Equalized Histogram", histImageEq);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return 0;
}
