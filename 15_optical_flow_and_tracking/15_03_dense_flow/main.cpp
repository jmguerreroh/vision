/**
 * @file main.cpp
 * @brief Dense optical flow using Gunnar Farneback's algorithm
 * @author José Miguel Guerrero Hernández
 *
 * @details Computes optical flow for every pixel in the frame using Farneback's
 *          polynomial expansion method. Unlike Lucas-Kanade (sparse), this
 *          approach produces a complete motion field.
 *
 *          Visualization:
 *          - Flow direction → Hue channel (color wheel)
 *          - Flow magnitude → Value channel (brightness)
 *          - Saturation is set to 1.0 (fully saturated)
 *
 * @see https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

namespace
{
// The video of the chapter is 1920x1080, and a window that size does not fit on
// a normal screen. The processing always runs at full resolution: only the copy
// sent to the screen is reduced, with INTER_AREA, the interpolation meant for
// shrinking. On a smaller video this does nothing
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

int main(int argc, char ** argv)
{
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/853889-hd_1920_1080_25fps.mp4 | Input file}"
    "{scale s | 0.5 | Factor applied to every frame BEFORE computing the flow. "
    "Farneback costs 392 ms per frame at 1920x1080 and 132 ms at half that, "
    "against the 40 ms of a 25 fps video. Use 1.0 for full resolution}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  const std::string filename =
    cv::samples::findFile(parser.get<std::string>("@input"), false);
  const double scale = parser.get<double>("scale");
  if (scale <= 0.0 || scale > 1.0) {
    std::cerr << "--scale must be greater than 0 and at most 1" << std::endl;
    return EXIT_FAILURE;
  }

  // Open video file
  cv::VideoCapture capture(filename);
  if (!capture.isOpened()) {
    std::cerr << "Unable to open file!" << std::endl;
    return EXIT_FAILURE;
  }

  // Read first frame and convert to grayscale
  // Unlike the on-screen reduction above, this one does change the result: the
  // flow is computed on the reduced frame, so its vectors are measured in the
  // pixels of that frame. It is the same thing the book does to draw the
  // figures of this chapter, and the reason is cost, measured here: at
  // 1920x1080 Farneback needs 392 ms per frame, ten times the 40 ms that a
  // 25 fps video leaves. Pass --scale=1.0 to see the difference
  cv::Mat frame1, prvs;
  capture >> frame1;
  if (scale != 1.0) {
    cv::resize(frame1, frame1, cv::Size(), scale, scale, cv::INTER_AREA);
  }
  cv::cvtColor(frame1, prvs, cv::COLOR_BGR2GRAY);
  std::cout << "Flow computed at " << frame1.cols << "x" << frame1.rows
            << " (--scale=" << scale << ")" << std::endl;

  // Main processing loop
  while (true) {
    cv::Mat frame2, next;

    // Capture the next frame
    capture >> frame2;
    if (frame2.empty()) {
      break;
    }
    if (scale != 1.0) {
      cv::resize(frame2, frame2, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    // Convert to grayscale
    cv::cvtColor(frame2, next, cv::COLOR_BGR2GRAY);

    // Compute dense optical flow using Farneback's algorithm
    cv::Mat flow(prvs.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(
      prvs,   // Previous grayscale frame
      next,   // Current grayscale frame
      flow,   // Output flow image (2-channel: u, v components)
      0.5,    // Pyramid scale (0.5 = classical pyramid, each level halves the resolution)
      3,      // Number of pyramid levels (more levels capture larger motions)
      15,     // Window size for averaging (larger = smoother but less precise)
      3,      // Number of iterations at each pyramid level
      5,      // Pixel neighborhood size for polynomial expansion
      1.2,    // Standard deviation of Gaussian for polynomial expansion smoothing
      0       // Flags (0 = default behavior)
    );

    // Split the optical flow into x and y components
    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
    // 'angle' is in degrees [0, 360], but Hue in 8-bit OpenCV lives in
    // [0, 180]. The odd-looking factor pre-divides so that the final
    // convertTo(..., 255) below lands exactly on H = angle / 2:
    //   angle * (1/360) * (180/255) * 255  =  angle / 2
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    // Create an HSV image representation of optical flow
    // H (Hue):        flow direction → color indicates movement direction
    // S (Saturation): fixed at 1.0  → fully saturated colors
    // V (Value):      flow magnitude → brightness indicates movement speed
    cv::Mat hsv_channels[3], hsv_image, hsv8, bgr;
    hsv_channels[0] = angle;                                // H: direction of motion
    hsv_channels[1] = cv::Mat::ones(angle.size(), CV_32F);  // S: max saturation
    hsv_channels[2] = magn_norm;                            // V: magnitude (bright = fast)
    cv::merge(hsv_channels, 3, hsv_image);
    hsv_image.convertTo(hsv8, CV_8U, 255.0);
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

    // Display the optical flow visualization
    showFit("frame2", bgr);

    // Wait for user input to continue or exit
    int keyboard = cv::waitKey(30);
    if (keyboard == 'q' || keyboard == 27) {
      break;
    }

    // Update previous frame for next iteration
    prvs = next;
  }

  return EXIT_SUCCESS;
}
