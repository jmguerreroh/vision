/**
 * @file main.cpp
 * @brief Background subtraction with the MOG2 Gaussian-mixture model
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - cv::createBackgroundSubtractorMOG2(): a LEARNED model of the background
 * - The foreground mask and its shadow detection (gray = shadow)
 * - Cleaning the mask with morphology (Chapter 9) and boxing the moving
 *   objects with findContours (Chapter 6)
 *
 * Difference with 12_01 (frame differencing): differencing compares each
 * frame against the PREVIOUS one, so an object that stops moving disappears
 * instantly. MOG2 instead models every pixel as a mixture of Gaussians
 * learned over MANY frames: it tolerates gradual lighting changes, adapts
 * to permanent scene changes, and can flag shadows separately.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>  // createBackgroundSubtractorMOG2
#include <iostream>
#include <string>
#include <vector>

namespace Config
{
constexpr int HISTORY = 500;             // Frames used to learn the background
constexpr double VAR_THRESHOLD = 16.0;   // Mahalanobis^2 to declare foreground
constexpr bool DETECT_SHADOWS = true;    // Mark shadows as gray (127)
constexpr double MIN_BLOB_AREA = 300.0;  // Ignore smaller foreground blobs
}

int main(int argc, char ** argv)
{
  // Command-line parser (same pattern as 12_01)
  const std::string keys =
    "{help h | | Show this help message}"
    "{@video | ../../data/vtest.avi | Input video file}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  const std::string filename =
    cv::samples::findFile(parser.get<std::string>("@video"), false);

  cv::VideoCapture cap(filename);
  if (!cap.isOpened()) {
    std::cerr << "Error: Cannot open video: " << filename << std::endl;
    return EXIT_FAILURE;
  }

  // ========================================
  // Create the background model
  // ========================================
  // Each pixel is described by up to 5 Gaussians in color space. Gaussians
  // that keep matching the incoming values gain weight and become
  // "background"; a pixel far (in Mahalanobis distance) from all background
  // Gaussians is declared foreground.
  //   history:      how fast the model forgets (larger = slower adaptation)
  //   varThreshold: squared distance to accept a pixel as background
  //   detectShadows: shadows darken the background without changing its
  //                  chromaticity; MOG2 uses that to label them 127
  cv::Ptr<cv::BackgroundSubtractorMOG2> subtractor =
    cv::createBackgroundSubtractorMOG2(Config::HISTORY, Config::VAR_THRESHOLD,
                                       Config::DETECT_SHADOWS);

  std::cout << "=== Background Subtraction (MOG2) ===" << std::endl;
  std::cout << "Video: " << filename << std::endl;
  std::cout << "Press 'q' or ESC to exit, SPACE to pause" << std::endl;

  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::Mat frame, foreground_mask;
  bool paused = false;

  while (true) {
    if (!paused) {
      cap >> frame;
      if (frame.empty()) {
        // Loop the video so the demo can run indefinitely
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        continue;
      }

      // apply() does two things at once: classifies every pixel of this
      // frame (output mask) and updates the background model with it.
      // The default learning rate (-1) decays as 1/history.
      subtractor->apply(frame, foreground_mask);

      // The raw mask has 3 values: 0 background, 127 shadow, 255 foreground.
      // Keep only confident foreground (drop the shadows) before cleanup
      cv::Mat moving;
      cv::threshold(foreground_mask, moving, 200, 255, cv::THRESH_BINARY);

      // Morphological opening (Chapter 9): remove isolated noise pixels,
      // then a closing to fill small holes inside the silhouettes
      cv::morphologyEx(moving, moving, cv::MORPH_OPEN, kernel);
      cv::morphologyEx(moving, moving, cv::MORPH_CLOSE, kernel);

      // Box each moving object (contours, Chapter 6)
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(moving.clone(), contours, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_SIMPLE);

      cv::Mat detections = frame.clone();
      int objects = 0;
      for (const auto & contour : contours) {
        if (cv::contourArea(contour) < Config::MIN_BLOB_AREA) {
          continue;  // Too small: residual noise
        }
        cv::rectangle(detections, cv::boundingRect(contour),
                      cv::Scalar(0, 255, 0), 2);
        ++objects;
      }

      cv::putText(detections, "Moving objects: " + std::to_string(objects),
                  cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(0, 255, 0), 2);

      // The learned background itself can be inspected -- useful to check
      // whether stopped objects are being absorbed into the model
      cv::Mat background;
      subtractor->getBackgroundImage(background);

      cv::imshow("1. Frame + detections", detections);
      cv::imshow("2. Raw MOG2 mask (gray = shadow)", foreground_mask);
      cv::imshow("3. Cleaned foreground", moving);
      if (!background.empty()) {
        cv::imshow("4. Learned background model", background);
      }
    }

    const int key = cv::waitKey(30);
    if (key == 'q' || key == 27) {
      break;
    } else if (key == ' ') {
      paused = !paused;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
