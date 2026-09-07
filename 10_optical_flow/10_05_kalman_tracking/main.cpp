/**
 * @file main.cpp
 * @brief 2D tracking with a Kalman filter (interactive mouse demo)
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Setting up cv::KalmanFilter with a constant-velocity motion model
 * - The predict / correct cycle that fuses model and measurement
 * - How the filter smooths a noisy measurement and bridges dropouts
 *
 * Move the mouse over the window: the cursor position (corrupted with
 * synthetic noise) is the MEASUREMENT; the filter maintains an estimate of
 * position AND velocity. Press 'h' to "hide" the target: with no
 * measurements the filter keeps predicting along the last velocity --
 * exactly what a tracker does when its object is briefly occluded.
 *
 * State (4D):       x = [px, py, vx, vy]
 * Measurement (2D): z = [px, py]           (we only observe the position)
 *
 * Transition model (constant velocity, time step dt folded into the matrix):
 *   px' = px + vx        F = [1 0 1 0]
 *   py' = py + vy            [0 1 0 1]
 *   vx' = vx                 [0 0 1 0]
 *   vy' = vy                 [0 0 0 1]
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>  // cv::KalmanFilter
#include <iostream>
#include <vector>

namespace Config
{
constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;
constexpr double MEASUREMENT_NOISE_PX = 8.0;  // Synthetic noise added to mouse
constexpr int TRAIL_LENGTH = 100;             // Points of trajectory drawn
}

/**
 * @brief Application state shared with the mouse callback
 */
struct KalmanApp
{
  cv::Point2f mouse_position{Config::WIDTH / 2.0f, Config::HEIGHT / 2.0f};
  bool mouse_seen = false;  // Becomes true after the first mouse event
};

// Global app state (required by the fixed OpenCV callback signature)
KalmanApp app;

static void onMouse(int /*event*/, int x, int y, int /*flags*/, void *)
{
  app.mouse_position = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
  app.mouse_seen = true;
}

/**
 * @brief Draws the last points of a trajectory as a fading polyline
 */
void drawTrail(
  cv::Mat & canvas, const std::vector<cv::Point> & trail, const cv::Scalar & color)
{
  for (size_t i = 1; i < trail.size(); ++i) {
    cv::line(canvas, trail[i - 1], trail[i], color, 2);
  }
}

int main(int argc, char ** argv)
{
  // Command-line arguments; --help prints the usage. This example builds
  // its own data, so it takes no input file
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  // ========================================
  // Configure the Kalman filter
  // ========================================
  // KalmanFilter(stateDims, measurementDims, controlDims)
  cv::KalmanFilter kalman(4, 2, 0);

  // F: transition matrix of the constant-velocity model (header diagram)
  kalman.transitionMatrix = (cv::Mat_<float>(4, 4) <<
    1, 0, 1, 0,
    0, 1, 0, 1,
    0, 0, 1, 0,
    0, 0, 0, 1);

  // H: measurement matrix -- we observe only the position components
  kalman.measurementMatrix = (cv::Mat_<float>(2, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0);

  // Q (process noise): how much we DISTRUST the constant-velocity model.
  // Larger Q = the filter follows measurements more nervously.
  cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1e-3));

  // R (measurement noise): how much we distrust the sensor. We set it to
  // the variance of the synthetic noise we inject, the "honest" value.
  cv::setIdentity(kalman.measurementNoiseCov,
                  cv::Scalar::all(Config::MEASUREMENT_NOISE_PX *
                                  Config::MEASUREMENT_NOISE_PX));

  // P: initial uncertainty of the state (large: we know nothing yet)
  cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1.0));

  // Initial state: center of the window, zero velocity
  kalman.statePost = (cv::Mat_<float>(4, 1) <<
    Config::WIDTH / 2.0f, Config::HEIGHT / 2.0f, 0.0f, 0.0f);

  // ========================================
  // Interactive loop
  // ========================================
  const std::string window_name = "Kalman Tracking";
  cv::namedWindow(window_name);
  cv::setMouseCallback(window_name, onMouse);

  std::cout << "=== Kalman Filter Tracking ===" << std::endl;
  std::cout << "Move the mouse over the window." << std::endl;
  std::cout << "  h : hold 'hidden' mode (predict without measurements)" << std::endl;
  std::cout << "  c : clear trails" << std::endl;
  std::cout << "  q/ESC : quit" << std::endl;

  cv::RNG rng(12345);  // Noise generator (fixed seed: reproducible runs)
  std::vector<cv::Point> measured_trail, filtered_trail;
  bool hidden = false;

  while (true) {
    // --- 1. PREDICT: advance the state with the motion model only ---
    // This is the filter's belief BEFORE looking at the sensor
    const cv::Mat prediction = kalman.predict();
    const cv::Point predicted_point(cvRound(prediction.at<float>(0)),
                                    cvRound(prediction.at<float>(1)));

    cv::Point filtered_point = predicted_point;
    cv::Point measured_point;

    if (!hidden && app.mouse_seen) {
      // --- 2. MEASURE: the mouse position + synthetic Gaussian noise ---
      // (a real tracker would measure with a detector; noise is inherent)
      measured_point = cv::Point(
        cvRound(app.mouse_position.x +
                rng.gaussian(Config::MEASUREMENT_NOISE_PX)),
        cvRound(app.mouse_position.y +
                rng.gaussian(Config::MEASUREMENT_NOISE_PX)));

      // --- 3. CORRECT: fuse prediction and measurement ---
      // The Kalman gain weighs them by their covariances (Q, R, P): a noisy
      // sensor pulls the estimate less than a precise one would
      const cv::Mat measurement = (cv::Mat_<float>(2, 1) <<
        static_cast<float>(measured_point.x),
        static_cast<float>(measured_point.y));
      const cv::Mat corrected = kalman.correct(measurement);

      filtered_point = cv::Point(cvRound(corrected.at<float>(0)),
                                 cvRound(corrected.at<float>(1)));

      measured_trail.push_back(measured_point);
      if (measured_trail.size() > Config::TRAIL_LENGTH) {
        measured_trail.erase(measured_trail.begin());
      }
    }
    // When hidden, we deliberately skip correct(): the state keeps evolving
    // with the last estimated velocity and its uncertainty (P) grows

    filtered_trail.push_back(filtered_point);
    if (filtered_trail.size() > Config::TRAIL_LENGTH) {
      filtered_trail.erase(filtered_trail.begin());
    }

    // --- Visualization ---
    cv::Mat canvas = cv::Mat::zeros(Config::HEIGHT, Config::WIDTH, CV_8UC3);
    drawTrail(canvas, measured_trail, cv::Scalar(0, 0, 255));    // Red: noisy
    drawTrail(canvas, filtered_trail, cv::Scalar(0, 255, 0));    // Green: filtered
    if (!hidden && app.mouse_seen) {
      cv::circle(canvas, measured_point, 4, cv::Scalar(0, 0, 255), cv::FILLED);
    }
    cv::circle(canvas, filtered_point, 6, cv::Scalar(0, 255, 0), 2);

    cv::putText(canvas,
                hidden ? "HIDDEN: predicting without measurements ('h' to release)"
                       : "Red: noisy measurement | Green: Kalman estimate",
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1);

    cv::imshow(window_name, canvas);

    const int key = cv::waitKey(20);
    if (key == 'q' || key == 27) {
      break;
    } else if (key == 'h') {
      hidden = !hidden;
      std::cout << (hidden ? "Measurements suppressed (pure prediction)"
                           : "Measurements restored") << std::endl;
    } else if (key == 'c') {
      measured_trail.clear();
      filtered_trail.clear();
    }
  }

  return EXIT_SUCCESS;
}
