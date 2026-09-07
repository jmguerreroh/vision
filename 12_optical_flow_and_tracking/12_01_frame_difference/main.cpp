/**
 * @file main.cpp
 * @brief Motion detection using accumulated frame differences
 * @author José Miguel Guerrero Hernández
 *
 * @details Detects motion by computing weighted differences between consecutive
 *          frames. More recent frames have higher weight, creating a temporal
 *          decay effect that highlights current movement.
 *
 *          Algorithm:
 *          1. Capture N consecutive frames
 *          2. For each frame k, compute |frame[k] - frame[0]|
 *          3. Weight each difference by k/(N-1) (recent = more weight)
 *          4. Sum all weighted differences
 *          5. Normalize and display as heatmap
 *
 *          Controls:
 *            q/ESC: Exit
 *            +/-: Increase/decrease number of frames
 *            c: Toggle color heatmap
 *            Space: Pause/resume
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
}  // namespace

/**
 * @brief Apply colormap to grayscale motion image
 * @param gray Input grayscale motion image
 * @return Colored heatmap visualization
 */
cv::Mat applyHeatmap(const cv::Mat & gray)
{
  cv::Mat colored;
  cv::applyColorMap(gray, colored, cv::COLORMAP_JET);
  return colored;
}

/**
 * @brief Draw info overlay on image
 * @param img Image to draw on (modified in place)
 * @param numFrames Number of frames being accumulated
 * @param useColor Whether color mode is enabled
 * @param fps Current frames per second
 * @param x0 Left edge to write from, so the text lands on the right half
 */
void drawInfo(cv::Mat & img, int num_frames, bool use_color, double fps, int x0)
{
  std::string info = "Frames: " + std::to_string(num_frames) +
    " | Color: " + (use_color ? "ON" : "OFF") +
    " | FPS: " + std::to_string(static_cast<int>(fps));
  cv::putText(img, info, cv::Point(x0 + 10, 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
  cv::putText(img, "+/-: frames | c: color | Space: pause | q: quit",
              cv::Point(x0 + 10, img.rows - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
}

int main(int argc, char ** argv)
{
  // Command-line parser
  const std::string keys =
    "{help h | | Show this help message}"
    "{@video | ../../data/853889-hd_1920_1080_25fps.mp4 | Input video file}"
    "{frames f | 4 | Number of frames to accumulate (2-20)}";

  cv::CommandLineParser parser(argc, argv, keys);

  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  std::string filename = cv::samples::findFile(parser.get<std::string>("@video"), false);
  int num_frames = parser.get<int>("frames");
  num_frames = std::max(2, std::min(20, num_frames));  // Clamp to valid range

  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  // Open video
  cv::VideoCapture cap(filename);
  if (!cap.isOpened()) {
    std::cerr << "Error: Cannot open video: " << filename << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Motion Detection (Frame Differences) ===" << std::endl;
  std::cout << "Video: " << filename << std::endl;
  std::cout << "Accumulating " << num_frames << " frames" << std::endl;

  bool use_color = true;
  bool paused = false;
  double fps = 0.0;
  int64 tick_start = cv::getTickCount();
  int frame_count = 0;

  while (true) {
    if (!paused) {
      std::vector<cv::Mat> frames;

      // Capture N frames
      for (int k = 0; k < num_frames; ++k) {
        cv::Mat frame, gray;
        cap >> frame;

        if (frame.empty()) {
          // Loop video
          cap.set(cv::CAP_PROP_POS_FRAMES, 0);
          cap >> frame;
          if (frame.empty()) {
            break;
          }
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        frames.push_back(gray);
      }

      if (static_cast<int>(frames.size()) < num_frames) {
        break;
      }

      // Compute weighted accumulated differences
      // Weight increases with frame distance: w[k] = k / (N-1)
      // This emphasizes recent motion over older motion
      cv::Mat diff_acc = cv::Mat::zeros(frames[0].size(), CV_32F);

      for (int k = 1; k < num_frames; ++k) {
        cv::Mat diff;
        cv::absdiff(frames[0], frames[k], diff);
        diff.convertTo(diff, CV_32F);

        // Weight: more recent frames contribute more
        float weight = static_cast<float>(k) / (num_frames - 1);
        diff_acc += weight * diff;
      }

      // Normalize for visualization
      cv::Mat diff_norm;
      cv::normalize(diff_acc, diff_norm, 0, 255, cv::NORM_MINMAX);
      diff_norm.convertTo(diff_norm, CV_8U);

      // Apply colormap if enabled
      cv::Mat display = use_color ? applyHeatmap(diff_norm) : diff_norm;
      if (!use_color) {
        cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
      }

      // Calculate FPS
      frame_count++;
      double elapsed = (cv::getTickCount() - tick_start) / cv::getTickFrequency();
      if (elapsed >= 1.0) {
        fps = frame_count / elapsed;
        frame_count = 0;
        tick_start = cv::getTickCount();
      }

      // Show original frame alongside motion
      cv::Mat original;
      cv::cvtColor(frames.back(), original, cv::COLOR_GRAY2BGR);

      cv::Mat combined;
      cv::hconcat(original, display, combined);

      // The labels are written on the reduced copy and not on the frame. Drawn
      // before the reduction they shrink with it: on this 1920x1080 video that
      // is a factor of 0.42, and a 0.6 font ends up under 6 px tall
      cv::Mat view = fitToScreen(combined);
      cv::putText(view, "Current Frame", cv::Point(10, 25),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
      drawInfo(view, num_frames, use_color, fps, view.cols / 2);
      cv::imshow("Motion Detection", view);
    }

    // Handle keyboard input
    int key = cv::waitKey(paused ? 0 : 30);
    if (key == 'q' || key == 27) {
      break;
    } else if (key == '+' || key == '=') {
      num_frames = std::min(20, num_frames + 1);
      std::cout << "Frames: " << num_frames << std::endl;
    } else if (key == '-' || key == '_') {
      num_frames = std::max(2, num_frames - 1);
      std::cout << "Frames: " << num_frames << std::endl;
    } else if (key == 'c') {
      use_color = !use_color;
      std::cout << "Color: " << (use_color ? "ON" : "OFF") << std::endl;
    } else if (key == ' ') {
      paused = !paused;
      std::cout << (paused ? "Paused" : "Resumed") << std::endl;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
