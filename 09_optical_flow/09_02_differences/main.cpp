/**
 * Optical flow using accumulated frame differences
 * @author José Miguel Guerrero
 *
 * This program computes optical flow using an accumulated difference method.
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char ** argv)
{
  // Define command line arguments with default values
  const std::string keys =
    "{ h help |      | print this help message }"
    "{ @video | vtest.avi | path to image file }"
    "{ @frames | 4 | number of frames to accumulate }";

  // Parse command-line arguments
  cv::CommandLineParser parser(argc, argv, keys);

  // Retrieve video file path and number of frames to accumulate
  std::string filename = cv::samples::findFile(parser.get<std::string>("@video"));
  int num_frames = parser.get<int>("@frames");

  // Check if arguments are valid
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  // Open the video file
  cv::VideoCapture cap(filename);
  if (!cap.isOpened()) {
    // Error handling if the video file cannot be opened
    std::cerr << "Error opening video!" << std::endl;
    return 1;
  }

  cv::Mat frame, gray, prev_gray, diff_acc;
  bool first_frame = true;

  while (true) {
    std::vector<cv::Mat> frames;

    // Capture multiple frames
    for (int k = 0; k < num_frames; ++k) {
      cap >> frame; // Read the next frame
      if (frame.empty()) {break;} // Exit loop if no more frames

      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); // Convert frame to grayscale
      frames.push_back(gray.clone()); // Store grayscale frame
    }

    // If the number of captured frames is less than required, stop processing
    if (frames.size() < num_frames) {break;}

    // Initialize the accumulated difference matrix with zeros
    diff_acc = cv::Mat::zeros(frames[0].size(), CV_32F);

    // Compute accumulated differences between frames
    for (int k = 1; k < frames.size(); ++k) {
      // Compute absolute difference between frames
      cv::Mat diff;
      cv::absdiff(frames[0], frames[k], diff);
      // Convert to floating point
      diff.convertTo(diff, CV_32F);
      // Compute alpha value
      float alpha = k / (num_frames - 1.0f);
      // Accumulate differences with alpha weight
      diff_acc += alpha * diff;
    }

    // Normalize the accumulated differences for visualization
    cv::Mat diff_acc_norm;
    cv::normalize(diff_acc, diff_acc_norm, 0, 255, cv::NORM_MINMAX);
    diff_acc_norm.convertTo(diff_acc_norm, CV_8U);

    // Display the accumulated optical flow
    cv::imshow("Accumulated Optical Flow", diff_acc_norm);

    // Wait for a key press, exit if 'q' or 'Esc' is pressed
    int key = cv::waitKey(30);
    if (key == 'q' || key == 27) {break;}
  }

  // Release the video capture and destroy all windows
  cap.release();
  cv::destroyAllWindows();
  return 0;
}
