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

  const std::string keys =
    "{ h help |      | print this help message }"
    "{ @video | vtest.avi | path to image file }"
    "{ @frames | 4 | number of frames to accumulate }";
  cv::CommandLineParser parser(argc, argv, keys);

  std::string filename = cv::samples::findFile(parser.get<std::string>("@video"));
  int num_frames = parser.get<int>("@frames");

  std::cout << "Using " << num_frames << " frames" << std::endl;

  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }


  cv::VideoCapture cap(filename);
  if (!cap.isOpened()) {
    std::cerr << "Error opening video!" << std::endl;
    return 1;
  }

  cv::Mat frame, gray, prev_gray, diff_acc;
  bool first_frame = true;

  while (true) {
    std::vector<cv::Mat> frames;

    // Capture multiple frames
    for (int k = 0; k < num_frames; ++k) {
      cap >> frame;
      if (frame.empty()) {break;}

      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      frames.push_back(gray.clone());
    }

    if (frames.size() < num_frames) {break;}

    // Compute accumulated differences
    diff_acc = cv::Mat::zeros(frames[0].size(), CV_32F);
    for (int k = 1; k < frames.size(); ++k) {
      cv::Mat diff;
      cv::absdiff(frames[0], frames[k], diff);
      diff.convertTo(diff, CV_32F);
      diff_acc += diff;
    }

    // Normalize for visualization
    cv::Mat diff_acc_norm;
    cv::normalize(diff_acc, diff_acc_norm, 0, 255, cv::NORM_MINMAX);
    diff_acc_norm.convertTo(diff_acc_norm, CV_8U);

    // Show result
    cv::imshow("Accumulated Optical Flow", diff_acc_norm);

    int key = cv::waitKey(30);
    if (key == 'q' || key == 27) {break;}
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
