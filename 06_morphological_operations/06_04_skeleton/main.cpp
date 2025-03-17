/**
 * Skeleton - sample code: based on Zhang-Suen thinning algorithm
 * @author José Miguel Guerrero
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

int countNeighbors(const cv::Mat & image, int x, int y)
{
  // Count the number of white (255) neighbors in the 3x3 window
  int count = 0;
  int dx[] = {-1, -1, 0, 1, 1, 1, 0, -1};
  int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

  for (int i = 0; i < 8; i++) {
    if (image.at<uchar>(x + dx[i], y + dy[i]) == 255) {
      count++;
    }
  }
  return count;
}

int countTransitions(const cv::Mat & image, int x, int y)
{
  // Count the number of 0 to 1 transitions in a circular scan of neighbors
  int dx[] = {-1, -1, 0, 1, 1, 1, 0, -1, -1};
  int dy[] = {0, 1, 1, 1, 0, -1, -1, -1, 0};
  int transitions = 0;

  for (int i = 0; i < 8; i++) {
    if (image.at<uchar>(x + dx[i], y + dy[i]) == 0 &&
      image.at<uchar>(x + dx[i + 1], y + dy[i + 1]) == 255)
    {
      transitions++;
    }
  }
  return transitions;
}

void zhangSuenThinning(cv::Mat & image)
{
  // Perform Zhang-Suen thinning on the input binary image
  cv::Mat temp = image.clone();
  bool changed;

  do {
    changed = false;
    std::vector<cv::Point> toRemove;

    // Step 1
    for (int x = 1; x < image.rows - 1; x++) {
      for (int y = 1; y < image.cols - 1; y++) {
        if (image.at<uchar>(x, y) == 255) {
          int neighbors = countNeighbors(image, x, y);
          int transitions = countTransitions(image, x, y);

          if (neighbors >= 2 && neighbors <= 6 && transitions == 1 &&
            (image.at<uchar>(x - 1, y) == 0 || image.at<uchar>(x,
            y + 1) == 0 || image.at<uchar>(x + 1, y) == 0) &&
            (image.at<uchar>(x, y + 1) == 0 || image.at<uchar>(x + 1, y) == 0 || image.at<uchar>(x,
            y - 1) == 0))
          {
            toRemove.push_back(cv::Point(y, x));
          }
        }
      }
    }
    // Remove the marked pixels
    for (auto p : toRemove) {
      temp.at<uchar>(p.y, p.x) = 0;
    }

    // Show iteration step
    cv::imshow("Thinning Process", temp);
    cv::waitKey(10);     // Pause for 10ms to visualize the changes

    // Step 2
    toRemove.clear();
    for (int x = 1; x < image.rows - 1; x++) {
      for (int y = 1; y < image.cols - 1; y++) {
        if (temp.at<uchar>(x, y) == 255) {
          int neighbors = countNeighbors(temp, x, y);
          int transitions = countTransitions(temp, x, y);

          if (neighbors >= 2 && neighbors <= 6 && transitions == 1 &&
            (temp.at<uchar>(x - 1, y) == 0 || temp.at<uchar>(x, y + 1) == 0 || temp.at<uchar>(x,
            y - 1) == 0) &&
            (temp.at<uchar>(x - 1, y) == 0 || temp.at<uchar>(x + 1, y) == 0 || temp.at<uchar>(x,
            y - 1) == 0))
          {
            toRemove.push_back(cv::Point(y, x));
          }
        }
      }
    }
    // Remove the marked pixels
    for (auto p : toRemove) {
      temp.at<uchar>(p.y, p.x) = 0;
    }

    // Show iteration step
    cv::imshow("Thinning Process", temp);
    cv::waitKey(10);     // Pause for 10ms to visualize the changes

    // Check if any pixel was removed
    changed = !toRemove.empty();
    temp.copyTo(image);

  } while (changed);
}

cv::Mat overlaySkeletonOnOriginal(const cv::Mat & original, const cv::Mat & skeleton)
{
  // Convert original grayscale image to BGR (color)
  cv::Mat coloredOriginal;
  cv::cvtColor(original, coloredOriginal, cv::COLOR_GRAY2BGR);

  // Overlay the skeleton in red
  for (int x = 0; x < skeleton.rows; x++) {
    for (int y = 0; y < skeleton.cols; y++) {
      if (skeleton.at<uchar>(x, y) == 255) {
        coloredOriginal.at<cv::Vec3b>(x, y) = cv::Vec3b(0, 0, 255);         // Red color
      }
    }
  }
  return coloredOriginal;
}

int main()
{
  // Load the input binary image
  cv::Mat original = cv::imread("../../data/star.jpg", cv::IMREAD_GRAYSCALE);
  if (original.empty()) {
    std::cout << "Error: Unable to load the image." << std::endl;
    return -1;
  }

  // Convert to binary image
  cv::Mat image;
  cv::threshold(original, image, 127, 255, cv::THRESH_BINARY);

  // Display original image
  cv::imshow("Original Image", image);
  cv::waitKey(10);   // Pause for 10ms to visualize the changes

  // Apply Zhang-Suen thinning
  zhangSuenThinning(image);

  // Overlay skeleton on original image
  cv::Mat finalOverlay = overlaySkeletonOnOriginal(original, image);

  // Display the final skeleton over the original image
  cv::imshow("Final Skeleton Overlay", finalOverlay);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}
