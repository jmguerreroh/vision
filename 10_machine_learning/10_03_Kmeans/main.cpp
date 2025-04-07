/**
 * KMeans demo sample
 * @author José Miguel Guerrero
 *
 * Reference: https://github.com/opencv/opencv/blob/master/samples/cpp/kmeans.cpp
 */

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

int main()
{
  const int MAX_CLUSTERS = 5;

  // Define colors for clusters
  cv::Scalar colorTab[] = {
    cv::Scalar(0, 0, 255),    // Red
    cv::Scalar(0, 255, 0),    // Green
    cv::Scalar(255, 100, 100),
    cv::Scalar(255, 0, 255),  // Magenta
    cv::Scalar(0, 255, 255)   // Yellow
  };

  cv::Mat img(500, 500, CV_8UC3);
  cv::RNG rng(12345); // Random number generator

  bool finish = false;
  while (!finish) {
    int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
    int i, sampleCount = rng.uniform(1, 1001);
    cv::Mat points(sampleCount, 1, CV_32FC2), labels;

    clusterCount = std::min(clusterCount, sampleCount);
    std::vector<cv::Point2f> centers;

    // Generate random sample points from a multi-Gaussian distribution
    for (k = 0; k < clusterCount; k++) {
      cv::Point center;
      center.x = rng.uniform(0, img.cols);
      center.y = rng.uniform(0, img.rows);
      cv::Mat pointChunk = points.rowRange(
        k * sampleCount / clusterCount,
        k == clusterCount - 1 ? sampleCount :
        (k + 1) * sampleCount / clusterCount);
      rng.fill(
        pointChunk, cv::RNG::NORMAL, cv::Scalar(center.x, center.y),
        cv::Scalar(img.cols * 0.05, img.rows * 0.05));
    }

    // Shuffle the sample points randomly
    cv::randShuffle(points, 1, &rng);

    // Apply Kmeans clustering algorithm
    double compactness = cv::kmeans(
      points, clusterCount, labels,
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
      3, cv::KMEANS_PP_CENTERS, centers);

    // Clear the image
    img = cv::Scalar::all(0);

    // Draw sample points with cluster colors
    for (i = 0; i < sampleCount; i++) {
      int clusterIdx = labels.at<int>(i);
      cv::Point ipt = points.at<cv::Point2f>(i);
      cv::circle(img, ipt, 2, colorTab[clusterIdx], cv::FILLED, cv::LINE_AA);
    }

    // Draw cluster centers
    for (i = 0; i < static_cast<int>(centers.size()); ++i) {
      cv::Point2f c = centers[i];
      cv::circle(img, c, 40, colorTab[i], 1, cv::LINE_AA);
    }

    std::cout << "Compactness: " << compactness << std::endl;

    // Display the clustered image
    cv::imshow("clusters", img);

    // Wait for user input, exit on 'ESC' or 'q'
    char key = static_cast<char>(cv::waitKey());
    if (key == 27 || key == 'q' || key == 'Q') {
      finish = true;
    }
  }

  return 0;
}
