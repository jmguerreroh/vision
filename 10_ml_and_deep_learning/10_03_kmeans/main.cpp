/**
 * @file main.cpp
 * @brief K-Means clustering demo using OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * @details Generates random 2D point clusters and applies K-Means clustering
 *          to partition them. Each iteration uses a random number of clusters
 *          (2-5) and random sample count (1-1000).
 *
 *          Visualization:
 *          - Each cluster is drawn in a different color
 *          - Cluster centers are shown as large circles
 *          - Press any key to generate a new random clustering
 *          - Press ESC or 'q' to exit
 *
 * @see https://github.com/opencv/opencv/blob/master/samples/cpp/kmeans.cpp
 */

#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

// Configuration constants
namespace Config
{
constexpr int MAX_CLUSTERS = 5;
const cv::Scalar COLOR_TAB[] = {
  cv::Scalar(0, 0, 255),
  cv::Scalar(0, 255, 0),
  cv::Scalar(255, 100, 100),
  cv::Scalar(255, 0, 255),
  cv::Scalar(0, 255, 255)
};
}

int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;

  std::cout << "K-Means Clustering Demo\n"
            << "  Press any key : generate new random clustering\n"
            << "  Press ESC / q : exit\n" << std::endl;

  cv::Mat img(500, 500, CV_8UC3);
  cv::RNG rng(12345);

  //-------------------------------------------------------------------------
  // Main loop
  //-------------------------------------------------------------------------
  bool finish = false;
  while (!finish) {
    int cluster_count = rng.uniform(2, Config::MAX_CLUSTERS + 1);
    int sample_count = rng.uniform(1, 1001);
    cv::Mat points(sample_count, 1, CV_32FC2), labels;

    cluster_count = std::min(cluster_count, sample_count);
    std::vector<cv::Point2f> centers;

    for (int k = 0; k < cluster_count; k++) {
      cv::Point center;
      center.x = rng.uniform(0, img.cols);
      center.y = rng.uniform(0, img.rows);
      cv::Mat point_chunk = points.rowRange(
                k * sample_count / cluster_count,
                k == cluster_count - 1 ? sample_count : (k + 1) * sample_count / cluster_count);
      rng.fill(point_chunk, cv::RNG::NORMAL, cv::Scalar(center.x, center.y),
                     cv::Scalar(img.cols * 0.05, img.rows * 0.05));
    }

    cv::randShuffle(points, 1, &rng);

    // cv::kmeans(data, K, labels, criteria, attempts, flags, centers)
    double compactness = cv::kmeans(
      points,         // Input data: N×1 matrix of 2D points (CV_32FC2)
      cluster_count,  // K: number of clusters to partition data into
      labels,         // Output: cluster index for each input sample
      cv::TermCriteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
        10,           // Maximum number of iterations
        1.0           // Desired accuracy (epsilon)
      ),
      3,              // Number of attempts with different initial centers
      cv::KMEANS_PP_CENTERS,  // Use kmeans++ center initialization
      centers         // Output: final cluster centers
    );

    img = cv::Scalar::all(0);

    for (int i = 0; i < sample_count; i++) {
      int cluster_idx = labels.at<int>(i);
      cv::Point ipt = points.at<cv::Point2f>(i);
      cv::circle(img, ipt, 2, Config::COLOR_TAB[cluster_idx], cv::FILLED, cv::LINE_AA);
    }

    for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
      cv::Point2f c = centers[i];
      cv::circle(img, c, 40, Config::COLOR_TAB[i], 1, cv::LINE_AA);
    }

    std::cout << "Compactness: " << compactness << std::endl;

    cv::imshow("clusters", img);

    char key = static_cast<char>(cv::waitKey(0));
    if (key == 27 || key == 'q' || key == 'Q') {
      finish = true;
    }
  }

  return EXIT_SUCCESS;
}
