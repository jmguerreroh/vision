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
constexpr int MAX_CLUSTERS = 5;  // Upper bound for the random number of clusters (K)

// One BGR color per cluster index (up to MAX_CLUSTERS).
// Used to paint both the individual points and their cluster center circle.
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

  //----------------------------------------------------------------------------
  // Main loop: each iteration generates a fresh random dataset and clusters it
  //----------------------------------------------------------------------------
  bool finish = false;
  while (!finish) {
    // Pick a random K in [2, MAX_CLUSTERS] and a random number of points
    int cluster_count = rng.uniform(2, Config::MAX_CLUSTERS + 1);
    int sample_count = rng.uniform(1, 1001);

    // points: N×1 matrix of 2-channel float (x, y) — the format expected by
    // cv::kmeans for 2D data. labels will hold the assigned cluster index per point.
    cv::Mat points(sample_count, 1, CV_32FC2), labels;

    // K cannot exceed the number of samples (degenerate case)
    cluster_count = std::min(cluster_count, sample_count);
    std::vector<cv::Point2f> centers;  // Will be filled by cv::kmeans with final centroids

    // --- Generate synthetic Gaussian clusters ---
    // Each of the K clusters is seeded at a random canvas position.
    // Points for cluster k are drawn from a 2D normal distribution centered
    // there with std-dev = 5% of canvas size, so clusters are compact blobs.
    // The points are split evenly across clusters using row ranges.
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

    // Shuffle the points so the initial order carries no cluster information;
    // this ensures the algorithm cannot exploit any ordering artifact
    cv::randShuffle(points, 1, &rng);

    // ----------------------------------------------------------------------------
    // cv::kmeans(data, K, labels, criteria, attempts, flags, centers)
    //
    // K-Means partitions N samples into K clusters by alternating two steps:
    //   E-step: assign each sample to its nearest centroid (Euclidean distance).
    //   M-step: recompute each centroid as the mean of all assigned samples.
    // The algorithm minimises the within-cluster sum of squared distances
    // (inertia), which equals the compactness value returned below.
    //
    // Parameters used:
    //   data    : N×1 CV_32FC2 matrix — each element is a 2D point (x, y)
    //   K       : cluster_count — number of clusters to find
    //   labels  : output N×1 CV_32S matrix — cluster index for every sample
    //   criteria: stop when iterations >= 10 OR centroid shift < 1.0 px
    //   attempts: run 3 times with different random initializations and keep
    //             the result with the lowest compactness (avoids local minima)
    //   flags   : KMEANS_PP_CENTERS uses the k-means++ seeding strategy, which
    //             spreads initial centers probabilistically to speed convergence
    //             and improve solution quality over purely random seeding
    //   centers : output vector of K final centroid positions (Point2f)
    // ----------------------------------------------------------------------------
    double compactness = cv::kmeans(
      points,         // Input data: N×1 matrix of 2D points (CV_32FC2)
      cluster_count,  // K: number of clusters to partition data into
      labels,         // Output: cluster index for each input sample
      cv::TermCriteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
        10,           // Maximum number of iterations
        1.0           // Desired accuracy (epsilon): stop if centroid shift < 1 px
      ),
      3,              // Number of attempts with different initial centers
      cv::KMEANS_PP_CENTERS,  // Use kmeans++ center initialization
      centers         // Output: final cluster centers
    );
    // compactness = sum over all points of squared distance to their assigned
    // centroid. Lower = tighter, better-separated clusters.

    img = cv::Scalar::all(0);  // Clear canvas to black

    // --- Draw individual points colored by their assigned cluster ---
    // labels.at<int>(i) gives the cluster index (0..K-1) for point i.
    // Each point is drawn as a small filled circle (radius 2).
    for (int i = 0; i < sample_count; i++) {
      int cluster_idx = labels.at<int>(i);
      cv::Point ipt = points.at<cv::Point2f>(i);
      cv::circle(img, ipt, 2, Config::COLOR_TAB[cluster_idx], cv::FILLED, cv::LINE_AA);
    }

    // --- Draw cluster centroids as large hollow circles ---
    // The centroid is the geometric mean of all points in the cluster.
    // A large ring (radius 40) makes centers easy to spot visually.
    for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
      cv::Point2f c = centers[i];
      cv::circle(img, c, 40, Config::COLOR_TAB[i], 1, cv::LINE_AA);
    }

    // Print compactness as a quality metric: lower values mean more compact
    // (tighter) clusters; useful to compare different K values or runs
    std::cout << "Compactness: " << compactness << std::endl;

    cv::imshow("clusters", img);

    char key = static_cast<char>(cv::waitKey(0));
    if (key == 27 || key == 'q' || key == 'Q') {
      finish = true;
    }
  }

  return EXIT_SUCCESS;
}
