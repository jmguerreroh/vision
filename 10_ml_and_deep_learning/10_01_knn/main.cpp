/**
 * @file main.cpp
 * @brief KNN (K-Nearest Neighbors) classification demo using OpenCV ML module
 * @author José Miguel Guerrero Hernández
 *
 * @details Interactive demo where the user adds 2D training points for two classes
 *          using the mouse, then runs KNN classification to visualize the decision
 *          boundaries for K=3 and K=15.
 *
 *          Controls:
 *            '0'/'1':     Switch to class #0 or #1
 *            Left click:  Add a new training point
 *            'r':         Run KNN classifier
 *            'i':         Clear all data
 *            ESC:         Exit
 *
 * @see https://docs.opencv.org/3.4/d5/d26/tutorial_py_knn_understanding.html
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

// Configuration constants
namespace Config
{
constexpr int MAX_CLASSES = 2;
constexpr int TEST_STEP = 5;
constexpr int IMG_WIDTH = 640;
constexpr int IMG_HEIGHT = 480;
const cv::Scalar WHITE_COLOR = cv::Scalar(255, 255, 255);
const std::string WINDOW_NAME = "points";
}

/**
 * @brief Application state for KNN classification
 */
struct KNNApp
{
  cv::Mat img;                                                // Display image
  cv::Mat img_dst;                                            // Classification output
  std::vector<cv::Point> trained_points;                      // Training point positions
  std::vector<int> trained_points_markers;                    // Training point class labels
  std::vector<cv::Vec3b> class_colors{Config::MAX_CLASSES};   // Colors per class
  int current_class = 0;                                      // Active class for new points
  std::vector<int> class_counters{Config::MAX_CLASSES, 0};    // Points per class counter
};

// Global app state (required for OpenCV callbacks)
KNNApp app;

/**
 * @brief Mouse callback function to record training points
 * @param event Mouse event type
 * @param x X coordinate of mouse position
 * @param y Y coordinate of mouse position
 */
static void onMouse(int event, int x, int y, int /*flags*/, void *)
{
  if (app.img.empty()) {
    return;
  }

  bool update_flag = false;

  // Record point position with left mouse button
  if (event == cv::EVENT_LBUTTONUP) {
    app.trained_points.push_back(cv::Point(x, y));
    app.trained_points_markers.push_back(app.current_class);
    app.class_counters[app.current_class]++;
    update_flag = true;
  }

  // Redraw the image with updated points
  if (update_flag) {
    app.img = cv::Scalar::all(0);

    for (std::size_t i = 0; i < app.trained_points.size(); i++) {
      cv::Vec3b c = app.class_colors[app.trained_points_markers[i]];
      cv::circle(app.img, app.trained_points[i], 5, cv::Scalar(c), -1);
    }

    cv::imshow(Config::WINDOW_NAME, app.img);
  }
}

/**
 * @brief Create and train a KNN model with given K value
 * @param K Number of nearest neighbors to consider
 */
static void runKNN(int K)
{
  // Create KNN classifier
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setDefaultK(K);
  knn->setIsClassifier(true);

  // Prepare training data
  cv::Mat samples;
  cv::Mat(app.trained_points).reshape(1,
    static_cast<int>(app.trained_points.size())).convertTo(samples, CV_32F);
  cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(
        samples, cv::ml::ROW_SAMPLE, cv::Mat(app.trained_points_markers));

  // Train the KNN model
  knn->train(train_data);

  // Predict classes for each pixel in the image
  cv::Mat test_sample(1, 2, CV_32FC1);
  for (int y = 0; y < app.img.rows; y += Config::TEST_STEP) {
    for (int x = 0; x < app.img.cols; x += Config::TEST_STEP) {
      test_sample.at<float>(0) = static_cast<float>(x);
      test_sample.at<float>(1) = static_cast<float>(y);
      int response = static_cast<int>(knn->predict(test_sample));
      app.img_dst.at<cv::Vec3b>(y, x) = app.class_colors[response];
    }
  }
}

int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;

  std::cout   << "Use:\n"
              << "  key 'Esc' - exit the program;\n"
              << "  key '0' .. '1' - switch to class #n\n"
              << "  left mouse button - to add new point;\n"
              << "  key 'r' - to run the ML model;\n"
              << "  key 'i' - to init (clear) the data." << std::endl;

  // Create window and initialize images
  cv::namedWindow(Config::WINDOW_NAME, 1);
  app.img.create(Config::IMG_HEIGHT, Config::IMG_WIDTH, CV_8UC3);
  app.img_dst.create(Config::IMG_HEIGHT, Config::IMG_WIDTH, CV_8UC3);

  cv::imshow(Config::WINDOW_NAME, app.img);
  cv::setMouseCallback(Config::WINDOW_NAME, onMouse);

  // Define colors for the classes
  app.class_colors[0] = cv::Vec3b(0, 255, 0);
  app.class_colors[1] = cv::Vec3b(0, 0, 255);

  // Main loop for user interaction
  bool finish = false;
  while (!finish) {
    char key = static_cast<char>(cv::waitKey(0));

    if (key == 27) {     // Exit on 'Esc' key
      finish = true;
    }

    if (key == 'i') {     // Reset data
      app.img = cv::Scalar::all(0);
      app.trained_points.clear();
      app.trained_points_markers.clear();
      app.class_counters.assign(Config::MAX_CLASSES, 0);
      cv::imshow(Config::WINDOW_NAME, app.img);
    }

    if (key == '0' || key == '1') {     // Switch class
      app.current_class = key - '0';
    }

    if (key == 'r') {     // Run KNN classification
      double minVal = 0;
      cv::minMaxLoc(app.class_counters, &minVal, 0, 0, 0);
      if (minVal == 0) {
        std::cout << "Each class should have at least 1 point" << std::endl;
        continue;
      }
      app.img.copyTo(app.img_dst);

      runKNN(3);
      cv::imshow("kNN 3", app.img_dst);

      runKNN(15);
      cv::imshow("kNN 15", app.img_dst);
    }
  }

  return EXIT_SUCCESS;
}
