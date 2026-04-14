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
constexpr int MAX_CLASSES = 2;          // Number of available classes (class 0 and class 1)
constexpr int TEST_STEP = 5;            // Pixel sampling step for prediction: classifies 1 out of every
                                        // 5 pixels in X and Y to reduce computational cost
constexpr int IMG_WIDTH = 640;          // Width of the demo canvas
constexpr int IMG_HEIGHT = 480;         // Height of the demo canvas
const cv::Scalar WHITE_COLOR = cv::Scalar(255, 255, 255);
const std::string WINDOW_NAME = "points";
}

/**
 * @brief Application state for KNN classification
 *
 * Holds all mutable application state in a single structure.
 * Declared as a global because OpenCV mouse callbacks do not allow passing
 * custom state — the function signature is fixed by the API.
 */
struct KNNApp
{
  cv::Mat img;                                                // Display image
  cv::Mat img_dst;                                            // Classification output
  std::vector<cv::Point> trained_points;                      // Training point positions
  std::vector<int> trained_points_markers;                    // Training point class labels
  std::vector<cv::Vec3b> class_colors{Config::MAX_CLASSES};   // Colors per class
  int current_class = 0;                                      // Active class for new points
  std::vector<int> class_counters{Config::MAX_CLASSES, 0};    // Points per class counter: every class has at least one training point before running the classifier
};

// Global app state (required for OpenCV callbacks)
KNNApp app;

/**
 * @brief Mouse callback: records training points on the canvas.
 *
 * OpenCV calls this function automatically for every mouse event on the
 * registered window. The signature (event, x, y, flags, userdata) is
 * imposed by the OpenCV API and cannot be changed.
 *
 * Flow on left-click:
 *   1. The position (x, y) is saved to trained_points.
 *   2. The active class is saved to trained_points_markers (same index).
 *   3. The counter for the active class is incremented.
 *   4. The entire image is redrawn: cleared to black, then every point
 *      is repainted as a filled circle with its class color.
 *
 * @param event  Mouse event type (cv::EVENT_LBUTTONUP, etc.)
 * @param x      X coordinate of the cursor (pixel column)
 * @param y      Y coordinate of the cursor (pixel row)
 */
static void onMouse(int event, int x, int y, int /*flags*/, void *)
{
  if (app.img.empty()) {
    return;
  }

  bool update_flag = false;

  // Only act on left-button release (LBUTTONUP) to avoid registering
  // multiple points during a drag gesture
  if (event == cv::EVENT_LBUTTONUP) {
    app.trained_points.push_back(cv::Point(x, y));           // 2D coordinates of the point
    app.trained_points_markers.push_back(app.current_class); // Class label
    app.class_counters[app.current_class]++;
    update_flag = true;
  }

  // Full redraw: clear and repaint all accumulated points so that
  // each point always shows the correct color for its class
  if (update_flag) {
    app.img = cv::Scalar::all(0);  // Clear to black

    for (std::size_t i = 0; i < app.trained_points.size(); i++) {
      cv::Vec3b c = app.class_colors[app.trained_points_markers[i]];
      cv::circle(app.img, app.trained_points[i], 5, cv::Scalar(c), -1);  // Radius 5, filled
    }

    cv::imshow(Config::WINDOW_NAME, app.img);
  }
}

/**
 * @brief Trains a KNN classifier and paints the decision-region map.
 *
 * KNN (K-Nearest Neighbors) is a non-parametric, lazy classifier:
 * it builds no explicit model during training — it simply memorizes
 * the examples. Classification of a new point is resolved by finding
 * its K nearest neighbors in feature space and assigning the majority
 * class among them (plurality voting).
 *
 * Main steps:
 *   1. Flatten the training data into a 2D matrix.
 *   2. Create and train the model.
 *   3. Classify every pixel (subsampled) of the canvas to paint
 *      the decision boundaries.
 *
 * @param K  Number of nearest neighbors to consider in the vote.
 *           Small values (K=3) yield more irregular/local boundaries;
 *           large values (K=15) smooth the boundaries at the cost of detail.
 */
static void runKNN(int K)
{
  // ------------------------------------------------------------------
  // 1. Create the KNN classifier
  //
  // cv::ml::KNearest uses an internal KD-tree for neighbor search.
  // ------------------------------------------------------------------
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setDefaultK(K);           // Number of neighbors queried per prediction
  knn->setIsClassifier(true);    // Classification mode (as opposed to regression)

  // ------------------------------------------------------------------
  // 2. Build the training sample matrix
  //
  // trained_points is a std::vector<cv::Point>, where each cv::Point holds
  // two ints (x, y). Wrapping it in cv::Mat produces a matrix of shape N×1
  // with type CV_32SC2 (2 channels, 32-bit int).
  //
  // reshape(1, N) converts it to N×2 with 1 channel: each row becomes one
  // sample [x, y] — this is the "flattening" required by the OpenCV ml API,
  // where each ROW is a sample and each COLUMN is a feature.
  //
  // convertTo(..., CV_32F) casts the data to 32-bit float, which is the type
  // required by cv::ml::KNearest::train().
  // ------------------------------------------------------------------
  cv::Mat samples;
  cv::Mat(app.trained_points).reshape(1,
    static_cast<int>(app.trained_points.size())).convertTo(samples, CV_32F);
  //        ^                ^
  //        |                +-- number of rows = number of points
  //        +------------------- channels=1 → from CV_32SC2 to single-channel matrix

  // cv::ml::TrainData bundles samples and labels into an object consumed by
  // the ml module. ROW_SAMPLE states that each row of 'samples' is one sample.
  cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(
        samples, cv::ml::ROW_SAMPLE, cv::Mat(app.trained_points_markers));

  // ------------------------------------------------------------------
  // 3. Train the model
  //
  // For KNN, "training" only stores the samples in the internal structure
  // (KD-tree). There are no parameters to optimize.
  // ------------------------------------------------------------------
  knn->train(train_data);

  // ------------------------------------------------------------------
  // 4. Dense prediction: classify every pixel of the canvas
  //
  // The image is scanned with a stride of TEST_STEP pixels to reduce the
  // number of predict() calls (classifying every single pixel would be slow).
  // Each predict() call returns the majority class among the K nearest
  // neighbors of the query point (x, y).
  // ------------------------------------------------------------------
  cv::Mat test_sample(1, 2, CV_32FC1);  // Reusable row vector [x, y]
  for (int y = 0; y < app.img.rows; y += Config::TEST_STEP) {
    for (int x = 0; x < app.img.cols; x += Config::TEST_STEP) {
      // Load the current pixel coordinates as the test sample
      test_sample.at<float>(0) = static_cast<float>(x);  // feature 0: column
      test_sample.at<float>(1) = static_cast<float>(y);  // feature 1: row

      // predict() runs the K-neighbor search and returns the winning class
      int response = static_cast<int>(knn->predict(test_sample));

      // Paint the pixel with the predicted class color to visualize
      // the decision region (the "zone of influence" of each class)
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

    if (key == 27) { // Exit on 'Esc' key
      finish = true;
    }

    if (key == 'i') { // Reset data
      app.img = cv::Scalar::all(0);
      app.trained_points.clear();
      app.trained_points_markers.clear();
      app.class_counters.assign(Config::MAX_CLASSES, 0);
      cv::imshow(Config::WINDOW_NAME, app.img);
    }

    if (key == '0' || key == '1') { // Switch class
      app.current_class = key - '0';
    }

    if (key == 'r') { // Run KNN classification
      // Verify that every class has at least 1 training point;
      // if any class is empty, KNN cannot classify correctly
      double minVal = 0;
      cv::minMaxLoc(app.class_counters, &minVal, 0, 0, 0);
      if (minVal == 0) {
        std::cout << "Each class should have at least 1 point" << std::endl;
        continue;
      }

      // Copy training points as the background of the decision map so that
      // the colored circles remain visible on top of the decision regions
      app.img.copyTo(app.img_dst);

      // Run KNN with K=3: more detailed, local decision boundaries
      // (sensitive to noise / isolated points)
      runKNN(3);
      cv::imshow("kNN 3", app.img_dst);   // Display the decision map for K=3

      // Re-copy the background before the second run, since runKNN writes
      // directly into img_dst
      app.img.copyTo(app.img_dst);

      // Run KNN with K=15: smoother boundaries, more robust to noise
      runKNN(15);
      cv::imshow("kNN 15", app.img_dst);  // Display the decision map for K=15
    }
  }

  return EXIT_SUCCESS;
}
