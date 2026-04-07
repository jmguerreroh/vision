/**
 * @file main.cpp
 * @brief Point classification demo comparing multiple ML classifiers
 * @author José Miguel Guerrero Hernández
 *
 * @details Interactive demo comparing 8 different ML classifiers on the same
 *          2D point dataset. The user adds training points with the mouse,
 *          then presses 'r' to visualize decision boundaries for all models.
 *
 *          Classifiers: Normal Bayes, KNN (K=3, K=15), SVM (C=1, C=10),
 *                       Decision Trees, Boosting, Random Forest, ANN-MLP, EM
 *
 *          Controls:
 *            '0'/'1':     Switch to class #0 or #1
 *            Left click:  Add a new training point
 *            'r':         Run all classifiers
 *            'i':         Clear all data
 *            ESC:         Exit
 *
 * @see https://github.com/opencv/opencv/blob/master/samples/cpp/points_classifier.cpp
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <iostream>
#include <cfloat>

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
 * @brief Application state for ML classification
 */
struct MLApp
{
  cv::Mat img;                                                // Display image
  cv::Mat img_dst;                                            // Classification output
  cv::RNG rng;                                                // Random number generator
  std::vector<cv::Point> trained_points;                      // Training point positions
  std::vector<int> trained_points_markers;                    // Training point class labels
  std::vector<cv::Vec3b> class_colors{Config::MAX_CLASSES};   // Colors per class
  int current_class = 0;                                      // Active class for new points
  std::vector<int> class_counters{Config::MAX_CLASSES, 0};    // Points per class counter
};

// Global app state (required for OpenCV callbacks)
MLApp app;

// Classifier flags
#define USE_NBC 1
#define USE_KNN 1
#define USE_SVM 1
#define USE_DT  1
#define USE_BT  1
#define USE_RF  1
#define USE_ANN 1
#define USE_EM  1

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

  if (event == cv::EVENT_LBUTTONUP) {
    app.trained_points.push_back(cv::Point(x, y));
    app.trained_points_markers.push_back(app.current_class);
    app.class_counters[app.current_class]++;
    update_flag = true;
  }

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
 * @brief Convert point vector to training sample matrix
 * @param pts Vector of 2D training points
 * @return CV_32F matrix suitable for ML training (N×2)
 */
static cv::Mat prepareTrainSamples(const std::vector<cv::Point> & pts)
{
  cv::Mat samples;
  cv::Mat(pts).reshape(1, static_cast<int>(pts.size())).convertTo(samples, CV_32F);
  return samples;
}

/**
 * @brief Create TrainData object from current training points
 * @return Pointer to TrainData ready for model training
 */
static cv::Ptr<cv::ml::TrainData> prepareTrainData()
{
  cv::Mat samples = prepareTrainSamples(app.trained_points);
  return cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE,
    cv::Mat(app.trained_points_markers));
}

/**
 * @brief Classify every pixel and paint the decision region colors
 * @param model Trained ML model to use for prediction
 * @param dst Output image where decision regions are painted
 */
static void predictAndPaint(const cv::Ptr<cv::ml::StatModel> & model, cv::Mat & dst)
{
  cv::Mat test_sample(1, 2, CV_32FC1);
  for (int y = 0; y < app.img.rows; y += Config::TEST_STEP) {
    for (int x = 0; x < app.img.cols; x += Config::TEST_STEP) {
      test_sample.at<float>(0) = static_cast<float>(x);
      test_sample.at<float>(1) = static_cast<float>(y);
      int response = static_cast<int>(model->predict(test_sample));
      dst.at<cv::Vec3b>(y, x) = app.class_colors[response];
    }
  }
}

// Classifier implementations
#if USE_NBC
/**
 * @brief Train a Normal Bayes classifier and paint its decision regions
 */
static void findDecisionBoundaryNBC()
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::NormalBayesClassifier> normal_bayes_classifier =
    cv::ml::StatModel::train<cv::ml::NormalBayesClassifier>(prepareTrainData());
  predictAndPaint(normal_bayes_classifier, app.img_dst);
}
#endif

#if USE_KNN
/**
 * @brief Create and train a KNN model with given K value
 * @param K Number of nearest neighbors to consider
 */
static void findDecisionBoundaryKNN(int K)
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setDefaultK(K);
  knn->setIsClassifier(true);
  knn->train(prepareTrainData());
  predictAndPaint(knn, app.img_dst);
}
#endif

#if USE_SVM
/**
 * @brief Create and train an SVM model with given C value
 * @param C Regularization parameter for SVM
 */
static void findDecisionBoundarySVM(double C)
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::POLY);
  svm->setDegree(0.5);
  svm->setGamma(1);
  svm->setCoef0(1);
  svm->setNu(0.5);
  svm->setP(0);
  svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000,
    0.01));
  svm->setC(C);
  svm->train(prepareTrainData());
  predictAndPaint(svm, app.img_dst);

  cv::Mat sv = svm->getSupportVectors();
  for (int i = 0; i < sv.rows; i++) {
    const float * support_vector = sv.ptr<float>(i);
    cv::circle(app.img_dst,
                   cv::Point(cv::saturate_cast<int>(support_vector[0]),
      cv::saturate_cast<int>(support_vector[1])),
                   5, Config::WHITE_COLOR, -1);
  }
}
#endif

#if USE_DT
/**
 * @brief Train a Decision Tree classifier and paint its decision regions
 */
static void findDecisionBoundaryDT()
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
  dtree->setMaxDepth(8);
  dtree->setMinSampleCount(2);
  dtree->setUseSurrogates(false);
  dtree->setCVFolds(0);
  dtree->setUse1SERule(false);
  dtree->setTruncatePrunedTree(false);
  dtree->train(prepareTrainData());
  predictAndPaint(dtree, app.img_dst);
}
#endif

#if USE_BT
/**
 * @brief Train a Boosting classifier and paint its decision regions
 */
static void findDecisionBoundaryBT()
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();
  boost->setBoostType(cv::ml::Boost::DISCRETE);
  boost->setWeakCount(100);
  boost->setWeightTrimRate(0.95);
  boost->setMaxDepth(2);
  boost->setUseSurrogates(false);
  boost->setPriors(cv::Mat());
  boost->train(prepareTrainData());
  predictAndPaint(boost, app.img_dst);
}
#endif

#if USE_RF
/**
 * @brief Train a Random Forest classifier and paint its decision regions
 */
static void findDecisionBoundaryRF()
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::RTrees> rtrees = cv::ml::RTrees::create();
  rtrees->setMaxDepth(4);
  rtrees->setMinSampleCount(2);
  rtrees->setRegressionAccuracy(0.f);
  rtrees->setUseSurrogates(false);
  rtrees->setMaxCategories(16);
  rtrees->setPriors(cv::Mat());
  rtrees->setCalculateVarImportance(false);
  rtrees->setActiveVarCount(1);
  rtrees->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 5, 0));
  rtrees->train(prepareTrainData());
  predictAndPaint(rtrees, app.img_dst);
}
#endif

#if USE_ANN
/**
 * @brief Create and train an ANN model with given layer sizes
 * @param layer_sizes Matrix containing the number of neurons in each layer
 */
static void findDecisionBoundaryANN(const cv::Mat & layer_sizes)
{
  app.img.copyTo(app.img_dst);
  cv::Mat train_classes = cv::Mat::zeros(static_cast<int>(app.trained_points.size()),
                                          static_cast<int>(app.class_colors.size()), CV_32FC1);
  for (int i = 0; i < train_classes.rows; i++) {
    train_classes.at<float>(i, app.trained_points_markers[i]) = 1.f;
  }

  cv::Mat samples = prepareTrainSamples(app.trained_points);
  cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE,
    train_classes);

  cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
  ann->setLayerSizes(layer_sizes);
  ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
  ann->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300,
    FLT_EPSILON));
  ann->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001);
  ann->train(tdata);
  predictAndPaint(ann, app.img_dst);
}
#endif

#if USE_EM
/**
 * @brief Train per-class EM (Gaussian Mixture) models and paint decision regions
 *        based on maximum log-likelihood
 */
static void findDecisionBoundaryEM()
{
  app.img.copyTo(app.img_dst);

  cv::Mat samples = prepareTrainSamples(app.trained_points);

  int nmodels = static_cast<int>(app.class_colors.size());
  std::vector<cv::Ptr<cv::ml::EM>> em_models(nmodels);
  cv::Mat model_samples;

  for (int i = 0; i < nmodels; i++) {
    const int component_count = 3;

    model_samples.release();
    for (int j = 0; j < samples.rows; j++) {
      if (app.trained_points_markers[j] == i) {
        model_samples.push_back(samples.row(j));
      }
    }

    if (!model_samples.empty()) {
      cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
      em->setClustersNumber(component_count);
      em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
      em->trainEM(model_samples, cv::noArray(), cv::noArray(), cv::noArray());
      em_models[i] = em;
    }
  }

  cv::Mat test_sample(1, 2, CV_32FC1);
  cv::Mat log_likelihoods(1, nmodels, CV_64FC1, cv::Scalar(-DBL_MAX));

  for (int y = 0; y < app.img.rows; y += Config::TEST_STEP) {
    for (int x = 0; x < app.img.cols; x += Config::TEST_STEP) {
      test_sample.at<float>(0) = static_cast<float>(x);
      test_sample.at<float>(1) = static_cast<float>(y);

      for (int i = 0; i < nmodels; i++) {
        if (!em_models[i].empty()) {
          log_likelihoods.at<double>(i) = em_models[i]->predict2(test_sample, cv::noArray())[0];
        }
      }
      cv::Point max_loc;
      cv::minMaxLoc(log_likelihoods, 0, 0, 0, &max_loc);
      app.img_dst.at<cv::Vec3b>(y, x) = app.class_colors[max_loc.x];
    }
  }
}
#endif

int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;

  std::cout   << "Use:" << std::endl
              << "  key '0' .. '1' - switch to class #n" << std::endl
              << "  left mouse button - to add new point;" << std::endl
              << "  key 'r' - to run the ML model;" << std::endl
              << "  key 'i' - to init (clear) the data." << std::endl << std::endl
              << "  ESC - to quit the demo." << std::endl;

  //-------------------------------------------------------------------------
  // Initialize image and UI elements
  //-------------------------------------------------------------------------
  cv::namedWindow(Config::WINDOW_NAME, 1);
  app.img.create(Config::IMG_HEIGHT, Config::IMG_WIDTH, CV_8UC3);
  app.img_dst.create(Config::IMG_HEIGHT, Config::IMG_WIDTH, CV_8UC3);

  cv::imshow(Config::WINDOW_NAME, app.img);
  cv::setMouseCallback(Config::WINDOW_NAME, onMouse);

  app.class_colors[0] = cv::Vec3b(0, 255, 0);
  app.class_colors[1] = cv::Vec3b(0, 0, 255);

  //-------------------------------------------------------------------------
  // Main loop
  //-------------------------------------------------------------------------
  bool finish = false;
  while (!finish) {
    char key = static_cast<char>(cv::waitKey(0));

    if (key == 27) {
      finish = true;
    }

    if (key == 'i') {
      app.img = cv::Scalar::all(0);
      app.trained_points.clear();
      app.trained_points_markers.clear();
      app.class_counters.assign(Config::MAX_CLASSES, 0);
      cv::imshow(Config::WINDOW_NAME, app.img);
    }

    if (key == '0' || key == '1') {
      app.current_class = key - '0';
    }

    if (key == 'r') {
      double minVal = 0;
      cv::minMaxLoc(app.class_counters, &minVal, 0, 0, 0);
      if (minVal == 0) {
        std::cout << "each class should have at least 1 point" << std::endl;
        continue;
      }
#if USE_NBC
      findDecisionBoundaryNBC();
      cv::imshow("NormalBayesClassifier", app.img_dst);
#endif

#if USE_KNN
      findDecisionBoundaryKNN(3);
      cv::imshow("kNN3", app.img_dst);

      findDecisionBoundaryKNN(15);
      cv::imshow("kNN15", app.img_dst);
#endif

#if USE_SVM
      findDecisionBoundarySVM(1);
      cv::imshow("classificationSVM1", app.img_dst);

      findDecisionBoundarySVM(10);
      cv::imshow("classificationSVM10", app.img_dst);
#endif

#if USE_DT
      findDecisionBoundaryDT();
      cv::imshow("DT", app.img_dst);
#endif

#if USE_BT
      findDecisionBoundaryBT();
      cv::imshow("BT", app.img_dst);
#endif

#if USE_RF
      findDecisionBoundaryRF();
      cv::imshow("RF", app.img_dst);
#endif

#if USE_ANN
      cv::Mat layer_sizes_1(1, 3, CV_32SC1);
      layer_sizes_1.at<int>(0) = 2;
      layer_sizes_1.at<int>(1) = 5;
      layer_sizes_1.at<int>(2) = static_cast<int>(app.class_colors.size());
      findDecisionBoundaryANN(layer_sizes_1);
      cv::imshow("ANN", app.img_dst);
#endif

#if USE_EM
      findDecisionBoundaryEM();
      cv::imshow("EM", app.img_dst);
#endif
    }
  }

  return EXIT_SUCCESS;
}
