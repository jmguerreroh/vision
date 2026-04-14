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
constexpr int MAX_CLASSES = 2;          // Number of classes (class 0 and class 1)
constexpr int TEST_STEP = 5;            // Pixel stride for dense prediction (1 of every 5 pixels
                                        // is classified to reduce computation)
constexpr int IMG_WIDTH = 640;          // Canvas width in pixels
constexpr int IMG_HEIGHT = 480;         // Canvas height in pixels
const cv::Scalar WHITE_COLOR = cv::Scalar(255, 255, 255);  // Used to highlight SVM support vectors
const std::string WINDOW_NAME = "points";
}

/**
 * @brief Application state for ML classification
 *
 * All mutable state lives here. Declared global because OpenCV mouse
 * callbacks have a fixed signature and cannot receive custom state otherwise.
 */
struct MLApp
{
  cv::Mat img;       // Canvas where training points are drawn
  cv::Mat img_dst;   // Destination image painted with the decision regions
  cv::RNG rng;       // Random number generator (unused here, available for extensions)

  // Training data: 2D pixel positions of each point added by the user
  std::vector<cv::Point> trained_points;

  // Class label (0 or 1) for each point in trained_points (parallel array)
  std::vector<int> trained_points_markers;

  // BGR color per class for drawing points and decision regions
  std::vector<cv::Vec3b> class_colors{Config::MAX_CLASSES};

  // Currently active class (assigned to the next mouse click)
  int current_class = 0;

  // Per-class point count; checked before training to ensure at least 1 point per class
  std::vector<int> class_counters{Config::MAX_CLASSES, 0};
};

// Global app state (required for OpenCV callbacks)
MLApp app;

// Preprocessor flags: set to 0 to disable a specific classifier
#define USE_NBC 1   // Normal Bayes Classifier
#define USE_KNN 1   // K-Nearest Neighbors (K=3 and K=15)
#define USE_SVM 1   // Support Vector Machine (C=1 and C=10)
#define USE_DT  1   // Decision Tree
#define USE_BT  1   // AdaBoost (Discrete Boosting)
#define USE_RF  1   // Random Forest
#define USE_ANN 1   // Artificial Neural Network (MLP with Backprop)
#define USE_EM  1   // Expectation-Maximization (Gaussian Mixture Model)

/**
 * @brief Mouse callback: records a training point on left-click.
 *
 * Stores the clicked pixel (x, y) together with the currently active class
 * label, then redraws all accumulated points as colored filled circles.
 *
 * @param event  OpenCV mouse event type
 * @param x      Cursor column (pixel x coordinate)
 * @param y      Cursor row    (pixel y coordinate)
 */
static void onMouse(int event, int x, int y, int /*flags*/, void *)
{
  if (app.img.empty()) {
    return;
  }

  bool update_flag = false;

  // Act only on button-release to avoid duplicate points during a drag
  if (event == cv::EVENT_LBUTTONUP) {
    app.trained_points.push_back(cv::Point(x, y));
    app.trained_points_markers.push_back(app.current_class);
    app.class_counters[app.current_class]++;
    update_flag = true;
  }

  if (update_flag) {
    app.img = cv::Scalar::all(0);  // Clear to black
    for (std::size_t i = 0; i < app.trained_points.size(); i++) {
      cv::Vec3b c = app.class_colors[app.trained_points_markers[i]];
      cv::circle(app.img, app.trained_points[i], 5, cv::Scalar(c), -1);  // Filled circle, radius 5
    }
    cv::imshow(Config::WINDOW_NAME, app.img);
  }
}

/**
 * @brief Flatten the point vector into an N×2 CV_32F matrix for ML training.
 *
 * std::vector<cv::Point> stores pairs of int (x, y). Wrapping it in cv::Mat
 * gives an N×1 matrix with 2-channel ints (CV_32SC2). reshape(1, N) converts
 * it to N×2 single-channel, and convertTo makes it CV_32F — the format all
 * OpenCV ML classifiers require for their sample matrices.
 *
 * @param pts  Vector of 2D training points
 * @return     N×2 CV_32F matrix where each row is one [x, y] sample
 */
static cv::Mat prepareTrainSamples(const std::vector<cv::Point> & pts)
{
  cv::Mat samples;
  cv::Mat(pts).reshape(1, static_cast<int>(pts.size())).convertTo(samples, CV_32F);
  return samples;
}

/**
 * @brief Bundle samples and labels into a cv::ml::TrainData object.
 *
 * cv::ml::TrainData is the unified input format consumed by all OpenCV ML
 * models. ROW_SAMPLE indicates that each row in the sample matrix is one
 * training example.
 *
 * @return  Pointer to TrainData ready to pass to model->train()
 */
static cv::Ptr<cv::ml::TrainData> prepareTrainData()
{
  cv::Mat samples = prepareTrainSamples(app.trained_points);
  return cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE,
    cv::Mat(app.trained_points_markers));
}

/**
 * @brief Run dense pixel-wise prediction and paint the decision regions.
 *
 * Iterates every TEST_STEP-th pixel, builds a 1×2 float row vector [x, y],
 * calls model->predict() (part of the cv::ml::StatModel interface shared by
 * all classifiers), and paints the pixel with the color of the returned class.
 *
 * Subsampling by TEST_STEP avoids calling predict() for every pixel, which
 * would be prohibitively slow for models like EM or ANN.
 *
 * @param model  Any trained cv::ml::StatModel (KNN, SVM, DT, …)
 * @param dst    Image to paint; must be CV_8UC3, same size as app.img
 */
static void predictAndPaint(const cv::Ptr<cv::ml::StatModel> & model, cv::Mat & dst)
{
  cv::Mat test_sample(1, 2, CV_32FC1);  // Reusable 1×2 query vector
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
 * @brief Normal Bayes Classifier decision boundary.
 *
 * Assumes each class follows a multivariate Gaussian distribution and
 * classifies by comparing the posterior probabilities:
 *   P(class | x) ∝ P(x | class) * P(class)
 * The decision boundary is a conic section (ellipse, parabola, or hyperbola)
 * depending on the covariance matrices of the two classes. Very fast to train
 * and predict; performs well when the Gaussian assumption holds.
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
 * @brief KNN decision boundary for a given K.
 *
 * Classifies each query point by majority vote among its K nearest training
 * neighbors (Euclidean distance). No explicit model is built; all training
 * samples are stored internally. Small K -> fine-grained, noisy boundaries;
 * large K -> smoother, more generalized boundaries.
 *
 * @param K  Number of nearest neighbors to consult per prediction
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
 * @brief SVM decision boundary for a given regularization constant C.
 *
 * Uses a polynomial kernel: K(x,y) = (gamma * x·y + coef0)^degree,
 * here (1 * x·y + 1)^0.5 — a square-root dot-product kernel that can
 * model non-linear boundaries without mapping to a very high-dimensional space.
 *
 * C controls the soft-margin trade-off:
 *   Small C (1)  -> wider margin, more misclassifications tolerated.
 *   Large C (10) -> narrower margin, fewer training errors (risk of overfitting).
 *
 * After training, support vectors (the critical boundary samples) are drawn
 * as white filled circles to make them visible.
 *
 * @param C  SVM regularization parameter
 */
static void findDecisionBoundarySVM(double C)
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);        // Soft-margin C-SVC
  svm->setKernel(cv::ml::SVM::POLY);       // Polynomial kernel
  svm->setDegree(0.5);                     // Exponent of the polynomial
  svm->setGamma(1);                        // Coefficient in the kernel formula
  svm->setCoef0(1);                        // Independent term in the kernel formula
  svm->setNu(0.5);                         // Not used for C_SVC, set for completeness
  svm->setP(0);                            // Not used for C_SVC
  svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000,
    0.01));
  svm->setC(C);
  svm->train(prepareTrainData());
  predictAndPaint(svm, app.img_dst);

  // Overlay support vectors as white filled circles
  // getSupportVectors() returns the compressed (alpha-weighted) vectors;
  // each row is one support vector [x, y] in feature space
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
 * @brief Decision Tree decision boundary.
 *
 * Recursively partitions the feature space with axis-aligned splits that
 * maximise class purity (Gini or entropy). Key parameters:
 *   maxDepth=8         : limits tree height to prevent overfitting.
 *   minSampleCount=2   : a node is not split if it has fewer than 2 samples.
 *   CVFolds=0          : disables cross-validation pruning (faster).
 *   use1SERule=false   : keep the tree that minimizes training error
 *                        rather than the simpler tree within 1 std-error.
 * Decision boundaries are piecewise constant and always axis-aligned
 * ("staircase" shape), which is characteristic of single decision trees.
 */
static void findDecisionBoundaryDT()
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
  dtree->setMaxDepth(8);              // Tree will not grow deeper than 8 levels
  dtree->setMinSampleCount(2);        // Minimum samples required to split a node
  dtree->setUseSurrogates(false);     // No surrogate splits (used for missing data)
  dtree->setCVFolds(0);               // No cross-validation pruning
  dtree->setUse1SERule(false);        // Keep the most accurate tree, not the simplest
  dtree->setTruncatePrunedTree(false);
  dtree->train(prepareTrainData());
  predictAndPaint(dtree, app.img_dst);
}
#endif

#if USE_BT
/**
 * @brief AdaBoost (Discrete Boosting) decision boundary.
 *
 * Boosting trains an ensemble of weak classifiers (shallow decision trees,
 * maxDepth=2) sequentially. Each new learner focuses on the samples that
 * previous learners misclassified by increasing their weights. The final
 * prediction is the weighted majority vote of all weak classifiers.
 *
 * Parameters:
 *   boostType=DISCRETE  : uses AdaBoost.M1 (binary labels).
 *   weakCount=100       : ensemble size — more trees = lower bias but slower.
 *   weightTrimRate=0.95 : ignore the 5% of samples with the lowest weights
 *                         to speed up each iteration.
 *   maxDepth=2          : each weak learner is a decision stump (depth 2).
 */
static void findDecisionBoundaryBT()
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();
  boost->setBoostType(cv::ml::Boost::DISCRETE); // AdaBoost with discrete (0/1) labels
  boost->setWeakCount(100);                     // Number of weak learners in the ensemble
  boost->setWeightTrimRate(0.95);               // Drop the 5% lowest-weight samples each round
  boost->setMaxDepth(2);                        // Shallow trees to keep each learner weak
  boost->setUseSurrogates(false);
  boost->setPriors(cv::Mat());                  // Uniform class priors
  boost->train(prepareTrainData());
  predictAndPaint(boost, app.img_dst);
}
#endif

#if USE_RF
/**
 * @brief Random Forest decision boundary.
 *
 * Trains an ensemble of decision trees on bootstrap samples of the data,
 * with a random subset of features considered at each split. Aggregating
 * many uncorrelated trees reduces variance compared to a single tree.
 *
 * Parameters:
 *   maxDepth=4          : limits individual tree depth.
 *   activeVarCount=1    : at each split, only 1 randomly chosen feature
 *                         (out of 2) is considered — standard sqrt(features)
 *                         heuristic for 2D data.
 *   MAX_ITER=5          : stop after 5 trees (small for a fast demo).
 *   calculateVarImportance=false: skip the extra computation of feature
 *                         importance scores since they are not displayed.
 */
static void findDecisionBoundaryRF()
{
  app.img.copyTo(app.img_dst);
  cv::Ptr<cv::ml::RTrees> rtrees = cv::ml::RTrees::create();
  rtrees->setMaxDepth(4);                   // Maximum depth of each individual tree
  rtrees->setMinSampleCount(2);
  rtrees->setRegressionAccuracy(0.f);       // Not relevant for classification
  rtrees->setUseSurrogates(false);
  rtrees->setMaxCategories(16);
  rtrees->setPriors(cv::Mat());             // Uniform class priors
  rtrees->setCalculateVarImportance(false); // Skip feature importance computation
  rtrees->setActiveVarCount(1);             // Random features per split (sqrt heuristic)
  rtrees->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 5, 0));  // 5 trees
  rtrees->train(prepareTrainData());
  predictAndPaint(rtrees, app.img_dst);
}
#endif

#if USE_ANN
/**
 * @brief ANN-MLP (Multilayer Perceptron) decision boundary.
 *
 * A fully-connected feedforward neural network trained with backpropagation.
 * Architecture is specified by layer_sizes: e.g. [2, 5, 2] means:
 *   - Input layer:  2 neurons (x and y features)
 *   - Hidden layer: 5 neurons with symmetric sigmoid activation
 *   - Output layer: 2 neurons (one per class)
 *
 * Unlike the other classifiers, ANN_MLP requires one-hot encoded labels
 * instead of integer class indices. The target matrix train_classes is
 * therefore built as an N×C float matrix where train_classes[i][c] = 1.0
 * if sample i belongs to class c, and 0.0 otherwise.
 *
 * Parameters:
 *   SIGMOID_SYM     : f(x) = 2/(1+exp(-alpha*x)) - 1  (range [-1, 1])
 *   BACKPROP, 0.001 : gradient descent with learning rate 0.001
 *   MAX_ITER=300    : training epochs
 *
 * @param layer_sizes  1×L CV_32SC1 matrix with neuron counts per layer
 */
static void findDecisionBoundaryANN(const cv::Mat & layer_sizes)
{
  app.img.copyTo(app.img_dst);

  // Build one-hot label matrix (N x num_classes)
  // ANN_MLP cannot use integer class indices; it needs continuous targets
  // so the loss function can compute gradients via backpropagation.
  cv::Mat train_classes = cv::Mat::zeros(static_cast<int>(app.trained_points.size()),
                                          static_cast<int>(app.class_colors.size()), CV_32FC1);
  for (int i = 0; i < train_classes.rows; i++) {
    train_classes.at<float>(i, app.trained_points_markers[i]) = 1.f;  // One-hot encoding
  }

  cv::Mat samples = prepareTrainSamples(app.trained_points);
  cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE,
    train_classes);

  cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
  ann->setLayerSizes(layer_sizes);       // Network topology: [input, hidden..., output]
  ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);  // Symmetric sigmoid, alpha=beta=1
  ann->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300,
    FLT_EPSILON));                       // Stop after 300 iterations or negligible improvement
  ann->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001);  // Stochastic gradient descent, lr=0.001
  ann->train(tdata);
  predictAndPaint(ann, app.img_dst);
}
#endif

#if USE_EM
/**
 * @brief EM (Expectation-Maximization) Gaussian Mixture Model boundary.
 *
 * Unlike the classifiers above, OpenCV's EM is unsupervised: it cannot be
 * trained with class labels. The workaround is to train one independent GMM
 * per class using only that class's samples, then classify each test pixel
 * by choosing the class whose GMM assigns the highest log-likelihood.
 *
 * Each per-class GMM models the density as a mixture of Gaussian components:
 *   p(x | class) = sum_k  pi_k * N(x; mu_k, Sigma_k)
 *
 * Parameters per model:
 *   nclusters=3         : each class distribution is approximated with up to
 *                         3 Gaussian components, allowing multimodal shapes.
 *   COV_MAT_DIAGONAL    : covariance matrices are diagonal (axis-aligned
 *                         ellipses), reducing parameters and preventing
 *                         degeneracy with few training points.
 *
 * Classification rule: argmax_c [ log p(x | class_c) ]
 *   predict2() returns [logLikelihood, label] — element [0] is used here.
 */
static void findDecisionBoundaryEM()
{
  app.img.copyTo(app.img_dst);

  cv::Mat samples = prepareTrainSamples(app.trained_points);

  int nmodels = static_cast<int>(app.class_colors.size());  // One GMM per class
  std::vector<cv::Ptr<cv::ml::EM>> em_models(nmodels);
  cv::Mat model_samples;

  // Train one EM model per class using only that class's samples
  for (int i = 0; i < nmodels; i++) {
    const int component_count = 3;  // Number of Gaussian components in the mixture

    model_samples.release();
    // Collect all training points belonging to class i
    for (int j = 0; j < samples.rows; j++) {
      if (app.trained_points_markers[j] == i) {
        model_samples.push_back(samples.row(j));
      }
    }

    if (!model_samples.empty()) {
      cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
      em->setClustersNumber(component_count);  // Number of mixture components
      // Diagonal covariance: fewer parameters, more stable with small datasets
      em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
      // trainEM runs the EM algorithm until convergence; noArray() means we
      // don't need the log-likelihoods, labels, or probs of the training set
      em->trainEM(model_samples, cv::noArray(), cv::noArray(), cv::noArray());
      em_models[i] = em;
    }
  }

  // Dense prediction: assign each pixel to the class with the highest log-likelihood
  cv::Mat test_sample(1, 2, CV_32FC1);
  cv::Mat log_likelihoods(1, nmodels, CV_64FC1, cv::Scalar(-DBL_MAX));  // Init to -inf

  for (int y = 0; y < app.img.rows; y += Config::TEST_STEP) {
    for (int x = 0; x < app.img.cols; x += Config::TEST_STEP) {
      test_sample.at<float>(0) = static_cast<float>(x);
      test_sample.at<float>(1) = static_cast<float>(y);

      // Query each per-class GMM and store its log-likelihood for this pixel
      for (int i = 0; i < nmodels; i++) {
        if (!em_models[i].empty()) {
          // predict2 returns [logLikelihood, componentLabel]; [0] is the log-likelihood
          log_likelihoods.at<double>(i) = em_models[i]->predict2(test_sample, cv::noArray())[0];
        }
      }
      // Pick the class with the highest log-likelihood (MAP decision rule)
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
      // Verify every class has at least one point before training
      double minVal = 0;
      cv::minMaxLoc(app.class_counters, &minVal, 0, 0, 0);
      if (minVal == 0) {
        std::cout << "each class should have at least 1 point" << std::endl;
        continue;
      }
      // Run each enabled classifier and display its decision map in a separate window.
      // img.copyTo(img_dst) inside each function resets the destination before
      // painting new regions, so windows are independent of each other.
#if USE_NBC
      findDecisionBoundaryNBC();
      cv::imshow("NormalBayesClassifier", app.img_dst);
#endif

#if USE_KNN
      // Two K values to compare: K=3 (local/noisy) vs K=15 (smooth/global)
      findDecisionBoundaryKNN(3);
      cv::imshow("kNN3", app.img_dst);

      findDecisionBoundaryKNN(15);
      cv::imshow("kNN15", app.img_dst);
#endif

#if USE_SVM
      // Two C values to compare: C=1 (wide margin) vs C=10 (tight margin)
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
      // Topology: 2 input features -> 5 hidden neurons -> 2 output classes
      cv::Mat layer_sizes_1(1, 3, CV_32SC1);
      layer_sizes_1.at<int>(0) = 2;                                         // Input: x, y
      layer_sizes_1.at<int>(1) = 5;                                         // Hidden layer
      layer_sizes_1.at<int>(2) = static_cast<int>(app.class_colors.size()); // Output: one per class
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
