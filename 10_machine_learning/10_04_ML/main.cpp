/**
 * Point classification demo sample
 * @author José Miguel Guerrero
 *
 * https://github.com/opencv/opencv/blob/master/samples/cpp/points_classifier.cpp
 */

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>

// Define constants for colors and window name
const cv::Scalar WHITE_COLOR = cv::Scalar(255, 255, 255);
const std::string winName = "points";
const int testStep = 5;

// Declare variables for images, random number generator, and training data
cv::Mat img, imgDst;
cv::RNG rng;

// Variables for storing the trained points and their class markers
std::vector<cv::Point> trainedPoints;
std::vector<int> trainedPointsMarkers;
const int MAX_CLASSES = 2;                      // Number of classes
std::vector<cv::Vec3b> classColors(MAX_CLASSES);// Colors for each class
int currentClass = 0;                           // Currently selected class
std::vector<int> classCounters(MAX_CLASSES);    // Counter for each class

// Define flags to enable/disable different classifiers
#define _NBC_ 1 // normal Bayessian classifier
#define _KNN_ 1 // k nearest neighbors classifier
#define _SVM_ 1 // support vectors machine
#define _DT_  1 // decision tree
#define _BT_  1 // ADA Boost
#define _RF_  1 // random forest
#define _ANN_ 1 // artificial neural networks
#define _EM_  1 // expectation-maximization

// Mouse callback function to collect training data
static void on_mouse(int event, int x, int y, int /*flags*/, void *)
{
  if (img.empty()) {
    return;
  }

  bool updateFlag = false;

  if (event == cv::EVENT_LBUTTONUP) {                 // When left mouse button is released
    trainedPoints.push_back(cv::Point(x, y));         // Add the point to the list
    trainedPointsMarkers.push_back(currentClass);     // Assign the point to the current class
    classCounters[currentClass]++;                    // Increment the class counter
    updateFlag = true;
  }

    // If the data is updated, redraw the points on the image
  if (updateFlag) {
    img = cv::Scalar::all(0);     // Clear the image
    for (std::size_t i = 0; i < trainedPoints.size(); i++) {
      cv::Vec3b c = classColors[trainedPointsMarkers[i]];
      cv::circle(img, trainedPoints[i], 5, cv::Scalar(c), -1);       // Draw the point
    }
        // Show the updated image
    cv::imshow(winName, img);
  }
}

// Prepare training samples in a format suitable for the classifier
static cv::Mat prepare_train_samples(const std::vector<cv::Point> & pts)
{
  cv::Mat samples;
  cv::Mat(pts).reshape(1, static_cast<int>(pts.size())).convertTo(samples, CV_32F);
  return samples;
}

// Create the training data object for ML models
static cv::Ptr<cv::ml::TrainData> prepare_train_data()
{
  cv::Mat samples = prepare_train_samples(trainedPoints);
  return cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE, cv::Mat(trainedPointsMarkers));
}

// Predict and paint the decision boundary on the image using a trained model
static void predict_and_paint(const cv::Ptr<cv::ml::StatModel> & model, cv::Mat & dst)
{
  cv::Mat testSample(1, 2, CV_32FC1);   // Create a test sample
  for (int y = 0; y < img.rows; y += testStep) {
    for (int x = 0; x < img.cols; x += testStep) {
      testSample.at<float>(0) = static_cast<float>(x);                    // Set the x-coordinate
      testSample.at<float>(1) = static_cast<float>(y);                    // Set the y-coordinate
      int response = static_cast<int>(model->predict(testSample));        // Predict the class
      dst.at<cv::Vec3b>(y, x) = classColors[response];                    // Paint the decision boundary
    }
  }
}

#if _NBC_
// Train and apply the Normal Bayes Classifier
static void find_decision_boundary_NBC()
{
  cv::Ptr<cv::ml::NormalBayesClassifier> normalBayesClassifier =
    cv::ml::StatModel::train<cv::ml::NormalBayesClassifier>(prepare_train_data());     // Train the classifier
  predict_and_paint(normalBayesClassifier, imgDst);   // Predict and paint the boundary
}
#endif


#if _KNN_
// Train and apply the k-Nearest Neighbors classifier
static void find_decision_boundary_KNN(int K)
{
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setDefaultK(K);   // Set the number of neighbors
  knn->setIsClassifier(true);   // Set as a classifier
  knn->train(prepare_train_data());   // Train the classifier
  predict_and_paint(knn, imgDst);   // Predict and paint the boundary
}
#endif

#if _SVM_
// Train and apply the Support Vector Machine classifier
static void find_decision_boundary_SVM(double C)
{
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::POLY);   // Set the kernel type - (LINEAR, POLY, RBF)
  svm->setDegree(0.5);
  svm->setGamma(1);
  svm->setCoef0(1);
  svm->setNu(0.5);
  svm->setP(0);
  svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000,
    0.01));
  svm->setC(C);
  svm->train(prepare_train_data());   // Train the classifier
  predict_and_paint(svm, imgDst);   // Predict and paint the boundary

  cv::Mat sv = svm->getSupportVectors();   // Get the support vectors
  for (int i = 0; i < sv.rows; i++) {
    const float * supportVector = sv.ptr<float>(i);
    circle(imgDst,
      cv::Point(cv::saturate_cast<int>(supportVector[0]), cv::saturate_cast<int>(supportVector[1])),
      5, cv::Scalar(255, 255, 255), -1);                                                                                                                 // Draw the support vectors
  }
}
#endif

#if _DT_
// Train and apply the Decision Tree classifier
static void find_decision_boundary_DT()
{
  cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
  dtree->setMaxDepth(8);
  dtree->setMinSampleCount(2);
  dtree->setUseSurrogates(false);
  dtree->setCVFolds(0);                 // the number of cross-validation folds
  dtree->setUse1SERule(false);
  dtree->setTruncatePrunedTree(false);
  dtree->train(prepare_train_data());   // Train the classifier
  predict_and_paint(dtree, imgDst);     // Predict and paint the boundary
}
#endif

#if _BT_
// Train and apply the Boosting classifier
static void find_decision_boundary_BT()
{
  cv::Ptr<cv::ml::Boost> boost = cv::ml::Boost::create();
  boost->setBoostType(cv::ml::Boost::DISCRETE);
  boost->setWeakCount(100);
  boost->setWeightTrimRate(0.95);
  boost->setMaxDepth(2);
  boost->setUseSurrogates(false);
  boost->setPriors(cv::Mat());
  boost->train(prepare_train_data()); // Train the classifier
  predict_and_paint(boost, imgDst); // Predict and paint the boundary
}

#endif

#if _RF_
// Train and apply the Random Forest classifier
static void find_decision_boundary_RF()
{
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
  rtrees->train(prepare_train_data()); // Train the classifier
  predict_and_paint(rtrees, imgDst); // Predict and paint the boundary
}
#endif

#if _ANN_
// Train and apply the Artificial Neural Network classifier
static void find_decision_boundary_ANN(const cv::Mat & layer_sizes)
{
  cv::Mat trainClasses = cv::Mat::zeros((int)trainedPoints.size(), (int)classColors.size(),
    CV_32FC1);  // Create the class labels
  for (int i = 0; i < trainClasses.rows; i++) {
    trainClasses.at<float>(i, trainedPointsMarkers[i]) = 1.f; // Set the class label
  }

  cv::Mat samples = prepare_train_samples(trainedPoints);
  cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE,
    trainClasses);

  cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
  ann->setLayerSizes(layer_sizes); // Set the layer sizes
  ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
  ann->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300,
    FLT_EPSILON));
  ann->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001);
  ann->train(tdata); // Train the classifier
  predict_and_paint(ann, imgDst); // Predict and paint the boundary
}
#endif

#if _EM_
// Train and apply the Expectation-Maximization classifier
static void find_decision_boundary_EM()
{
  img.copyTo(imgDst);

  cv::Mat samples = prepare_train_samples(trainedPoints);

  int i, j, nmodels = (int)classColors.size();
  std::vector<cv::Ptr<cv::ml::EM>> em_models(nmodels);
  cv::Mat modelSamples;

  for (i = 0; i < nmodels; i++) {
    const int componentCount = 3;

    modelSamples.release();
    for (j = 0; j < samples.rows; j++) {
      if (trainedPointsMarkers[j] == i) {
        modelSamples.push_back(samples.row(j));
      }
    }

    // Learn models
    if (!modelSamples.empty() ) {
      cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
      em->setClustersNumber(componentCount);
      em->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
      em->trainEM(modelSamples, cv::noArray(), cv::noArray(), cv::noArray());
      em_models[i] = em;
    }
  }

  // Classify coordinate plane points using the bayes classifier, i.e.
  // y(x) = arg max_i=1_modelsCount likelihoods_i(x)
  cv::Mat testSample(1, 2, CV_32FC1);
  cv::Mat logLikelihoods(1, nmodels, CV_64FC1, cv::Scalar(-DBL_MAX));

  for (int y = 0; y < img.rows; y += testStep) {
    for (int x = 0; x < img.cols; x += testStep) {
      testSample.at<float>(0) = (float)x;
      testSample.at<float>(1) = (float)y;

      for (i = 0; i < nmodels; i++) {
        if (!em_models[i].empty() ) {
          logLikelihoods.at<double>(i) = em_models[i]->predict2(testSample, cv::noArray())[0];
        }
      }
      cv::Point maxLoc;
      cv::minMaxLoc(logLikelihoods, 0, 0, 0, &maxLoc);
      imgDst.at<cv::Vec3b>(y, x) = classColors[maxLoc.x];
    }
  }
}
#endif

int main()
{
  std::cout << "Use:" << std::endl
            << "  key '0' .. '1' - switch to class #n" << std::endl
            << "  left mouse button - to add new point;" << std::endl
            << "  key 'r' - to run the ML model;" << std::endl
            << "  key 'i' - to init (clear) the data." << std::endl << std::endl;

  // Initialize image and UI elements
  cv::namedWindow("points", 1);
  img.create(480, 640, CV_8UC3);
  imgDst.create(480, 640, CV_8UC3);

  cv::imshow("points", img);
  cv::setMouseCallback("points", on_mouse);

  // Initialize colors for the classes
  classColors[0] = cv::Vec3b(0, 255, 0);
  classColors[1] = cv::Vec3b(0, 0, 255);

  bool finish = false;
  while (!finish) {
    // Wait for user input
    char key = (char)cv::waitKey();

    if (key == 27) {finish = true;} // Exit on 'Esc' key

    if (key == 'i') {    // init
      img = cv::Scalar::all(0);

      trainedPoints.clear();
      trainedPointsMarkers.clear();
      classCounters.assign(MAX_CLASSES, 0);

      imshow(winName, img);
    }

    if (key == '0' || key == '1') {
      currentClass = key - '0';
    }

    if (key == 'r') {    // run
      double minVal = 0;
      cv::minMaxLoc(classCounters, &minVal, 0, 0, 0);
      if (minVal == 0) {
        printf("each class should have at least 1 point\n");
        continue;
      }
      img.copyTo(imgDst);

#if _NBC_
      find_decision_boundary_NBC();
      imshow("NormalBayesClassifier", imgDst);
#endif

#if _KNN_
      find_decision_boundary_KNN(3);
      imshow("kNN", imgDst);

      find_decision_boundary_KNN(15);
      imshow("kNN2", imgDst);
#endif

#if _SVM_
      //(1)-(2)separable and not sets

      find_decision_boundary_SVM(1);
      imshow("classificationSVM1", imgDst);

      find_decision_boundary_SVM(10);
      imshow("classificationSVM2", imgDst);
#endif

#if _DT_
      find_decision_boundary_DT();
      imshow("DT", imgDst);
#endif

#if _BT_
      find_decision_boundary_BT();
      imshow("BT", imgDst);
#endif

#if _RF_
      find_decision_boundary_RF();
      imshow("RF", imgDst);
#endif

#if _ANN_
      cv::Mat layer_sizes1(1, 3, CV_32SC1);
      layer_sizes1.at<int>(0) = 2;
      layer_sizes1.at<int>(1) = 5;
      layer_sizes1.at<int>(2) = (int)classColors.size();
      find_decision_boundary_ANN(layer_sizes1);
      imshow("ANN", imgDst);
#endif

#if _EM_
      find_decision_boundary_EM();
      imshow("EM", imgDst);
#endif
    }
  }

  return 0;
}
