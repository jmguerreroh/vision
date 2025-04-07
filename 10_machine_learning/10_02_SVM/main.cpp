/**
 * SVM demo sample
 * @author José Miguel Guerrero
 *
 * https://docs.opencv.org/3.4/d0/dcc/tutorial_non_linear_svms.html
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

// Function to display help information
static void help()
{
  std::cout << std::endl <<
    "--------------------------------------------------------------------------" << std::endl <<
    "This program shows Support Vector Machines for Non-Linearly Separable Data. " << std::endl <<
    "--------------------------------------------------------------------------" << std::endl <<
    std::endl;
}

int main()
{
  help();
  const int NTRAINING_SAMPLES = 100;    // Number of training samples per class
  const float FRAC_LINEAR_SEP = 0.9f;   // Fraction of samples that are linearly separable

  // Image dimensions for visualization
  const int WIDTH = 512, HEIGHT = 512;
  cv::Mat I = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

  //--------------------- 1. Set up training data randomly ---------------------------------------
  // Initialize training data and labels
  cv::Mat trainData(2 * NTRAINING_SAMPLES, 2, CV_32F);
  cv::Mat labels(2 * NTRAINING_SAMPLES, 1, CV_32S);
  cv::RNG rng(100);   // Random number generator

  int nLinearSamples = static_cast<int>(FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

  // Generate random points for class 1 - x coordinate in [0, 0.4) and y in [0, 1)
  cv::Mat trainClass = trainData.rowRange(0, nLinearSamples);
  rng.fill(trainClass.colRange(0, 1), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(0.4 * WIDTH));
  rng.fill(trainClass.colRange(1, 2), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(HEIGHT));

  // Generate random points for class 2 - x coordinate in [0.6, 1) and y in [0, 1)
  trainClass = trainData.rowRange(2 * NTRAINING_SAMPLES - nLinearSamples, 2 * NTRAINING_SAMPLES);
  rng.fill(trainClass.colRange(0, 1), cv::RNG::UNIFORM, cv::Scalar(0.6 * WIDTH), cv::Scalar(WIDTH));
  rng.fill(trainClass.colRange(1, 2), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(HEIGHT));

  // Generate non-linearly separable data - x coordinate in [0.4, 0.6) and y in [0, 1)
  trainClass = trainData.rowRange(nLinearSamples, 2 * NTRAINING_SAMPLES - nLinearSamples);
  rng.fill(trainClass.colRange(0, 1), cv::RNG::UNIFORM, cv::Scalar(0.4 * WIDTH),
    cv::Scalar(0.6 * WIDTH));
  rng.fill(trainClass.colRange(1, 2), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(HEIGHT));

  // Assign class labels
  labels.rowRange(0, NTRAINING_SAMPLES).setTo(1);                       // Class 1
  labels.rowRange(NTRAINING_SAMPLES, 2 * NTRAINING_SAMPLES).setTo(2);   // Class 2

  //------------------------ 2. Set up the support vector machines parameters --------------------
  // See https://docs.opencv.org/3.4/d1/d73/tutorial_introduction_to_svm.html

  // Create and configure SVM model
  std::cout << "Starting training process" << std::endl;
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setC(0.1);
  svm->setKernel(cv::ml::SVM::LINEAR);
  svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, (int)1e7, 1e-6));

  //------------------------ 3. Train the SVM model ----------------------------------------------
  svm->train(trainData, cv::ml::ROW_SAMPLE, labels);
  std::cout << "Finished training process" << std::endl;

  //------------------------ 4. Show the decision regions ----------------------------------------
  // Visualize decision regions
  cv::Vec3b green(0, 100, 0), blue(100, 0, 0);
  for (int i = 0; i < I.rows; i++) {
    for (int j = 0; j < I.cols; j++) {
      cv::Mat sampleMat = (cv::Mat_<float>(1, 2) << j, i);
      float response = svm->predict(sampleMat);
      I.at<cv::Vec3b>(i, j) = (response == 1) ? green : blue;
    }
  }

  //----------------------- 5. Show the training data --------------------------------------------
  int thickness = -1;
  for (int i = 0; i < NTRAINING_SAMPLES; i++) {
    cv::circle(I, cv::Point(static_cast<int>(trainData.at<float>(i, 0)),
                            static_cast<int>(trainData.at<float>(i, 1))),
                3, cv::Scalar(0, 255, 0), thickness);
  }
  for (int i = NTRAINING_SAMPLES; i < 2 * NTRAINING_SAMPLES; i++) {
    cv::circle(I, cv::Point(static_cast<int>(trainData.at<float>(i, 0)),
                            static_cast<int>(trainData.at<float>(i, 1))),
                3, cv::Scalar(255, 0, 0), thickness);
  }

  //------------------------ 6. Highlight and show the support vectors ------------------------------
  thickness = 2;
  cv::Mat sv = svm->getUncompressedSupportVectors();
  for (int i = 0; i < sv.rows; i++) {
    const float *v = sv.ptr<float>(i);
    cv::circle(I, cv::Point(static_cast<int>(v[0]), static_cast<int>(v[1])), 6,
      cv::Scalar(128, 128, 128), thickness);
  }

  // Save and display the result
  cv::imwrite("result.png", I);
  cv::imshow("SVM for Non-Linear Training Data", I);
  cv::waitKey();
  return 0;
}
