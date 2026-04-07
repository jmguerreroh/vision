/**
 * @file main.cpp
 * @brief SVM (Support Vector Machine) demo for non-linearly separable data
 * @author José Miguel Guerrero Hernández
 *
 * @details Generates random 2D training points with two classes (linearly separable
 *          region + overlap zone), trains a linear SVM classifier, and visualizes
 *          the decision regions and support vectors.
 *
 *          Steps:
 *          1. Generate random training data (90% linearly separable, 10% overlap)
 *          2. Train a linear C-SVC with C=0.1
 *          3. Classify every pixel to visualize decision boundaries
 *          4. Highlight support vectors with gray circles
 *
 * @see https://docs.opencv.org/3.4/d0/dcc/tutorial_non_linear_svms.html
 */

#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

// Configuration constants
namespace Config
{
constexpr int NTRAINING_SAMPLES = 100;
constexpr float FRAC_LINEAR_SEP = 0.9f;
constexpr int WIDTH = 512;
constexpr int HEIGHT = 512;
}

/**
 * @brief Display help information
 */
static void help()
{
  std::cout   << std::endl
              << "--------------------------------------------------------------------------\n"
              << "This program shows Support Vector Machines for Non-Linearly Separable Data. "
              << "--------------------------------------------------------------------------\n"
              << std::endl;
}

int main(int argc, char ** argv)
{
  (void)argc;
  (void)argv;

  help();

  cv::Mat I = cv::Mat::zeros(Config::HEIGHT, Config::WIDTH, CV_8UC3);

  //-------------------------------------------------------------------------
  // 1. Set up training data randomly
  //-------------------------------------------------------------------------
  cv::Mat train_data(2 * Config::NTRAINING_SAMPLES, 2, CV_32F);
  cv::Mat labels(2 * Config::NTRAINING_SAMPLES, 1, CV_32S);
  cv::RNG rng(100);

  int n_linear_samples = static_cast<int>(Config::FRAC_LINEAR_SEP * Config::NTRAINING_SAMPLES);

  cv::Mat train_class = train_data.rowRange(0, n_linear_samples);
  rng.fill(train_class.colRange(0, 1), cv::RNG::UNIFORM, cv::Scalar(0),
    cv::Scalar(0.4 * Config::WIDTH));
  rng.fill(train_class.colRange(1, 2), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(Config::HEIGHT));

  train_class = train_data.rowRange(2 * Config::NTRAINING_SAMPLES - n_linear_samples,
    2 * Config::NTRAINING_SAMPLES);
  rng.fill(train_class.colRange(0, 1), cv::RNG::UNIFORM, cv::Scalar(0.6 * Config::WIDTH),
    cv::Scalar(Config::WIDTH));
  rng.fill(train_class.colRange(1, 2), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(Config::HEIGHT));

  train_class = train_data.rowRange(n_linear_samples,
    2 * Config::NTRAINING_SAMPLES - n_linear_samples);
  rng.fill(train_class.colRange(0, 1), cv::RNG::UNIFORM, cv::Scalar(0.4 * Config::WIDTH),
    cv::Scalar(0.6 * Config::WIDTH));
  rng.fill(train_class.colRange(1, 2), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(Config::HEIGHT));

  labels.rowRange(0, Config::NTRAINING_SAMPLES).setTo(1);
  labels.rowRange(Config::NTRAINING_SAMPLES, 2 * Config::NTRAINING_SAMPLES).setTo(2);

  //-------------------------------------------------------------------------
  // 2. Set up the support vector machines parameters
  //-------------------------------------------------------------------------
  std::cout << "Starting training process" << std::endl;
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setC(0.1);
  svm->setKernel(cv::ml::SVM::LINEAR);
  svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, static_cast<int>(1e7), 1e-6));

  //-------------------------------------------------------------------------
  // 3. Train the SVM model
  //-------------------------------------------------------------------------
  svm->train(train_data, cv::ml::ROW_SAMPLE, labels);
  std::cout << "Finished training process" << std::endl;

  //-------------------------------------------------------------------------
  // 4. Show the decision regions
  //-------------------------------------------------------------------------
  cv::Vec3b green(0, 100, 0), blue(100, 0, 0);
  for (int i = 0; i < I.rows; i++) {
    for (int j = 0; j < I.cols; j++) {
      cv::Mat sample_mat = (cv::Mat_<float>(1, 2) << j, i);
      float response = svm->predict(sample_mat);
      I.at<cv::Vec3b>(i, j) = (response == 1) ? green : blue;
    }
  }

  //-------------------------------------------------------------------------
  // 5. Show the training data
  //-------------------------------------------------------------------------
  int thickness = -1;
  for (int i = 0; i < Config::NTRAINING_SAMPLES; i++) {
    cv::circle(I, cv::Point(static_cast<int>(train_data.at<float>(i, 0)),
                                static_cast<int>(train_data.at<float>(i, 1))),
                   3, cv::Scalar(0, 255, 0), thickness);
  }
  for (int i = Config::NTRAINING_SAMPLES; i < 2 * Config::NTRAINING_SAMPLES; i++) {
    cv::circle(I, cv::Point(static_cast<int>(train_data.at<float>(i, 0)),
                                static_cast<int>(train_data.at<float>(i, 1))),
                   3, cv::Scalar(255, 0, 0), thickness);
  }

  //-------------------------------------------------------------------------
  // 6. Highlight and show the support vectors
  //-------------------------------------------------------------------------
  thickness = 2;
  cv::Mat sv = svm->getUncompressedSupportVectors();
  for (int i = 0; i < sv.rows; i++) {
    const float *v = sv.ptr<float>(i);
    cv::circle(I, cv::Point(static_cast<int>(v[0]), static_cast<int>(v[1])), 6,
                   cv::Scalar(128, 128, 128), thickness);
  }

  // Save and display the result
  cv::imwrite("../../data/svm_result.png", I);
  cv::imshow("SVM for Non-Linear Training Data", I);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
