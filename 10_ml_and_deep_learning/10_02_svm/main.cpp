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
constexpr int NTRAINING_SAMPLES = 100;   // Number of training samples per class
constexpr float FRAC_LINEAR_SEP = 0.9f;  // Fraction of samples placed in the linearly separable
                                         // region (0.9 = 90% clean, 10% overlapping/noise)
constexpr int WIDTH = 512;               // Canvas width in pixels
constexpr int HEIGHT = 512;              // Canvas height in pixels
}

/**
 * @brief Display help information
 */
static void help()
{
  std::cout   << std::endl
              << "--------------------------------------------------------------------------\n"
              << "This program shows Support Vector Machines for Non-Linearly Separable Data.\n"
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
  //
  // Two classes are created with 100 samples each (200 rows total).
  // Each sample has 2 features (x, y) stored as CV_32F so the SVM can
  // consume them directly.
  //
  // Class layout (x axis split into three zones):
  //   [0,   0.4*W)   -> class 1  (n_linear_samples points, left region)
  //   (0.6*W, W]     -> class 2  (n_linear_samples points, right region)
  //   [0.4*W, 0.6*W] -> overlap zone (remaining points from both classes)
  //
  // The overlap zone makes the data non-linearly separable, which is the
  // key motivation for using a soft-margin SVM (C_SVC with small C).
  //-------------------------------------------------------------------------
  cv::Mat train_data(2 * Config::NTRAINING_SAMPLES, 2, CV_32F);
  cv::Mat labels(2 * Config::NTRAINING_SAMPLES, 1, CV_32S);
  cv::RNG rng(100);  // Fixed seed for reproducibility

  // Number of samples placed in the clean (linearly separable) regions
  int n_linear_samples = static_cast<int>(Config::FRAC_LINEAR_SEP * Config::NTRAINING_SAMPLES);

  // --- Class 1: left clean region (rows 0 .. n_linear_samples-1) ---
  cv::Mat train_class = train_data.rowRange(0, n_linear_samples);
  rng.fill(train_class.colRange(0, 1), cv::RNG::UNIFORM, cv::Scalar(0),
    cv::Scalar(0.4 * Config::WIDTH));
  rng.fill(train_class.colRange(1, 2), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(Config::HEIGHT));

  // --- Class 2: right clean region (last n_linear_samples rows of the first half) ---
  train_class = train_data.rowRange(2 * Config::NTRAINING_SAMPLES - n_linear_samples,
    2 * Config::NTRAINING_SAMPLES);
  rng.fill(train_class.colRange(0, 1), cv::RNG::UNIFORM, cv::Scalar(0.6 * Config::WIDTH),
    cv::Scalar(Config::WIDTH));
  rng.fill(train_class.colRange(1, 2), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(Config::HEIGHT));

  // --- Overlap zone: remaining samples from both classes in the middle band ---
  // These points intentionally mix the two classes, creating a non-linearly
  // separable region that forces the SVM to tolerate misclassifications.
  train_class = train_data.rowRange(n_linear_samples,
    2 * Config::NTRAINING_SAMPLES - n_linear_samples);
  rng.fill(train_class.colRange(0, 1), cv::RNG::UNIFORM, cv::Scalar(0.4 * Config::WIDTH),
    cv::Scalar(0.6 * Config::WIDTH));
  rng.fill(train_class.colRange(1, 2), cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(Config::HEIGHT));

  // Assign integer labels: first 100 rows -> class 1, next 100 rows -> class 2
  labels.rowRange(0, Config::NTRAINING_SAMPLES).setTo(1);
  labels.rowRange(Config::NTRAINING_SAMPLES, 2 * Config::NTRAINING_SAMPLES).setTo(2);

  //-------------------------------------------------------------------------
  // 2. Set up the support vector machine parameters
  //
  // SVM finds the hyperplane that maximizes the margin between classes.
  // For non-separable data a soft-margin formulation is used:
  //
  //   minimize  (1/2)||w||^2 + C * sum(xi_i)
  //   subject to  y_i(w·x_i + b) >= 1 - xi_i,  xi_i >= 0
  //
  // Key parameters:
  //   Type   C_SVC  : C-Support Vector Classification (multi-class capable)
  //   C      0.1    : Regularization parameter. Small C -> wider margin but
  //                   more misclassifications allowed (soft margin).
  //                   Large C -> fewer misclassifications but narrower margin
  //                   (risk of overfitting).
  //   Kernel LINEAR : decision boundary is a straight hyperplane (w·x + b = 0).
  //                   Suitable here because the classes are mostly separable.
  //   TermCriteria  : stop after 1e7 iterations or when the change in the
  //                   objective function is smaller than 1e-6.
  //-------------------------------------------------------------------------
  std::cout << "Starting training process" << std::endl;
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);     // Soft-margin classification
  svm->setC(0.1);                       // Regularization: trade-off margin vs. errors
  svm->setKernel(cv::ml::SVM::LINEAR);  // Linear hyperplane as decision boundary
  svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, static_cast<int>(1e7), 1e-6));

  //-------------------------------------------------------------------------
  // 3. Train the SVM model
  //
  // train_data is already in ROW_SAMPLE layout (each row = one sample,
  // each column = one feature). The solver uses Sequential Minimal
  // Optimization (SMO) internally to find the optimal w and b.
  //-------------------------------------------------------------------------
  svm->train(train_data, cv::ml::ROW_SAMPLE, labels);
  std::cout << "Finished training process" << std::endl;

  //-------------------------------------------------------------------------
  // 4. Show the decision regions
  //
  // Every pixel (j, i) is treated as a 2D test sample [j, i] and fed to
  // svm->predict(). The returned value is the class label (1 or 2).
  // Pixels are painted green for class 1 and blue for class 2, producing
  // a dense visualization of the decision boundary.
  //
  // Note: this pixel-wise loop is the straightforward approach; for large
  // images a vectorized call with all pixels at once would be faster.
  //-------------------------------------------------------------------------
  cv::Vec3b green(0, 100, 0), blue(100, 0, 0);
  for (int i = 0; i < I.rows; i++) {
    for (int j = 0; j < I.cols; j++) {
      // Build a 1x2 float matrix with the pixel coordinates as features
      cv::Mat sample_mat = (cv::Mat_<float>(1, 2) << j, i);
      float response = svm->predict(sample_mat);  // Returns the winning class label
      I.at<cv::Vec3b>(i, j) = (response == 1) ? green : blue;
    }
  }

  //-------------------------------------------------------------------------
  // 5. Show the training data
  //
  // Draw each training point as a small filled circle on top of the
  // decision-region background so we can see how samples relate to
  // the learned boundary.
  //   Bright green (0, 255, 0) -> class 1
  //   Bright blue  (255, 0, 0) -> class 2
  //-------------------------------------------------------------------------
  int thickness = -1;  // Negative thickness = filled circle
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
  //
  // Support vectors are the training samples that lie closest to (or inside)
  // the margin. They are the only samples that actually define the decision
  // boundary: all other samples could be removed without changing the result.
  //
  // getUncompressedSupportVectors() returns the raw feature vectors of those
  // samples as a matrix (one row per support vector, one column per feature).
  // They are drawn as gray rings to distinguish them from regular points.
  //
  // With C=0.1 (soft margin) many points in the overlap zone will become
  // support vectors because the solver needs them to describe the boundary
  // under the relaxed constraints.
  //-------------------------------------------------------------------------
  thickness = 2;  // Hollow circle (outline only) to overlay on top of points
  cv::Mat sv = svm->getUncompressedSupportVectors();
  for (int i = 0; i < sv.rows; i++) {
    const float *v = sv.ptr<float>(i);  // Row pointer: v[0]=x, v[1]=y
    cv::circle(I, cv::Point(static_cast<int>(v[0]), static_cast<int>(v[1])), 6,
                   cv::Scalar(128, 128, 128), thickness);
  }

  // Save and display the result
  cv::imwrite("../../data/svm_result.png", I);
  cv::imshow("SVM for Non-Linear Training Data", I);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
