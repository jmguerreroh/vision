/**
 * @file main.cpp
 * @brief Handwritten digit classification with evaluation metrics
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Building a real dataset from an image mosaic (data/digits.png)
 * - A proper TRAIN / TEST split (never evaluate on the training data!)
 * - Training the classifiers of 16_01 and 16_02 (KNN and SVM) on real data
 * - Computing the metrics of the book: accuracy, confusion matrix and
 *   per-class precision / recall
 *
 * The dataset: digits.png is a 2000x1000 mosaic of 5000 handwritten digits
 * from the MNIST-style OpenCV samples -- a grid of 100x50 cells of 20x20
 * pixels. Rows are ordered by digit: the first 5 rows are '0's, the next
 * 5 rows are '1's, and so on (500 samples per digit).
 *
 * The feature vector: simply the 400 raw pixel intensities of each cell.
 * No descriptors, no learning of features -- and yet KNN already exceeds
 * 90% accuracy. Deep learning (next examples) starts from this baseline.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

namespace Config
{
constexpr int CELL_SIZE = 20;      // Each digit is a 20x20 pixel cell
constexpr int NUM_CLASSES = 10;    // Digits 0..9
constexpr int KNN_K = 5;           // Neighbors for the KNN vote
constexpr float TRAIN_RATIO = 0.5f;  // Left half train, right half test
}

/**
 * @brief Slices the mosaic into per-digit samples and splits train/test
 * @param mosaic Grayscale digits.png image
 * @param train_samples Output train matrix (one flattened digit per row, CV_32F)
 * @param train_labels Output train labels (CV_32S)
 * @param test_samples Output test matrix
 * @param test_labels Output test labels
 *
 * The split is done BY COLUMNS (left half of the mosaic for training, right
 * half for testing) so both sets contain the same amount of every digit.
 * Evaluating on the same samples used for training would report a
 * misleadingly high score -- the model has already seen those exact images.
 */
void buildDataset(
  const cv::Mat & mosaic,
  cv::Mat & train_samples, cv::Mat & train_labels,
  cv::Mat & test_samples, cv::Mat & test_labels)
{
  const int cells_per_row = mosaic.cols / Config::CELL_SIZE;   // 100
  const int cells_per_col = mosaic.rows / Config::CELL_SIZE;   // 50
  const int rows_per_digit = cells_per_col / Config::NUM_CLASSES;  // 5
  const int train_columns = static_cast<int>(cells_per_row * Config::TRAIN_RATIO);

  for (int cell_row = 0; cell_row < cells_per_col; ++cell_row) {
    // Which digit is this row of the mosaic? (5 consecutive rows per digit)
    const int digit = cell_row / rows_per_digit;

    for (int cell_col = 0; cell_col < cells_per_row; ++cell_col) {
      // Extract the 20x20 cell as an ROI (02_03) and flatten it into a
      // single row of 400 float features -- the same reshape+convertTo
      // pattern used to feed TrainData in 16_01
      const cv::Rect cell_rect(cell_col * Config::CELL_SIZE,
                               cell_row * Config::CELL_SIZE,
                               Config::CELL_SIZE, Config::CELL_SIZE);
      cv::Mat feature_row;
      mosaic(cell_rect).clone().reshape(1, 1).convertTo(feature_row, CV_32F,
                                                        1.0 / 255.0);

      if (cell_col < train_columns) {
        train_samples.push_back(feature_row);
        train_labels.push_back(digit);
      } else {
        test_samples.push_back(feature_row);
        test_labels.push_back(digit);
      }
    }
  }
}

/**
 * @brief Fills the confusion matrix and returns the global accuracy
 * @param predictions Predicted label per test sample (CV_32F from predict())
 * @param test_labels Ground-truth label per test sample (CV_32S)
 * @param confusion Output NUM_CLASSES x NUM_CLASSES matrix:
 *                  confusion[true][predicted] = count
 * @return Fraction of correctly classified samples (accuracy)
 */
double evaluate(
  const cv::Mat & predictions, const cv::Mat & test_labels, cv::Mat & confusion)
{
  confusion = cv::Mat::zeros(Config::NUM_CLASSES, Config::NUM_CLASSES, CV_32S);
  int correct = 0;

  for (int i = 0; i < test_labels.rows; ++i) {
    const int truth = test_labels.at<int>(i);
    const int predicted = static_cast<int>(predictions.at<float>(i));
    confusion.at<int>(truth, predicted)++;
    if (truth == predicted) {
      ++correct;
    }
  }

  return static_cast<double>(correct) / test_labels.rows;
}

/**
 * @brief Prints the confusion matrix with per-class recall and precision
 *
 * Reading the matrix: row = TRUE digit, column = PREDICTED digit. A perfect
 * classifier only fills the diagonal. Off-diagonal cells reveal WHICH
 * mistakes happen (e.g. 4s classified as 9s), something the single accuracy
 * number can never tell you.
 *
 * Per class:
 *   recall    = diagonal / row sum    ("of the real 4s, how many found?")
 *   precision = diagonal / column sum ("of the predicted 4s, how many right?")
 */
void printConfusionMatrix(const cv::Mat & confusion)
{
  std::cout << "\nConfusion matrix (rows = truth, cols = predicted):\n\n     ";
  for (int j = 0; j < Config::NUM_CLASSES; ++j) {
    std::cout << std::setw(5) << j;
  }
  std::cout << "  | recall\n";
  std::cout << std::string(5 + 5 * Config::NUM_CLASSES + 10, '-') << std::endl;

  for (int i = 0; i < Config::NUM_CLASSES; ++i) {
    int row_sum = 0;
    std::cout << "  " << i << " |";
    for (int j = 0; j < Config::NUM_CLASSES; ++j) {
      std::cout << std::setw(5) << confusion.at<int>(i, j);
      row_sum += confusion.at<int>(i, j);
    }
    const double recall = 100.0 * confusion.at<int>(i, i) / row_sum;
    std::cout << "  | " << std::fixed << std::setprecision(1) << recall << "%\n";
  }

  std::cout << "prec:";
  for (int j = 0; j < Config::NUM_CLASSES; ++j) {
    int col_sum = 0;
    for (int i = 0; i < Config::NUM_CLASSES; ++i) {
      col_sum += confusion.at<int>(i, j);
    }
    const double precision =
      col_sum > 0 ? 100.0 * confusion.at<int>(j, j) / col_sum : 0.0;
    std::cout << std::setw(5) << static_cast<int>(precision + 0.5);
  }
  std::cout << "  (%)\n";
}

int main(int argc, char ** argv)
{
  // Load the dataset mosaic
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/digits.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  const cv::Mat mosaic = cv::imread(cv::samples::findFile(image_path, false),
                                    cv::IMREAD_GRAYSCALE);

  if (mosaic.empty()) {
    std::cerr << "Error: Could not load '" << image_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Digit Classification with Metrics ===" << std::endl;

  // ========================================
  // Build the dataset
  // ========================================
  cv::Mat train_samples, test_samples;
  cv::Mat train_labels, test_labels;
  buildDataset(mosaic, train_samples, train_labels, test_samples, test_labels);

  std::cout << "Train samples: " << train_samples.rows
            << "   Test samples: " << test_samples.rows
            << "   Features per sample: " << train_samples.cols << std::endl;

  cv::imshow("digits.png (5000 samples)", mosaic);

  // ========================================
  // Classifier 1: KNN (as introduced in 16_01)
  // ========================================
  std::cout << "\n[KNN] training (K = " << Config::KNN_K << ")..." << std::endl;
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setDefaultK(Config::KNN_K);
  knn->setIsClassifier(true);
  knn->train(train_samples, cv::ml::ROW_SAMPLE, train_labels);

  cv::Mat knn_predictions;
  knn->predict(test_samples, knn_predictions);

  cv::Mat knn_confusion;
  const double knn_accuracy = evaluate(knn_predictions, test_labels, knn_confusion);
  std::cout << "[KNN] accuracy on TEST set: "
            << std::fixed << std::setprecision(2) << 100.0 * knn_accuracy
            << "%" << std::endl;
  printConfusionMatrix(knn_confusion);

  // ========================================
  // Classifier 2: SVM (as introduced in 16_02)
  // ========================================
  // Linear kernel: fast to train on 2500 samples of 400 features. An RBF
  // kernel with tuned (C, gamma) reaches a few points more at the cost of
  // a much longer training -- a good experiment for the reader.
  std::cout << "\n[SVM] training (linear kernel)..." << std::endl;
  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::LINEAR);
  svm->setC(1.0);
  svm->setTermCriteria(cv::TermCriteria(
    cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6));
  svm->train(train_samples, cv::ml::ROW_SAMPLE, train_labels);

  cv::Mat svm_predictions;
  svm->predict(test_samples, svm_predictions);

  cv::Mat svm_confusion;
  const double svm_accuracy = evaluate(svm_predictions, test_labels, svm_confusion);
  std::cout << "[SVM] accuracy on TEST set: "
            << 100.0 * svm_accuracy << "%" << std::endl;
  printConfusionMatrix(svm_confusion);

  // ========================================
  // Wrap-up
  // ========================================
  std::cout << "\nThings to observe:" << std::endl;
  std::cout << "  - The off-diagonal peaks: which digits get confused with which"
            << std::endl;
  std::cout << "  - Raw pixels already work surprisingly well on clean data;"
            << std::endl;
  std::cout << "    real images need the descriptors of Chapters 11 and 12 or learned"
            << std::endl;
  std::cout << "    features (CNNs, next examples)" << std::endl;
  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
