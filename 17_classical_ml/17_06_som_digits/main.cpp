/**
 * @file main.cpp
 * @brief Self-organizing map (Kohonen) over the handwritten digits
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - A SOM implemented by hand: OpenCV has no self-organizing map, and the
 *   algorithm is short enough that writing it is the best way to see it
 * - The two decaying schedules that make it work: learning rate and
 *   neighbourhood radius
 * - Reading the trained map: every neuron is a codebook vector, and drawing
 *   it as a 20x20 image turns the map into an atlas of the dataset
 * - Measuring what the map is worth: quantization error, its accuracy when
 *   used as a classifier, and how many neighbouring neurons disagree
 *
 * The dataset and the train/test split are the same as 17_03, so the accuracy
 * printed here can be put next to the KNN and SVM of that example. The
 * comparison is not fair on purpose: KNN keeps the 2500 training samples,
 * while the SOM compresses them into a handful of prototypes and still gets
 * close. That compression is the point.
 *
 * What to look for in the map window: digits that look alike land in
 * neighbouring cells, because the update drags the whole neighbourhood of the
 * winner and not only the winner. No other clustering method guarantees that.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace Config
{
constexpr int CELL_SIZE = 20;        // Each digit is a 20x20 cell of the mosaic
constexpr int NUM_CLASSES = 10;
constexpr float TRAIN_RATIO = 0.5f;  // Left half train, right half test (17_03)
constexpr int MAP_ROWS = 12;         // The map: 12x12 = 144 neurons
constexpr int MAP_COLS = 12;
constexpr int EPOCHS = 50;          // About ten seconds of training
constexpr double ALPHA_INITIAL = 0.5;    // Learning rate at the start...
constexpr double ALPHA_FINAL = 0.01;     // ...and at the end
constexpr double SIGMA_FINAL = 0.6;      // Final neighbourhood radius, in cells
constexpr int ZOOM = 3;                  // Scale of the map window
constexpr uint64_t SEED = 7;             // Fixed: the run is reproducible
}

/**
 * @brief Slices digits.png into samples, split by columns as in 17_03
 */
void buildDataset(
  const cv::Mat & mosaic,
  cv::Mat & train_samples, cv::Mat & train_labels,
  cv::Mat & test_samples, cv::Mat & test_labels)
{
  const int cells_per_row = mosaic.cols / Config::CELL_SIZE;      // 100
  const int cells_per_col = mosaic.rows / Config::CELL_SIZE;      // 50
  const int rows_per_digit = cells_per_col / Config::NUM_CLASSES;  // 5
  const int train_columns = static_cast<int>(cells_per_row * Config::TRAIN_RATIO);

  for (int cell_row = 0; cell_row < cells_per_col; ++cell_row) {
    const int digit = cell_row / rows_per_digit;
    for (int cell_col = 0; cell_col < cells_per_row; ++cell_col) {
      const cv::Rect cell(cell_col * Config::CELL_SIZE, cell_row * Config::CELL_SIZE,
        Config::CELL_SIZE, Config::CELL_SIZE);
      cv::Mat feature_row;
      mosaic(cell).clone().reshape(1, 1).convertTo(feature_row, CV_32F, 1.0 / 255.0);
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
 * @brief Index of the best matching unit: the neuron closest to the sample
 * @param weights One codebook vector per row
 * @param sample Row vector with the features
 * @param distance Output, the distance to that neuron
 *
 * This is the competitive step, and it is a plain nearest-neighbour search.
 * The squared distance is enough to compare, but the square root is kept
 * because the quantization error is reported in the units of the data.
 */
int bestMatchingUnit(const cv::Mat & weights, const cv::Mat & sample, double & distance)
{
  int best = 0;
  double best_distance = std::numeric_limits<double>::max();
  for (int i = 0; i < weights.rows; ++i) {
    const double d = cv::norm(sample, weights.row(i), cv::NORM_L2);
    if (d < best_distance) {
      best_distance = d;
      best = i;
    }
  }
  distance = best_distance;
  return best;
}

int main(int argc, char ** argv)
{
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/digits.png | Mosaic of handwritten digits}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  const std::string path = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  const cv::Mat mosaic = cv::imread(cv::samples::findFile(path, false),
    cv::IMREAD_GRAYSCALE);
  if (mosaic.empty()) {
    std::cerr << "Error: could not load '" << path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat train_samples, train_labels, test_samples, test_labels;
  buildDataset(mosaic, train_samples, train_labels, test_samples, test_labels);

  const int dimension = train_samples.cols;
  const int num_neurons = Config::MAP_ROWS * Config::MAP_COLS;
  std::cout << "=== Self-organizing map over the digits ===" << std::endl;
  std::cout << "Train: " << train_samples.rows << " samples of " << dimension
            << " features   Map: " << Config::MAP_ROWS << "x" << Config::MAP_COLS
            << " = " << num_neurons << " neurons" << std::endl;

  // ========================================
  // Initialisation
  // ========================================
  // Each codebook starts as a copy of a random training sample. Starting from
  // real data instead of noise costs nothing and avoids dead neurons, the ones
  // that are so far from everything that they never win and never learn
  cv::RNG rng(Config::SEED);
  cv::Mat weights(num_neurons, dimension, CV_32F);
  for (int j = 0; j < num_neurons; ++j) {
    train_samples.row(rng.uniform(0, train_samples.rows)).copyTo(weights.row(j));
  }

  // ========================================
  // Training
  // ========================================
  // Two things decay with time, and both must: the learning rate, so the map
  // settles instead of chasing the last sample, and the neighbourhood radius,
  // which starts covering half the map (that is what unfolds it globally) and
  // ends below one cell (that is what refines each neuron on its own)
  const double sigma_initial = std::max(Config::MAP_ROWS, Config::MAP_COLS) / 2.0;
  std::vector<int> order(train_samples.rows);
  std::iota(order.begin(), order.end(), 0);

  const double tick = static_cast<double>(cv::getTickCount());
  for (int epoch = 0; epoch < Config::EPOCHS; ++epoch) {
    // Presenting the samples always in the same order would bias the map
    // towards whatever comes last, and the mosaic is sorted by digit
    cv::randShuffle(cv::Mat(order), 1.0, &rng);

    const double progress = static_cast<double>(epoch) / (Config::EPOCHS - 1);
    const double alpha = Config::ALPHA_INITIAL *
      std::pow(Config::ALPHA_FINAL / Config::ALPHA_INITIAL, progress);
    const double sigma = sigma_initial *
      std::pow(Config::SIGMA_FINAL / sigma_initial, progress);

    for (const int index : order) {
      const cv::Mat sample = train_samples.row(index);
      double distance = 0.0;
      const int winner = bestMatchingUnit(weights, sample, distance);
      const int winner_row = winner / Config::MAP_COLS;
      const int winner_col = winner % Config::MAP_COLS;

      // Cooperative step: the winner and its neighbours move towards the
      // sample, by an amount that falls off as a Gaussian ON THE MAP, not in
      // the feature space. That is what preserves the topology
      for (int j = 0; j < num_neurons; ++j) {
        const double dr = j / Config::MAP_COLS - winner_row;
        const double dc = j % Config::MAP_COLS - winner_col;
        const double grid_distance2 = dr * dr + dc * dc;
        if (grid_distance2 > 9.0 * sigma * sigma) {
          continue;                     // beyond 3 sigma the update is noise
        }
        const double h = std::exp(-grid_distance2 / (2.0 * sigma * sigma));
        weights.row(j) += static_cast<float>(alpha * h) * (sample - weights.row(j));
      }
    }
  }
  const double seconds = (static_cast<double>(cv::getTickCount()) - tick) /
    cv::getTickFrequency();
  std::cout << "Trained in " << std::fixed << std::setprecision(1) << seconds
            << " s (" << Config::EPOCHS << " epochs)" << std::endl;

  // ========================================
  // Labelling the map
  // ========================================
  // The SOM is UNSUPERVISED: it never saw a label. Labels are used only now,
  // to read the result: each neuron takes the majority digit among the
  // training samples that chose it as their winner
  cv::Mat votes = cv::Mat::zeros(num_neurons, Config::NUM_CLASSES, CV_32S);
  double quantization_error = 0.0;
  for (int i = 0; i < train_samples.rows; ++i) {
    double distance = 0.0;
    const int winner = bestMatchingUnit(weights, train_samples.row(i), distance);
    votes.at<int>(winner, train_labels.at<int>(i))++;
    quantization_error += distance;
  }
  quantization_error /= train_samples.rows;

  std::vector<int> neuron_label(num_neurons, -1);
  for (int j = 0; j < num_neurons; ++j) {
    double best = 0.0;
    cv::Point where;
    cv::minMaxLoc(votes.row(j), nullptr, &best, nullptr, &where);
    neuron_label[j] = (best > 0) ? where.x : -1;   // -1: nobody chose it
  }

  std::cout << "\nMap of labels (- is a neuron nobody voted for):" << std::endl;
  for (int r = 0; r < Config::MAP_ROWS; ++r) {
    std::cout << "  ";
    for (int c = 0; c < Config::MAP_COLS; ++c) {
      const int label = neuron_label[r * Config::MAP_COLS + c];
      std::cout << (label < 0 ? "-" : std::to_string(label)) << " ";
    }
    std::cout << std::endl;
  }

  // ========================================
  // What the map is worth
  // ========================================
  // Used as a classifier: a test sample takes the label of its winner. It is
  // a 1-NN against 100 prototypes instead of against 2500 samples
  int correct = 0;
  for (int i = 0; i < test_samples.rows; ++i) {
    double distance = 0.0;
    const int winner = bestMatchingUnit(weights, test_samples.row(i), distance);
    if (neuron_label[winner] == test_labels.at<int>(i)) {
      ++correct;
    }
  }
  const double accuracy = 100.0 * correct / test_samples.rows;

  // Topology: how often two neurons side by side carry different digits, and
  // which pair of digits does it most. Frontiers are unavoidable (ten classes
  // on a square grid), but they should fall between digits that LOOK alike
  int borders = 0, adjacencies = 0;
  cv::Mat confusable = cv::Mat::zeros(Config::NUM_CLASSES, Config::NUM_CLASSES, CV_32S);
  for (int r = 0; r < Config::MAP_ROWS; ++r) {
    for (int c = 0; c < Config::MAP_COLS; ++c) {
      const int here = neuron_label[r * Config::MAP_COLS + c];
      if (here < 0) {
        continue;
      }
      const int neighbours[2][2] = {{r, c + 1}, {r + 1, c}};
      for (const auto & n : neighbours) {
        if (n[0] >= Config::MAP_ROWS || n[1] >= Config::MAP_COLS) {
          continue;
        }
        const int there = neuron_label[n[0] * Config::MAP_COLS + n[1]];
        if (there < 0) {
          continue;
        }
        ++adjacencies;
        if (here != there) {
          ++borders;
          confusable.at<int>(std::min(here, there), std::max(here, there))++;
        }
      }
    }
  }
  double most = 0.0;
  cv::Point pair;
  cv::minMaxLoc(confusable, nullptr, &most, nullptr, &pair);

  std::cout << "\nMean quantization error: " << std::setprecision(3)
            << quantization_error << std::endl;
  std::cout << "Accuracy on the TEST set (label of the winning neuron): "
            << std::setprecision(2) << accuracy << " %" << std::endl;
  std::cout << "Neighbouring neurons with different digits: " << borders << " of "
            << adjacencies << std::endl;
  std::cout << "The digits that touch the most on the map: " << pair.y << " and "
            << pair.x << " (" << static_cast<int>(most) << " frontiers)" << std::endl;

  // ========================================
  // The map, drawn
  // ========================================
  // Every codebook vector is 400 numbers, which is exactly a 20x20 image: the
  // prototype the neuron has learned. Putting them in their grid position
  // turns the weights into something readable
  cv::Mat atlas(Config::MAP_ROWS * Config::CELL_SIZE,
    Config::MAP_COLS * Config::CELL_SIZE, CV_8U);
  for (int j = 0; j < num_neurons; ++j) {
    cv::Mat prototype = weights.row(j).clone().reshape(1, Config::CELL_SIZE);
    prototype.convertTo(prototype, CV_8U, 255.0);
    const cv::Rect cell((j % Config::MAP_COLS) * Config::CELL_SIZE,
      (j / Config::MAP_COLS) * Config::CELL_SIZE,
      Config::CELL_SIZE, Config::CELL_SIZE);
    prototype.copyTo(atlas(cell));
  }
  cv::Mat display;
  cv::resize(atlas, display, cv::Size(), Config::ZOOM, Config::ZOOM,
    cv::INTER_NEAREST);
  cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);
  for (int j = 0; j < num_neurons; ++j) {
    if (neuron_label[j] < 0) {
      continue;
    }
    const cv::Point corner((j % Config::MAP_COLS) * Config::CELL_SIZE * Config::ZOOM + 4,
      (j / Config::MAP_COLS + 1) * Config::CELL_SIZE * Config::ZOOM - 4);
    cv::putText(display, std::to_string(neuron_label[j]), corner,
      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 200, 255), 1, cv::LINE_AA);
  }
  for (int k = 1; k < Config::MAP_COLS; ++k) {
    const int x = k * Config::CELL_SIZE * Config::ZOOM;
    cv::line(display, cv::Point(x, 0), cv::Point(x, display.rows),
      cv::Scalar(60, 60, 60), 1);
  }
  for (int k = 1; k < Config::MAP_ROWS; ++k) {
    const int y = k * Config::CELL_SIZE * Config::ZOOM;
    cv::line(display, cv::Point(0, y), cv::Point(display.cols, y),
      cv::Scalar(60, 60, 60), 1);
  }

  cv::imshow("SOM: prototype of every neuron (its label in orange)", display);
  std::cout << "\nEach cell is the codebook of one neuron drawn as an image."
    "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);
  return EXIT_SUCCESS;
}
