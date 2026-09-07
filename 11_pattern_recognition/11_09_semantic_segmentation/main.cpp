/**
 * @file main.cpp
 * @brief Semantic segmentation with DeepLabV3 (one class label per pixel)
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - cv::dnn::readNetFromONNX() with a segmentation network, whose output is
 *   not a list of boxes but a score map: 21 x H x W
 * - The preprocessing a torchvision model expects, normalisation included,
 *   which blobFromImage cannot do on its own
 * - Turning the score volume into a label image with an argmax per pixel
 * - The PASCAL VOC colour palette, and blending the mask with the photo
 *
 * Detection (11_07, 11_08) answers "what is there and roughly where".
 * Semantic segmentation answers the same question PIXEL BY PIXEL: every
 * pixel gets the label of one of the 21 classes, background included.
 *
 * What it does NOT do is tell apart two objects of the same class: three
 * people side by side come out as a single "person" region, because the
 * network has no notion of instance. Separating them is instance
 * segmentation, which needs another kind of model (Mask R-CNN, YOLO-seg).
 *
 * The model is DeepLabV3 with a MobileNetV3 backbone, exported from
 * torchvision by export_model.py. Run that script once (or let CMake run it)
 * before this example.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace Config
{
const char * MODEL = "../../data/models/deeplabv3/deeplabv3_mobilenetv3.onnx";
const char * CLASSES = "../../data/models/deeplabv3/voc.names";
constexpr int INPUT_SIZE = 384;      // The size the model was exported with
// ImageNet statistics: every torchvision model was trained on inputs
// normalised with these, and feeding it anything else quietly ruins the result
const cv::Scalar MEAN(0.485, 0.456, 0.406);
const cv::Scalar STD(0.229, 0.224, 0.225);
constexpr double OVERLAY = 0.55;     // Weight of the mask over the photo
constexpr double MIN_SHARE = 0.5;    // Classes below this % are not listed
const char * WINDOW_NAME = "Semantic segmentation (DeepLabV3)";
}

/**
 * @brief Reads the class names, one per line
 */
std::vector<std::string> loadClassNames(const std::string & path)
{
  std::vector<std::string> names;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty()) {
      names.push_back(line);
    }
  }
  return names;
}

/**
 * @brief The PASCAL VOC palette, generated the way the dataset defines it
 *
 * The colour of class i comes from interleaving the bits of i, which is why
 * consecutive classes get very different colours. Hardcoding the 21 triplets
 * would work too, but this is the actual rule.
 */
std::vector<cv::Vec3b> vocPalette(int num_classes)
{
  std::vector<cv::Vec3b> palette(num_classes);
  for (int i = 0; i < num_classes; ++i) {
    int id = i, r = 0, g = 0, b = 0;
    for (int shift = 7; shift >= 0; --shift) {
      r |= ((id >> 0) & 1) << shift;
      g |= ((id >> 1) & 1) << shift;
      b |= ((id >> 2) & 1) << shift;
      id >>= 3;
    }
    palette[i] = cv::Vec3b(static_cast<uchar>(b), static_cast<uchar>(g),
      static_cast<uchar>(r));                      // BGR, as OpenCV stores it
  }
  return palette;
}

/**
 * @brief Builds the input blob with the normalisation torchvision expects
 *
 * blobFromImage computes scalefactor * (pixel - mean), which covers the
 * division by 255 and the subtraction of the mean, but it has no per-channel
 * standard deviation. That last division has to be done by hand over the
 * planes of the blob, and forgetting it is the classic reason why a model
 * ported from Python returns nonsense in C++.
 */
cv::Mat makeBlob(const cv::Mat & frame)
{
  cv::Mat blob = cv::dnn::blobFromImage(
    frame, 1.0 / 255.0, cv::Size(Config::INPUT_SIZE, Config::INPUT_SIZE),
    Config::MEAN * 255.0, /*swapRB=*/ true, /*crop=*/ false);

  // NCHW layout: plane c holds INPUT_SIZE * INPUT_SIZE consecutive floats
  for (int c = 0; c < 3; ++c) {
    cv::Mat plane(Config::INPUT_SIZE, Config::INPUT_SIZE, CV_32F,
      blob.ptr<float>(0, c));
    plane /= Config::STD[c];
  }
  return blob;
}

/**
 * @brief Label of every pixel: the class with the highest score
 * @param scores Network output, 1 x C x H x W
 * @return CV_8U image of H x W with the class index
 *
 * Done channel by channel with matrix operations instead of a triple loop:
 * keep the best score so far and, wherever the new channel beats it, write
 * the new score and the new label. C passes over the image, no branches.
 */
cv::Mat argmaxPerPixel(const cv::Mat & scores, int & num_classes)
{
  num_classes = scores.size[1];
  const int height = scores.size[2], width = scores.size[3];

  cv::Mat best(height, width, CV_32F, const_cast<float *>(scores.ptr<float>(0, 0)));
  best = best.clone();
  cv::Mat labels = cv::Mat::zeros(height, width, CV_8U);

  for (int c = 1; c < num_classes; ++c) {
    const cv::Mat channel(height, width, CV_32F,
      const_cast<float *>(scores.ptr<float>(0, c)));
    cv::Mat better;
    cv::compare(channel, best, better, cv::CMP_GT);
    channel.copyTo(best, better);
    labels.setTo(c, better);
  }
  return labels;
}

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h | | Show this help message}"
    "{@input | ../../data/messi5.jpg | Input image or video}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  const std::vector<std::string> classes = loadClassNames(Config::CLASSES);
  if (classes.empty()) {
    std::cerr << "Error: cannot read " << Config::CLASSES << std::endl;
    std::cerr << "Export the model first with: python3 export_model.py" << std::endl;
    return EXIT_FAILURE;
  }

  cv::dnn::Net net;
  try {
    net = cv::dnn::readNetFromONNX(Config::MODEL);
  } catch (const cv::Exception &) {
    const bool exists = std::ifstream(Config::MODEL).good();
    std::cerr << "Error: cannot load " << Config::MODEL << std::endl;
    std::cerr << (exists ?
      "The file is there, so the problem is the OpenCV version reading it: " +
      cv::getVersionString() :
      "Export it first with: python3 export_model.py") << std::endl;
    return EXIT_FAILURE;
  }
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  const std::string input_path = parser.get<std::string>("@input");
  cv::VideoCapture cap(cv::samples::findFile(input_path, false));
  if (!cap.isOpened()) {
    std::cerr << "Error: cannot open " << input_path << std::endl;
    return EXIT_FAILURE;
  }

  const std::vector<cv::Vec3b> palette = vocPalette(static_cast<int>(classes.size()));
  std::cout << "=== Semantic segmentation with DeepLabV3 ===" << std::endl;
  std::cout << "Input: " << input_path << "   Classes: " << classes.size()
            << std::endl;
  std::cout << "Press 'q' or ESC to exit" << std::endl;

  cv::Mat frame;
  while (cap.read(frame)) {
    // ---- Inference -----------------------------------------------------
    net.setInput(makeBlob(frame));
    const double tick = static_cast<double>(cv::getTickCount());
    const cv::Mat scores = net.forward();
    const double ms = (static_cast<double>(cv::getTickCount()) - tick) /
      cv::getTickFrequency() * 1000.0;

    // ---- From scores to labels, and from labels to colour --------------
    int num_classes = 0;
    cv::Mat labels = argmaxPerPixel(scores, num_classes);

    // The network worked at INPUT_SIZE: the label image has to go back to
    // the size of the frame, and NEAREST is compulsory because interpolating
    // between class 7 and class 15 would invent class 11
    cv::resize(labels, labels, frame.size(), 0, 0, cv::INTER_NEAREST);

    cv::Mat coloured(frame.size(), CV_8UC3);
    for (int y = 0; y < labels.rows; ++y) {
      const uchar * row = labels.ptr<uchar>(y);
      cv::Vec3b * out = coloured.ptr<cv::Vec3b>(y);
      for (int x = 0; x < labels.cols; ++x) {
        out[x] = palette[row[x]];
      }
    }

    cv::Mat blended;
    cv::addWeighted(coloured, Config::OVERLAY, frame, 1.0 - Config::OVERLAY, 0.0,
      blended);

    // ---- What came out, in numbers -------------------------------------
    // The share of the image each class takes. Background included: it is a
    // class like any other for the network, and usually the largest one
    const double total = static_cast<double>(labels.total());
    std::cout << "\nInference: " << std::fixed << std::setprecision(0) << ms
              << " ms" << std::endl;
    int legend_line = 0;
    for (int c = 0; c < num_classes; ++c) {
      const double share = 100.0 * cv::countNonZero(labels == c) / total;
      if (share < Config::MIN_SHARE) {
        continue;
      }
      std::cout << "  " << std::setw(12) << std::left << classes[c] << " "
                << std::setprecision(1) << share << " %" << std::endl;
      // Legend on the image, in the colour the mask uses for that class
      const cv::Scalar colour(palette[c][0], palette[c][1], palette[c][2]);
      cv::rectangle(blended, cv::Rect(10, 10 + 22 * legend_line, 16, 16), colour,
        cv::FILLED);
      cv::putText(blended, classes[c], cv::Point(32, 24 + 22 * legend_line),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
      ++legend_line;
    }

    cv::imshow(Config::WINDOW_NAME, blended);
    cv::imshow("Label map", coloured);

    const int key = cv::waitKey(cap.get(cv::CAP_PROP_FRAME_COUNT) > 1 ? 1 : 0);
    if (key == 'q' || key == 27) {
      break;
    }
  }

  std::cout << "\nEvery pixel carries one label, so two objects of the same "
    "class\nmelt into a single region: that is the line between semantic and "
    "instance\nsegmentation." << std::endl;
  return EXIT_SUCCESS;
}
