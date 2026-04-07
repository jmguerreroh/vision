/**
 * @file main.cpp
 * @brief Simple YOLO v4-tiny object detection using OpenCV DNN module
 * @author José Miguel Guerrero Hernández
 *
 * @details Simplified YOLO example that detects objects in images or video
 *          using YOLOv4-tiny with the OpenCV DNN module.
 *
 *          Pipeline:
 *          1. Load YOLOv4-tiny model (Darknet format)
 *          2. Create blob from input image/frame
 *          3. Forward pass through the network
 *          4. Filter detections by confidence threshold
 *          5. Apply Non-Maximum Suppression (NMS)
 *          6. Draw bounding boxes with class labels
 *
 *          Usage:
 *            ./yolo_simple                         (default: ../../data/vtest.avi)
 *            ./yolo_simple ../../data/fruits.jpg
 *            ./yolo_simple ../../data/vtest.avi
 *
 *          Download model:
 *            chmod +x download_model.sh && ./download_model.sh
 *
 * @see https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Detection thresholds and input dimensions
namespace Config
{
constexpr float CONF_THRESHOLD = 0.5f;   // Minimum confidence to keep a detection
constexpr float NMS_THRESHOLD = 0.4f;    // Non-Maximum Suppression threshold
constexpr int INPUT_WIDTH = 416;         // Network input width
constexpr int INPUT_HEIGHT = 416;        // Network input height
}  // namespace Config

/**
 * @brief Load class names from a text file (one class per line)
 * @param path Path to the class names file
 * @return Vector of class name strings
 */
std::vector<std::string> loadClassNames(const std::string & path)
{
  std::vector<std::string> names;
  std::ifstream ifs(path);
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty()) {
      names.push_back(line);
    }
  }
  return names;
}

/**
 * @brief Draw a labeled bounding box on the image
 * @param frame Image to draw on
 * @param label Text label (class name + confidence)
 * @param box Bounding box rectangle
 * @param color Box and label color
 */
void drawBox(
  cv::Mat & frame, const std::string & label,
  const cv::Rect & box, const cv::Scalar & color)
{
  cv::rectangle(frame, box, color, 2);

  int baseline = 0;
  cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                        0.5, 1, &baseline);
  int top = std::max(box.y, label_size.height);
  cv::rectangle(frame,
                cv::Point(box.x, top - label_size.height - 4),
                cv::Point(box.x + label_size.width, top + 2),
                color, cv::FILLED);
  cv::putText(frame, label, cv::Point(box.x, top - 2),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

/**
 * @brief Post-process network output: filter by confidence and apply NMS
 * @param frame Input image (used for scaling boxes to image size)
 * @param outputs Raw network outputs from net.forward()
 * @param classes Vector of class names
 */
void postprocess(
  cv::Mat & frame,
  const std::vector<cv::Mat> & outputs,
  const std::vector<std::string> & classes)
{
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  // Parse each detection from all output layers
  for (const auto & output : outputs) {
    const float * data = reinterpret_cast<const float *>(output.data);
    for (int i = 0; i < output.rows; ++i, data += output.cols) {
      // Columns 5..N contain class scores; first 5 are [cx, cy, w, h, objectness]
      cv::Mat scores = output.row(i).colRange(5, output.cols);
      cv::Point max_loc;
      double max_conf;
      cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);

      if (max_conf > Config::CONF_THRESHOLD) {
        // Convert from normalized [0,1] center coords to pixel coordinates
        int cx = static_cast<int>(data[0] * frame.cols);
        int cy = static_cast<int>(data[1] * frame.rows);
        int w = static_cast<int>(data[2] * frame.cols);
        int h = static_cast<int>(data[3] * frame.rows);

        class_ids.push_back(max_loc.x);
        confidences.push_back(static_cast<float>(max_conf));
        boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
      }
    }
  }

  // Apply Non-Maximum Suppression to remove overlapping detections
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, Config::CONF_THRESHOLD,
                    Config::NMS_THRESHOLD, indices);

  // Generate a consistent color per class
  for (int idx : indices) {
    const cv::Rect & box = boxes[idx];
    int cid = class_ids[idx];
    cv::Scalar color(
      (cid * 72) % 256,
      (cid * 49 + 111) % 256,
      (cid * 137 + 67) % 256);

    std::string label = (cid < static_cast<int>(classes.size()) ? classes[cid] : "?") +
      ": " + cv::format("%.0f%%", confidences[idx] * 100);
    drawBox(frame, label, box, color);
  }
}

int main(int argc, char ** argv)
{
  // Command-line parser
  const std::string keys =
    "{help h   |      | Print help message}"
    "{@input   | ../../data/vtest.avi | Path to input image or video}";

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Simple YOLO v4-tiny object detection example");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  std::string input_path = parser.get<std::string>("@input");

  //--- Load class names and network model ---------------------------------
  std::vector<std::string> classes = loadClassNames("../../data/models/yolov4/coco.names");
  if (classes.empty()) {
    std::cerr << "Error: cannot load ../../data/models/yolov4/coco.names" << std::endl;
    return EXIT_FAILURE;
  }
  cv::dnn::Net net = cv::dnn::readNetFromDarknet("../../data/models/yolov4/yolov4-tiny.cfg",
                                                  "../../data/models/yolov4/yolov4-tiny.weights");
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  // Get names of the output layers (YOLO has multiple output layers)
  std::vector<std::string> out_names = net.getUnconnectedOutLayersNames();

  //--- Open input source --------------------------------------------------
  cv::VideoCapture cap(input_path);
  if (!cap.isOpened()) {
    std::cerr << "Error: cannot open " << input_path << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== YOLO v4-tiny Object Detection ===" << std::endl;
  std::cout << "Input: " << input_path << std::endl;
  std::cout << "Classes loaded: " << classes.size() << std::endl;
  std::cout << "Press 'q' or ESC to exit" << std::endl;

  //--- Main processing loop -----------------------------------------------
  cv::Mat frame;
  while (true) {
    cap >> frame;
    if (frame.empty()) {
      break;
    }

    // Create 4D blob: normalize [0,1], resize to 416x416, swap R↔B
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0,
                           cv::Size(Config::INPUT_WIDTH, Config::INPUT_HEIGHT),
                           cv::Scalar(), true, false);

    // Forward pass
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, out_names);

    // Post-process: filter + NMS + draw
    postprocess(frame, outputs, classes);

    // Show inference time
    std::vector<double> layer_times;
    double t = net.getPerfProfile(layer_times) /
      (cv::getTickFrequency() / 1000.0);
    cv::putText(frame, cv::format("%.1f ms", t), cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    cv::imshow("YOLO v4-tiny", frame);

    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
