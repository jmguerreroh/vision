/**
 * @file main.cpp
 * @brief YOLO11 (Ultralytics) object detection using OpenCV DNN with ONNX model
 * @author José Miguel Guerrero Hernández
 *
 * @details Detects objects in images or video using a YOLO11 model exported to
 *          ONNX format from Ultralytics. Uses OpenCV DNN module for inference.
 *
 *          IMPORTANT: Requires OpenCV >= 4.9 for ONNX model loading.
 *
 *          YOLO11 ONNX output format:
 *          - Shape: [1, 84, 8400] for 80 COCO classes
 *            - 84 = 4 (cx, cy, w, h) + 80 (class scores)
 *            - 8400 = total number of candidate detections
 *          - Box coordinates are in pixel space relative to input size (640x640)
 *          - No objectness score (unlike YOLOv3/v4/v5)
 *
 *          Pipeline:
 *          1. Load YOLO11 ONNX model
 *          2. Create blob from input (resize + normalize)
 *          3. Forward pass → output [1, 84, 8400]
 *          4. Transpose to [8400, 84] for easier parsing
 *          5. Filter by confidence, rescale boxes, apply NMS
 *          6. Draw bounding boxes
 *
 *          Usage:
 *            ./yolo11                              (default: ../../data/vtest.avi)
 *            ./yolo11 ../../data/fruits.jpg
 *
 *          Export model (requires Python + ultralytics):
 *            python3 export_model.py
 *
 * @see https://docs.ultralytics.com/models/yolo11/
 */

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Detection thresholds and network input size
namespace Config
{
constexpr float CONF_THRESHOLD = 0.25f;  // Minimum confidence to keep a detection
constexpr float NMS_THRESHOLD = 0.45f;   // Non-Maximum Suppression IoU threshold
constexpr int INPUT_SIZE = 640;          // YOLO11 default input size (square)
constexpr int NUM_CLASSES = 80;          // COCO dataset classes
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
 * @brief Post-process YOLO11 ONNX output
 *
 *        YOLO11 output shape is [1, 84, 8400]:
 *        - Rows 0-3: cx, cy, w, h (pixel coordinates in input size space)
 *        - Rows 4-83: class confidence scores (no objectness)
 *
 *        We transpose to [8400, 84] so each row = one detection.
 *
 * @param frame Original image (for scaling boxes)
 * @param output Raw network output (single Mat, shape [1, 84, 8400])
 * @param classes Vector of class names
 * @param x_scale Scale factor from input size to original width
 * @param y_scale Scale factor from input size to original height
 */
void postprocess(
  cv::Mat & frame,
  const cv::Mat & output,
  const std::vector<std::string> & classes,
  float x_scale, float y_scale)
{
  // Reshape from [1, 84, 8400] to [84, 8400], then transpose to [8400, 84]
  cv::Mat det = output.reshape(1, output.size[1]);  // [84, 8400]
  cv::transpose(det, det);                           // [8400, 84]

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < det.rows; ++i) {
    // Columns 4..83 are class scores (no objectness in YOLO11)
    cv::Mat scores = det.row(i).colRange(4, 4 + Config::NUM_CLASSES);
    cv::Point max_loc;
    double max_conf;
    cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);

    if (max_conf > Config::CONF_THRESHOLD) {
      // Extract box: cx, cy, w, h (in 640x640 pixel space)
      float cx = det.at<float>(i, 0);
      float cy = det.at<float>(i, 1);
      float w = det.at<float>(i, 2);
      float h = det.at<float>(i, 3);

      // Scale back to original image dimensions
      int left = static_cast<int>((cx - w / 2.0f) * x_scale);
      int top = static_cast<int>((cy - h / 2.0f) * y_scale);
      int width = static_cast<int>(w * x_scale);
      int height = static_cast<int>(h * y_scale);

      class_ids.push_back(max_loc.x);
      confidences.push_back(static_cast<float>(max_conf));
      boxes.emplace_back(left, top, width, height);
    }
  }

  // Apply Non-Maximum Suppression
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, Config::CONF_THRESHOLD,
                    Config::NMS_THRESHOLD, indices);

  // Draw surviving detections with per-class colors
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
  parser.about("YOLO11 (Ultralytics) object detection via ONNX");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  std::string input_path = parser.get<std::string>("@input");

  //--- Load class names and ONNX model ------------------------------------
  std::vector<std::string> classes = loadClassNames("../../data/models/yolo11/coco.names");
  if (classes.empty()) {
    std::cerr << "Error: cannot load ../../data/models/yolo11/coco.names" << std::endl;
    return EXIT_FAILURE;
  }

  cv::dnn::Net net = cv::dnn::readNetFromONNX("../../data/models/yolo11/yolo11n.onnx");
  if (net.empty()) {
    std::cerr << "Error: cannot load cfg/yolo11n.onnx" << std::endl;
    std::cerr << "Run: python3 export_model.py" << std::endl;
    return EXIT_FAILURE;
  }

  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  //--- Open input source --------------------------------------------------
  cv::VideoCapture cap(input_path);
  if (!cap.isOpened()) {
    std::cerr << "Error: cannot open " << input_path << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== YOLO11 (Ultralytics) Object Detection ===" << std::endl;
  std::cout << "Input: " << input_path << std::endl;
  std::cout << "Model: yolo11n.onnx (nano)" << std::endl;
  std::cout << "Classes: " << classes.size() << std::endl;
  std::cout << "Press 'q' or ESC to exit" << std::endl;

  //--- Main processing loop -----------------------------------------------
  cv::Mat frame;
  while (true) {
    cap >> frame;
    if (frame.empty()) {
      break;
    }

    // Scale factors to map detections from input size back to original image
    float x_scale = static_cast<float>(frame.cols) / Config::INPUT_SIZE;
    float y_scale = static_cast<float>(frame.rows) / Config::INPUT_SIZE;

    // Create blob: normalize [0,1], resize to 640x640, swap R↔B (BGR→RGB)
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0,
                           cv::Size(Config::INPUT_SIZE, Config::INPUT_SIZE),
                           cv::Scalar(), true, false);

    // Forward pass → output shape [1, 84, 8400]
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Post-process: transpose, filter, NMS, draw
    postprocess(frame, outputs[0], classes, x_scale, y_scale);

    // Show inference time
    std::vector<double> layer_times;
    double t = net.getPerfProfile(layer_times) /
      (cv::getTickFrequency() / 1000.0);
    cv::putText(frame, cv::format("YOLO11n: %.1f ms", t), cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    cv::imshow("YOLO11 (Ultralytics)", frame);

    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
