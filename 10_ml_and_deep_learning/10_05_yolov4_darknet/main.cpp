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
 *            ./yolov4 (default: ../../data/vtest.avi)
 *            ./yolov4 ../../data/fruits.jpg
 *            ./yolov4 ../../data/vtest.avi
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
// Minimum objectness * class-score to keep a detection.
// Detections below this value are discarded before NMS.
constexpr float CONF_THRESHOLD = 0.5f;

// IoU threshold for Non-Maximum Suppression.
// If two boxes overlap by more than this fraction, the one with
// lower confidence is suppressed. Lower values remove more boxes.
constexpr float NMS_THRESHOLD = 0.4f;

// Spatial resolution of the network input tensor.
// YOLOv4-tiny was trained with 416x416; changing this affects accuracy
// and speed (larger = more accurate but slower).
constexpr int INPUT_WIDTH = 416;
constexpr int INPUT_HEIGHT = 416;
}  // namespace Config

/**
 * @brief Load class names from a text file (one class per line).
 *
 * COCO class names (coco.names) list 80 object categories, one per line.
 * The line index matches the class index returned by the network.
 *
 * @param path  Path to the .names file
 * @return      Vector of class name strings ordered by class index
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
 * @brief Draw a labeled bounding box on the image.
 *
 * Draws:
 *   1. A colored rectangle around the detected object.
 *   2. A filled rectangle as background for the text label.
 *   3. The label text (class name + confidence %) in white.
 *
 * The label background is clipped to stay inside the image when the box
 * is near the top edge (top = max(box.y, label_size.height)).
 *
 * @param frame   Image to draw on (modified in-place)
 * @param label   Text to display (e.g. "dog: 87%")
 * @param box     Bounding box in pixel coordinates
 * @param color   BGR color for the rectangle and label background
 */
void drawBox(
  cv::Mat & frame, const std::string & label,
  const cv::Rect & box, const cv::Scalar & color)
{
  cv::rectangle(frame, box, color, 2);  // Bounding box outline, thickness=2

  // Measure label text size to size the background rectangle
  int baseline = 0;
  cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                        0.5, 1, &baseline);
  int top = std::max(box.y, label_size.height);  // Prevent label from going off-screen
  cv::rectangle(frame,
                cv::Point(box.x, top - label_size.height - 4),
                cv::Point(box.x + label_size.width, top + 2),
                color, cv::FILLED);              // Filled background for readability
  cv::putText(frame, label, cv::Point(box.x, top - 2),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);  // White text
}

/**
 * @brief Post-process YOLO output: parse detections, filter by confidence, apply NMS.
 *
 * YOLOv4-tiny has two output layers (different scales: 13x13 and 26x26 grid
 * cells). Each layer produces a matrix of shape (N_anchors * grid_h * grid_w) x (5 + C)
 * where each row encodes one anchor-box prediction:
 *
 *   [cx, cy, w, h, objectness, score_class0, score_class1, ..., score_classC]
 *
 *   - cx, cy : box center, normalized to [0, 1] relative to the image size.
 *   - w,  h  : box dimensions, normalized to [0, 1].
 *   - objectness: P(object exists in this cell).
 *   - score_k : P(class k | object), already multiplied by objectness in YOLO v4.
 *
 * Steps:
 *   1. Scan every row of every output layer.
 *   2. Find the class with the highest score via minMaxLoc on columns 5..end.
 *   3. Keep the detection if max_score > CONF_THRESHOLD.
 *   4. Convert normalized coords to pixel coords and build a cv::Rect.
 *   5. Run NMSBoxes to suppress redundant overlapping boxes (IoU > NMS_THRESHOLD).
 *   6. Draw surviving detections with a per-class color.
 *
 * @param frame    Current video frame (used for coord scaling and drawing)
 * @param outputs  Raw output matrices from net.forward()
 * @param classes  Class name list (indexed by class id)
 */
void postprocess(
  cv::Mat & frame,
  const std::vector<cv::Mat> & outputs,
  const std::vector<std::string> & classes)
{
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  // ---------------------------------------------------------------------
  // Step 1-4: parse all anchor predictions from all output layers
  // ---------------------------------------------------------------------
  for (const auto & output : outputs) {
    const float * data = reinterpret_cast<const float *>(output.data);
    for (int i = 0; i < output.rows; ++i, data += output.cols) {
      // Columns 0-4: [cx, cy, w, h, objectness]
      // Columns 5..end: one score per class
      cv::Mat scores = output.row(i).colRange(5, output.cols);
      cv::Point max_loc;   // Column index = class id of the winning class
      double max_conf;
      cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);

      if (max_conf > Config::CONF_THRESHOLD) {
        // Scale normalized coordinates [0,1] back to pixel space
        int cx = static_cast<int>(data[0] * frame.cols);  // Box center x
        int cy = static_cast<int>(data[1] * frame.rows);  // Box center y
        int w = static_cast<int>(data[2] * frame.cols);   // Box width
        int h = static_cast<int>(data[3] * frame.rows);   // Box height

        class_ids.push_back(max_loc.x);
        confidences.push_back(static_cast<float>(max_conf));
        // cv::Rect expects top-left corner: (cx - w/2, cy - h/2)
        boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
      }
    }
  }

  // ---------------------------------------------------------------------
  // Step 5: Non-Maximum Suppression
  //
  // NMSBoxes suppresses boxes with IoU > NMS_THRESHOLD, keeping only the
  // highest-confidence box among overlapping candidates for the same object.
  // 'indices' holds the surviving box indices after suppression.
  // ---------------------------------------------------------------------
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, Config::CONF_THRESHOLD,
                    Config::NMS_THRESHOLD, indices);

  // ---------------------------------------------------------------------
  // Step 6: draw surviving detections
  //
  // Generate a visually distinct BGR color for each class id using a
  // deterministic hash so the same class always gets the same color.
  // ---------------------------------------------------------------------
  for (int idx : indices) {
    const cv::Rect & box = boxes[idx];
    int cid = class_ids[idx];
    cv::Scalar color(
      (cid * 72) % 256,
      (cid * 49 + 111) % 256,
      (cid * 137 + 67) % 256);

    // Format label as "class_name: XX%"
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

  // ---------------------------------------------------------------------
  // Load class names and network model
  // coco.names: 80 COCO object categories, one per line.
  // yolov4-tiny.cfg:     Darknet architecture definition.
  // yolov4-tiny.weights: Pre-trained weights (download via download_model.sh).
  // ---------------------------------------------------------------------
  std::vector<std::string> classes = loadClassNames("../../data/models/yolov4/coco.names");
  if (classes.empty()) {
    std::cerr << "Error: cannot load ../../data/models/yolov4/coco.names" << std::endl;
    return EXIT_FAILURE;
  }
  // readNetFromDarknet parses the .cfg file to build the layer graph and
  // loads the corresponding pre-trained weights from the .weights binary.
  cv::dnn::Net net = cv::dnn::readNetFromDarknet("../../data/models/yolov4/yolov4-tiny.cfg",
                                                  "../../data/models/yolov4/yolov4-tiny.weights");
  // DNN_BACKEND_OPENCV uses OpenCV's own optimized backend (no GPU required).
  // DNN_TARGET_CPU runs inference on the CPU; switch to DNN_TARGET_CUDA for GPU.
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  // Retrieve the names of unconnected output layers.
  // YOLOv4-tiny has two detection heads (at strides 32 and 16) so this
  // returns two layer names; both are needed to capture detections at
  // different scales (large and small objects).
  std::vector<std::string> out_names = net.getUnconnectedOutLayersNames();

  // Open input source
  cv::VideoCapture cap(input_path);
  if (!cap.isOpened()) {
    std::cerr << "Error: cannot open " << input_path << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== YOLO v4-tiny Object Detection ===" << std::endl;
  std::cout << "Input: " << input_path << std::endl;
  std::cout << "Classes loaded: " << classes.size() << std::endl;
  std::cout << "Press 'q' or ESC to exit" << std::endl;

  // Main processing loop
  cv::Mat frame;
  while (true) {
    cap >> frame;     // Grab the next frame (or image for single-image inputs)
    if (frame.empty()) {
      break;          // End of video or failed read
    }

    // ---------------------------------------------------------------------
    // Preprocessing: create the network input blob
    //
    // blobFromImage performs four operations in one call:
    //   1. Resize the frame to INPUT_WIDTH x INPUT_HEIGHT (416x416).
    //   2. Scale pixel values from [0, 255] to [0.0, 1.0] (scale = 1/255).
    //   3. Swap R and B channels (swapRB=true) because OpenCV uses BGR
    //      but the model was trained on RGB images.
    //   4. Pack the result into a 4D tensor: [batch=1, channels=3, H, W].
    // No mean subtraction is applied (Scalar() = zero mean).
    // crop=false means the image is stretched to fit, not center-cropped.
    // ---------------------------------------------------------------------
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0,
                           cv::Size(Config::INPUT_WIDTH, Config::INPUT_HEIGHT),
                           cv::Scalar(), true, false);

    // ---------------------------------------------------------------------
    // Forward pass
    //
    // setInput feeds the blob into the network's first layer.
    // forward() runs inference and collects the output tensors of the
    // unconnected layers (the two YOLO detection heads).
    // ---------------------------------------------------------------------
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, out_names);

    // ---------------------------------------------------------------------
    // Post-processing
    //
    // Parse raw predictions, apply confidence threshold, run NMS, draw boxes.
    // ---------------------------------------------------------------------
    postprocess(frame, outputs, classes);

    // ---------------------------------------------------------------------
    // Display inference time

    // getPerfProfile returns the total number of ticks spent in all layers;
    // dividing by (tickFrequency / 1000) converts it to milliseconds.
    // ---------------------------------------------------------------------
    std::vector<double> layer_times;
    double t = net.getPerfProfile(layer_times) /
      (cv::getTickFrequency() / 1000.0);
    cv::putText(frame, cv::format("%.1f ms", t), cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    cv::imshow("YOLO v4-tiny", frame);

    // waitKey(1) keeps the GUI responsive; 'q'/ESC stop the loop
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
