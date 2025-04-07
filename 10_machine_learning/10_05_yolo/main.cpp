// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html
// https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.cpp

// Usage example:  ./yolo --video=run.mp4
//                 ./yolo --image=bird.jpg

// Download weights mandatory: wget https://pjreddie.com/media/files/yolov3.weights

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char * keys =
  "{help h usage ? | | Usage examples: \n\t\t./yolo --image=dog.jpg \n\t\t./yolo --video=run.mp4}"
  "{image i        |<none>| input image   }"
  "{video v        |<none>| input video   }"
  "{device d       |<cpu>| input device   }"
;

// Initialize the parameters
float confThreshold = 0.5;  // Confidence threshold
float nmsThreshold = 0.4;   // Non-maximum suppression threshold
int inpWidth = 416;         // Width of network's input image
int inpHeight = 416;        // Height of network's input image
std::vector<std::string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat & frame, const std::vector<cv::Mat> & out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat & frame);

// Get the names of the output layers
std::vector<std::string> getOutputsNames(const cv::dnn::Net & net);

int main(int argc, char ** argv)
{
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  // Load names of classes
  std::string classesFile = "cfg/coco.names";
  std::ifstream ifs(classesFile.c_str());
  std::string line;
  while (std::getline(ifs, line)) {classes.push_back(line);}

  std::string device = "cpu";
  device = parser.get<std::string>("device");

  // Give the configuration and weight files for the model
  std::string modelConfiguration = "cfg/yolov3.cfg";
  std::string modelWeights = "cfg/yolov3.weights";

  // Load the network
  cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);

  if (device == "cpu") {
    std::cout << "Using CPU device" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
  } else if (device == "gpu") {
    std::cout << "Using GPU device" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  }

  // Open a video file or an image file or a camera stream.
  std::string str, outputFile;
  cv::VideoCapture cap;
  cv::VideoWriter video;
  cv::Mat frame, blob;

  try {
    outputFile = "yolo_out_cpp.avi";
    if (parser.has("image")) {
      // Open the image file
      str = parser.get<std::string>("image");
      std::ifstream ifile(str);
      if (!ifile) {throw("error");}
      cap.open(str);
      str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg");
      outputFile = str;
    } else if (parser.has("video")) {
      // Open the video file
      str = parser.get<std::string>("video");
      std::ifstream ifile(str);
      if (!ifile) {throw("error");}
      cap.open(str);
      str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
      outputFile = str;
    }
    // Open the webcam
    else {cap.open(parser.get<int>("device"));}

  } catch (...) {
    std::cout << "Could not open the input image/video stream" << std::endl;
    return 0;
  }

  // Get the video writer initialized to save the output video
  if (!parser.has("image")) {
    video.open(
      outputFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 28,
      cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
  }

  // Create a window
  static const std::string kWinName = "Deep learning object detection in OpenCV";
  cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

  // Process frames
  while (cv::waitKey(1) < 0) {
    // get frame from the video
    cap >> frame;

    // Stop the program if reached end of video
    if (frame.empty()) {
      std::cout << "Done processing !!!" << std::endl;
      std::cout << "Output file is stored as " << outputFile << std::endl;
      cv::waitKey(3000);
      break;
    }
    // Create a 4D blob from a frame
    cv::dnn::blobFromImage(
      frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(
        0, 0,
        0), true, false);

    // Sets the input to the network
    net.setInput(blob);

    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);

    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time for a frame : %.2f ms", t);
    putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

    // Write the frame with the detection boxes
    cv::Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);
    if (parser.has("image")) {cv::imwrite(outputFile, detectedFrame);} else {
      video.write(detectedFrame);
    }

    imshow(kWinName, frame);
  }

  cap.release();
  if (!parser.has("image")) {video.release();}

  return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat & frame, const std::vector<cv::Mat> & outs)
{
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (std::size_t i = 0; i < outs.size(); ++i) {
    // Scan through all the bounding boxes output from the network and keep only the
    // ones with high confidence scores. Assign the box's class label as the class
    // with the highest score for the box.
    float * data = (float *)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
      cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      cv::Point classIdPoint;
      double confidence;
      // Get the value and location of the maximum score
      cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confThreshold) {
        int centerX = (int)(data[0] * frame.cols);
        int centerY = (int)(data[1] * frame.rows);
        int width = (int)(data[2] * frame.cols);
        int height = (int)(data[3] * frame.rows);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

        classIds.push_back(classIdPoint.x);
        confidences.push_back((float)confidence);
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
  }

  // Perform non maximum suppression to eliminate redundant overlapping boxes with
  // lower confidences
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    int idx = indices[i];
    cv::Rect box = boxes[idx];
    drawPred(
      classIds[idx], confidences[idx], box.x, box.y,
      box.x + box.width, box.y + box.height, frame);
  }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat & frame)
{
  //Draw a rectangle displaying the bounding box
  cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

  //Get the label for the class name and its confidence
  std::string label = cv::format("%.2f", conf);
  if (!classes.empty()) {
    CV_Assert(classId < (int)classes.size());
    label = classes[classId] + ":" + label;
  }

  //Display the label at the top of the bounding box
  int baseLine;
  cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = std::max(top, labelSize.height);
  rectangle(
    frame,
    cv::Point(left, top - round(1.5 * labelSize.height)),
    cv::Point(left + round(1.5 * labelSize.width), top + baseLine),
    cv::Scalar(255, 255, 255), cv::FILLED);
  cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75,
    cv::Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
std::vector<std::string> getOutputsNames(const cv::dnn::Net & net)
{
  static std::vector<std::string> names;
  if (names.empty()) {
    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    std::vector<int> outLayers = net.getUnconnectedOutLayers();

    //get the names of all the layers in the network
    std::vector<std::string> layersNames = net.getLayerNames();

    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (std::size_t i = 0; i < outLayers.size(); ++i) {
      names[i] = layersNames[outLayers[i] - 1];
    }
  }
  return names;
}
