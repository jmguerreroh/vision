/**
 * KNN demo sample
 * @author José Miguel Guerrero
 */

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>
#include <vector>
#include <iostream>

// Define color constants
const cv::Scalar WHITE_COLOR = cv::Scalar(255, 255, 255);
const std::string winName = "points";
const int testStep = 5;

// Declare global variables
cv::Mat img, imgDst;
std::vector<cv::Point> trainedPoints;
std::vector<int> trainedPointsMarkers;
const int MAX_CLASSES = 2;
std::vector<cv::Vec3b> classColors(MAX_CLASSES);
int currentClass = 0;
std::vector<int> classCounters(MAX_CLASSES);

// Mouse callback function to record training points
static void on_mouse(int event, int x, int y, int /*flags*/, void *)
{
  if (img.empty()) {
    return;
  }

  int updateFlag = 0;

  // Record point position with left mouse button
  if (event == cv::EVENT_LBUTTONUP) {
    trainedPoints.push_back(cv::Point(x, y));
    trainedPointsMarkers.push_back(currentClass);
    classCounters[currentClass]++;
    updateFlag = true;
  }

  // Redraw the image with updated points
  if (updateFlag) {
    img = cv::Scalar::all(0);

    for (std::size_t i = 0; i < trainedPoints.size(); i++) {
      cv::Vec3b c = classColors[trainedPointsMarkers[i]];
      cv::circle(img, trainedPoints[i], 5, cv::Scalar(c), -1);
    }

    cv::imshow(winName, img);
  }
}

// Function to create and train a KNN model with given K value
static void KNN(int K)
{
  // Create KNN classifier
  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setDefaultK(K);
  knn->setIsClassifier(true);

  // Prepare training data
  cv::Mat samples;
  cv::Mat(trainedPoints).reshape(1, (int)trainedPoints.size()).convertTo(samples, CV_32F);
  cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE,
    cv::Mat(trainedPointsMarkers));

  // Train the KNN model
  knn->train(train_data);

  // Predict classes for each pixel in the image
  cv::Mat testSample(1, 2, CV_32FC1);
  for (int y = 0; y < img.rows; y += testStep) {
    for (int x = 0; x < img.cols; x += testStep) {
      testSample.at<float>(0) = (float)x;
      testSample.at<float>(1) = (float)y;
      int response = (int)knn->predict(testSample);
      imgDst.at<cv::Vec3b>(y, x) = classColors[response];
    }
  }
}

int main()
{
  std::cout   << "Use:\n"
              << "  key 'Esc' - exit the program;\n"
              << "  key '0' .. '1' - switch to class #n\n"
              << "  left mouse button - to add new point;\n"
              << "  key 'r' - to run the ML model;\n"
              << "  key 'i' - to init (clear) the data." << std::endl;

  // Create window and initialize images
  cv::namedWindow("points", 1);
  img.create(480, 640, CV_8UC3);
  imgDst.create(480, 640, CV_8UC3);

  cv::imshow("points", img);
  cv::setMouseCallback("points", on_mouse);

  // Define colors for the classes
  classColors[0] = cv::Vec3b(0, 255, 0);
  classColors[1] = cv::Vec3b(0, 0, 255);

  // Main loop for user interaction
  bool finish = false;
  while (!finish) {
    char key = (char)cv::waitKey();

    if (key == 27) { // Exit on 'Esc' key
      finish = true;
    }

    if (key == 'i') { // Reset data
      img = cv::Scalar::all(0);
      trainedPoints.clear();
      trainedPointsMarkers.clear();
      classCounters.assign(MAX_CLASSES, 0);
      cv::imshow(winName, img);
    }

    if (key == '0' || key == '1') { // Switch class
      currentClass = key - '0';
    }

    if (key == 'r') { // Run KNN classification
      double minVal = 0;
      cv::minMaxLoc(classCounters, &minVal, 0, 0, 0);
      if (minVal == 0) {
        printf("Each class should have at least 1 point\n");
        continue;
      }
      img.copyTo(imgDst);

      KNN(3);
      cv::imshow("kNN 3", imgDst);

      KNN(15);
      cv::imshow("kNN 15", imgDst);
    }
  }

  return 0;
}
