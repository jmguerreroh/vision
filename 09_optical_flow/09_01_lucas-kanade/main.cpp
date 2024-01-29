/**
 * Optical flow using Lucas-Kanade demo sample
 * @author José Miguel Guerrero
 *
 * https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
 *
 * OpenCV provides all these in a single function, cv.calcOpticalFlowPyrLK(). Here, we create a simple application which tracks some points in a video.
 * To decide the points, we use cv.goodFeaturesToTrack(). We take the first frame, detect some Shi-Tomasi corner points in it, then we iteratively track
 * those points using Lucas-Kanade optical flow. For the function cv.calcOpticalFlowPyrLK() we pass the previous frame, previous points and next frame.
 * It returns next points along with some status numbers which has a value of 1 if next point is found, else zero. We iteratively pass these next points
 * as previous points in next step. See the code below:
 */

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  const string about =
    "This sample demonstrates Lucas-Kanade Optical Flow calculation.\n";
  const string keys =
    "{ h help |      | print this help message }"
    "{ @image | vtest.avi | path to image file }";
  CommandLineParser parser(argc, argv, keys);
  parser.about(about);
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  string filename = samples::findFile(parser.get<string>("@image"));
  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  VideoCapture capture(filename);
  if (!capture.isOpened()) {
    //error in opening the video input
    cerr << "Unable to open file!" << endl;
    return 0;
  }

  // Create some random colors
  vector<Scalar> colors;
  RNG rng;
  for (int i = 0; i < 100; i++) {
    int r = rng.uniform(0, 256);
    int g = rng.uniform(0, 256);
    int b = rng.uniform(0, 256);
    colors.push_back(Scalar(r, g, b));
  }

  Mat old_frame, old_gray;
  vector<Point2f> p0, p1;

  // Take first frame and find corners in it
  capture >> old_frame;
  cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
  goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

  // Create a mask image for drawing purposes
  Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

  while (true) {
    Mat frame, frame_gray;

    capture >> frame;
    if (frame.empty()) {
      break;
    }
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    // calculate optical flow
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) +(TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);

    vector<Point2f> good_new;
    for (uint i = 0; i < p0.size(); i++) {
      // Select good points
      if (status[i] == 1) {
        good_new.push_back(p1[i]);
        // draw the tracks
        line(mask, p1[i], p0[i], colors[i], 2);
        circle(frame, p1[i], 5, colors[i], -1);
      }
    }
    Mat img;
    add(frame, mask, img);

    imshow("Frame", img);

    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27) {
      break;
    }

    // Now update the previous frame and previous points
    old_gray = frame_gray.clone();
    p0 = good_new;
  }
}
