/**
 * Optical flow using Gunner Farneback's algorithm demo sample
 * @author José Miguel Guerrero
 *
 * https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
 *
 * Lucas-Kanade method computes optical flow for a sparse feature set (in our example, corners detected using Shi-Tomasi algorithm).
 * OpenCV provides another algorithm to find the dense optical flow. It computes the optical flow for all the points in the frame.
 * It is based on Gunner Farneback's algorithm which is explained in "Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunner Farneback in 2003.
 *
 * Below sample shows how to find the dense optical flow using above algorithm. We get a 2-channel array with optical flow vectors, (u,v).
 * We find their magnitude and direction. We color code the result for better visualization. Direction corresponds to Hue value of the image.
 * Magnitude corresponds to Value plane
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
  const string keys =
    "{ h help |      | print this help message }"
    "{ @image | vtest.avi | path to image file }";
  CommandLineParser parser(argc, argv, keys);

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

  Mat frame1, prvs;
  capture >> frame1;
  cvtColor(frame1, prvs, COLOR_BGR2GRAY);

  while (true) {
    Mat frame2, next;
    capture >> frame2;
    if (frame2.empty()) {
      break;
    }
    cvtColor(frame2, next, COLOR_BGR2GRAY);

    Mat flow(prvs.size(), CV_32FC2);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    // visualization
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    //build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);

    imshow("frame2", bgr);

    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27) {
      break;
    }

    prvs = next;
  }
}
