/**
 * Contours - sample code
 * @author José Miguel Guerrero
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
  // Create image variables
  Mat image, gray, edges, gauss;

  // Read image
  image = imread("../../data/coins.jpg", IMREAD_COLOR);

  // Convert image to grayscale
  cvtColor(image, gray, COLOR_BGR2GRAY);

  // Gaussian blur
  GaussianBlur(gray, gauss, Size(5, 5), 0);
  imshow("Gaussian Blur", gauss);

  // Image processing
  Canny(gauss, edges, 50, 100, 3);

  // Contours
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
  cout << contours.size() << " contours found." << endl;

  // Drawing contours
  Mat drawing = Mat::zeros(edges.size(), CV_8UC3);
  drawContours(drawing, contours, -1, (0, 0, 255), 2, LINE_8, hierarchy, 1);
  imshow("Contours", drawing);

  // Wait to press a key
  waitKey(0);

  return 0;
}
