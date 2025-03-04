/**
 * Moments - sample code
 * @author José Miguel Guerrero
 *
 * https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

// Show all value moments
void show_moment(Moments m)
{
  cout << "Spatial moments:" << endl;
  cout << "  m00 = " << m.m00 << ", m10 = " << m.m10 << ", m01 = " << m.m01 << ", m20 = " <<
    m.m20 << ", m11 = " << m.m11 << endl;
  cout << "  m02 = " << m.m02 << ", m30 = " << m.m30 << ", m21 = " << m.m21 << ", m12 = " <<
    m.m12 << ", m03 = " << m.m03 << endl;

  cout << "Central moments:" << endl;
  cout << "  mu20 = " << m.mu20 << ", mu11 = " << m.mu11 << ", mu02 = " << m.mu02 << ", mu30 = " <<
    m.mu30 << ", mu21 = " << m.mu21 << endl;
  cout << "  mu12 = " << m.mu12 << ", mu03 = " << m.mu03 << endl;

  cout << "Central normalized moments:" << endl;
  cout << "  nu20 = " << m.nu20 << ", nu11 = " << m.nu11 << ", nu02 = " << m.nu02 << ", nu30 = " <<
    m.nu30 << ", nu21 = " << m.nu21 << endl;
  cout << "  nu12 = " << m.nu12 << ", nu03 = " << m.nu03 << endl;
}

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
  // iterate through all the top-level contours,
  // draw each connected component with its own random color
  // If for the contour i there are no next, previous, parent, or nested contours, the corresponding elements of hierarchy[i] will be negative.
  int idx = 0;
  while (idx >= 0) {
    Scalar color(rand() & 255, rand() & 255, rand() & 255);
    drawContours(drawing, contours, idx, color, FILLED, 8, hierarchy, 1);       // Last value navigates into the hierarchy
    idx = hierarchy[idx][0];
  }
  imshow("Contours", drawing);

  // Calculate moments
  vector<Moments> mu(contours.size() );
  for (size_t i = 0; i < contours.size(); i++) {
    mu[i] = moments(contours[i]);
  }

  // Show moments
  for (int i = 0; i < contours.size(); i++) {
    cout << "*****************************" << endl;
    cout << "       Contour[" << i << "]" << endl;
    cout << "*****************************" << endl;
    show_moment(mu[i]);
  }

  // Use contourArea and arcLength fucntions to calculate Area and Perimeter length using OpenCV
  cout << "\t Info: Area and Contour Length \n";
  for (size_t i = 0; i < contours.size(); i++) {
    cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) <<
      mu[i].m00
         << " - Area OpenCV: " << contourArea(contours[i]) << " - Length: " << arcLength(
      contours[i], true) << endl;
  }

  // Wait to press a key
  waitKey(0);

  return 0;
}
