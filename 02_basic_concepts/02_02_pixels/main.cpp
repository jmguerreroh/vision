/**
 * Cycle through pixels - sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  // Load an image
  Mat src = imread("../../data/lena.jpg", IMREAD_COLOR);
  if (src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  // Show image
  namedWindow("Pixel Demo", WINDOW_AUTOSIZE);
  imshow("Pixel Demo", src);

  // ------------
  // Method 1:
  // Read pixel values using Vec3b: vector of 3 values
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      // You can now access the pixel value with cv::Vec3b
      cout <<
        (uint)src.at<Vec3b>(i, j)[0] << " " <<
        (uint)src.at<Vec3b>(i, j)[1] << " " <<
        (uint)src.at<Vec3b>(i, j)[2] << endl;
    }
  }

  // ------------
  // Method 2:
  // Read pixel values using split channels
  vector<Mat> three_channels;
  split(src, three_channels);

  // Now I can access each channel separately
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      cout <<
        (uint)three_channels[0].at<uchar>(i, j) << " " <<
        (uint)three_channels[1].at<uchar>(i, j) << " " <<
        (uint)three_channels[2].at<uchar>(i, j) << endl;
    }
  }

  imshow("Blue channel", three_channels[0]);
  imshow("Green channel", three_channels[1]);
  imshow("Red channel", three_channels[2]);

  // Create new image combining channels
  vector<Mat> channels;
  channels.push_back(three_channels[0]);
  channels.push_back(three_channels[1]);
  channels.push_back(three_channels[2]);

  Mat new_image;
  merge(channels, new_image);
  imshow("New image", new_image);

  waitKey(0);
  return 0;
}
