/**
 * Logic transformation - sample code
 * @author José Miguel Guerrero
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main()
{
  // Create two input matrices filled with zeros
  Mat im1 = Mat::zeros(Size(400, 400), CV_8UC1);
  Mat im2 = Mat::zeros(Size(400, 400), CV_8UC1);

  // Draw a circle on images moving 10px in image 2
  circle(im1, Point(200, 200), 100.0, Scalar(255, 255, 255), 1, 8);
  circle(im2, Point(210, 200), 100.0, Scalar(255, 255, 255), 1, 8);

  // Display circles
  imshow("Circle 1", im1);
  imshow("circle 2", im2);

  // Output images
  Mat output1, output2;

  // Compute the bitwise AND of input images and store them in the output1 image
  bitwise_and(im1, im2, output1);

  // Compute the bitwise OR of input images and store them in the output2 image
  bitwise_or(im1, im2, output2);

  // Display the output images
  imshow("bitwise_and", output1);
  imshow("bitwise_or", output2);

  waitKey(0);
  return 0;
}
