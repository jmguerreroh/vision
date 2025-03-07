/**
 * Sobel edges - sample code
 * @author José Miguel Guerrero
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  // Read image
  Mat src = imread("../../data/lena.jpg", 0);
  if (src.empty()) {
    cout << "the image is not exist" << endl;
    return -1;
  }

  resize(src, src, Size(512, 512));

  // ------------
  // Method 1:
  // Use masks to calculate the gradient

  // Masks
  Mat SobelGx = (Mat_<char>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1);

  Mat SobelGy = (Mat_<char>(3, 3) <<
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1);

  // Applying masks
  Mat SobelHorizontal, SobelVertical;
  Mat grad_masks;
  // Gradient X and Y
  filter2D(src, SobelVertical, src.depth(), SobelGx);
  filter2D(src, SobelHorizontal, src.depth(), SobelGy);
  // Combine both masks
  addWeighted(SobelHorizontal, 0.5, SobelVertical, 0.5, 0, grad_masks);

  // Show images
  imshow("Original", src);
  imshow("Horizontal edges", SobelHorizontal);
  imshow("Vertical edges", SobelVertical);
  imshow("Edges", grad_masks);

  // ------------
  // Method 2:
  // Use Sobel to calculate the gradient
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  Mat grad;
  // Gradient X and Y
  Sobel(src, grad_x, src.depth(), 1, 0, 3);
  Sobel(src, grad_y, src.depth(), 0, 1, 3);
  // Convert to absolute values
  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);
  // Combine both masks
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

  // Show images
  imshow("Horizontal edges (Sobel)", grad_y);
  imshow("Vertical edges (Sobel)", grad_x);
  imshow("Edges (Sobel)", grad);

  waitKey();

  return 0;
}
