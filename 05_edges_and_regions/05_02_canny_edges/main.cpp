/**
 * Canny edges - sample code
 * @author José Miguel Guerrero
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

int main()
{
  // Create image variables
  Mat image, edges;

  // Read image
  image = imread("../../data/lena.jpg", IMREAD_COLOR);

  // Image processing
  Canny(image, edges, 0, 100, 3);

  // Show images
  imshow("TEST IMAGE", image);
  imshow("CANNY EDGES", edges);

  // Wait to press a key
  waitKey(0);

  return 0;
}
