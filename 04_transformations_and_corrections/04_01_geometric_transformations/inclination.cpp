/**
 * Warp + rotate - sample code
 * @author José Miguel Guerrero
 *
 * https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
  CommandLineParser parser(argc, argv, "{@input | lena.jpg | input image}");
  Mat src = imread(samples::findFile(parser.get<String>("@input") ) );
  if (src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  Point2f srcTri[3];
  srcTri[0] = Point2f(0.f, 0.f);
  srcTri[1] = Point2f(src.cols - 1.f, 0.f);
  srcTri[2] = Point2f(0.f, src.rows - 1.f);

  Point2f dstTri[3];
  dstTri[0] = Point2f(0.f, src.rows * 0.33f);
  dstTri[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
  dstTri[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);

  Mat warp_mat = getAffineTransform(srcTri, dstTri);
  Mat warp_dst = Mat::zeros(src.rows, src.cols, src.type() );

  warpAffine(src, warp_dst, warp_mat, warp_dst.size() );

  Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
  double angle = -50.0;
  double scale = 0.6;

  Mat rot_mat = getRotationMatrix2D(center, angle, scale);
  Mat warp_rotate_dst;
  warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );

  imshow("Source image", src);
  imshow("Warp", warp_dst);
  imshow("Warp + Rotate", warp_rotate_dst);

  waitKey();
  return 0;
}
