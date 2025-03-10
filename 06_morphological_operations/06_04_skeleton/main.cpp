/**
 * Skeleton - sample code: based on Hang Suen and Guo Hall
 * @author José Miguel Guerrero
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define THINNING_ZHANGSUEN 1
#define THINNING_GUOHALL 2

// Applies a thinning iteration to a binary image
static void thinningIteration(Mat img, int iter, int thinningType)
{
  Mat marker = Mat::zeros(img.size(), CV_8UC1);

  if (thinningType == THINNING_ZHANGSUEN) {
    for (int i = 1; i < img.rows - 1; i++) {
      for (int j = 1; j < img.cols - 1; j++) {
        uchar p2 = img.at<uchar>(i - 1, j);
        uchar p3 = img.at<uchar>(i - 1, j + 1);
        uchar p4 = img.at<uchar>(i, j + 1);
        uchar p5 = img.at<uchar>(i + 1, j + 1);
        uchar p6 = img.at<uchar>(i + 1, j);
        uchar p7 = img.at<uchar>(i + 1, j - 1);
        uchar p8 = img.at<uchar>(i, j - 1);
        uchar p9 = img.at<uchar>(i - 1, j - 1);

        int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
          (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
          (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
          (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
        int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
        int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
        int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

        if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
          marker.at<uchar>(i, j) = 1;
        }
      }
    }
  }

  if (thinningType == THINNING_GUOHALL) {
    for (int i = 1; i < img.rows - 1; i++) {
      for (int j = 1; j < img.cols - 1; j++) {
        uchar p2 = img.at<uchar>(i - 1, j);
        uchar p3 = img.at<uchar>(i - 1, j + 1);
        uchar p4 = img.at<uchar>(i, j + 1);
        uchar p5 = img.at<uchar>(i + 1, j + 1);
        uchar p6 = img.at<uchar>(i + 1, j);
        uchar p7 = img.at<uchar>(i + 1, j - 1);
        uchar p8 = img.at<uchar>(i, j - 1);
        uchar p9 = img.at<uchar>(i - 1, j - 1);

        int C = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) + ((!p6) & (p7 | p8)) +
          ((!p8) & (p9 | p2));
        int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
        int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
        int N = N1 < N2 ? N1 : N2;
        int m = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);

        if ((C == 1) && ((N >= 2) && ((N <= 3)) & (m == 0))) {
          marker.at<uchar>(i, j) = 1;
        }
      }
    }
  }
  img &= ~marker;
}

// Apply the thinning procedure to a given image
void thinning(InputArray input, OutputArray output, int thinningType)
{
  Mat processed = input.getMat().clone();
  // Enforce the range of the input image to be in between 0 - 255
  processed /= 255;

  Mat prev = Mat::zeros(processed.size(), CV_8UC1);
  Mat diff, temp;

  do {
    thinningIteration(processed, 0, thinningType);
    thinningIteration(processed, 1, thinningType);
    absdiff(processed, prev, diff);
    processed.copyTo(prev);

    //// Displays the animation at each iteration of the algorithm
    temp = processed * 255;
    imshow("Original Skeleton Final", temp);
    waitKey(10);
    //// end animation
  } while (countNonZero(diff) > 0);

  processed *= 255;
  output.assign(processed);
}


int main(int argc, char ** argv)
{
  Mat image = imread("../../data/star.jpg", IMREAD_GRAYSCALE);

  if (image.empty()) {
    printf("No image data \n");
    return -1;
  }

  Mat src = image.clone();
  thinning(src, src, 2);

  // Draw red skeleton over the original image
  cvtColor(image, image, COLOR_GRAY2BGR);

  for (int i = 0; i < image.cols; i++) {
    for (int j = 0; j < image.rows; j++) {
      Scalar intensity = src.at<uchar>(j, i);
      if (intensity.val[0] == 255) {
        image.at<Vec3b>(j, i) = Vec3b(0, 0, 255);
      }
    }
  }

  imshow("OpenCV Skeleton Final", src);
  imshow("Original Skeleton Final", image);
  waitKey(0);

  return 0;
}
