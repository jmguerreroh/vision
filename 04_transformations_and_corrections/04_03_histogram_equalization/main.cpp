/**
 * Histogram equalization - sample code
 * @author José Miguel Guerrero
 *
 * https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
 */

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{
  // Read image
  CommandLineParser parser(argc, argv, "{@input | lena.jpg | input image}");
  Mat src = imread(samples::findFile(parser.get<String>("@input") ), IMREAD_COLOR);
  if (src.empty() ) {
    return EXIT_FAILURE;
  }

  // Split BGR planes
  vector<Mat> bgr_planes;
  split(src, bgr_planes);

  // Establish the number of bins
  int histSize = 256;
  // Set the ranges ( for B,G,R) )
  float range[] = {0, 256};       //the upper boundary is exclusive
  const float * histRange = {range};
  bool uniform = true, accumulate = false;

  // Compute the histograms for each channel
  Mat b_hist, g_hist, r_hist;
  calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
  calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
  calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

  // Draw the histograms for B, G and R
  int hist_w = 512, hist_h = 400;
  int bin_w = cvRound( (double) hist_w / histSize);

  Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0) );

  // normalize the histograms between 0 and histImage.rows
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  // Draw the intensity line for histograms
  for (int i = 1; i < histSize; i++) {
    line(
      histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1)) ),
      Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i)) ),
      Scalar(255, 0, 0), 2, 8, 0);
    line(
      histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1)) ),
      Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i)) ),
      Scalar(0, 255, 0), 2, 8, 0);
    line(
      histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1)) ),
      Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i)) ),
      Scalar(0, 0, 255), 2, 8, 0);
  }

  // Show images
  imshow("Source image", src);
  imshow("calcHist Source", histImage);

  // Equalization
  Mat b_eqhist, g_eqhist, r_eqhist;
  equalizeHist(bgr_planes[0], b_eqhist);
  equalizeHist(bgr_planes[1], g_eqhist);
  equalizeHist(bgr_planes[2], r_eqhist);

  // Compute the histograms for each channel
  Mat b_histeq, g_histeq, r_histeq;
  calcHist(&b_eqhist, 1, 0, Mat(), b_histeq, 1, &histSize, &histRange, uniform, accumulate);
  calcHist(&g_eqhist, 1, 0, Mat(), g_histeq, 1, &histSize, &histRange, uniform, accumulate);
  calcHist(&r_eqhist, 1, 0, Mat(), r_histeq, 1, &histSize, &histRange, uniform, accumulate);

  // normalize the histograms between 0 and histImage.rows
  Mat histImageEq(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0) );
  normalize(b_histeq, b_histeq, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(g_histeq, g_histeq, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(r_histeq, r_histeq, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  // Draw the intensity line for equalized histograms
  for (int i = 1; i < histSize; i++) {
    line(
      histImageEq, Point(bin_w * (i - 1), hist_h - cvRound(b_histeq.at<float>(i - 1)) ),
      Point(bin_w * (i), hist_h - cvRound(b_histeq.at<float>(i)) ),
      Scalar(255, 0, 0), 2, 8, 0);
    line(
      histImageEq, Point(bin_w * (i - 1), hist_h - cvRound(g_histeq.at<float>(i - 1)) ),
      Point(bin_w * (i), hist_h - cvRound(g_histeq.at<float>(i)) ),
      Scalar(0, 255, 0), 2, 8, 0);
    line(
      histImageEq, Point(bin_w * (i - 1), hist_h - cvRound(r_histeq.at<float>(i - 1)) ),
      Point(bin_w * (i), hist_h - cvRound(r_histeq.at<float>(i)) ),
      Scalar(0, 0, 255), 2, 8, 0);
  }

  // Create Equalized image
  vector<Mat> equalized;
  equalized.push_back(b_eqhist);
  equalized.push_back(g_eqhist);
  equalized.push_back(r_eqhist);
  // Merge channels
  Mat equalized_image;
  merge(equalized, equalized_image);

  // Show images
  imshow("Equalized image", equalized_image);
  imshow("calcHist Equalized", histImageEq);
  waitKey();

  return EXIT_SUCCESS;
}
