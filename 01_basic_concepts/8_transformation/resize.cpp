
/**
 * Resize sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {
  // Load an image
  Mat src = imread( "../../images/cat-small.jpg", IMREAD_COLOR );
  if ( src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  // Create windows
  namedWindow( "Original image", WINDOW_AUTOSIZE );
  namedWindow( "Resampling", WINDOW_AUTOSIZE );

  imshow( "Original image", src );

  // Resize
  Mat resize_dst;
  resize(src, resize_dst, cv::Size(), 10, 10, INTER_NEAREST);
  imshow( "Resampling", resize_dst );

  waitKey();
  return 0;
}