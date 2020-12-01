
/**
 * Resize sample code
 * @author José Miguel Guerrero
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// Global variables
Mat src, resize_dst;

int resize_size = 10;
int resize_inter = 0;
int const max_elem = 2;



/**
 * Resize function
 */
void Resize( int, void* ) {
  int resize_type = 0;
  if ( resize_inter == 0 ){ resize_type = INTER_NEAREST; }
  else if ( resize_inter == 1 ){ resize_type = INTER_LINEAR; }
  else if ( resize_inter == 2 ) { resize_type = INTER_CUBIC; }

  cv::resize(src, resize_dst, cv::Size(), resize_size, resize_size, resize_type);

  imshow( "Resize Demo", resize_dst );
}



int main( int argc, char** argv ) {
  // Load an image
  src = imread( "../../images/cat-small.jpg", IMREAD_COLOR );
  if ( src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  // Create windows
  namedWindow( "Resize Demo" );

  // Create Resize Trackbar
  createTrackbar( "Element:\n 0: Nearest \n 1: Linear \n 2: Cubic", "Resize Demo",
          &resize_inter, max_elem,
          Resize );


  // Default start
  Resize( 0, 0 );

  waitKey(0);
  return 0;
}