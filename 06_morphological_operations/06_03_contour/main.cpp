/**
 * Contour sample code
 * @author José Miguel Guerrero
 */
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

// Global variables
Mat src, dst;
int morph_elem = 0;
int morph_size = 1;
int morph_operator = 0;
int const max_operator = 1;
int const max_elem = 2;
int const max_kernel_size = 21;
const char* window_name = "Erode and Dilate Demo";


void ErodeDilate( int, void* ) {
  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  if (morph_operator == 0){
    erode( src, dst, element );
    dst = src - dst;
  } else {
    dilate( src, dst, element );
    dst = dst - src;
  }
  imshow( window_name, dst );
}

int main( int argc, char** argv ) {
  // Load an image
  CommandLineParser parser( argc, argv, "{@input | ../../images_and_videos/horse.png | input image}" );
  src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_COLOR );
  if( src.empty() ) {
    cout << "Could not open or find the image!\n" << endl;
    cout << "Usage: " << argv[0] << " <Input image>" << endl;
    return -1;
  }

  // Create windows
  namedWindow( window_name, WINDOW_AUTOSIZE ); // Create window

  createTrackbar("Operator:\n 0: In - 1: Out ", window_name, &morph_operator, max_operator, ErodeDilate );
  createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
                  &morph_elem, max_elem,
                  ErodeDilate );
  createTrackbar( "Kernel size:\n 2n +1", window_name,
                  &morph_size, max_kernel_size,
                  ErodeDilate );
          
  // Default start
  ErodeDilate( 0, 0 );
  
  waitKey(0);
  return 0;
}