/**
 * Opening and Closing sample code
 * @author José Miguel Guerrero
 */
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;

Mat src, dst;
int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 1;
int const max_elem = 2;
int const max_kernel_size = 21;
const char* window_name = "Morphology Transformations Demo";

void Morphology_Operations( int, void* ) {
  // Since MORPH_X : 2,3,4,5 and 6
  int operation = morph_operator + 2;
  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  morphologyEx( src, dst, operation, element );
  imshow( window_name, dst );
}

int main( int argc, char** argv ) {
  CommandLineParser parser( argc, argv, "{@input | ../../images_and_videos/crop.png | input image}" );
  src = imread( samples::findFile( parser.get<String>( "@input" ) ), IMREAD_COLOR );
  if (src.empty())  {
    std::cout << "Could not open or find the image!\n" << std::endl;
    std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  namedWindow( window_name, WINDOW_AUTOSIZE ); // Create window
  createTrackbar("Operator:\n 0: Opening - 1: Closing ", window_name, &morph_operator, max_operator, Morphology_Operations );
  createTrackbar( "Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
                  &morph_elem, max_elem,
                  Morphology_Operations );
  createTrackbar( "Kernel size:\n 2n +1", window_name,
                  &morph_size, max_kernel_size,
                  Morphology_Operations );
                  
  Morphology_Operations( 0, 0 );
  waitKey(0);
  return 0;
}
