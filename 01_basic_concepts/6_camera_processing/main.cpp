/**
 * Camera processing sample code
 * @author José Miguel Guerrero
 */

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;



int main( int argc, char** argv ) {
    // Image
    Mat frame, edges;

    // Initialize videocapture
    VideoCapture cap;

    // Open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API

    // Open selected camera using selected API
    cap.open( deviceID, apiID );

    // Check if we succeeded
    if ( !cap.isOpened() ) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    // Grab and write loop
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
    // Wait for a key with timeout long enough to show images
    while ( waitKey(5) <= 0 ) {
        // Wait for a new frame from camera and store it into 'frame'
        cap.read( frame );

        // Check if we succeeded
        if ( frame.empty() ) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        // Show live 
        imshow( "Live", frame );
        // Image processing
        Canny( frame, edges, 0, 100, 3 );
        // Show image processing 
        imshow( "Live edges", edges );
    }

    // The camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}