/**
 * ICP demo sample
 * @author José Miguel Guerrero
 *
 * https://github.com/opencv/opencv_contrib/blob/master/modules/surface_matching/samples/ppf_load_match.cpp
 */

#include "opencv2/surface_matching.hpp"
#include <iostream>
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/viz.hpp"

using namespace std;
using namespace cv;
using namespace ppf_match_3d;

static void help(const string & errorMessage)
{
  cout << "Program init error : " << errorMessage << endl;
  cout << "\nUsage : ppf_matching [input model file] [input scene file]" << endl;
  cout << "\nPlease start again with new parameters" << endl;
}

int show_image3D(string window_name, string filename)
{
  // Load the depth map (the depth map of kinect2 used here)
  Mat depth = cv::viz::readCloud(filename);
  //initialization
  viz::Viz3d window(window_name);
  // Display coordinate system
  window.showWidget(window_name, viz::WCoordinateSystem());

  viz::WCloud cloud(depth);
  window.showWidget(window_name, cloud);
  window.spin();
  return 0;
}

int main(int argc, char ** argv)
{
  // welcome message
  cout << "****************************************************" << endl;
  cout << "* Surface Matching demonstration : demonstrates the use of surface matching"
    " using point pair features." << endl;
  cout << "* The sample loads a model and a scene, where the model lies in a different"
    " pose than the training.\n* It then trains the model and searches for it in the"
    " input scene. The detected poses are further refined by ICP\n* and printed to the "
    " standard output." << endl;
  cout << "****************************************************" << endl;

  if (argc < 3) {
    help("Not enough input arguments");
    exit(1);
  }

#if (defined __x86_64__ || defined _M_X64)
  cout << "Running on 64 bits" << endl;
#else
  cout << "Running on 32 bits" << endl;
#endif

#ifdef _OPENMP
  cout << "Running with OpenMP" << endl;
#else
  cout << "Running without OpenMP and without TBB" << endl;
#endif

  string modelFileName = (string)argv[1];
  string sceneFileName = (string)argv[2];
  show_image3D("model", modelFileName);
  show_image3D("scene", sceneFileName);

  Mat pc = loadPLYSimple(modelFileName.c_str(), 1);

  // Now train the model
  cout << "Training..." << endl;
  int64 tick1 = cv::getTickCount();
  ppf_match_3d::PPF3DDetector detector(0.025, 0.05);
  detector.trainModel(pc);
  int64 tick2 = cv::getTickCount();
  cout << endl << "Training complete in "
       << (double)(tick2 - tick1) / cv::getTickFrequency()
       << " sec" << endl << "Loading model..." << endl;

  // Read the scene
  Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);

  // Match the model to the scene and get the pose
  cout << endl << "Starting matching..." << endl;
  vector<Pose3DPtr> results;
  tick1 = cv::getTickCount();
  detector.match(pcTest, results, 1.0 / 40.0, 0.05);
  tick2 = cv::getTickCount();
  cout << endl << "PPF Elapsed Time " <<
    (tick2 - tick1) / cv::getTickFrequency() << " sec" << endl;

  //check results size from match call above
  size_t results_size = results.size();
  cout << "Number of matching poses: " << results_size;
  if (results_size == 0) {
    cout << endl << "No matching poses found. Exiting." << endl;
    exit(0);
  }

  // Get only first N results - but adjust to results size if num of results are less than that specified by N
  size_t N = 2;
  if (results_size < N) {
    cout << endl << "Reducing matching poses to be reported (as specified in code): "
         << N << " to the number of matches found: " << results_size << endl;
    N = results_size;
  }
  vector<Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

  // Create an instance of ICP
  ICP icp(100, 0.005f, 2.5f, 8);
  int64 t1 = cv::getTickCount();

  // Register for all selected poses
  cout << endl << "Performing ICP on " << N << " poses..." << endl;
  icp.registerModelToScene(pc, pcTest, resultsSub);
  int64 t2 = cv::getTickCount();

  cout << endl << "ICP Elapsed Time " <<
    (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

  cout << "Poses: " << endl;
  // debug first five poses
  for (size_t i = 0; i < resultsSub.size(); i++) {
    Pose3DPtr result = resultsSub[i];
    cout << "\nPose Result " << i << endl;
    result->printPose();
    if (i == 0) {
      Mat pct = transformPCPose(pc, result->pose);
      writePLY(pct, "para6700PCTrans.ply");
    }
  }
  return 0;

}
