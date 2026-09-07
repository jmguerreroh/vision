/**
 * @file main.cpp
 * @brief Object tracking: CamShift (color histogram) vs CSRT (learned filter)
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - cv::calcHist() + cv::calcBackProject(): the colour model of an object
 * - cv::CamShift(): the mean-shift window that also adapts size and angle
 * - cv::TrackerCSRT: a discriminative filter learned online
 * - Running both on the same clip and timing them, which is the only fair way
 *   to compare a tracker against another
 *
 * The two families answer the same question with opposite assumptions.
 * CamShift knows nothing about the object except ITS COLOUR HISTOGRAM: it
 * climbs the back-projection towards the mode, so it is very fast and it
 * survives deformations and rotations. CSRT LEARNS the appearance of the
 * patch and updates that model frame by frame: it is much more precise and
 * much slower.
 *
 * On the default clip the difference is brutal and the example measures it.
 * The target is a face, and the restaurant behind it is lit in the same warm
 * hue as skin, so the colour model is NOT discriminative: CamShift keeps
 * finding more object outside the box, the window grows several times its
 * initial area and ends up covering half the shot. CSRT stays on the face.
 * That is not a bug in CamShift, it is its assumption failing: colour alone
 * is enough only when the colour of the object is rare in the scene. Run it
 * with --select on something of a distinctive colour and the same code
 * tracks perfectly.
 *
 * The default clip has a scene cut at frame 98. The example stops right
 * before it (see Config::LAST_FRAME) because a cut is not tracking, it is a
 * new video: raise that limit and you will see both boxes stay behind on a
 * face that is no longer there, which is the failure mode every tracker has.
 *
 * @note cv::TrackerCSRT lives in the tracking module of opencv_contrib. The
 *       CamShift half of the example only needs plain OpenCV.
 */

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>  // cv::CamShift
#include <opencv2/tracking.hpp>        // cv::TrackerCSRT (opencv_contrib)
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace Config
{
constexpr int START_FRAME = 20;      // The clip fades in: nothing to track before
constexpr int LAST_FRAME = 97;       // Scene cut at 98
// Face of the first shot, measured on frame 20 of Megamind.avi
const cv::Rect DEFAULT_ROI(255, 200, 120, 130);
constexpr int HUE_BINS = 30;         // Histogram resolution of the hue channel
constexpr int S_MIN = 60;            // Below this saturation the hue is noise
constexpr int V_MIN = 40;            // And in the dark it is meaningless
constexpr int V_MAX = 250;
constexpr int MAX_ITERATIONS = 10;   // Mean-shift iterations per frame
constexpr double EPSILON = 1.0;      // ...or until the window moves less than this
const char * WINDOW_NAME = "CamShift (green) vs CSRT (red)";
}

/**
 * @brief Hue histogram of the object, used as its colour model
 *
 * Only the HUE channel is used: it is the component of HSV that does not
 * change when the light does (Chapter 3), which is exactly what a tracker
 * needs. Pixels that are too dark or too grey are masked out because their
 * hue is numerically unstable, and letting them in poisons the model.
 */
cv::Mat hueHistogram(const cv::Mat & frame_hsv, const cv::Rect & roi)
{
  cv::Mat mask;
  cv::inRange(frame_hsv, cv::Scalar(0, Config::S_MIN, Config::V_MIN),
    cv::Scalar(180, 255, Config::V_MAX), mask);

  const int channels[] = {0};
  const int histogram_size[] = {Config::HUE_BINS};
  const float hue_range[] = {0, 180};
  const float * ranges[] = {hue_range};

  cv::Mat histogram;
  const cv::Mat roi_hsv = frame_hsv(roi);
  const cv::Mat roi_mask = mask(roi);
  cv::calcHist(&roi_hsv, 1, channels, roi_mask, histogram, 1, histogram_size, ranges);
  cv::normalize(histogram, histogram, 0, 255, cv::NORM_MINMAX);
  return histogram;
}

int main(int argc, char ** argv)
{
  const std::string keys =
    "{help h  | | Show this help message}"
    "{select  | | Pick the object with the mouse instead of using the default box}"
    "{@video  | ../../data/Megamind.avi | Input video file}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  const std::string filename =
    cv::samples::findFile(parser.get<std::string>("@video"), false);
  cv::VideoCapture cap(filename);
  if (!cap.isOpened()) {
    std::cerr << "Error: cannot open video: " << filename << std::endl;
    return EXIT_FAILURE;
  }

  // Skip the fade-in
  cv::Mat frame;
  for (int i = 0; i <= Config::START_FRAME; ++i) {
    if (!cap.read(frame)) {
      std::cerr << "Error: the video is shorter than START_FRAME" << std::endl;
      return EXIT_FAILURE;
    }
  }

  // ========================================
  // The object to track
  // ========================================
  cv::Rect roi = Config::DEFAULT_ROI;
  if (parser.has("select")) {
    roi = cv::selectROI("Select the object and press ENTER", frame, false);
    cv::destroyWindow("Select the object and press ENTER");
  }
  if (roi.width < 5 || roi.height < 5 || (roi & cv::Rect(cv::Point(), frame.size())) != roi) {
    std::cerr << "Error: invalid region" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Tracking: CamShift vs CSRT ===" << std::endl;
  std::cout << "Video: " << filename << std::endl;
  std::cout << "Initial box: " << roi << " at frame " << Config::START_FRAME << std::endl;

  // ========================================
  // Initialise both trackers on the same box
  // ========================================
  cv::Mat frame_hsv;
  cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
  const cv::Mat histogram = hueHistogram(frame_hsv, roi);
  cv::Rect camshift_window = roi;

  cv::Ptr<cv::Tracker> csrt = cv::TrackerCSRT::create();
  csrt->init(frame, roi);
  cv::Rect csrt_box = roi;

  const cv::TermCriteria criteria(
    cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
    Config::MAX_ITERATIONS, Config::EPSILON);

  double camshift_ms = 0.0, csrt_ms = 0.0, separation = 0.0;
  int frames = 0, csrt_lost = 0;
  const double initial_area = static_cast<double>(roi.area());
  double last_camshift_area = initial_area;

  std::cout << "Press 'q' or ESC to stop early" << std::endl;

  while (cap.read(frame) && Config::START_FRAME + frames + 1 <= Config::LAST_FRAME) {
    ++frames;
    cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

    // ---- CamShift ----------------------------------------------------
    // The back-projection turns the frame into a probability map: every
    // pixel is replaced by how often ITS hue appeared inside the object
    double tick = static_cast<double>(cv::getTickCount());
    cv::Mat back_projection;
    const int channels[] = {0};
    const float hue_range[] = {0, 180};
    const float * ranges[] = {hue_range};
    cv::calcBackProject(&frame_hsv, 1, channels, histogram, back_projection, ranges);

    cv::Mat mask;
    cv::inRange(frame_hsv, cv::Scalar(0, Config::S_MIN, Config::V_MIN),
      cv::Scalar(180, 255, Config::V_MAX), mask);
    back_projection &= mask;   // ignore the pixels whose hue means nothing

    // CamShift returns the rotated rectangle and UPDATES camshift_window with
    // the new position and size, which is what mean-shift alone cannot do
    const cv::RotatedRect camshift_result =
      cv::CamShift(back_projection, camshift_window, criteria);
    camshift_ms += (static_cast<double>(cv::getTickCount()) - tick) /
      cv::getTickFrequency() * 1000.0;

    // ---- CSRT --------------------------------------------------------
    tick = static_cast<double>(cv::getTickCount());
    const bool csrt_ok = csrt->update(frame, csrt_box);
    csrt_ms += (static_cast<double>(cv::getTickCount()) - tick) /
      cv::getTickFrequency() * 1000.0;
    if (!csrt_ok) {
      ++csrt_lost;
    }

    // How far apart the two answers are, in pixels between centres, and how
    // much the CamShift window has grown: those two numbers are the whole
    // comparison
    const cv::Point2f csrt_centre(csrt_box.x + csrt_box.width / 2.0f,
      csrt_box.y + csrt_box.height / 2.0f);
    separation += cv::norm(camshift_result.center - csrt_centre);
    last_camshift_area = camshift_result.size.area();

    // ---- Draw --------------------------------------------------------
    cv::Point2f corners[4];
    camshift_result.points(corners);
    for (int i = 0; i < 4; ++i) {
      cv::line(frame, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 220, 0), 2,
        cv::LINE_AA);
    }
    cv::rectangle(frame, csrt_box, cv::Scalar(0, 0, 220), 2);
    cv::putText(frame, "CamShift", corners[1] + cv::Point2f(0, -6),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 220, 0), 1, cv::LINE_AA);
    cv::putText(frame, csrt_ok ? "CSRT" : "CSRT (lost)",
      cv::Point(csrt_box.x, csrt_box.y + csrt_box.height + 16),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 220), 1, cv::LINE_AA);

    cv::imshow(Config::WINDOW_NAME, frame);
    cv::imshow("CamShift back-projection", back_projection);

    const int key = cv::waitKey(30);
    if (key == 'q' || key == 27) {
      break;
    }
  }

  // ========================================
  // The comparison, in numbers
  // ========================================
  if (frames > 0) {
    std::cout << "\nFrames tracked: " << frames << std::endl;
    std::cout << "CamShift: " << camshift_ms / frames << " ms/frame" << std::endl;
    std::cout << "CSRT:     " << csrt_ms / frames << " ms/frame  ("
              << csrt_ms / camshift_ms << "x slower)" << std::endl;
    std::cout << "CSRT reported a loss in " << csrt_lost << " frames" << std::endl;
    std::cout << "Mean distance between the two centres: " << separation / frames
              << " px" << std::endl;
    std::cout << "CamShift window: " << initial_area << " px2 at the start, "
              << last_camshift_area << " px2 at the end ("
              << last_camshift_area / initial_area << "x)" << std::endl;
    std::cout << "\nCSRT pays an order of magnitude more per frame and stays on "
      "the face.\nCamShift is nearly free, but here it grows over everything that "
      "shares\nthe hue of skin, and the two answers drift apart. Look at the "
      "back-\nprojection window: whatever is bright in it is what CamShift "
      "believes\nthe object to be. Colour is enough only when the colour is "
      "rare in the\nscene, which is exactly the assumption to check before "
      "choosing this\nfamily of tracker." << std::endl;
  }

  std::cout << "Press any key to exit..." << std::endl;
  cv::waitKey(0);
  return EXIT_SUCCESS;
}
