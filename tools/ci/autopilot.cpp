/**
 * @file autopilot.cpp
 * @brief Plays the user of the examples, so that they can run unattended
 * @author José Miguel Guerrero Hernández
 *
 * Every example ends waiting for a person: a key in cv::waitKey(0), a box in
 * cv::selectROI, the window of a PCL or viz visualizer closed by hand, ENTER in
 * 15_11. With nobody there a CI run would hang, and killing it on a timeout
 * would say nothing about whether the example worked.
 *
 * Loaded with LD_PRELOAD, this library replaces those calls with a user who
 * always does the same thing:
 *
 *   - cv::waitKey(0) answers ESC at once. A timed cv::waitKey lets the loop run
 *     AUTOPILOT_FRAMES iterations (30 by default) and then answers ESC, which
 *     every example takes as "quit". The real waitKey still runs, capped at
 *     1 ms, so the windows are really drawn.
 *   - cv::selectROI draws the image and returns its central quarter.
 *   - cv::viz::Viz3d::spin runs a few frames and returns, as if the window had
 *     been closed.
 *   - A PCL visualizer reports itself stopped after AUTOPILOT_SPINS calls (20 by
 *     default), and every few frames it receives a Return key press, which is
 *     what the "Press ENTER" waits of 15_11 listen to.
 *
 * The real functions are reached through dlsym(RTLD_NEXT), by their mangled
 * name. A name that does not exist in the loaded libraries (a signature of
 * another OpenCV version, or viz where it is not installed) just leaves that
 * replacement unused.
 *
 * The file is compiled twice. Without AUTOPILOT_PCL it only touches OpenCV,
 * and can be preloaded into any example. With it, it also replaces the PCL
 * visualizer, and it has to be linked against PCL and VTK: the VTK headers
 * leave static objects whose symbols an example without VTK cannot resolve.
 * run_examples.py picks one or the other from the libraries each binary needs.
 */

#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#if __has_include(<opencv2/viz.hpp>)
#include <opencv2/viz.hpp>
#define AUTOPILOT_VIZ 1
#endif

#ifdef AUTOPILOT_PCL
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkCommand.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#endif

namespace
{

constexpr int ESC = 27;

int envInt(const char * name, int fallback)
{
  const char * value = std::getenv(name);
  return value ? std::atoi(value) : fallback;
}

const int FRAMES = envInt("AUTOPILOT_FRAMES", 30);
const int SPINS = envInt("AUTOPILOT_SPINS", 20);

template<typename Fn>
Fn real(const char * mangled)
{
  void * symbol = dlsym(RTLD_NEXT, mangled);
  if (!symbol) {
    std::cerr << "[autopilot] missing symbol " << mangled << std::endl;
    std::abort();
  }
  return reinterpret_cast<Fn>(symbol);
}

#ifdef AUTOPILOT_PCL
// Counts calls per object, so that two visualizers in one process each get
// their own budget.
int tick(const void * object)
{
  static std::mutex mutex;
  static std::map<const void *, int> calls;
  std::lock_guard<std::mutex> lock(mutex);
  return ++calls[object];
}
#endif

}  // namespace

// --- OpenCV highgui ------------------------------------------------------------

namespace cv
{

int waitKey(int delay)
{
  static auto real_wait = real<int (*)(int)>("_ZN2cv7waitKeyEi");
  static int timed = 0;
  real_wait(1);
  if (delay <= 0 || ++timed > FRAMES) {
    return ESC;
  }
  return -1;
}

static Rect centralQuarter(const String & window, InputArray img)
{
  imshow(window, img);
  const Size size = img.size();
  std::cerr << "[autopilot] selectROI answers the central quarter" << std::endl;
  return Rect(size.width / 4, size.height / 4, size.width / 2, size.height / 2);
}

// OpenCV 4.6 (Ubuntu 24.04) has no printNotice; 4.10 (Ubuntu 26.04) added it.
// Both signatures are defined, and the one the library exports takes over.
Rect selectROI(const String & window, InputArray img, bool, bool)
{
  return centralQuarter(window, img);
}

Rect selectROI(const String & window, InputArray img, bool, bool, bool)
{
  return centralQuarter(window, img);
}

Rect selectROI(InputArray img, bool, bool)
{
  return centralQuarter("ROI selector", img);
}

Rect selectROI(InputArray img, bool, bool, bool)
{
  return centralQuarter("ROI selector", img);
}

#ifdef AUTOPILOT_VIZ
void viz::Viz3d::spin()
{
  for (int i = 0; i < SPINS; ++i) {
    spinOnce(10, true);
  }
}
#endif

}  // namespace cv

// --- PCL visualization ---------------------------------------------------------

#ifdef AUTOPILOT_PCL

namespace pcl
{
namespace visualization
{

void PCLVisualizer::spinOnce(int time, bool force_redraw)
{
  static auto real_spin = real<void (*)(PCLVisualizer *, int, bool)>(
    "_ZN3pcl13visualization13PCLVisualizer8spinOnceEib");
  real_spin(this, std::min(time, 10), force_redraw);

  // A Return every fifth frame, as the key press of someone reading the
  // "Press ENTER" prompt. Keys no example listens to are harmless.
  if (tick(this) % 5 == 0) {
    vtkRenderWindowInteractor * interactor = getRenderWindow()->GetInteractor();
    if (interactor) {
      interactor->SetKeyEventInformation(0, 0, '\r', 1, "Return");
      interactor->InvokeEvent(vtkCommand::KeyPressEvent, nullptr);
      interactor->InvokeEvent(vtkCommand::KeyReleaseEvent, nullptr);
    }
  }
}

void PCLVisualizer::spin()
{
  for (int i = 0; i < SPINS; ++i) {
    spinOnce(10, true);
  }
}

bool PCLVisualizer::wasStopped() const
{
  static auto real_stopped = real<bool (*)(const PCLVisualizer *)>(
    "_ZNK3pcl13visualization13PCLVisualizer10wasStoppedEv");
  return real_stopped(this) || tick(reinterpret_cast<const char *>(this) + 1) > SPINS;
}

bool CloudViewer::wasStopped(int millis)
{
  static auto real_stopped = real<bool (*)(CloudViewer *, int)>(
    "_ZN3pcl13visualization11CloudViewer10wasStoppedEi");
  return real_stopped(this, millis) || tick(this) > SPINS;
}

}  // namespace visualization
}  // namespace pcl

#endif  // AUTOPILOT_PCL
