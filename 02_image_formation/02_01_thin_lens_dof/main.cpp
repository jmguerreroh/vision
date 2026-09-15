/**
 * @file main.cpp
 * @brief Thin lens geometry and depth of field, from the camera parameters
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - How to solve the thin lens equation 1/f = 1/u + 1/v for the image distance
 * - How to obtain the lateral magnification m = -v/u
 * - How the circle of confusion grows away from the plane in focus
 * - How the depth of field depends on the criterion chosen for that circle
 *
 * The last point is the reason this example exists. "Acceptably sharp" is not
 * a property of the lens: it is a threshold that somebody has to choose, and
 * the same camera has two very different depths of field depending on which
 * one is used:
 *
 *   - The PHOTOGRAPHIC criterion, c <= 0.03 mm for the 35 mm format, comes
 *     from what the eye resolves on an enlarged print at normal viewing
 *     distance. It is the number every depth-of-field table uses.
 *   - The PIXEL criterion, c <= one pixel, is the one that matters in computer
 *     vision: a blur disc that fits inside a pixel cannot be told from a point,
 *     and anything smaller is wasted.
 *
 * On a full-frame sensor the pixel is around five times smaller than 0.03 mm,
 * so the pixel criterion gives a much narrower depth of field. This example
 * prints both side by side so the difference is a measurement and not a claim.
 *
 * Usage: ./02_01_thin_lens_dof
 *        ./02_01_thin_lens_dof --focal=50 --fnumber=2.8 --distance=2000
 *        ./02_01_thin_lens_dof --help
 *
 * @note Everything here is arithmetic on the camera parameters: no image is
 *       read and no OpenCV image structure is used. cv::CommandLineParser is
 *       used for the arguments, the same as in every other example.
 */

#include <opencv2/core.hpp>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace
{

// Every length is in millimetres. Keeping a single unit everywhere avoids the
// conversion mistakes that this kind of formula invites, and the printing
// function is the only place where metres appear
struct DepthOfField
{
  double near_limit;
  double far_limit;
  double hyperfocal;
  bool reaches_infinity;
};

/// Image distance from the thin lens equation, 1/f = 1/u + 1/v.
/// Only valid for u > f: at u = f the rays leave parallel and there is no image
double imageDistance(double focal, double object_distance)
{
  return object_distance * focal / (object_distance - focal);
}

/// Lateral magnification. Negative because the real image is inverted
double magnification(double focal, double object_distance)
{
  return -imageDistance(focal, object_distance) / object_distance;
}

/// Diameter of the circle of confusion left on the sensor by a point at
/// other_distance, when the lens is focused at focus_distance.
/// The aperture diameter is focal/f_number, and the disc is the section of the
/// cone of light at the distance between the sensor and where the point
/// actually converges
double circleOfConfusion(double focal, double f_number,
  double focus_distance, double other_distance)
{
  const double aperture = focal / f_number;
  const double v_focus = imageDistance(focal, focus_distance);
  const double v_other = imageDistance(focal, other_distance);
  return aperture * std::abs(v_other - v_focus) / v_other;
}

/// Depth of field for a given maximum circle of confusion.
/// Solving c(d) = c_max gives the classic closed form, written around the
/// hyperfocal distance H: focused at H, everything from H/2 to infinity is
/// within the criterion, and there is no point focusing further away
DepthOfField depthOfField(double focal, double f_number,
  double focus_distance, double c_max)
{
  DepthOfField dof;
  dof.hyperfocal = focal * focal / (f_number * c_max) + focal;

  const double numerator = focus_distance * (dof.hyperfocal - focal);
  dof.near_limit = numerator / (dof.hyperfocal + focus_distance - 2.0 * focal);

  // Focused at or beyond the hyperfocal distance the far limit runs off to
  // infinity: the denominator goes to zero and then negative
  if (focus_distance >= dof.hyperfocal) {
    dof.reaches_infinity = true;
    dof.far_limit = std::numeric_limits<double>::infinity();
  } else {
    dof.reaches_infinity = false;
    dof.far_limit = numerator / (dof.hyperfocal - focus_distance);
  }
  return dof;
}

void printDepthOfField(const std::string & label, double c_max, const DepthOfField & dof)
{
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  " << std::left << std::setw(26) << label
            << "c <= " << std::setprecision(4) << std::setw(8) << c_max << " mm"
            << std::setprecision(3)
            << "   near " << std::setw(7) << dof.near_limit / 1000.0 << " m";
  if (dof.reaches_infinity) {
    std::cout << "   far  infinity";
  } else {
    std::cout << "   far  " << std::setw(7) << dof.far_limit / 1000.0 << " m"
              << "   width " << std::setw(7)
              << (dof.far_limit - dof.near_limit) / 1000.0 << " m";
  }
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char ** argv)
{
  // Defaults reproduce the figure of the book: a 50 mm lens at f/2.8 focused
  // at 2 m on a full-frame sensor of 36 mm by 6000 px
  cv::CommandLineParser parser(argc, argv,
    "{help h    |      | Show this help message}"
    "{focal     | 50.0 | Focal length, in mm}"
    "{fnumber   | 2.8  | f-number of the diaphragm (focal / aperture diameter)}"
    "{distance  | 2000 | Distance to the plane in focus, in mm}"
    "{sensor    | 36.0 | Sensor width, in mm}"
    "{pixels    | 6000 | Sensor width, in pixels}"
    "{coc       | 0.03 | Photographic circle of confusion, in mm}");

  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }

  // Without this, a malformed value is reported by the parser but the example
  // carries on with the default, which is the hardest kind of failure to find
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  const double focal = parser.get<double>("focal");
  const double f_number = parser.get<double>("fnumber");
  const double distance = parser.get<double>("distance");
  const double sensor_width = parser.get<double>("sensor");
  const double sensor_pixels = parser.get<double>("pixels");
  const double photographic_coc = parser.get<double>("coc");

  // The formulas below divide by (u - f) and by f_number, and take the object
  // side as positive. Anything else is not a camera and the results would be
  // silently meaningless rather than obviously wrong
  if (focal <= 0.0 || f_number <= 0.0 || sensor_width <= 0.0 || sensor_pixels <= 0.0) {
    std::cerr << "Error: focal, fnumber, sensor and pixels must all be positive."
              << std::endl;
    return EXIT_FAILURE;
  }
  if (distance <= focal) {
    std::cerr << "Error: the object must be farther than the focal length ("
              << focal << " mm). At u = f the rays leave the lens parallel and "
              << "no image is formed." << std::endl;
    return EXIT_FAILURE;
  }

  const double pixel_size = sensor_width / sensor_pixels;
  const double v = imageDistance(focal, distance);
  const double m = magnification(focal, distance);

  std::cout << "Camera" << std::endl;
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  focal length            " << focal << " mm" << std::endl;
  std::cout << "  f-number                f/" << f_number << std::endl;
  std::cout << "  aperture diameter       " << focal / f_number << " mm" << std::endl;
  std::cout << "  focused at              " << distance / 1000.0 << " m" << std::endl;
  std::cout << "  sensor                  " << sensor_width << " mm / "
            << sensor_pixels << " px" << std::endl;
  std::cout << "  pixel size              " << std::setprecision(4) << pixel_size
            << " mm" << std::setprecision(3) << std::endl;

  std::cout << "\nThin lens, 1/f = 1/u + 1/v" << std::endl;
  std::cout << "  image distance v        " << v << " mm" << std::endl;
  std::cout << "  magnification m         " << m
            << "  (negative: the real image is inverted)" << std::endl;

  std::cout << "\nDepth of field, by criterion" << std::endl;
  printDepthOfField("photographic (35 mm)", photographic_coc,
    depthOfField(focal, f_number, distance, photographic_coc));
  printDepthOfField("one pixel", pixel_size,
    depthOfField(focal, f_number, distance, pixel_size));

  // The ratio between the two criteria is also the ratio between the two
  // f-numbers that make them agree, because c is inversely proportional to N.
  // That is the whole content of the comparison, and it is worth printing
  // rather than leaving the reader to notice it
  const double ratio = photographic_coc / pixel_size;
  std::cout << "\n  The photographic criterion is " << std::setprecision(1) << ratio
            << " times more permissive than the pixel." << std::endl;
  std::cout << "  Since c is inversely proportional to the f-number, closing down to f/"
            << std::setprecision(1) << f_number * ratio
            << " gives the pixel criterion the same depth of field" << std::endl;
  std::cout << "  the photographic one gave at f/" << f_number
            << ", at the cost of " << std::setprecision(1) << ratio * ratio
            << " times less light." << std::endl;

  // A point away from the plane in focus, to show the circle growing
  std::cout << "\nCircle of confusion away from the plane in focus" << std::endl;
  std::cout << std::setprecision(4);
  for (const double factor : {0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 2.0}) {
    const double other = distance * factor;
    if (other <= focal) {
      continue;
    }
    std::cout << "  at " << std::setprecision(2) << std::setw(6) << other / 1000.0
              << " m   c = " << std::setprecision(4) << std::setw(8)
              << circleOfConfusion(focal, f_number, distance, other) << " mm"
              << std::endl;
  }

  return EXIT_SUCCESS;
}
