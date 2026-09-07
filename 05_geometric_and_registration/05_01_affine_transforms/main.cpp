/**
 * @file main.cpp
 * @brief Geometric transformations in OpenCV
 * @author José Miguel Guerrero Hernández
 *
 * This example demonstrates:
 * - Translation: shifting an image by (tx, ty) pixels
 * - Rotation: rotating around a center point with optional scaling
 * - Resize: scaling images with different interpolation methods
 * - Affine warp: mapping triangular regions for general affine deformations
 *
 * @note Affine transformations preserve parallel lines and use a 2x3 matrix:
 *       | a  b  tx |
 *       | c  d  ty |
 *
 * @see https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
 */

#include <string>
#include <algorithm>
#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace
{
// The images this example works on are about 1400 px on the long side, and
// several windows at that size do not fit on a normal screen. The processing
// always runs at full resolution: only the copy sent to the screen is reduced,
// with INTER_AREA, which is the interpolation meant for shrinking
constexpr int MAX_DISPLAY_SIDE = 800;

void showFit(const std::string & window, const cv::Mat & image)
{
  const int side = std::max(image.cols, image.rows);
  if (side <= MAX_DISPLAY_SIDE || image.empty()) {
    cv::imshow(window, image);
    return;
  }
  const double factor = static_cast<double>(MAX_DISPLAY_SIDE) / side;
  cv::Mat reduced;
  cv::resize(image, reduced, cv::Size(), factor, factor, cv::INTER_AREA);
  cv::imshow(window, reduced);
}
}  // namespace

// Transformation parameters as named constants
namespace TransformParams
{
constexpr float TRANSLATION_X = 100.0f;
constexpr float TRANSLATION_Y = 100.0f;
constexpr double ROTATION_ANGLE = -50.0;
constexpr double ROTATION_SCALE = 0.6;
constexpr int RESIZE_SCALE_UP = 2;
constexpr int RESIZE_SCALE_DOWN = 2;
}

/**
 * @brief Demonstrates translation transformation on an image
 * @param src Input source image to be translated
 *
 * Shifts the image by (100, 100) pixels using a 2x3 affine transformation matrix.
 */
void demoTranslation(const cv::Mat & src)
{
  std::cout << "\n=== Translation ===" << std::endl;
  std::cout << "Shifting image by (" << TransformParams::TRANSLATION_X << ", "
            << TransformParams::TRANSLATION_Y << ") pixels" << std::endl;

  cv::Mat translation_dst;

  // Build the 2x3 translation matrix using OpenCV API:
  // | 1  0  tx |
  // | 0  1  ty |

  // Alternative (NOT recommended): Using raw array
  // float data[6] = {1, 0, TransformParams::TRANSLATION_X, 0, 1, TransformParams::TRANSLATION_Y};
  // cv::Mat trans_mat(2, 3, CV_32F, data);
  // Problems: 1) Less readable (matrix structure not clear)
  //           2) Array must remain in scope while Mat is used
  //           3) Error-prone with row-major ordering

  // Recommended: Use initializer list syntax (clearer and safer)
  // NOTE: Why 2x3 instead of 3x3?
  // Affine transformations use 2x3 matrices (used by warpAffine):
  //   | a  b  tx |
  //   | c  d  ty |
  // The full homogeneous form is 3x3, but the third row [0, 0, 1] is always
  // constant for affine transforms, so OpenCV omits it for efficiency.
  // Use 3x3 matrices only for perspective transformations (warpPerspective),
  // where parallel lines can converge and the third row has non-constant values.
  cv::Mat trans_mat = (cv::Mat_<float>(2, 3) <<
    1, 0, TransformParams::TRANSLATION_X,
    0, 1, TransformParams::TRANSLATION_Y);

  // Apply the affine transformation to translate the image
  // INTER_LINEAR provides good quality/performance balance
  cv::warpAffine(src, translation_dst, trans_mat, src.size(), cv::INTER_LINEAR);
  showFit("Translation", translation_dst);
}

/**
 * @brief Demonstrates rotation transformation on an image
 * @param src Input source image to be rotated
 *
 * Rotates the image -50 degrees around its center with 0.6x scaling factor.
 */
void demoRotation(const cv::Mat & src)
{
  std::cout << "\n=== Rotation ===" << std::endl;
  std::cout << "Rotating " << TransformParams::ROTATION_ANGLE << " degrees with "
            << TransformParams::ROTATION_SCALE << "x scale" << std::endl;

  cv::Mat rotation_dst;

  // Define the rotation center (center of the image)
  cv::Point center(src.cols / 2, src.rows / 2);

  // Compute the 2x3 rotation matrix
  cv::Mat rot_mat = cv::getRotationMatrix2D(center,
                                            TransformParams::ROTATION_ANGLE,
                                            TransformParams::ROTATION_SCALE);

  // Apply the affine transformation to rotate the image
  // INTER_LINEAR provides good quality for rotations
  cv::warpAffine(src, rotation_dst, rot_mat, src.size(), cv::INTER_LINEAR);
  showFit("Rotation", rotation_dst);
}

/**
 * @brief Demonstrates image resize operations with different interpolation methods
 * @param src Input source image to be resized
 *
 * Shows upscaling (x2) with multiple interpolation methods:
 * - INTER_NEAREST: Fastest, but lowest quality (blocky)
 * - INTER_LINEAR: Good balance of speed and quality (default)
 * - INTER_CUBIC: Best quality for upscaling, slower
 * And downscaling (/2) with INTER_AREA (recommended for downsampling)
 */
void demoResize(const cv::Mat & src)
{
  std::cout << "\n=== Resize ===" << std::endl;

  // Upscale with INTER_LINEAR (recommended for most upscaling)
  // NOTE: cv::resize has TWO ways to specify output size:
  // 1) Using scale factors (fx, fy): cv::Size() is EMPTY, size calculated as src.size * fx/fy
  // 2) Using explicit size: cv::Size(w, h) has VALUES, fx=0 fy=0 are IGNORED
  cv::Mat resize_up_linear;
  cv::resize(src, resize_up_linear, cv::Size(),  // Empty Size → use scale factors below
             TransformParams::RESIZE_SCALE_UP,   // fx = 2 (horizontal scale)
             TransformParams::RESIZE_SCALE_UP,   // fy = 2 (vertical scale)
             cv::INTER_LINEAR);
  std::cout << "Upscale x" << TransformParams::RESIZE_SCALE_UP
            << " using INTER_LINEAR: "
            << src.cols << "x" << src.rows << " -> "
            << resize_up_linear.cols << "x" << resize_up_linear.rows << std::endl;
  showFit("Resize x2 (INTER_LINEAR)", resize_up_linear);

  // Upscale with INTER_CUBIC (best quality, slower)
  cv::Mat resize_up_cubic;
  cv::resize(src, resize_up_cubic, cv::Size(),
             TransformParams::RESIZE_SCALE_UP,
             TransformParams::RESIZE_SCALE_UP,
             cv::INTER_CUBIC);
  showFit("Resize x2 (INTER_CUBIC)", resize_up_cubic);

  // Downscale with INTER_AREA (recommended for shrinking images)
  cv::Mat resize_down;
  // Using explicit size: cv::Size has VALUES (width, height)
  // When using explicit size, fx and fy MUST be 0 (they are ignored)
  cv::resize(src, resize_down,
             cv::Size(src.cols / TransformParams::RESIZE_SCALE_DOWN,  // explicit width
                      src.rows / TransformParams::RESIZE_SCALE_DOWN), // explicit height
             0, 0,  // fx=0, fy=0 → ignored when Size is specified
             cv::INTER_AREA);
  std::cout << "Downscale /" << TransformParams::RESIZE_SCALE_DOWN
            << " using INTER_AREA: "
            << src.cols << "x" << src.rows << " -> "
            << resize_down.cols << "x" << resize_down.rows << std::endl;
  showFit("Resize /2 (INTER_AREA)", resize_down);
}

/**
 * @brief Demonstrates affine warp transformation for general deformations
 * @param src Input source image to be warped
 *
 * Applies a general affine transformation by mapping triangular regions,
 * creating a complex deformation effect, then applies an additional rotation
 * to demonstrate combined transformations.
 */
void demoAffineWarp(const cv::Mat & src)
{
  std::cout << "\n=== Affine Warp (General Deformation) ===" << std::endl;
  std::cout << "Applying general affine transformation + rotation" << std::endl;

  // Define Source Triangle Points
  // These are the original corner positions of the image region to transform
  cv::Point2f src_tri[3];
  src_tri[0] = cv::Point2f(0.f, 0.f);                  // Top-left corner
  src_tri[1] = cv::Point2f(src.cols - 1.f, 0.f);       // Top-right corner
  src_tri[2] = cv::Point2f(0.f, src.rows - 1.f);       // Bottom-left corner

  // Define Destination Triangle Points
  // These points define where the source corners will be mapped to,
  // creating a general affine deformation effect
  cv::Point2f dst_tri[3];
  dst_tri[0] = cv::Point2f(0.f, src.rows * 0.33f);              // New top-left
  dst_tri[1] = cv::Point2f(src.cols * 0.85f, src.rows * 0.25f); // New top-right
  dst_tri[2] = cv::Point2f(src.cols * 0.15f, src.rows * 0.7f);  // New bottom-left

  // Apply Affine Warp Transformation
  // Computes and applies the transformation matrix from source to destination
  cv::Mat warp_mat = cv::getAffineTransform(src_tri, dst_tri);
  cv::Mat warp_dst;
  cv::warpAffine(src, warp_dst, warp_mat, src.size(), cv::INTER_LINEAR);
  showFit("Affine Warp (Deformation)", warp_dst);

  // Apply Rotation to warped image (reusing transformation parameters)
  cv::Point center(warp_dst.cols / 2, warp_dst.rows / 2);
  cv::Mat rot_mat = cv::getRotationMatrix2D(center,
                                            TransformParams::ROTATION_ANGLE,
                                            TransformParams::ROTATION_SCALE);
  cv::Mat warp_rotate_dst;
  cv::warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size(), cv::INTER_LINEAR);
  showFit("Warp + Rotate", warp_rotate_dst);
}

int main(int argc, char ** argv)
{
  // Load input image
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/starry_night.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string image_path = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }
  cv::Mat src = cv::imread(cv::samples::findFile(image_path, false), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << image_path << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "=== Geometric Transformations Demo ===" << std::endl;
  std::cout << "Image loaded: " << src.cols << "x" << src.rows << " pixels" << std::endl;

  // Display original image
  showFit("Original", src);

  // Run all transformation demos
  demoTranslation(src);
  demoRotation(src);
  demoResize(src);
  demoAffineWarp(src);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
