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

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

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
  cv::imshow("Translation", translation_dst);
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
  cv::imshow("Rotation", rotation_dst);
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
  cv::imshow("Resize x2 (INTER_LINEAR)", resize_up_linear);

  // Upscale with INTER_CUBIC (best quality, slower)
  cv::Mat resize_up_cubic;
  cv::resize(src, resize_up_cubic, cv::Size(),
             TransformParams::RESIZE_SCALE_UP,
             TransformParams::RESIZE_SCALE_UP,
             cv::INTER_CUBIC);
  cv::imshow("Resize x2 (INTER_CUBIC)", resize_up_cubic);

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
  cv::imshow("Resize /2 (INTER_AREA)", resize_down);
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
  cv::Point2f srcTri[3];
  srcTri[0] = cv::Point2f(0.f, 0.f);                  // Top-left corner
  srcTri[1] = cv::Point2f(src.cols - 1.f, 0.f);       // Top-right corner
  srcTri[2] = cv::Point2f(0.f, src.rows - 1.f);       // Bottom-left corner

  // Define Destination Triangle Points
  // These points define where the source corners will be mapped to,
  // creating a general affine deformation effect
  cv::Point2f dstTri[3];
  dstTri[0] = cv::Point2f(0.f, src.rows * 0.33f);              // New top-left
  dstTri[1] = cv::Point2f(src.cols * 0.85f, src.rows * 0.25f); // New top-right
  dstTri[2] = cv::Point2f(src.cols * 0.15f, src.rows * 0.7f);  // New bottom-left

  // Apply Affine Warp Transformation
  // Computes and applies the transformation matrix from source to destination
  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
  cv::Mat warp_dst;
  cv::warpAffine(src, warp_dst, warp_mat, src.size(), cv::INTER_LINEAR);
  cv::imshow("Affine Warp (Deformation)", warp_dst);

  // Apply Rotation to warped image (reusing transformation parameters)
  cv::Point center(warp_dst.cols / 2, warp_dst.rows / 2);
  cv::Mat rot_mat = cv::getRotationMatrix2D(center,
                                            TransformParams::ROTATION_ANGLE,
                                            TransformParams::ROTATION_SCALE);
  cv::Mat warp_rotate_dst;
  cv::warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size(), cv::INTER_LINEAR);
  cv::imshow("Warp + Rotate", warp_rotate_dst);
}

int main(int argc, char ** argv)
{
  // Load input image
  const std::string imagePath = argc >= 2 ? argv[1] : "lena.jpg";
  cv::Mat src = cv::imread(cv::samples::findFile(imagePath), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image!" << std::endl;
    std::cerr << "Path: " << imagePath << std::endl;
    std::cerr << "Usage: " << argv[0] << " <Input image>" << std::endl;
    return -1;
  }

  std::cout << "=== Geometric Transformations Demo ===" << std::endl;
  std::cout << "Image loaded: " << src.cols << "x" << src.rows << " pixels" << std::endl;

  // Display original image
  cv::imshow("Original", src);

  // Run all transformation demos
  demoTranslation(src);
  demoRotation(src);
  demoResize(src);
  demoAffineWarp(src);

  std::cout << "\nPress any key to exit..." << std::endl;
  cv::waitKey(0);

  return 0;
}
