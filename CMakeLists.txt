cmake_minimum_required(VERSION 3.16)
project(vision_examples LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- All executables land in <root>/vision_examples/bin/ ---------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/vision_examples/bin)

# --- Global compiler flags ----------------------------------------------------
add_compile_options(-Wall -Wextra -O2)

# --- Dependencies -------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(MPI COMPONENTS C CXX)  # Must precede PCL so VTK::mpi resolves MPI::MPI_C

find_package(PCL 1.10 REQUIRED COMPONENTS
  common io visualization keypoints features
  filters registration recognition kdtree sample_consensus)

# --- Helper: add an OpenCV example --------------------------------------------
# Usage: add_cv_example(<relative/path/to/dir> [EXTRA_LIBS lib1 lib2 ...])
#
# Rules:
#   main.cpp       → executable named <dir_name>
#   other.cpp      → executable named <dir_name>_<stem>   (e.g. frequencies.cpp
#                    in 03_01_fourier → 03_01_fourier_frequencies)
function(add_cv_example REL_PATH)
  cmake_parse_arguments(ARG "" "" "EXTRA_LIBS" ${ARGN})
  get_filename_component(DIR_NAME ${REL_PATH} NAME)
  file(GLOB SRCS "${CMAKE_SOURCE_DIR}/${REL_PATH}/*.cpp")
  foreach(SRC ${SRCS})
    get_filename_component(STEM ${SRC} NAME_WE)
    if(STEM STREQUAL "main")
      set(TGT ${DIR_NAME})
    else()
      set(TGT "${DIR_NAME}_${STEM}")
    endif()
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} PRIVATE ${OpenCV_LIBS} ${ARG_EXTRA_LIBS})
  endforeach()
endfunction()

# --- Helper: add a PCL example ------------------------------------------------
# Usage: add_pcl_example(<relative/path/to/dir>)
#
# Executable is named <dir_name>.
# -Wno-array-bounds suppresses a GCC 13 false positive triggered by Eigen SIMD.
# -Wl,--copy-dt-needed-entries resolves VTK symbols that PCL links transitively.
function(add_pcl_example REL_PATH)
  get_filename_component(DIR_NAME ${REL_PATH} NAME)
  add_executable(${DIR_NAME} "${CMAKE_SOURCE_DIR}/${REL_PATH}/main.cpp")
  target_include_directories(${DIR_NAME} PRIVATE
    ${PCL_INCLUDE_DIRS}
    /usr/include/vtk-9.1)
  target_link_libraries(${DIR_NAME} PRIVATE ${PCL_LIBRARIES})
  target_compile_definitions(${DIR_NAME} PRIVATE ${PCL_DEFINITIONS})
  target_compile_options(${DIR_NAME} PRIVATE -Wno-array-bounds)
  target_link_options(${DIR_NAME} PRIVATE -Wl,--copy-dt-needed-entries)
endfunction()

# --- 02 Basic Concepts --------------------------------------------------------
add_cv_example(02_basic_concepts/02_01_read_image)
add_cv_example(02_basic_concepts/02_02_pixels)
add_cv_example(02_basic_concepts/02_03_color_spaces)
add_cv_example(02_basic_concepts/02_04_video_capture)

# --- 03 Image Transforms ------------------------------------------------------
# 03_01_fourier has two sources → 03_01_fourier + 03_01_fourier_frequencies
add_cv_example(03_image_transforms/03_01_fourier)
add_cv_example(03_image_transforms/03_02_cosine)
add_cv_example(03_image_transforms/03_03_wavelets)
add_cv_example(03_image_transforms/03_04_point_operations)
add_cv_example(03_image_transforms/03_05_convolution)
add_cv_example(03_image_transforms/03_06_bitwise_operations)

# --- 04 Geometric and Enhancement ---------------------------------------------
add_cv_example(04_geometric_and_enhancement/04_01_affine_transforms)
add_cv_example(04_geometric_and_enhancement/04_02_smoothing)
add_cv_example(04_geometric_and_enhancement/04_03_histogram_equalization)
add_cv_example(04_geometric_and_enhancement/04_04_histogram_comparison)

# --- 05 Segmentation and Features ---------------------------------------------
add_cv_example(05_segmentation_and_features/05_01_sobel)
add_cv_example(05_segmentation_and_features/05_02_canny)
add_cv_example(05_segmentation_and_features/05_03_laplacian)
add_cv_example(05_segmentation_and_features/05_04_hough_lines)
add_cv_example(05_segmentation_and_features/05_05_hough_circles)
add_cv_example(05_segmentation_and_features/05_06_threshold)
add_cv_example(05_segmentation_and_features/05_07_contours)
add_cv_example(05_segmentation_and_features/05_08_moments)
add_cv_example(05_segmentation_and_features/05_09_hu_moments)
add_cv_example(05_segmentation_and_features/05_10_chain_code)
add_cv_example(05_segmentation_and_features/05_11_harris_corners)
add_cv_example(05_segmentation_and_features/05_12_good_features)
add_cv_example(05_segmentation_and_features/05_13_orb)
add_cv_example(05_segmentation_and_features/05_14_ransac_matching)
add_cv_example(05_segmentation_and_features/05_15_connected_components)

# --- 06 Morphological Operations ----------------------------------------------
add_cv_example(06_morphological_operations/06_01_erode_dilate)
add_cv_example(06_morphological_operations/06_02_opening_closing)
add_cv_example(06_morphological_operations/06_03_morphological_gradient)
add_cv_example(06_morphological_operations/06_04_skeletonization)
add_cv_example(06_morphological_operations/06_05_flood_fill)

# --- 07 Camera Calibration ----------------------------------------------------
add_cv_example(07_camera_calibration/07_01_camera_calibration)

# --- 08 Stereo and 3D ---------------------------------------------------------
add_cv_example(08_stereo_and_3d/08_01_stereo_disparity)
add_cv_example(08_stereo_and_3d/08_02_opencv_icp)

add_pcl_example(08_stereo_and_3d/08_03_pcl_write)
add_pcl_example(08_stereo_and_3d/08_04_pcl_read)
add_pcl_example(08_stereo_and_3d/08_05_pcl_simple_visualizer)
add_pcl_example(08_stereo_and_3d/08_06_pcl_advanced_visualizer)
add_pcl_example(08_stereo_and_3d/08_07_pcl_icp)
add_pcl_example(08_stereo_and_3d/08_08_pcl_registration)
add_pcl_example(08_stereo_and_3d/08_09_pcl_correspondence)
add_pcl_example(08_stereo_and_3d/08_10_pcl_ransac_registration)

# --- 09 Optical Flow ----------------------------------------------------------
add_cv_example(09_optical_flow/09_01_frame_difference)
add_cv_example(09_optical_flow/09_02_sparse_flow)
add_cv_example(09_optical_flow/09_03_dense_flow)

# --- 10 ML and Deep Learning --------------------------------------------------
add_cv_example(10_ml_and_deep_learning/10_01_knn)
add_cv_example(10_ml_and_deep_learning/10_02_svm)
add_cv_example(10_ml_and_deep_learning/10_03_kmeans)
add_cv_example(10_ml_and_deep_learning/10_04_ml_comparison)
add_cv_example(10_ml_and_deep_learning/10_05_yolov4_darknet
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 10_05_yolov4_darknet PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/10_ml_and_deep_learning/10_05_yolov4_darknet/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/10_ml_and_deep_learning/10_05_yolov4_darknet
  COMMENT "Downloading YOLOv4-tiny model if needed..."
)
add_cv_example(10_ml_and_deep_learning/10_06_yolo_ultralytics
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 10_06_yolo_ultralytics PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/10_ml_and_deep_learning/10_06_yolo_ultralytics/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/10_ml_and_deep_learning/10_06_yolo_ultralytics
  COMMENT "Exporting YOLO11 model if needed..."
)