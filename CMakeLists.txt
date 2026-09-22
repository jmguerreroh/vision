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
find_package(jsoncpp CONFIG)  # Must precede PCL so VTK::jsoncpp resolves JsonCpp::JsonCpp

# VTK headers are needed to resolve the symbols PCL links transitively. The
# directory carries the version in its name (vtk-9.1 on Ubuntu 22.04, vtk-9.3 on
# 24.04) and lives elsewhere on other distributions, so it is asked for instead
# of hardcoded.
#
# VTK 9 exports targets and leaves VTK_INCLUDE_DIRS empty, so it has to be
# rebuilt from the prefix and the version. Checked, not assumed: writing
# ${VTK_INCLUDE_DIRS} alone dropped the path without a single warning, which is
# the same silent failure this block is here to avoid.
find_package(VTK QUIET)
if(VTK_FOUND AND NOT VTK_INCLUDE_DIRS)
  set(VTK_INCLUDE_DIRS
      "${VTK_PREFIX_PATH}/include/vtk-${VTK_VERSION_MAJOR}.${VTK_VERSION_MINOR}")
endif()
if(VTK_INCLUDE_DIRS AND NOT EXISTS "${VTK_INCLUDE_DIRS}")
  message(WARNING "VTK headers not found at ${VTK_INCLUDE_DIRS}; "
                  "the PCL examples may fail to compile")
endif()

# Optional: PCL examples (chapter 15) are skipped if PCL isn't installed.
find_package(PCL 1.10 QUIET COMPONENTS
  common io visualization keypoints features
  filters registration recognition kdtree sample_consensus
  segmentation search)
if(NOT PCL_FOUND)
  message(WARNING "PCL 1.10 not found - skipping chapter 15 PCL examples. "
    "To install it on Ubuntu/Debian: sudo apt install libpcl-dev")
endif()

# --- Helper: add an OpenCV example --------------------------------------------
# Usage: add_cv_example(<relative/path/to/dir> [EXTRA_LIBS lib1 lib2 ...])
#
# Rules:
#   main.cpp       → executable named <dir_name>
#   other.cpp      → executable named <dir_name>_<stem>   (e.g. frequencies.cpp
#                    in 06_01_dft → 06_01_dft_frequencies)
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
    ${VTK_INCLUDE_DIRS})
  target_link_libraries(${DIR_NAME} PRIVATE ${PCL_LIBRARIES})
  target_compile_definitions(${DIR_NAME} PRIVATE ${PCL_DEFINITIONS})
  target_compile_options(${DIR_NAME} PRIVATE -Wno-array-bounds)
  target_link_options(${DIR_NAME} PRIVATE -Wl,--copy-dt-needed-entries)
endfunction()

# Sections and order follow the book, chapter by chapter. The number in the
# heading is the book chapter, which is also the prefix of the folder.

# --- 02 Image formation --------------------------------------------------------
add_cv_example(02_image_formation/02_01_thin_lens_dof)

# --- 03 The digital image and color --------------------------------------------
add_cv_example(03_digital_image_and_color/03_01_read_image)
add_cv_example(03_digital_image_and_color/03_02_color_spaces)
add_cv_example(03_digital_image_and_color/03_03_mat_copy_roi)
add_cv_example(03_digital_image_and_color/03_04_pixel_access)
add_cv_example(03_digital_image_and_color/03_05_video_capture)

# --- 04 Pixel operations and spatial filtering ---------------------------------
add_cv_example(04_pixel_and_filtering/04_01_point_operations)
add_cv_example(04_pixel_and_filtering/04_02_convolution)
add_cv_example(04_pixel_and_filtering/04_03_bitwise_operations)
add_cv_example(04_pixel_and_filtering/04_04_intensity_transforms)
add_cv_example(04_pixel_and_filtering/04_05_smoothing)

# --- 05 The histogram ----------------------------------------------------------
add_cv_example(05_histogram/05_01_histogram_equalization)
add_cv_example(05_histogram/05_02_histogram_matching)
add_cv_example(05_histogram/05_03_histogram_comparison)

# --- 06 Frequency-domain transforms --------------------------------------------
# 06_01_dft has two sources -> 06_01_dft + 06_01_dft_frequencies
add_cv_example(06_frequency/06_01_dft)
add_cv_example(06_frequency/06_02_dct)
add_cv_example(06_frequency/06_03_wavelet_denoising)
add_cv_example(06_frequency/06_04_gabor_bank)
add_cv_example(06_frequency/06_05_homomorphic_filter)

# --- 07 Geometric transforms and registration ----------------------------------
add_cv_example(07_geometric_and_registration/07_01_affine_transforms)
add_cv_example(07_geometric_and_registration/07_02_perspective_correction)

# --- 08 Edge detection ---------------------------------------------------------
add_cv_example(08_edge_detection/08_01_sobel_edges)
add_cv_example(08_edge_detection/08_02_canny_edges)
add_cv_example(08_edge_detection/08_03_laplacian_edges)
add_cv_example(08_edge_detection/08_04_contour_extraction)
add_cv_example(08_edge_detection/08_05_chain_code)

# --- 09 Model fitting ----------------------------------------------------------
add_cv_example(09_model_fitting/09_01_hough_lines)
add_cv_example(09_model_fitting/09_02_hough_circles)

# --- 10 Region segmentation ----------------------------------------------------
add_cv_example(10_region_segmentation/10_01_threshold)
add_cv_example(10_region_segmentation/10_02_connected_components)
add_cv_example(10_region_segmentation/10_03_color_segmentation)

# --- 11 Morphological operations -----------------------------------------------
add_cv_example(11_morphological_operations/11_01_erode_dilate)
add_cv_example(11_morphological_operations/11_02_opening_closing)
add_cv_example(11_morphological_operations/11_03_morphological_gradient)
add_cv_example(11_morphological_operations/11_04_hit_or_miss)
add_cv_example(11_morphological_operations/11_05_skeletonization)
add_cv_example(11_morphological_operations/11_06_flood_fill)
add_cv_example(11_morphological_operations/11_07_tophat_illumination)
add_cv_example(11_morphological_operations/11_08_distance_watershed)

# --- 12 Region descriptors -----------------------------------------------------
add_cv_example(12_region_descriptors/12_01_region_moments)
add_cv_example(12_region_descriptors/12_02_hu_moments)
add_cv_example(12_region_descriptors/12_03_convex_hull)

# --- 13 Keypoints --------------------------------------------------------------
add_cv_example(13_keypoints/13_01_harris_corners)
add_cv_example(13_keypoints/13_02_shi_tomasi_corners)
add_cv_example(13_keypoints/13_03_orb_features)
add_cv_example(13_keypoints/13_04_ransac_matching)

# --- 14 Camera geometry and calibration ----------------------------------------
add_cv_example(14_camera_calibration/14_01_camera_calibration)
add_cv_example(14_camera_calibration/14_02_pose_estimation)
add_cv_example(14_camera_calibration/14_03_stereo_calibration)

# --- 15 3D vision and point clouds ---------------------------------------------
add_cv_example(15_3d_and_point_clouds/15_01_epipolar_geometry)
add_cv_example(15_3d_and_point_clouds/15_02_stereo_disparity)
add_cv_example(15_3d_and_point_clouds/15_03_stereo_to_pointcloud)
add_cv_example(15_3d_and_point_clouds/15_04_opencv_icp)
if(PCL_FOUND)
  add_pcl_example(15_3d_and_point_clouds/15_05_pcl_write)
  add_pcl_example(15_3d_and_point_clouds/15_06_pcl_read)
  add_pcl_example(15_3d_and_point_clouds/15_07_pcl_simple_visualizer)
  add_pcl_example(15_3d_and_point_clouds/15_08_pcl_advanced_visualizer)
  add_pcl_example(15_3d_and_point_clouds/15_09_pcl_icp)
  add_pcl_example(15_3d_and_point_clouds/15_10_pcl_ransac_model_fitting)
  add_pcl_example(15_3d_and_point_clouds/15_11_pcl_registration)
  add_pcl_example(15_3d_and_point_clouds/15_12_pcl_correspondence)
  add_pcl_example(15_3d_and_point_clouds/15_13_pcl_plane_clustering)
endif()

# --- 16 Optical flow and tracking ----------------------------------------------
add_cv_example(16_optical_flow_and_tracking/16_01_frame_difference)
add_cv_example(16_optical_flow_and_tracking/16_02_sparse_flow)
add_cv_example(16_optical_flow_and_tracking/16_03_dense_flow)
add_cv_example(16_optical_flow_and_tracking/16_04_background_subtraction)
add_cv_example(16_optical_flow_and_tracking/16_05_kalman_tracking)
add_cv_example(16_optical_flow_and_tracking/16_06_object_tracking)

# --- 17 Classical machine learning ---------------------------------------------
add_cv_example(17_classical_ml/17_01_knn_classification)
add_cv_example(17_classical_ml/17_02_svm_classification)
add_cv_example(17_classical_ml/17_03_digits_classification)
add_cv_example(17_classical_ml/17_04_kmeans_clustering)
add_cv_example(17_classical_ml/17_05_classifier_comparison)
add_cv_example(17_classical_ml/17_06_som_digits)

# --- 18 Deep learning ----------------------------------------------------------
# The three of them fetch or export their model on first build, each with the
# script that lives beside its source
add_cv_example(18_deep_learning/18_01_yolov4_darknet
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 18_01_yolov4_darknet PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/18_deep_learning/18_01_yolov4_darknet/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/18_deep_learning/18_01_yolov4_darknet
  COMMENT "Downloading YOLOv4-tiny model if needed..."
)

add_cv_example(18_deep_learning/18_02_yolo_ultralytics
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 18_02_yolo_ultralytics PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/18_deep_learning/18_02_yolo_ultralytics/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/18_deep_learning/18_02_yolo_ultralytics
  COMMENT "Exporting YOLO11 model if needed..."
)

add_cv_example(18_deep_learning/18_03_semantic_segmentation)
add_custom_command(
  TARGET 18_03_semantic_segmentation PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/18_deep_learning/18_03_semantic_segmentation/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/18_deep_learning/18_03_semantic_segmentation
  COMMENT "Exporting DeepLabV3 model if needed..."
)
