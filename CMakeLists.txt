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
  filters registration recognition kdtree sample_consensus
  segmentation search)

# --- Helper: add an OpenCV example --------------------------------------------
# Usage: add_cv_example(<relative/path/to/dir> [EXTRA_LIBS lib1 lib2 ...])
#
# Rules:
#   main.cpp       → executable named <dir_name>
#   other.cpp      → executable named <dir_name>_<stem>   (e.g. frequencies.cpp
#                    in 05_01_dft → 05_01_dft_frequencies)
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

# Sections and order follow the book, chapter by chapter. The number in the
# heading is the book chapter, which is also the prefix of the folder.

# --- 02 Image formation --------------------------------------------------------
add_cv_example(02_image_formation/02_01_read_image)
add_cv_example(02_image_formation/02_02_color_spaces)
add_cv_example(02_image_formation/02_03_mat_copy_roi)
add_cv_example(02_image_formation/02_04_pixel_access)
add_cv_example(02_image_formation/02_05_video_capture)

# --- 03 Pixel operations and spatial filtering ---------------------------------
add_cv_example(03_pixel_and_filtering/03_01_point_operations)
add_cv_example(03_pixel_and_filtering/03_02_convolution)
add_cv_example(03_pixel_and_filtering/03_03_bitwise_operations)
add_cv_example(03_pixel_and_filtering/03_04_intensity_transforms)
add_cv_example(03_pixel_and_filtering/03_05_smoothing)

# --- 04 The histogram ----------------------------------------------------------
add_cv_example(04_histogram/04_01_histogram_equalization)
add_cv_example(04_histogram/04_02_histogram_matching)
add_cv_example(04_histogram/04_03_histogram_comparison)

# --- 05 Frequency-domain transforms --------------------------------------------
# 05_01_dft has two sources -> 05_01_dft + 05_01_dft_frequencies
add_cv_example(05_frequency/05_01_dft)
add_cv_example(05_frequency/05_02_dct)
add_cv_example(05_frequency/05_03_wavelet_denoising)
add_cv_example(05_frequency/05_04_gabor_bank)
add_cv_example(05_frequency/05_05_homomorphic_filter)

# --- 06 Geometric transforms and registration ----------------------------------
add_cv_example(06_geometric_and_registration/06_01_affine_transforms)
add_cv_example(06_geometric_and_registration/06_02_perspective_correction)

# --- 07 Edge detection ---------------------------------------------------------
add_cv_example(07_edge_detection/07_01_sobel_edges)
add_cv_example(07_edge_detection/07_02_canny_edges)
add_cv_example(07_edge_detection/07_03_laplacian_edges)
add_cv_example(07_edge_detection/07_04_contour_extraction)
add_cv_example(07_edge_detection/07_05_chain_code)

# --- 08 Model fitting ----------------------------------------------------------
add_cv_example(08_model_fitting/08_01_hough_lines)
add_cv_example(08_model_fitting/08_02_hough_circles)

# --- 09 Region segmentation ----------------------------------------------------
add_cv_example(09_region_segmentation/09_01_threshold)
add_cv_example(09_region_segmentation/09_02_connected_components)
add_cv_example(09_region_segmentation/09_03_color_segmentation)

# --- 10 Morphological operations -----------------------------------------------
add_cv_example(10_morphological_operations/10_01_erode_dilate)
add_cv_example(10_morphological_operations/10_02_opening_closing)
add_cv_example(10_morphological_operations/10_03_morphological_gradient)
add_cv_example(10_morphological_operations/10_04_hit_or_miss)
add_cv_example(10_morphological_operations/10_05_skeletonization)
add_cv_example(10_morphological_operations/10_06_flood_fill)
add_cv_example(10_morphological_operations/10_07_tophat_illumination)
add_cv_example(10_morphological_operations/10_08_distance_watershed)

# --- 11 Region descriptors -----------------------------------------------------
add_cv_example(11_region_descriptors/11_01_region_moments)
add_cv_example(11_region_descriptors/11_02_hu_moments)

# --- 12 Keypoints --------------------------------------------------------------
add_cv_example(12_keypoints/12_01_harris_corners)
add_cv_example(12_keypoints/12_02_shi_tomasi_corners)
add_cv_example(12_keypoints/12_03_orb_features)
add_cv_example(12_keypoints/12_04_ransac_matching)

# --- 13 Camera geometry and calibration ----------------------------------------
add_cv_example(13_camera_calibration/13_01_camera_calibration)
add_cv_example(13_camera_calibration/13_02_pose_estimation)
add_cv_example(13_camera_calibration/13_03_stereo_calibration)

# --- 14 3D vision and point clouds ---------------------------------------------
add_cv_example(14_3d_and_point_clouds/14_01_epipolar_geometry)
add_cv_example(14_3d_and_point_clouds/14_02_stereo_disparity)
add_cv_example(14_3d_and_point_clouds/14_03_stereo_to_pointcloud)
add_cv_example(14_3d_and_point_clouds/14_04_opencv_icp)
add_pcl_example(14_3d_and_point_clouds/14_05_pcl_write)
add_pcl_example(14_3d_and_point_clouds/14_06_pcl_read)
add_pcl_example(14_3d_and_point_clouds/14_07_pcl_simple_visualizer)
add_pcl_example(14_3d_and_point_clouds/14_08_pcl_advanced_visualizer)
add_pcl_example(14_3d_and_point_clouds/14_09_pcl_icp)
add_pcl_example(14_3d_and_point_clouds/14_10_pcl_ransac_model_fitting)
add_pcl_example(14_3d_and_point_clouds/14_11_pcl_registration)
add_pcl_example(14_3d_and_point_clouds/14_12_pcl_correspondence)
add_pcl_example(14_3d_and_point_clouds/14_13_pcl_plane_clustering)

# --- 15 Optical flow and tracking ----------------------------------------------
add_cv_example(15_optical_flow_and_tracking/15_01_frame_difference)
add_cv_example(15_optical_flow_and_tracking/15_02_sparse_flow)
add_cv_example(15_optical_flow_and_tracking/15_03_dense_flow)
add_cv_example(15_optical_flow_and_tracking/15_04_background_subtraction)
add_cv_example(15_optical_flow_and_tracking/15_05_kalman_tracking)
add_cv_example(15_optical_flow_and_tracking/15_06_object_tracking)

# --- 16 Classical machine learning ---------------------------------------------
add_cv_example(16_classical_ml/16_01_knn_classification)
add_cv_example(16_classical_ml/16_02_svm_classification)
add_cv_example(16_classical_ml/16_03_digits_classification)
add_cv_example(16_classical_ml/16_04_kmeans_clustering)
add_cv_example(16_classical_ml/16_05_classifier_comparison)
add_cv_example(16_classical_ml/16_06_som_digits)

# The last three fetch or export their model on first build, each with the
# script that lives beside its source
add_cv_example(17_deep_learning/17_01_yolov4_darknet
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 17_01_yolov4_darknet PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/17_deep_learning/17_01_yolov4_darknet/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/17_deep_learning/17_01_yolov4_darknet
  COMMENT "Downloading YOLOv4-tiny model if needed..."
)

add_cv_example(17_deep_learning/17_02_yolo_ultralytics
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 17_02_yolo_ultralytics PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/17_deep_learning/17_02_yolo_ultralytics/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/17_deep_learning/17_02_yolo_ultralytics
  COMMENT "Exporting YOLO11 model if needed..."
)


# --- 17 Deep learning ----------------------------------------------------------
add_cv_example(17_deep_learning/17_03_semantic_segmentation)
add_custom_command(
  TARGET 17_03_semantic_segmentation PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/17_deep_learning/17_03_semantic_segmentation/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/17_deep_learning/17_03_semantic_segmentation
  COMMENT "Exporting DeepLabV3 model if needed..."
)
