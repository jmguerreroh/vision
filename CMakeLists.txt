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
#                    in 04_01_dft → 04_01_dft_frequencies)
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

# --- 02 Image formation -------------------------------------------------------
add_cv_example(02_image_formation/02_01_read_image)
add_cv_example(02_image_formation/02_02_color_spaces)
add_cv_example(02_image_formation/02_03_mat_copy_roi)
add_cv_example(02_image_formation/02_04_pixel_access)
add_cv_example(02_image_formation/02_05_video_capture)

# --- 03 Spatial and radiometric transforms ------------------------------------
add_cv_example(03_spatial_and_radiometric/03_01_point_operations)
add_cv_example(03_spatial_and_radiometric/03_02_convolution)
add_cv_example(03_spatial_and_radiometric/03_03_bitwise_operations)
add_cv_example(03_spatial_and_radiometric/03_04_intensity_transforms)
add_cv_example(03_spatial_and_radiometric/03_05_smoothing)
add_cv_example(03_spatial_and_radiometric/03_06_histogram_equalization)
add_cv_example(03_spatial_and_radiometric/03_07_histogram_matching)
add_cv_example(03_spatial_and_radiometric/03_08_histogram_comparison)

# --- 04 Frequency-domain transforms -------------------------------------------
# 04_01_dft has two sources -> 04_01_dft + 04_01_dft_frequencies
add_cv_example(04_frequency/04_01_dft)
add_cv_example(04_frequency/04_02_dct)
add_cv_example(04_frequency/04_03_wavelet_denoising)
add_cv_example(04_frequency/04_04_gabor_bank)
add_cv_example(04_frequency/04_05_homomorphic_filter)

# --- 05 Geometric transforms and registration ---------------------------------
add_cv_example(05_geometric_and_registration/05_01_affine_transforms)
add_cv_example(05_geometric_and_registration/05_02_perspective_correction)

# --- 06 Edge detection and model fitting --------------------------------------
# Order follows the book: edges -> contour description -> model fitting
add_cv_example(06_edges_and_model_fitting/06_01_sobel_edges)
add_cv_example(06_edges_and_model_fitting/06_02_canny_edges)
add_cv_example(06_edges_and_model_fitting/06_03_laplacian_edges)
add_cv_example(06_edges_and_model_fitting/06_04_contour_extraction)
add_cv_example(06_edges_and_model_fitting/06_05_chain_code)
add_cv_example(06_edges_and_model_fitting/06_06_hough_lines)
add_cv_example(06_edges_and_model_fitting/06_07_hough_circles)

# --- 07 Region segmentation ---------------------------------------------------
add_cv_example(07_region_segmentation/07_01_threshold)
add_cv_example(07_region_segmentation/07_02_connected_components)
add_cv_example(07_region_segmentation/07_03_color_segmentation)

# --- 08 Morphological operations ----------------------------------------------
add_cv_example(08_morphological_operations/08_01_erode_dilate)
add_cv_example(08_morphological_operations/08_02_opening_closing)
add_cv_example(08_morphological_operations/08_03_morphological_gradient)
add_cv_example(08_morphological_operations/08_04_hit_or_miss)
add_cv_example(08_morphological_operations/08_05_skeletonization)
add_cv_example(08_morphological_operations/08_06_flood_fill)
add_cv_example(08_morphological_operations/08_07_tophat_illumination)
add_cv_example(08_morphological_operations/08_08_distance_watershed)

# --- 09 Descriptors and keypoints ---------------------------------------------
add_cv_example(09_features_and_keypoints/09_01_region_moments)
add_cv_example(09_features_and_keypoints/09_02_hu_moments)
add_cv_example(09_features_and_keypoints/09_03_harris_corners)
add_cv_example(09_features_and_keypoints/09_04_shi_tomasi_corners)
add_cv_example(09_features_and_keypoints/09_05_orb_features)
add_cv_example(09_features_and_keypoints/09_06_ransac_matching)

# --- 10 Camera geometry and calibration ---------------------------------------
add_cv_example(10_camera_calibration/10_01_camera_calibration)
add_cv_example(10_camera_calibration/10_02_pose_estimation)
add_cv_example(10_camera_calibration/10_03_stereo_calibration)

# --- 11 3D vision and point clouds --------------------------------------------
add_cv_example(11_3d_and_point_clouds/11_01_epipolar_geometry)
add_cv_example(11_3d_and_point_clouds/11_02_stereo_disparity)
add_cv_example(11_3d_and_point_clouds/11_03_stereo_to_pointcloud)
add_cv_example(11_3d_and_point_clouds/11_04_opencv_icp)
add_pcl_example(11_3d_and_point_clouds/11_05_pcl_write)
add_pcl_example(11_3d_and_point_clouds/11_06_pcl_read)
add_pcl_example(11_3d_and_point_clouds/11_07_pcl_simple_visualizer)
add_pcl_example(11_3d_and_point_clouds/11_08_pcl_advanced_visualizer)
add_pcl_example(11_3d_and_point_clouds/11_09_pcl_icp)
add_pcl_example(11_3d_and_point_clouds/11_10_pcl_ransac_model_fitting)
add_pcl_example(11_3d_and_point_clouds/11_11_pcl_registration)
add_pcl_example(11_3d_and_point_clouds/11_12_pcl_correspondence)
add_pcl_example(11_3d_and_point_clouds/11_13_pcl_plane_clustering)

# --- 12 Optical flow and tracking ---------------------------------------------
add_cv_example(12_optical_flow_and_tracking/12_01_frame_difference)
add_cv_example(12_optical_flow_and_tracking/12_02_sparse_flow)
add_cv_example(12_optical_flow_and_tracking/12_03_dense_flow)
add_cv_example(12_optical_flow_and_tracking/12_04_background_subtraction)
add_cv_example(12_optical_flow_and_tracking/12_05_kalman_tracking)
add_cv_example(12_optical_flow_and_tracking/12_06_object_tracking)

# --- 13 Machine learning ------------------------------------------------------
add_cv_example(13_machine_learning/13_01_knn_classification)
add_cv_example(13_machine_learning/13_02_svm_classification)
add_cv_example(13_machine_learning/13_03_digits_classification)
add_cv_example(13_machine_learning/13_04_kmeans_clustering)
add_cv_example(13_machine_learning/13_05_classifier_comparison)
add_cv_example(13_machine_learning/13_06_som_digits)

# The last three fetch or export their model on first build, each with the
# script that lives beside its source
add_cv_example(13_machine_learning/13_07_yolov4_darknet
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 13_07_yolov4_darknet PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/13_machine_learning/13_07_yolov4_darknet/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/13_machine_learning/13_07_yolov4_darknet
  COMMENT "Downloading YOLOv4-tiny model if needed..."
)

add_cv_example(13_machine_learning/13_08_yolo_ultralytics
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 13_08_yolo_ultralytics PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/13_machine_learning/13_08_yolo_ultralytics/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/13_machine_learning/13_08_yolo_ultralytics
  COMMENT "Exporting YOLO11 model if needed..."
)

add_cv_example(13_machine_learning/13_09_semantic_segmentation)
add_custom_command(
  TARGET 13_09_semantic_segmentation PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/13_machine_learning/13_09_semantic_segmentation/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/13_machine_learning/13_09_semantic_segmentation
  COMMENT "Exporting DeepLabV3 model if needed..."
)
