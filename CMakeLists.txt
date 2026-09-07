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
#                    in 03_01_dft → 03_01_dft_frequencies)
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

# --- 02 Image formation --------------------------------------------------------
add_cv_example(02_image_formation/02_01_read_image)
add_cv_example(02_image_formation/02_02_color_spaces)
add_cv_example(02_image_formation/02_03_mat_copy_roi)
add_cv_example(02_image_formation/02_04_pixel_access)
add_cv_example(02_image_formation/02_05_video_capture)

# --- 03 Frequency-domain transforms -------------------------------------------
# 03_01_dft has two sources -> 03_01_dft + 03_01_dft_frequencies
add_cv_example(03_frequency/03_01_dft)
add_cv_example(03_frequency/03_02_dct)
add_cv_example(03_frequency/03_03_wavelet_denoising)
add_cv_example(03_frequency/03_04_gabor_bank)

# --- 04 Spatial, geometric and radiometric transforms -------------------------
add_cv_example(04_spatial_geometric_and_radiometric/04_01_point_operations)
add_cv_example(04_spatial_geometric_and_radiometric/04_02_convolution)
add_cv_example(04_spatial_geometric_and_radiometric/04_03_bitwise_operations)
add_cv_example(04_spatial_geometric_and_radiometric/04_04_affine_transforms)
add_cv_example(04_spatial_geometric_and_radiometric/04_05_perspective_correction)
add_cv_example(04_spatial_geometric_and_radiometric/04_06_intensity_transforms)
add_cv_example(04_spatial_geometric_and_radiometric/04_07_smoothing)
add_cv_example(04_spatial_geometric_and_radiometric/04_08_homomorphic_filter)
add_cv_example(04_spatial_geometric_and_radiometric/04_09_histogram_equalization)
add_cv_example(04_spatial_geometric_and_radiometric/04_10_histogram_matching)
add_cv_example(04_spatial_geometric_and_radiometric/04_11_histogram_comparison)

# --- 05 Edges and model fitting -----------------------------------------------
# Order follows the book: edges -> contour description -> model fitting
add_cv_example(05_edges_and_model_fitting/05_01_sobel_edges)
add_cv_example(05_edges_and_model_fitting/05_02_canny_edges)
add_cv_example(05_edges_and_model_fitting/05_03_laplacian_edges)
add_cv_example(05_edges_and_model_fitting/05_04_contour_extraction)
add_cv_example(05_edges_and_model_fitting/05_05_chain_code)
add_cv_example(05_edges_and_model_fitting/05_06_hough_lines)
add_cv_example(05_edges_and_model_fitting/05_07_hough_circles)

# --- 06 Regions, descriptors and keypoints ------------------------------------
add_cv_example(06_regions_and_features/06_01_threshold)
add_cv_example(06_regions_and_features/06_02_connected_components)
add_cv_example(06_regions_and_features/06_03_color_segmentation)
add_cv_example(06_regions_and_features/06_04_region_moments)
add_cv_example(06_regions_and_features/06_05_hu_moments)
add_cv_example(06_regions_and_features/06_06_harris_corners)
add_cv_example(06_regions_and_features/06_07_shi_tomasi_corners)
add_cv_example(06_regions_and_features/06_08_orb_features)
add_cv_example(06_regions_and_features/06_09_ransac_matching)

# --- 07 Morphological Operations ----------------------------------------------
add_cv_example(07_morphological_operations/07_01_erode_dilate)
add_cv_example(07_morphological_operations/07_02_opening_closing)
add_cv_example(07_morphological_operations/07_03_morphological_gradient)
add_cv_example(07_morphological_operations/07_04_hit_or_miss)
add_cv_example(07_morphological_operations/07_05_skeletonization)
add_cv_example(07_morphological_operations/07_06_flood_fill)
add_cv_example(07_morphological_operations/07_07_tophat_illumination)
add_cv_example(07_morphological_operations/07_08_distance_watershed)

# --- 08 Camera Calibration ----------------------------------------------------
add_cv_example(08_camera_calibration/08_01_camera_calibration)
add_cv_example(08_camera_calibration/08_02_pose_estimation)
add_cv_example(08_camera_calibration/08_03_stereo_calibration)

# --- 09 3D vision and point clouds --------------------------------------------
add_cv_example(09_3d_and_point_clouds/09_01_epipolar_geometry)
add_cv_example(09_3d_and_point_clouds/09_02_stereo_disparity)
add_cv_example(09_3d_and_point_clouds/09_03_stereo_to_pointcloud)
add_cv_example(09_3d_and_point_clouds/09_04_opencv_icp)
add_pcl_example(09_3d_and_point_clouds/09_05_pcl_write)
add_pcl_example(09_3d_and_point_clouds/09_06_pcl_read)
add_pcl_example(09_3d_and_point_clouds/09_07_pcl_simple_visualizer)
add_pcl_example(09_3d_and_point_clouds/09_08_pcl_advanced_visualizer)
add_pcl_example(09_3d_and_point_clouds/09_09_pcl_icp)
add_pcl_example(09_3d_and_point_clouds/09_10_pcl_ransac_model_fitting)
add_pcl_example(09_3d_and_point_clouds/09_11_pcl_registration)
add_pcl_example(09_3d_and_point_clouds/09_12_pcl_correspondence)
add_pcl_example(09_3d_and_point_clouds/09_13_pcl_plane_clustering)

# --- 10 Optical Flow ----------------------------------------------------------
add_cv_example(10_optical_flow/10_01_frame_difference)
add_cv_example(10_optical_flow/10_02_sparse_flow)
add_cv_example(10_optical_flow/10_03_dense_flow)
add_cv_example(10_optical_flow/10_04_background_subtraction)
add_cv_example(10_optical_flow/10_05_kalman_tracking)
add_cv_example(10_optical_flow/10_06_object_tracking)

# --- 11 Pattern recognition ---------------------------------------------------
add_cv_example(11_pattern_recognition/11_01_knn_classification)
add_cv_example(11_pattern_recognition/11_02_svm_classification)
add_cv_example(11_pattern_recognition/11_03_digits_classification)
add_cv_example(11_pattern_recognition/11_04_kmeans_clustering)
add_cv_example(11_pattern_recognition/11_05_classifier_comparison)
add_cv_example(11_pattern_recognition/11_06_som_digits)
add_cv_example(11_pattern_recognition/11_07_yolov4_darknet
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 11_07_yolov4_darknet PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/11_pattern_recognition/11_07_yolov4_darknet/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/11_pattern_recognition/11_07_yolov4_darknet
  COMMENT "Downloading YOLOv4-tiny model if needed..."
)

add_cv_example(11_pattern_recognition/11_08_yolo_ultralytics
  EXTRA_LIBS Threads::Threads)
add_custom_command(
  TARGET 11_08_yolo_ultralytics PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/11_pattern_recognition/11_08_yolo_ultralytics/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/11_pattern_recognition/11_08_yolo_ultralytics
  COMMENT "Exporting YOLO11 model if needed..."
)

add_cv_example(11_pattern_recognition/11_09_semantic_segmentation)
add_custom_command(
  TARGET 11_09_semantic_segmentation PRE_BUILD
  COMMAND bash ${CMAKE_SOURCE_DIR}/11_pattern_recognition/11_09_semantic_segmentation/download_model.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/11_pattern_recognition/11_09_semantic_segmentation
  COMMENT "Exporting DeepLabV3 model if needed..."
)

