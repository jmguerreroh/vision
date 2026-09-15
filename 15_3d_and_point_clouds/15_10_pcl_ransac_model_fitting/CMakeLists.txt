# CMakeLists.txt for PCL RANSAC example
# Author: José Miguel Guerrero Hernández

cmake_minimum_required(VERSION 3.16)
project(15_10_pcl_ransac_model_fitting LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix for VTK 9.1 MPI issue
find_package(MPI QUIET)

# Find PCL with sample_consensus module
find_package(PCL 1.10 REQUIRED COMPONENTS
  common
  io
  visualization
  features
  filters
  recognition
  kdtree
  console
  sample_consensus
)

# Executable
add_executable(15_10_pcl_ransac_model_fitting main.cpp)

# Link PCL
target_include_directories(15_10_pcl_ransac_model_fitting PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(15_10_pcl_ransac_model_fitting PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(15_10_pcl_ransac_model_fitting PRIVATE ${PCL_DEFINITIONS})