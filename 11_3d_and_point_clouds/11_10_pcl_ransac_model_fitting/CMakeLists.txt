# CMakeLists.txt for PCL RANSAC example
# Author: José Miguel Guerrero Hernández

cmake_minimum_required(VERSION 3.16)
project(pcl_ransac_example LANGUAGES C CXX)

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
add_executable(pcl_ransac main.cpp)

# Link PCL
target_include_directories(pcl_ransac PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_ransac PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(pcl_ransac PRIVATE ${PCL_DEFINITIONS})