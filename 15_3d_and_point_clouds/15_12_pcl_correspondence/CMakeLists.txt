# CMakeLists.txt for PCL Correspondence Grouping example
# Author: José Miguel Guerrero Hernández

cmake_minimum_required(VERSION 3.16)
project(15_12_pcl_correspondence LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix for VTK 9.1 MPI issue
find_package(MPI QUIET)

# Find PCL with recognition module
find_package(PCL 1.10 REQUIRED COMPONENTS
  common
  io
  visualization
  features
  filters
  recognition
  kdtree
  console
)

# Executable
add_executable(15_12_pcl_correspondence main.cpp)

# Link PCL
target_include_directories(15_12_pcl_correspondence PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(15_12_pcl_correspondence PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(15_12_pcl_correspondence PRIVATE ${PCL_DEFINITIONS})