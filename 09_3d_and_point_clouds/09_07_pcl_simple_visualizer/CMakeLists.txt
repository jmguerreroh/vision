# CMakeLists.txt for PCL Simple Visualizer example
# Author: José Miguel Guerrero Hernández

cmake_minimum_required(VERSION 3.16)
project(pcl_simple_visualizer LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix for VTK 9.1 MPI issue
find_package(MPI QUIET)

# Find PCL with visualization
find_package(PCL 1.10 REQUIRED COMPONENTS common io visualization)

# Executable
add_executable(pcl_visualizer main.cpp)

# Link PCL
target_include_directories(pcl_visualizer PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_visualizer PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(pcl_visualizer PRIVATE ${PCL_DEFINITIONS})