# CMakeLists.txt for PCL Read example
# Author: José Miguel Guerrero Hernández

cmake_minimum_required(VERSION 3.16)
project(pcl_read_example LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix for VTK 9.1 MPI issue
find_package(MPI QUIET)

# Find PCL
find_package(PCL 1.10 REQUIRED COMPONENTS common io)

# Executable
add_executable(pcd_read main.cpp)

# Link PCL (modern CMake approach)
target_include_directories(pcd_read PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcd_read PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(pcd_read PRIVATE ${PCL_DEFINITIONS})
