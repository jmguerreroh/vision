# CMakeLists.txt for PCL Read example
# Author: José Miguel Guerrero Hernández

cmake_minimum_required(VERSION 3.16)
project(14_06_pcl_read LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix for VTK 9.1 MPI issue
find_package(MPI QUIET)

# Find PCL
find_package(PCL 1.10 REQUIRED COMPONENTS common io)

# Executable
add_executable(14_06_pcl_read main.cpp)

# Link PCL (modern CMake approach)
target_include_directories(14_06_pcl_read PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(14_06_pcl_read PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(14_06_pcl_read PRIVATE ${PCL_DEFINITIONS})
