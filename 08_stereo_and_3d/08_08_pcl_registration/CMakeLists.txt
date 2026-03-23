# CMakeLists.txt for PCL Incremental Registration example
# Author: José Miguel Guerrero Hernández

cmake_minimum_required(VERSION 3.16)
project(pcl_registration_example LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix for VTK 9.1 MPI issue
find_package(MPI QUIET)

# Find PCL with registration module
find_package(PCL 1.10 REQUIRED COMPONENTS
  common
  io
  visualization
  features
  filters
  registration
)

# Executable
add_executable(pcl_inc_registration main.cpp)

# Link PCL
target_include_directories(pcl_inc_registration PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_inc_registration PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(pcl_inc_registration PRIVATE ${PCL_DEFINITIONS})