cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(MY_PCL_PROJECT)
find_package(PCL 1.4 REQUIRED COMPONENTS common io visualization features filters recognition kdtrees console sample_consensus) 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(pcl_ransac main.cpp)
target_link_libraries(pcl_ransac ${PCL_LIBRARIES})