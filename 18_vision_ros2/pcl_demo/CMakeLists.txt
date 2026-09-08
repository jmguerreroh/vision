cmake_minimum_required(VERSION 3.8)
project(pcl_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

# Create executable
add_executable(pcl_processing
  src/main.cpp
  src/pcl_processing.cpp
)

# Include directories
target_include_directories(pcl_processing PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

# Link ament dependencies
ament_target_dependencies(pcl_processing
  rclcpp
  sensor_msgs
  pcl_ros
)

# Link non-ament libraries
target_link_libraries(pcl_processing
  ${PCL_LIBRARIES}
)

# Install targets
install(TARGETS pcl_processing
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

# Export include dirs and dependencies for other packages
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  pcl_ros
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
