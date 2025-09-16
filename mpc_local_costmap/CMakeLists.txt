cmake_minimum_required(VERSION 3.22)
project(mpc_local_costmap)

# Set CMake policy for modern find_package behavior
cmake_policy(SET CMP0074 NEW)

# Enable warnings for GCC or Clang
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

############################################################
# Find Required Packages
############################################################

# Core build system for ROS 2
find_package(ament_cmake REQUIRED)

# ROS 2 packages
find_package(rclcpp REQUIRED)


############################################################
# Include Directories
############################################################

include_directories(
  include                           # Local headers
  include/mpc_local_costmap
)

#####################################################################
#                        Declare Executables                        #
#####################################################################

# # Main clustering node
# add_executable(mpc_local_costmap
#   src/mpc_local_costmap.cpp
# )

# # Link ROS 2 dependencies
# ament_target_dependencies(mpc_local_costmap
#   rclcpp
#   std_msgs
#   nav2_costmap_2d
#   nav2_ros_common
#   rclcpp_components
# )

# # Use C++17 standard
# target_compile_features(mpc_local_costmap PUBLIC cxx_std_17)

#####################################################################
#                           Installation                            #
#####################################################################

# Install the node binary into the ROS 2 lib directory
install(TARGETS
  # mpc_local_costmap
  DESTINATION lib/${PROJECT_NAME}
)

#####################################################################
#                      Linting and Testing                          #
#####################################################################

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Optionally disable some linters (uncomment if needed)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

#####################################################################
#                           Finalization                            #
#####################################################################

# Install launch files
install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
