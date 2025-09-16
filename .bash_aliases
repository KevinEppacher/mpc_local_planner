# ROS 2 Sourcing (Adjust the path to your installation)
alias src_ros2="source /opt/ros/jazzy/setup.bash"  # Or foxy, galactic, etc.
alias src_ws="source ~/ros2_ws/install/setup.bash"  # Your custom workspace setup

# ROS 2 Launch & Nodes
alias rlaunch="ros2 launch"
alias rrun="ros2 run"

# ROS 2 Topic & Service Helpers
alias rtopic="ros2 topic"
alias rservice="ros2 service"
alias raction="ros2 action"
alias rparam="ros2 param"
alias rnode="ros2 node"

# Quickly list all ROS 2 nodes
alias rnodelist="ros2 node list"
alias rtopiclist="ros2 topic list"
alias rservicelist="ros2 service list"

# Quickly monitor ROS 2 nodes & topics
alias rtopicinfo="ros2 topic info"
alias rtopicecho="ros2 topic echo"
alias rtopichz="ros2 topic hz"

# Logging and Debugging
alias rlog="ros2 bag play"
alias rbagrec="ros2 bag record -a"  # Records all topics
alias rbaginfo="ros2 bag info"
alias rbagplay="ros2 bag play"

# Colcon build & workspace management
alias cbuild="colcon build --symlink-install"
alias cclean="rm -rf build/ install/ log/"
alias ctest="colcon test --event-handlers console_cohesion+"
alias src="source install/setup.bash"
alias cb="colcon build && source install/setup.bash"
alias cbs="colcon build --symlink-install && source install/setup.bash"
alias s="source install/setup.bash"

# Selective build
alias cbps="colcon build --symlink-install --packages-select"
