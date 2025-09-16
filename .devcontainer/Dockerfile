# Stage 1
FROM nvidia/cuda:12.6.1-devel-ubuntu24.04 AS cuda

# Stage 2
FROM osrf/ros:jazzy-desktop-full

# Take CUDA from stage 1 (symlink is stable)
COPY --from=cuda /usr/local/cuda /usr/local/cuda

# (Optional) Ensure symlink – usually already present
RUN ln -sf /usr/local/cuda /usr/local/cuda-12 || true

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.4"

# ROS environment
ENV ROS_DISTRO=jazzy
SHELL ["/bin/bash", "-lc"]

# ---- Create user like your host ----
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
RUN if ! getent group ${USER_GID} >/dev/null; then groupadd -g ${USER_GID} ${USERNAME}; fi && \
    if ! id -u ${USER_UID} >/dev/null 2>&1; then useradd -m -s /bin/bash -u ${USER_UID} -g ${USER_GID} ${USERNAME}; fi && \
    apt-get update && apt-get install -y sudo && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}

# (Optional) Webcam / video group
RUN sudo usermod --append --groups video ${USERNAME}

# Update packages
RUN sudo apt update && sudo apt upgrade -y

########################################################
# Custom installations
########################################################

RUN sudo apt update && sudo apt upgrade -y && \
    sudo apt install -y \
    python3-pip

# Install PyTorch first
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN pip install pypose

RUN sudo apt-get update && sudo apt-get install -y \
    ros-${ROS_DISTRO}-turtlebot3-gazebo \
    ros-${ROS_DISTRO}-turtlebot3-bringup \
    ros-${ROS_DISTRO}-turtlebot3-navigation2 \
    ros-${ROS_DISTRO}-turtlebot3-teleop \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-map-server \
    ros-${ROS_DISTRO}-nav2-amcl \
    ros-${ROS_DISTRO}-nav2-lifecycle-manager \
    ros-${ROS_DISTRO}-turtlebot3-cartographer \
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-ros-gz-sim \
    ros-${ROS_DISTRO}-plotjuggler \
    ros-${ROS_DISTRO}-plotjuggler-ros \
    ros-${ROS_DISTRO}-tf-transformations \
    ros-${ROS_DISTRO}-test-msgs


########################################################
# End Custom installations
########################################################

# Source ROS setup
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source install/setup.bash" >> ~/.bashrc

