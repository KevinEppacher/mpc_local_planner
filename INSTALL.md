# Local Installation

Follow these steps to set up the environment locally:

1. **Update package lists and upgrade the system:**
    ```sh
    sudo apt update && sudo apt upgrade -y
    ```

2. **Install pip if not already available:**
    ```sh
    sudo apt install -y python3-pip
    ```

3. **Allow pip to overwrite system packages if required:**
    ```sh
    export PIP_BREAK_SYSTEM_PACKAGES=1
    ```

4. **Install PyTorch with CUDA 12.6 support:**
    ```sh
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

5. **Install PyPose:**
    ```sh
    pip install pypose
    ```

6. **Update package lists to ensure ROS dependencies are available:**
    ```sh
    sudo apt update
    ```

7. **Install ROS 2 dependencies**  
   *(Replace `<ros-distro>` with your ROS version, e.g. `humble` or `jazzy`):*
    ```sh
    sudo apt install -y \
      ros-<ros-distro>-turtlebot3-gazebo \
      ros-<ros-distro>-turtlebot3-bringup \
      ros-<ros-distro>-turtlebot3-navigation2 \
      ros-<ros-distro>-turtlebot3-teleop \
      ros-<ros-distro>-navigation2 \
      ros-<ros-distro>-nav2-map-server \
      ros-<ros-distro>-nav2-amcl \
      ros-<ros-distro>-nav2-lifecycle-manager \
      ros-<ros-distro>-turtlebot3-cartographer \
      ros-<ros-distro>-ros-gz \
      ros-<ros-distro>-ros-gz-sim \
      ros-<ros-distro>-plotjuggler \
      ros-<ros-distro>-plotjuggler-ros \
      ros-<ros-distro>-tf-transformations \
      ros-<ros-distro>-test-msgs
    ```

8. **(Optional) Use rosdep to resolve any remaining dependencies:**
    ```sh
    sudo rosdep init || true   # Only required once; ignore if already initialized
    rosdep update
    rosdep install --from-paths src --ignore-src -r
    ```

---

## All-in-One Installation with Docker (Recommended)

**Requirements:**
- Docker
- Nvidia Container Toolkit

If Docker is not installed, follow:  
[Install Docker](docker_install.md)

If Nvidia Container Toolkit is not installed, follow:  
[Install Nvidia Container Toolkit](nvidia_container_toolkit_install.md)

**Start the development container:**
```sh
cd ~ros2_ws/.devcontainer
docker compose up optimization_container
```

**VS Code Integration:**
1. Open Visual Studio Code.
2. Go to the **Docker** tab (install the Docker extension if needed).
3. Locate the container (e.g., `ros2/optimization`).
4. Right-click and select **Attach to Container**.
5. A new VS Code window will open inside the container.
6. Open the terminal within the container.
7. Navigate to the `/app` folder and open it in VS Code.

**Tip:**  
Use ROS 2 aliases for common tasks (e.g., building, sourcing, ...).
```sh
cd ~ros2_ws
source .bash_aliases
```