#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch file arguments
    declare_args = [
        DeclareLaunchArgument(
            'tb3_model', default_value='burger',
            description='TurtleBot3 model: burger | waffle | waffle_pi'
        ),
            DeclareLaunchArgument(
            'gui',
            default_value='false',
            description='Enable GUI (Gazebo + RViz)'
        ),
    ]

    # Paths
    world_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 
        'worlds', 
        'turtlebot3_dqn_stage2.world'
    )

    gazebo_ros2_launch_path = os.path.join(
        get_package_share_directory('gazebo_ros2'), 
        'launch', 
        'gazebo_ros2.launch.py'
    )

    rviz_config = os.path.join(
        get_package_share_directory('mpc_local_planner'), 
        'rviz', 
        'rviz.rviz'
    )

    tb3_model_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 
        'models'
    )

    robot_state_publisher_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 
        'launch',
        'robot_state_publisher.launch.py'
    )

    spawn_tb3_launch_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'launch',
        'spawn_turtlebot3.launch.py'
    )

    # LaunchConfigurations
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='-2.0')
    y_pose = LaunchConfiguration('y_pose', default='-0.5')
    tb3_model = LaunchConfiguration('tb3_model', default='burger')
    gui = LaunchConfiguration('gui')

    # Set TB3 model env var
    set_tb3_model = SetEnvironmentVariable(
        name='TURTLEBOT3_MODEL',
        value=tb3_model
    )

    # Include gazebo_ros2
    gazebo_ros2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_ros2_launch_path),
        launch_arguments={
            'world_path': world_path,
            'gz_sim_resource_path': tb3_model_path,
            'gui': gui,
            'verbose': '4',
            'run_on_start': 'true',
        }.items(),
    )

    # TB3 robot state publisher
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_state_publisher_path),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # Spawn TB3
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(spawn_tb3_launch_path),
        launch_arguments={
            'x_pose': x_pose, 
            'y_pose': y_pose
        }.items()
    )

    rviz_node = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output='screen'
    )

    return LaunchDescription(
        declare_args + 
        [
            set_tb3_model,
            gazebo_ros2_launch,
            robot_state_publisher_cmd,
            spawn_turtlebot_cmd,
            rviz_node
        ]
    )
