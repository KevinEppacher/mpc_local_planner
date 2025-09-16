#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'mpc_local_planner'

    robot_bringup_launch_path = os.path.join(
        get_package_share_directory(pkg_name),
        'launch',
        'bringup.launch.py'
    )

    navigation_launch_path = os.path.join(
        get_package_share_directory(pkg_name),
        'launch',
        'navigation_stack.launch.py'
    )

    mpc_parameters_path = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'mpc_params.yaml'
    )

    mpc_local_costmap_launch_path = os.path.join(
        get_package_share_directory("mpc_local_costmap"),
        'launch',
        'mpc_local_costmap.launch.py'
    )

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_bringup_launch_path)
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_launch_path)
    )

    mpc_local_costmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(mpc_local_costmap_launch_path)
    )

    delayed_nav2 = TimerAction(period=4.0, actions=[nav2])

    mpc_controller_node = Node(
        package='mpc_local_planner',
        executable='mpc_local_planner',
        name='mpc_local_planner',
        output='screen',
        namespace='mpc',
        emulate_tty=True,
        parameters=[
            {'use_sim_time': True},
            mpc_parameters_path
        ]
    )

    goal_to_plan_node = Node(
        package='mpc_local_planner',
        executable='goal_to_plan',
        name='goal_to_plan',
        output='screen',
        namespace='mpc',
        parameters=[
            {'use_sim_time': True},
        ]
    )


    ld = LaunchDescription()
    ld.add_action(bringup)
    ld.add_action(delayed_nav2)
    ld.add_action(mpc_controller_node)
    ld.add_action(mpc_local_costmap)
    ld.add_action(goal_to_plan_node)
    return ld
