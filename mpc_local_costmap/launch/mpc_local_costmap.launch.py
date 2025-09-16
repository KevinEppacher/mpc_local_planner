#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from nav2_common.launch import RewrittenYaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():

    use_respawn = LaunchConfiguration('use_respawn')

    declare_use_respawn_cmd = DeclareLaunchArgument(
        'use_respawn',
        default_value='False',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.',
    )

    mpc_local_costmap_param_path = os.path.join(
        get_package_share_directory("mpc_local_costmap"),
        'config',
        'mpc_local_costmap.yaml'
    )

    lifecycle_nodes = [
        'local_costmap/local_costmap',
        'planner_server'
    ]

    
    mpc_local_costmap_node = LifecycleNode(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        output='screen',
        emulate_tty=True,
        namespace='local_costmap',
        parameters=[
            {'use_sim_time': True},
            mpc_local_costmap_param_path
        ]
    )

    planner_server_node = LifecycleNode(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        emulate_tty=True,
        namespace='',
        respawn=use_respawn,
        respawn_delay=2.0,
        parameters=[
            {'use_sim_time': True},
            mpc_local_costmap_param_path
        ],
    )

    lcm = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_local_costmap',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': lifecycle_nodes,
            'bond_timeout': 0.0,
            'bond_respawn': True
        }],
        # arguments=['--ros-args', '--log-level', 'debug'],
        emulate_tty=True
    )


    ld = LaunchDescription()
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(mpc_local_costmap_node)
    ld.add_action(planner_server_node)
    ld.add_action(lcm)

    return ld
