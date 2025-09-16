import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Declare launch file arguments
    tb3_model_arg = DeclareLaunchArgument(
        'tb3_model', default_value='burger',
        description='TurtleBot3 model: burger | waffle | waffle_pi'
    )

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Flag to enable use_sim_time'
    )

    nav2_localization_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'localization_launch.py'
    )

    localization_params_path = os.path.join(
        get_package_share_directory('mpc_local_planner'),
        'config',
        'amcl_param.yaml'
    )

    nav2_params_path = os.path.join(
        get_package_share_directory('mpc_local_planner'),
        'config',
        'burger.yaml'
    )

    map_file_path = os.path.join(
        get_package_share_directory('mpc_local_planner'),
        'map',
        'my_map.yaml'
    )

    nav2_navigation_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'navigation_launch.py'
    )

    # Launch configurations
    tb3_model = LaunchConfiguration('tb3_model', default='burger')

    # Set TB3 model env var
    set_tb3_model = SetEnvironmentVariable(
        name='TURTLEBOT3_MODEL',
        value=tb3_model
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_localization_launch_path),
        launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': localization_params_path,
                'map': map_file_path,
        }.items()
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_navigation_launch_path),
        launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': nav2_params_path,
        }.items()
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(tb3_model_arg)
    launchDescriptionObject.add_action(sim_time_arg)
    launchDescriptionObject.add_action(localization_launch)
    # launchDescriptionObject.add_action(navigation_launch)
    launchDescriptionObject.add_action(set_tb3_model)

    return launchDescriptionObject