from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mpc_local_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/**/*.py', recursive=True)),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/**/*.rviz', recursive=True)),
        (os.path.join('share', package_name, 'map'), glob('map/**/*.yaml', recursive=True)),
        (os.path.join('share', package_name, 'map'), glob('map/**/*.pgm', recursive=True)),
        (os.path.join('share', package_name, 'config'), glob('config/**/*.yaml', recursive=True)),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='your@adress.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_local_planner = mpc_local_planner.mpc_local_planner_node:main',
            'goal_to_plan = mpc_local_planner.goal_to_plan:main',

            # # Test scripts
            # 'test_pypose = mpc_local_planner.test.test_pypose:main',
            # 'test_mpc_controller = mpc_local_planner.test.test_mpc:main',
            # 'multiple_shooting_nmpc_tutorial = mpc_local_planner.test.multiple_shooting_nmpc_tutorial:main',
            # 'multiple_shooting_nmpc_obstacle_avoidance_refrence_tracking = mpc_local_planner.test.multiple_shooting_nmpc_obstacle_avoidance_refrence_tracking:main',
            # 'multiple_shooting_nmpc_obstacle_avoidance_refrence_tracking_casadi = mpc_local_planner.test.multiple_shooting_nmpc_obstacle_avoidance_refrence_tracking_casadi:main',
        ],
    },
)
