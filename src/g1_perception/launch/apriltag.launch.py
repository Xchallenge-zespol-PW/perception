"""Launch the AprilTag node with config/apriltag.yaml."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _launch_setup(context, *args, **kwargs):
    share = get_package_share_directory('g1_perception')
    config_name = LaunchConfiguration('config').perform(context)
    params_path = os.path.join(share, 'config', config_name)
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f'Config not found: {params_path}')

    use_sim_time = LaunchConfiguration('use_sim_time')
    parameters = [
        params_path,
        {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)},
    ]
    # Optional overrides; empty keeps the YAML value.
    tag_size = LaunchConfiguration('tag_size_m').perform(context).strip()
    if tag_size:
        parameters.append({'tag_size_m': float(tag_size)})
    target = LaunchConfiguration('target_frame').perform(context).strip()
    if target:
        parameters.append({'target_frame': target})

    return [
        Node(
            package='g1_perception',
            executable='apriltag_node',
            name='apriltag_node',
            output='screen',
            parameters=parameters,
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value='apriltag.yaml',
            description='YAML under share/g1_perception/config/',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use /clock (only with ros2 bag play --clock)',
        ),
        DeclareLaunchArgument(
            'tag_size_m',
            default_value='',
            description='Override tag_size_m (side of the black square, m)',
        ),
        DeclareLaunchArgument(
            'target_frame',
            default_value='',
            description='Override target_frame (e.g. pelvis)',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
