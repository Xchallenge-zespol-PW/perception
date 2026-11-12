"""Launch the ball node (exercise 2) with config/balls.yaml."""

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

    # Optional overrides; an empty value keeps the YAML.
    camera_ns = LaunchConfiguration('camera_ns').perform(context).strip()
    if camera_ns:
        camera_ns = camera_ns.rstrip('/')
        parameters.append({
            'color_topic': f'{camera_ns}/color/image_raw',
            'depth_topic': f'{camera_ns}/aligned_depth_to_color/image_raw',
            'camera_info_topic': f'{camera_ns}/color/camera_info',
        })
    world_frame = LaunchConfiguration('world_frame').perform(context).strip()
    if world_frame:
        parameters.append(
            {'world_frame': '' if world_frame == 'none' else world_frame}
        )
    for name in ('model_path', 'device'):
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            parameters.append({name: value})

    return [
        Node(
            package='g1_perception',
            executable='ball_node',
            name='ball_node',
            output='screen',
            parameters=parameters,
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config', default_value='balls.yaml',
            description='YAML under share/g1_perception/config/',
        ),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use /clock (only with ros2 bag play --clock)',
        ),
        DeclareLaunchArgument(
            'camera_ns', default_value='',
            description=(
                'Camera topic prefix, e.g. /camera/camera on the laptop '
                '(default: topics from the YAML, /camera/d435)'
            ),
        ),
        DeclareLaunchArgument(
            'world_frame', default_value='',
            description='Override world_frame (none = camera frame, no robot TF)',
        ),
        DeclareLaunchArgument(
            'model_path', default_value='',
            description='Override model_path (e.g. /ws/models/cubes_seg_v4_aug.onnx)',
        ),
        DeclareLaunchArgument(
            'device', default_value='',
            description='Override device (e.g. cpu with an .onnx model)',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
