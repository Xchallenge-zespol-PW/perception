"""Launch the cube perception node with a chosen config/*.yaml."""

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
        raise FileNotFoundError(
            f'Config not found: {params_path} '
            f'(pass config:=params_laptop.yaml or params_robot.yaml)'
        )

    use_sim_time = LaunchConfiguration('use_sim_time')
    parameters = [
        params_path,
        {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)},
    ]

    # Optional overrides (ros2 launch … model_path:=… rois_mode:=none).
    # Empty default keeps the value from the YAML params file.
    model_path = LaunchConfiguration('model_path').perform(context).strip()
    if model_path:
        parameters.append({'model_path': model_path})
    rois_mode = LaunchConfiguration('rois_mode').perform(context).strip()
    if rois_mode:
        parameters.append({'rois_mode': rois_mode})

    return [
        Node(
            package='g1_perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=parameters,
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value='params_laptop.yaml',
            description=(
                'YAML under share/g1_perception/config/ '
                '(params_laptop.yaml or params_robot.yaml)'
            ),
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use /clock (only with ros2 bag play --clock, no --loop)',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='',
            description=(
                'Override perception_node model_path from the YAML '
                '(e.g. model_path:=/ws/models/cubes_seg.onnx)'
            ),
        ),
        DeclareLaunchArgument(
            'rois_mode',
            default_value='',
            description=(
                'Override rois_mode from the YAML '
                '(none = free YOLO+tracker; image_roi = fixed ROIs)'
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
