"""
Bring up the full Roboception stack.

Usage:
  ros2 launch roboception_bringup full_system.launch.py \\
    phone_url:=http://192.168.0.102:8080

Optional overrides:
  ros2 launch roboception_bringup full_system.launch.py \\
    phone_url:=http://192.168.0.102:8080 \\
    vision_conf:=0.5 \\
    log_level:=debug
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_phone_ip = os.environ.get('PHONE_IP', '').strip()
_default_url = f'http://{_phone_ip}:8080' if _phone_ip else None


def generate_launch_description():
    # ── Launch arguments ──────────────────────────────────────────
    phone_url = LaunchConfiguration('phone_url')
    vision_conf = LaunchConfiguration('vision_conf')
    log_level = LaunchConfiguration('log_level')

    _url_arg = {'default_value': _default_url} if _default_url else {}

    declared_args = [
        DeclareLaunchArgument(
            'phone_url',
            description='IP Webcam base URL — or set PHONE_IP env var instead',
            **_url_arg,
        ),
        DeclareLaunchArgument(
            'vision_conf',
            default_value='0.35',
            description='YOLOv8 confidence threshold',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='ROS log level (debug, info, warn, error)',
        ),
    ]

    # ── Bridges ───────────────────────────────────────────────────
    camera_bridge = Node(
        package='phone_bridge',
        executable='camera_bridge',
        name='camera_bridge',
        parameters=[{'phone_url': phone_url}],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen',
    )

    audio_bridge = Node(
        package='phone_bridge',
        executable='audio_bridge',
        name='audio_bridge',
        parameters=[{'phone_url': phone_url}],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen',
    )

    imu_bridge = Node(
        package='phone_bridge',
        executable='imu_bridge',
        name='imu_bridge',
        parameters=[{'phone_url': phone_url}],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen',
    )

    # ── ML inference ──────────────────────────────────────────────
    vision_detector = Node(
        package='ml_inference',
        executable='vision_detector',
        name='vision_detector',
        parameters=[{'conf_threshold': vision_conf}],
        output='screen',
    )

    audio_classifier = Node(
        package='ml_inference',
        executable='audio_classifier',
        name='audio_classifier',
        output='screen',
    )

    imu_classifier = Node(
        package='ml_inference',
        executable='imu_classifier',
        name='imu_classifier',
        output='screen',
    )

    # ── Fusion ────────────────────────────────────────────────────
    aggregator = Node(
        package='fusion',
        executable='aggregator',
        name='aggregator',
        output='screen',
    )

    # ── Dashboard ─────────────────────────────────────────────────
    dashboard = Node(
        package='dashboard',
        executable='dashboard',
        name='dashboard',
        output='screen',
    )

    return LaunchDescription(declared_args + [
        camera_bridge,
        audio_bridge,
        imu_bridge,
        vision_detector,
        audio_classifier,
        imu_classifier,
        aggregator,
        dashboard,
    ])
