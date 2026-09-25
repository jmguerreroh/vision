# Chains the depth_image_proc nodes described in Chapter 19 so that a raw depth
# image plus a colour image come out as a coloured point cloud on
# /stereo/points, which is the input pcl_demo expects.
#
#   depth 16UC1, mm --convert_metric--> 32FC1, m --+
#                                                  +--> point_cloud_xyzrgb --> /stereo/points
#   colour image ----------------------------------+
#
# Both stages exist as standalone nodes, so this file adds no code: it only
# fixes the remappings that would otherwise have to be typed by hand.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    depth_topic = LaunchConfiguration('depth_topic')
    depth_info_topic = LaunchConfiguration('depth_info_topic')
    rgb_topic = LaunchConfiguration('rgb_topic')
    rgb_info_topic = LaunchConfiguration('rgb_info_topic')
    points_topic = LaunchConfiguration('points_topic')

    arguments = [
        DeclareLaunchArgument(
            'depth_topic', default_value='/stereo/depth',
            description='Raw depth image, 16UC1 in millimetres'),
        DeclareLaunchArgument(
            'depth_info_topic', default_value='/stereo/camera_info',
            description='CameraInfo of the depth camera'),
        DeclareLaunchArgument(
            'rgb_topic', default_value='/color/image',
            description='Rectified colour image, already registered on the depth one'),
        DeclareLaunchArgument(
            'rgb_info_topic', default_value='/color/camera_info',
            description='CameraInfo of the colour camera'),
        DeclareLaunchArgument(
            'points_topic', default_value='/stereo/points',
            description='Resulting XYZRGB cloud'),
    ]

    # Raw units to metres. The rest of the chain only works on 32FC1.
    convert_metric = Node(
        package='depth_image_proc',
        executable='convert_metric_node',
        name='convert_metric',
        remappings=[
            ('image_raw', depth_topic),
            ('camera_info', depth_info_topic),
            ('image', '/stereo/converted_depth'),
        ],
        output='screen',
    )

    # Back-projection with K plus the colour of the homologous pixel.
    point_cloud_xyzrgb = Node(
        package='depth_image_proc',
        executable='point_cloud_xyzrgb_node',
        name='point_cloud_xyzrgb',
        remappings=[
            ('depth_registered/image_rect', '/stereo/converted_depth'),
            ('rgb/image_rect_color', rgb_topic),
            ('rgb/camera_info', rgb_info_topic),
            ('points', points_topic),
        ],
        output='screen',
    )

    return LaunchDescription(arguments + [convert_metric, point_cloud_xyzrgb])
