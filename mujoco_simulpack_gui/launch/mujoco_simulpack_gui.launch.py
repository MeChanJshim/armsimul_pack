from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port = LaunchConfiguration("port")
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="18100"),
        Node(
            package="mujoco_simulpack_gui",
            executable="mujoco_simulpack_gui",
            name="mujoco_simulpack_gui",
            output="screen",
            parameters=[{"port": port}],
        ),
    ])
