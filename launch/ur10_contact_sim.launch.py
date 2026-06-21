from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")

    default_config = PathJoinSubstitution(
        [FindPackageShare("armsimul_pack"), "config", "ur10_contact_sim.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="YAML configuration file for the MuJoCo UR10e simulation.",
            ),
            Node(
                package="armsimul_pack",
                executable="ur10_contact_sim",
                name="ur10_contact_sim",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
