from glob import glob
from setuptools import find_packages, setup

package_name = "mujoco_simulpack"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/models/ur10e", glob("models/ur10e/*.xml")),
        (f"share/{package_name}/models/ur10e/assets", glob("models/ur10e/assets/*")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="jay",
    maintainer_email="user@example.com",
    description="ROS2 MuJoCo contact simulation package for a UR10e arm.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "realtime_joint_position_demo = mujoco_simulpack.realtime_joint_position_demo:main",
            "ur10_contact_sim = mujoco_simulpack.ur10_contact_sim_node:main",
            "send_joint_position = mujoco_simulpack.send_joint_position:main",
        ],
    },
)
