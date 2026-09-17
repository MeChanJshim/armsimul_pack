from glob import glob
from setuptools import find_packages, setup

package_name = "mujoco_simulpack_gui"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/web", glob("web/*")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jay",
    maintainer_email="user@example.com",
    description="Web GUI for MuJoCo simulator parameters and response analysis.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mujoco_simulpack_gui = mujoco_simulpack_gui.gui_node:main",
        ],
    },
)
