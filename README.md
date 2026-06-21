# armsimul_pack

`armsimul_pack` is a ROS2 Humble package for UR10e contact simulation in MuJoCo.
It accepts joint position commands, runs the UR10e MuJoCo model, publishes joint
states, and publishes an estimated end-effector contact wrench from MuJoCo
contact forces.

## Features

- UR10e MuJoCo model packaged under `models/ur10e`.
- Position control through ROS2 topics.
- End-effector contact force output as `geometry_msgs/WrenchStamped`.
- Runtime settings are collected in `config/ur10_contact_sim.yaml`.
- Optional MuJoCo viewer through config.

## Dependencies

System packages:

```bash
sudo apt update
sudo apt install -y ros-humble-desktop python3-pip python3-colcon-common-extensions
```

Python package:

```bash
python3 -m pip install --user mujoco PyYAML
```

This workspace already has ROS2 Humble and `colcon` installed. If MuJoCo is not
installed, the simulation node will print an error telling you to install the
`mujoco` Python package.

## Build

```bash
cd ~/armcon_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select armsimul_pack
source install/setup.bash
```

## Run

```bash
ros2 launch armsimul_pack ur10_contact_sim.launch.py
```

To enable the MuJoCo viewer, edit:

```yaml
use_viewer: true
```

in `config/ur10_contact_sim.yaml`, then rebuild or run with an installed config
file override:

```bash
ros2 launch armsimul_pack ur10_contact_sim.launch.py config_file:=/absolute/path/to/ur10_contact_sim.yaml
```

## Command The Robot

Send a one-shot joint position command in radians:

```bash
ros2 run armsimul_pack send_joint_position -1.57 -1.57 1.57 -1.57 -1.57 0.0
```

For a real-time control loop, publish six joint targets in radians continuously:

```bash
ros2 topic pub --rate 100 /armsimul/joint_position_cmd_array std_msgs/msg/Float64MultiArray "{data: [-1.57, -1.57, 1.57, -1.57, -1.57, 0.0]}"
```

A built-in real-time demo publisher is also available:

```bash
ros2 run armsimul_pack realtime_joint_position_demo --rate 100 --amplitude 0.15
```

Or publish directly:

```bash
ros2 topic pub --once /armsimul/joint_position_cmd sensor_msgs/msg/JointState "{name: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint], position: [-1.57, -1.57, 1.57, -1.57, -1.57, 0.0]}"
```

The simulator also accepts `trajectory_msgs/JointTrajectory` on:

```text
/joint_trajectory_controller/joint_trajectory
```

It uses the last point in the received trajectory as the current position target.

## Published Topics

- `/joint_states` (`sensor_msgs/JointState`): simulated UR10e joint state.
- `/armsimul/joint_position_cmd_array` (`std_msgs/Float64MultiArray`): simple
  six-value joint position command input for real-time controllers.
- `/armsimul/ee_wrench` (`geometry_msgs/WrenchStamped`): summed contact wrench
  involving the configured EE body names.
- `/armsimul/contact_state` (`std_msgs/Bool`): true when EE contact force is over
  the configured deadband.

## Configuration

All main settings are in `config/ur10_contact_sim.yaml`.

Important fields:

- `model_file`: MuJoCo scene XML path. Relative paths are resolved inside the
  installed package share directory.
- `use_viewer`: opens the MuJoCo passive viewer when true.
- `publish_rate_hz`: ROS publish rate for joint states and wrench.
- `simulation_rate_hz`: MuJoCo step timer rate.
- `joint_position_command_topic`: `sensor_msgs/JointState` position command.
- `joint_trajectory_command_topic`: `trajectory_msgs/JointTrajectory` command.
- `ee_body_names`: MuJoCo body names treated as the end-effector for contact
  force summation. Default is `wrist_3_link`.
- `contact_force_deadband`: small force threshold in newtons.

## Model Editing

The default scene is:

```text
models/ur10e/ur10e_contact_scene.xml
```

It includes `ur10e.xml`, a floor, and a movable contact box. Change the box
position, size, friction, or add new objects in this XML file. After editing
model/config files in `src`, rebuild the package:

```bash
cd ~/armcon_ws
colcon build --packages-select armsimul_pack
source install/setup.bash
```

## Notes

- Position commands are clipped by `position_limits_lower` and
  `position_limits_upper`.
- The simulator keeps the last commanded joint target. On startup this is
  `initial_positions`, so the arm should hold the configured initial pose.
- The wrench is computed from MuJoCo contact forces, not from a physical F/T
  sensor driver. For a real robot, bridge your hardware F/T sensor separately
  and keep topic names consistent with this package.
