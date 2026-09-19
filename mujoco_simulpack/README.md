# mujoco_simulpack

`mujoco_simulpack` is a ROS2 Humble package for UR10e contact simulation in MuJoCo.
It accepts joint position commands, runs the UR10e MuJoCo model, publishes joint
states, and publishes an estimated end-effector contact wrench from MuJoCo
contact forces.

## Features

- UR10e MuJoCo model packaged under `models/ur10e`.
- Position control through ROS2 topics.
- Optional inverse-dynamics acceleration control using torque motor actuators.
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
colcon build --packages-select mujoco_simulpack
source install/setup.bash
```

## Run

```bash
ros2 launch mujoco_simulpack ur10_contact_sim.launch.py
```

Run the inverse-dynamics acceleration controller:

```bash
ros2 launch mujoco_simulpack ur10_contact_sim.launch.py \
  config_file:=/home/jay/armcon_ws/src/armsimul_pack/mujoco_simulpack/config/ur10_contact_sim_inverse_dynamics.yaml
```

To enable the MuJoCo viewer, edit:

```yaml
use_viewer: true
```

in `config/ur10_contact_sim.yaml`, then rebuild or run with an installed config
file override:

```bash
ros2 launch mujoco_simulpack ur10_contact_sim.launch.py config_file:=/absolute/path/to/ur10_contact_sim.yaml
```

## Command The Robot

Send a one-shot joint position command in radians:

```bash
ros2 run mujoco_simulpack send_joint_position -1.57 -1.57 1.57 -1.57 -1.57 0.0
```

For a real-time control loop, publish six joint targets in radians continuously:

```bash
ros2 topic pub --rate 100 /armsimul/joint_position_cmd_array std_msgs/msg/Float64MultiArray "{data: [-1.57, -1.57, 1.57, -1.57, -1.57, 0.0]}"
```

A built-in real-time demo publisher is also available:

```bash
ros2 run mujoco_simulpack realtime_joint_position_demo --rate 100 --amplitude 0.15
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
- `/armsimul/ft_sensor_wrench` (`geometry_msgs/WrenchStamped`): the same contact
  wrench expressed in the configured virtual FT sensor frame. The torque is
  shifted from each MuJoCo contact point to the sensor origin before rotation.
- `/armsimul/contact_state` (`std_msgs/Bool`): true when EE contact force is over
  the configured deadband.

## Configuration

All main settings are in `config/ur10_contact_sim.yaml`.

Important fields:

- `model_file`: MuJoCo scene XML path. Relative paths are resolved inside the
  installed package share directory.
- `control_mode`: `position_pd` or `inverse_dynamics_accel`.
- `use_viewer`: opens the MuJoCo passive viewer when true.
- `publish_rate_hz`: ROS publish rate for joint states and wrench.
- `simulation_rate_hz`: MuJoCo step timer rate.
- `joint_position_command_topic`: `sensor_msgs/JointState` position command.
- `joint_trajectory_command_topic`: `trajectory_msgs/JointTrajectory` command.
- `ee_body_names`: MuJoCo body names treated as the end-effector for contact
  force summation. The contact scene includes `wrist_3_link` and the attached
  `nrs_spindle`.
- `ft_sensor_site_name`: MuJoCo site used as the virtual FT sensor origin and
  coordinate frame. The default is `attachment_site`.
- `contact_force_deadband`: small force threshold in newtons.
- `force_axis_sign`: per-axis sensor sign applied to the published force
  vector, for example `[-1.0, 1.0, 1.0]` flips only the X force direction.

The force-axis sign changes the reported `/armsimul/ee_wrench` sensor value and
does not change the physical contact solution. Use it when the simulator's
contact-force convention is opposite to the force controller convention.

For contact-point estimation, subscribe to `/armsimul/ft_sensor_wrench`.
Its `header.frame_id` identifies the configured sensor site, and its torque is
computed about that site rather than about the individual contact point.

## Simulator GUI

The companion `mujoco_simulpack_gui` package provides a browser-based control
and analysis page. It can start and stop simulator processes, apply controller
and contact parameters at runtime, save/load JSON setting profiles, and plot
joint position, effort, contact force, and contact state.

Start it with:

```bash
ros2 launch mujoco_simulpack_gui mujoco_simulpack_gui.launch.py
```

Then open:

```text
http://127.0.0.1:18100
```

The GUI's Contact Model section includes `+X/-X`, `+Y/-Y`, and `+Z/-Z` sensor
force direction selectors. Apply the setting after changing it, then restart
the simulator only when changing model or launch-level settings.

## Control Modes

### `position_pd`

This is the original control path. The model file is:

```text
models/ur10e/ur10e_contact_scene.xml
```

The XML uses position-servo style actuators. The command sent through
`/armsimul/joint_position_cmd_array` is interpreted as target joint position:

```text
data.ctrl = q_des
```

The actuator torque is generated by MuJoCo from the XML gain/bias settings:

```text
tau = Kp_xml (q_des - q) - Kd_xml qdot
```

### `inverse_dynamics_accel`

This mode is intended to behave more like IsaacSim joint acceleration control. The model file is:

```text
models/ur10e/ur10e_torque_contact_scene.xml
```

This scene includes:

```text
models/ur10e/ur10e_torque.xml
```

The XML uses torque motor actuators. The command topic is still a joint position target, but the simulator internally converts it to desired joint acceleration and then to torque:

```text
q_des -> qdd_des -> mj_inverse() -> tau_cmd -> data.ctrl
```

The desired acceleration is computed per joint as:

```text
omega_n = 2*pi*accel_natural_frequency_hz
Kp_acc  = omega_n^2
Kd_acc  = 2*zeta*omega_n

qdd_des = Kp_acc*(q_des - q) + Kd_acc*(qd_des - qd)
```

where:

```text
q_des  = commanded joint position
q      = current joint position
qd_des = estimated target velocity from command updates
qd     = current joint velocity
zeta   = accel_damping_ratio
```

Then MuJoCo inverse dynamics is used:

```text
data.qacc[dof] = qdd_des
mujoco.mj_inverse(model, data)
tau_cmd = data.qfrc_inverse[dof]
```

Finally torque is clipped and sent to torque motors:

```text
tau_cmd = clip(tau_cmd, -torque_limit, torque_limit)
data.ctrl[actuator] = tau_cmd
```

The acceleration command is also clipped:

```text
qdd_des = clip(qdd_des, -accel_limit, accel_limit)
```

## Natural Frequency And Damping Tuning

For `inverse_dynamics_accel`, tune the following parameters:

```yaml
control_mode: "inverse_dynamics_accel"
model_file: "models/ur10e/ur10e_torque_contact_scene.xml"
accel_natural_frequency_hz: [6.0, 6.0, 6.0, 8.0, 8.0, 8.0]
accel_damping_ratio: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
accel_limit: [80.0, 80.0, 80.0, 120.0, 120.0, 120.0]
torque_limit: [330.0, 330.0, 150.0, 56.0, 56.0, 56.0]
```

These parameters are intentionally kept only in
`config/ur10_contact_sim_inverse_dynamics.yaml`. The default
`config/ur10_contact_sim.yaml` is for `position_pd` and does not include unused
inverse-dynamics tuning values.

For each joint, the requested second-order error dynamics are approximately:

```text
e = q_des - q
e_ddot + 2*zeta*omega_n*e_dot + omega_n^2*e = 0
```

So the intuitive tuning rules are:

```text
natural_frequency_hz higher -> faster tracking, more torque demand
damping_ratio = 1.0        -> near critically damped
damping_ratio < 1.0        -> faster but can overshoot
damping_ratio > 1.0        -> slower and more damped
```

Convert desired bandwidth to natural frequency conservatively. For a well-damped second-order system, closed-loop bandwidth is usually on the same order as `natural_frequency_hz`, but not exactly identical:

```text
omega_n = 2*pi*natural_frequency_hz
```

Start with low values:

```text
shoulder/elbow: 3-6 Hz
wrist:          5-10 Hz
zeta:           1.0
```

Then use `Y2SYS_ID` sine sweep to measure Cartesian bandwidth. If tracking is too slow, raise `accel_natural_frequency_hz`. If the response overshoots or oscillates, increase `accel_damping_ratio` or reduce `natural_frequency_hz`.

Always check `/armsimul/joint_states.effort`. If effort approaches `torque_limit`, the requested natural frequency is not physically achievable with the configured torque limits.

## Recommended Inverse-Dynamics Experiment

1. Start the simulator in inverse-dynamics mode:

```bash
ros2 launch mujoco_simulpack ur10_contact_sim.launch.py \
  config_file:=/home/jay/armcon_ws/src/armsimul_pack/mujoco_simulpack/config/ur10_contact_sim_inverse_dynamics.yaml
```

2. Start `Y2RobMotion`:

```bash
ros2 launch Y2RobMotion ur_mujocoArm.launch.py
```

3. Run `Y2SYS_ID`:

```bash
ros2 launch Y2SYS_ID cartesian_id.launch.py
```

4. Inspect the YAML result:

```text
/home/jay/armcon_ws/src/armcon_pack/Y2SYS_ID/results/*_id_*.yaml
```

5. Increase or decrease `accel_natural_frequency_hz` and repeat.

Use small Cartesian amplitudes first, for example 2-5 mm, because high acceleration gains can produce large joint torques through IK.

## Model Editing

The default scene is:

```text
models/ur10e/ur10e_contact_scene.xml
```

It includes `ur10e.xml`, a floor, and a movable contact box. Change the box
position, size, friction, or add new objects in this XML file. After editing
model/config files in `src`, rebuild the package:

The inverse-dynamics torque scene is:

```text
models/ur10e/ur10e_torque_contact_scene.xml
```

It includes `ur10e_torque.xml`, which uses `<motor>` torque actuators instead of position-servo actuators.

```bash
cd ~/armcon_ws
colcon build --packages-select mujoco_simulpack
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

### Contact body axes

The MuJoCo viewer starts with coordinate axes hidden. Press **F6** to toggle only the selected contact body's local XYZ frame (red, green, blue), using the small frame dimensions in the scene XML. The arrows follow the body pose and are visual decorations, so they do not affect collisions. `viewer_frame_body` selects a body explicitly; by default the last existing entry in `ee_body_names` is used (`nrs_spindle` in the spindle model, or `wrist_3_link` in the model without a spindle). F6 toggles this overlay on/off instead of cycling through native all-body frames. Restart the simulator to load this viewer change.

Small contact-body axes are drawn as short colored arrows beyond the body bounds, with thin stems along the same axes from the true body origin. This prevents the spindle mesh from hiding the arrows. F6 logs `Contact body axes: shown/hidden`; visibility does not depend on contact detection.
