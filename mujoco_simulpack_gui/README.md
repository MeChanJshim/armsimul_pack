# mujoco_simulpack_gui

`mujoco_simulpack_gui` is the browser-based management and monitoring GUI for
the `mujoco_simulpack` ROS2 Humble simulator.

## Features

- Start and stop the MuJoCo simulator and joint demo process.
- Configure controller, inverse-dynamics, joint-limit, and contact settings.
- Change contact solver parameters: `solimp`, `solref`, friction, and `condim`.
- Select the reported sensor force direction independently for X, Y, and Z.
- Save and load ROS parameter YAML profiles under the simulator package's
  `config/profiles/` directory (the source package in a symlink workspace).
  The GUI selects the default folder on page load. Use the folder picker to
  choose another location, then select a YAML file and load it.
  Use `.yaml` filenames; saved files are passed directly to the next launch.
  Existing JSON profiles in `~/.ros/mujoco_simulpack_gui/settings` are copied
  to YAML on GUI startup without overwriting an existing YAML or deleting JSON.
- Saving or loading a profile stages it for the next simulator launch without
  changing a running simulator. Start the simulator from the GUI after loading
  the profile to apply the complete configuration from startup.
- Monitor joint positions, joint efforts, contact force, contact state, and
  recent samples through plots.

## Build

```bash
cd ~/armcon_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mujoco_simulpack mujoco_simulpack_gui --symlink-install
source install/setup.bash
```

## Run

Start the simulator first, or let the GUI start it from the Simulator Nodes
panel:

```bash
ros2 launch mujoco_simulpack_gui mujoco_simulpack_gui.launch.py
```

Open the page at:

```text
http://127.0.0.1:18100
```

To use another port:

```bash
ros2 launch mujoco_simulpack_gui mujoco_simulpack_gui.launch.py port:=18101
```

## Runtime Settings

The individual **Apply** buttons send live ROS parameter updates to
`/ur10_contact_sim`. Contact settings
are applied to environment geoms, while robot-link geoms are left unchanged.

This prevents friction and `condim` settings from unintentionally locking the
robot's own links together.

The sensor direction selectors multiply the published force vector by
`[Fx_sign, Fy_sign, Fz_sign]`, where each sign is `+1` or `-1`. For example,
selecting `-Z` changes a measured `Fz=-10 N` into `Fz=+10 N` for downstream
force control. The physical MuJoCo contact direction is unchanged.

If contact behavior is abnormal, first check for duplicate simulator or
controller nodes:

```bash
ros2 node list
pgrep -af 'ur10_contact_sim'
pgrep -af 'mujoco_simulpack_gui'
```

After verifying the unwanted PID, stop it with:

```bash
kill -9 PID
```

If there are no duplicate nodes and contact is still behaving unexpectedly,
return to the Contact Model section and press **Apply contact model** again.
This reapplies the current `solimp`, `solref`, friction, `condim`, and sensor
force-direction values to the running simulator.

## Related Topics

The GUI monitors:

- `/armsimul/joint_states`
- `/armsimul/ee_wrench`
- `/armsimul/contact_state`

It controls the simulator through its parameter service and can launch:

- `mujoco_simulpack/ur10_contact_sim.launch.py`
- `mujoco_simulpack/realtime_joint_position_demo`
