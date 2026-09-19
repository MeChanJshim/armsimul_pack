from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Wrench, WrenchStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory


class Ur10ContactSim(Node):
    def __init__(self) -> None:
        super().__init__("ur10_contact_sim")

        default_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        default_actuator_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]
        default_initial_positions = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        default_lower_limits = [-6.2831, -6.2831, -3.1415, -6.2831, -6.2831, -6.2831]
        default_upper_limits = [6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831]

        self.declare_parameters(
            namespace="",
            parameters=[
                ("model_file", "models/ur10e/ur10e_contact_scene.xml"),
                ("use_viewer", False),
                ("viewer_frame_body", ""),
                ("publish_rate_hz", 100.0),
                ("simulation_rate_hz", 1000.0),
                ("control_mode", "position_pd"),
                ("accel_natural_frequency_hz", [6.0, 6.0, 6.0, 8.0, 8.0, 8.0]),
                ("accel_damping_ratio", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ("accel_limit", [80.0, 80.0, 80.0, 120.0, 120.0, 120.0]),
                ("torque_limit", [330.0, 330.0, 150.0, 56.0, 56.0, 56.0]),
                ("joint_names", default_joint_names),
                ("actuator_names", default_actuator_names),
                ("initial_positions", default_initial_positions),
                ("position_limits_lower", default_lower_limits),
                ("position_limits_upper", default_upper_limits),
                ("joint_state_topic", "/joint_states"),
                ("joint_position_command_topic", "/armsimul/joint_position_cmd"),
                ("joint_position_array_command_topic", "/armsimul/joint_position_cmd_array"),
                ("joint_trajectory_command_topic", "/joint_trajectory_controller/joint_trajectory"),
                ("ee_wrench_topic", "/armsimul/ee_wrench"),
                ("ft_sensor_wrench_topic", "/armsimul/ft_sensor_wrench"),
                ("ft_sensor_wrench_raw_topic", "/armsimul/ft_sensor_wrench_raw"),
                ("contact_state_topic", "/armsimul/contact_state"),
                ("frame_id", "base"),
                ("ee_frame_id", "attachment_site"),
                ("ft_sensor_site_name", "attachment_site"),
                ("ee_body_names", ["wrist_3_link", "nrs_spindle"]),
                ("contact_force_deadband", 0.01),
                ("contact_detection_enabled", True),
                ("force_axis_sign", [1.0, 1.0, 1.0]),
                ("solimp", [0.9, 0.95, 0.001, 0.5, 2.0]),
                ("solref", [0.02, 1.0]),
                ("friction", [1.0, 0.1, 0.001]),
                ("condim", 4),
            ],
        )

        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "Python package 'mujoco' is not installed. Install it with "
                "`python3 -m pip install --user mujoco`."
            ) from exc

        self.mujoco = mujoco
        self.model_path = self._resolve_model_path(self.get_parameter("model_file").value)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.joint_names = list(self.get_parameter("joint_names").value)
        self.actuator_names = list(self.get_parameter("actuator_names").value)
        self.initial_positions = np.array(self.get_parameter("initial_positions").value, dtype=float)
        self.lower_limits = np.array(self.get_parameter("position_limits_lower").value, dtype=float)
        self.upper_limits = np.array(self.get_parameter("position_limits_upper").value, dtype=float)
        self.control_mode = str(self.get_parameter("control_mode").value)
        self.accel_natural_frequency_hz = np.array(
            self.get_parameter("accel_natural_frequency_hz").value,
            dtype=float,
        )
        self.accel_damping_ratio = np.array(
            self.get_parameter("accel_damping_ratio").value,
            dtype=float,
        )
        self.accel_limit = np.array(self.get_parameter("accel_limit").value, dtype=float)
        self.torque_limit = np.array(self.get_parameter("torque_limit").value, dtype=float)

        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self._validate_config()

        self.joint_qpos_addr = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.joint_dof_addr = [self.model.joint(name).dofadr[0] for name in self.joint_names]
        self.actuator_ids = [self.model.actuator(name).id for name in self.actuator_names]
        self.body_name_by_geom = self._build_body_name_by_geom()
        self.contact_geom_ids = self._build_environment_geom_ids()

        self.target_positions = self.initial_positions.copy()
        self.previous_target_positions = self.initial_positions.copy()
        for index, qpos_addr in enumerate(self.joint_qpos_addr):
            self.data.qpos[qpos_addr] = self.initial_positions[index]
        if self.control_mode == "position_pd":
            for index, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = self.initial_positions[index]
        else:
            self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.last_command_time = self.get_clock().now()
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.ee_frame_id = str(self.get_parameter("ee_frame_id").value)
        self.ft_sensor_site_name = str(self.get_parameter("ft_sensor_site_name").value)
        try:
            self.ft_sensor_site_id = self.model.site(self.ft_sensor_site_name).id
        except Exception as exc:
            raise ValueError(
                f"MuJoCo site '{self.ft_sensor_site_name}' was not found; "
                "set ft_sensor_site_name to a valid sensor site."
            ) from exc
        self.ee_body_names = set(self.get_parameter("ee_body_names").value)
        self.force_deadband = float(self.get_parameter("contact_force_deadband").value)
        self.contact_detection_enabled = bool(self.get_parameter("contact_detection_enabled").value)
        self.force_axis_sign = np.asarray(
            self.get_parameter("force_axis_sign").value, dtype=float
        )
        self._apply_contact_model_parameters()

        self.joint_state_pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_state_topic").value), 10
        )
        self.wrench_pub = self.create_publisher(
            WrenchStamped, str(self.get_parameter("ee_wrench_topic").value), 10
        )
        self.ft_sensor_wrench_pub = self.create_publisher(
            WrenchStamped, str(self.get_parameter("ft_sensor_wrench_topic").value), 10
        )
        self.ft_sensor_wrench_raw_pub = self.create_publisher(
            Wrench, str(self.get_parameter("ft_sensor_wrench_raw_topic").value), 10
        )
        self.contact_pub = self.create_publisher(
            Bool, str(self.get_parameter("contact_state_topic").value), 10
        )

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_position_command_topic").value),
            self._joint_state_command_cb,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("joint_position_array_command_topic").value),
            self._joint_position_array_command_cb,
            10,
        )
        self.create_subscription(
            JointTrajectory,
            str(self.get_parameter("joint_trajectory_command_topic").value),
            self._trajectory_command_cb,
            10,
        )

        sim_rate = float(self.get_parameter("simulation_rate_hz").value)
        pub_rate = float(self.get_parameter("publish_rate_hz").value)
        self.sim_period = 1.0 / sim_rate
        self.sim_timer = self.create_timer(1.0 / sim_rate, self._step_simulation)
        self.publish_timer = self.create_timer(1.0 / pub_rate, self._publish_state)

        self.viewer = None
        self.viewer_body_axes_visible = False
        if bool(self.get_parameter("use_viewer").value):
            from mujoco import viewer

            requested_body = str(self.get_parameter("viewer_frame_body").value)
            candidates = ([requested_body] if requested_body else
                          list(reversed(self.get_parameter("ee_body_names").value)))
            self.viewer_frame_body_id = -1
            for name in candidates:
                body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    self.viewer_frame_body_id = body_id
                    break
            if self.viewer_frame_body_id < 0:
                raise ValueError(f"No viewer frame body found among {candidates}")
            self.viewer_axis_offsets = self._viewer_axis_surface_offsets()
            self.viewer = viewer.launch_passive(
                self.model, self.data, key_callback=self._viewer_key_callback)
            self._update_viewer_body_axes()

        self.get_logger().info(f"Loaded MuJoCo model: {self.model_path}")
        self.get_logger().info(f"Control mode: {self.control_mode}")

    def _resolve_model_path(self, model_file: str) -> Path:
        path = Path(model_file).expanduser()
        if path.is_absolute():
            return path
        return Path(get_package_share_directory("mujoco_simulpack")) / path

    def _validate_config(self) -> None:
        count = len(self.joint_names)
        required_lengths = {
            "actuator_names": self.actuator_names,
            "initial_positions": self.initial_positions,
            "position_limits_lower": self.lower_limits,
            "position_limits_upper": self.upper_limits,
            "accel_natural_frequency_hz": self.accel_natural_frequency_hz,
            "accel_damping_ratio": self.accel_damping_ratio,
            "accel_limit": self.accel_limit,
            "torque_limit": self.torque_limit,
        }
        for name, values in required_lengths.items():
            if len(values) != count:
                raise ValueError(f"Parameter '{name}' must have {count} entries.")
        if count == 0:
            raise ValueError("Parameter 'joint_names' must not be empty.")
        if self.control_mode not in ("position_pd", "inverse_dynamics_accel"):
            raise ValueError(
                "Parameter 'control_mode' must be 'position_pd' or 'inverse_dynamics_accel'."
            )

    def _on_parameters_changed(self, parameters):
        try:
            values = {parameter.name: parameter.value for parameter in parameters}
            if "simulation_rate_hz" in values or "publish_rate_hz" in values:
                simulation_rate = float(values.get("simulation_rate_hz", self.sim_period ** -1))
                publish_rate = float(values.get("publish_rate_hz", 1.0 / self.publish_timer.timer_period_ns * 1e9))
                if simulation_rate <= 0.0 or publish_rate <= 0.0:
                    return SetParametersResult(successful=False, reason="Rates must be positive")
                if hasattr(self, "sim_timer"):
                    self.destroy_timer(self.sim_timer)
                    self.sim_timer = self.create_timer(1.0 / simulation_rate, self._step_simulation)
                if hasattr(self, "publish_timer"):
                    self.destroy_timer(self.publish_timer)
                    self.publish_timer = self.create_timer(1.0 / publish_rate, self._publish_state)
                self.sim_period = 1.0 / simulation_rate
            if "contact_force_deadband" in values:
                deadband = float(values["contact_force_deadband"])
                if deadband < 0.0:
                    return SetParametersResult(successful=False, reason="contact_force_deadband must not be negative")
                self.force_deadband = deadband
            if "contact_detection_enabled" in values:
                self.contact_detection_enabled = bool(values["contact_detection_enabled"])
            if "force_axis_sign" in values:
                force_axis_sign = np.asarray(values["force_axis_sign"], dtype=float)
                if len(force_axis_sign) != 3 or not np.all(np.isfinite(force_axis_sign)):
                    return SetParametersResult(
                        successful=False,
                        reason="force_axis_sign must contain three finite values",
                    )
                if not np.all(np.isin(force_axis_sign, (-1.0, 1.0))):
                    return SetParametersResult(
                        successful=False,
                        reason="force_axis_sign values must be -1 or 1",
                    )
                self.force_axis_sign = force_axis_sign
            if "ee_body_names" in values:
                body_names = [str(name) for name in values["ee_body_names"]]
                if not body_names:
                    return SetParametersResult(successful=False, reason="ee_body_names must not be empty")
                self.ee_body_names = set(body_names)
            for parameter_name, width in (("solimp", 5), ("solref", 2)):
                if parameter_name in values:
                    contact_values = np.asarray(values[parameter_name], dtype=float)
                    if len(contact_values) != width or not np.all(np.isfinite(contact_values)):
                        return SetParametersResult(
                            successful=False,
                            reason=f"{parameter_name} must contain {width} finite values",
                        )
                    if parameter_name == "solimp":
                        self.model.geom_solimp[self.contact_geom_ids, :] = contact_values
                    else:
                        self.model.geom_solref[self.contact_geom_ids, :] = contact_values
            if "friction" in values:
                friction = np.asarray(values["friction"], dtype=float)
                if len(friction) != 3 or not np.all(np.isfinite(friction)) or np.any(friction < 0.0):
                    return SetParametersResult(
                        successful=False, reason="friction must contain three non-negative values")
                self.model.geom_friction[self.contact_geom_ids, :] = friction
            if "condim" in values:
                condim = int(values["condim"])
                if condim not in (1, 3, 4, 6):
                    return SetParametersResult(
                        successful=False, reason="condim must be one of 1, 3, 4, or 6")
                self.model.geom_condim[self.contact_geom_ids] = condim
            if "control_mode" in values:
                if values["control_mode"] not in ("position_pd", "inverse_dynamics_accel"):
                    return SetParametersResult(
                        successful=False, reason="Unsupported control_mode")
                self.control_mode = str(values["control_mode"])

            arrays = {
                "accel_natural_frequency_hz": "accel_natural_frequency_hz",
                "accel_damping_ratio": "accel_damping_ratio",
                "accel_limit": "accel_limit",
                "torque_limit": "torque_limit",
            }
            for parameter_name, attribute_name in arrays.items():
                if parameter_name in values:
                    array = np.asarray(values[parameter_name], dtype=float)
                    if len(array) != len(self.joint_names) or not np.all(np.isfinite(array)):
                        return SetParametersResult(
                            successful=False,
                            reason=f"{parameter_name} must contain six finite values",
                        )
                    if parameter_name in ("accel_natural_frequency_hz", "accel_limit", "torque_limit") and np.any(array <= 0.0):
                        return SetParametersResult(
                            successful=False, reason=f"{parameter_name} must be positive")
                    if parameter_name == "accel_damping_ratio" and np.any(array < 0.0):
                        return SetParametersResult(
                            successful=False, reason="accel_damping_ratio must not be negative")
                    setattr(self, attribute_name, array)
            return SetParametersResult(successful=True)
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))

    def _build_body_name_by_geom(self) -> Dict[int, str]:
        body_names = {}
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_names[geom_id] = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
        return body_names

    def _build_environment_geom_ids(self) -> np.ndarray:
        """Return geoms outside the robot kinematic tree for contact settings."""
        robot_body_ids = set()
        for joint_name in self.joint_names:
            body_id = int(self.model.joint(joint_name).bodyid)
            while body_id >= 0:
                robot_body_ids.add(body_id)
                parent_id = int(self.model.body_parentid[body_id])
                if parent_id == body_id:
                    break
                body_id = parent_id

        environment_ids = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) not in robot_body_ids
        ]
        return np.asarray(environment_ids, dtype=int)

    def _apply_contact_model_parameters(self) -> None:
        """Apply contact settings only to environment geoms, never robot links."""
        if self.contact_geom_ids.size == 0:
            return
        self.model.geom_solimp[self.contact_geom_ids, :] = np.asarray(
            self.get_parameter("solimp").value, dtype=float
        )
        self.model.geom_solref[self.contact_geom_ids, :] = np.asarray(
            self.get_parameter("solref").value, dtype=float
        )
        self.model.geom_friction[self.contact_geom_ids, :] = np.asarray(
            self.get_parameter("friction").value, dtype=float
        )
        self.model.geom_condim[self.contact_geom_ids] = int(
            self.get_parameter("condim").value
        )

    def _joint_state_command_cb(self, msg: JointState) -> None:
        if not msg.position:
            return
        if msg.name:
            position_by_name = dict(zip(msg.name, msg.position))
            positions = [position_by_name.get(name, self.target_positions[i]) for i, name in enumerate(self.joint_names)]
        else:
            positions = list(msg.position[: len(self.joint_names)])
        self._set_target_positions(positions)

    def _joint_position_array_command_cb(self, msg: Float64MultiArray) -> None:
        self._set_target_positions(msg.data[: len(self.joint_names)])

    def _trajectory_command_cb(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return
        point = msg.points[-1]
        if not point.positions:
            return
        position_by_name = dict(zip(msg.joint_names, point.positions))
        positions = [position_by_name.get(name, self.target_positions[i]) for i, name in enumerate(self.joint_names)]
        self._set_target_positions(positions)

    def _set_target_positions(self, positions: Iterable[float]) -> None:
        target = np.array(list(positions), dtype=float)
        if len(target) != len(self.joint_names):
            self.get_logger().warn(
                f"Ignoring command with {len(target)} positions; expected {len(self.joint_names)}."
            )
            return
        self.target_positions = np.clip(target, self.lower_limits, self.upper_limits)
        self.last_command_time = self.get_clock().now()

    def _step_simulation(self) -> None:
        if not rclpy.ok():
            return

        if self.control_mode == "inverse_dynamics_accel":
            self._apply_inverse_dynamics_accel_control()
        else:
            self._apply_position_pd_control()

        self.mujoco.mj_step(self.model, self.data)
        self.previous_target_positions = self.target_positions.copy()
        if self.viewer is not None:
            self._update_viewer_body_axes()
            self.viewer.sync()

    def _viewer_axis_surface_offsets(self) -> np.ndarray:
        """Place the short axis arrows outside the contact body's solid mesh."""
        maximum = np.zeros(3)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] != self.viewer_frame_body_id:
                continue
            center, half_size = self.model.geom_aabb[geom_id].reshape(2, 3)
            rotation = np.empty(9)
            self.mujoco.mju_quat2Mat(rotation, self.model.geom_quat[geom_id])
            corners = (center + signs * half_size) @ rotation.reshape(3, 3).T
            corners += self.model.geom_pos[geom_id]
            maximum = np.maximum(maximum, corners.max(axis=0))
        gap = self.model.stat.meansize * self.model.vis.scale.framewidth
        return maximum + gap

    def _viewer_key_callback(self, keycode: int) -> None:
        from glfw import KEY_F6

        if keycode == KEY_F6:
            self.viewer_body_axes_visible = not self.viewer_body_axes_visible
            self.get_logger().info("Contact body axes: " + ("shown" if self.viewer_body_axes_visible else "hidden"))
            if self.viewer is not None:
                self._update_viewer_body_axes()

    def _update_viewer_body_axes(self) -> None:
        """Draw only the contact body's local frame as visual-only arrows."""
        with self.viewer.lock():
            self.viewer.opt.frame = self.mujoco.mjtFrame.mjFRAME_NONE
            scene = self.viewer.user_scn
            scene.ngeom = 0
            if not self.viewer_body_axes_visible:
                return
            body_id = self.viewer_frame_body_id
            origin = self.data.xpos[body_id]
            rotation = self.data.xmat[body_id].reshape(3, 3)
            length = self.model.stat.meansize * self.model.vis.scale.framelength
            width = self.model.stat.meansize * self.model.vis.scale.framewidth
            colors = ((1.0, 0.15, 0.15, 1.0), (0.15, 1.0, 0.25, 1.0), (0.2, 0.45, 1.0, 1.0))
            for axis, color in enumerate(colors):
                if scene.ngeom + 2 > scene.maxgeom:
                    break
                # Keep the true body origin and direction; extend a thin stem
                # through the mesh to a short, visible arrow outside the body.
                start = origin + self.viewer_axis_offsets[axis] * rotation[:, axis]
                stem = scene.geoms[scene.ngeom]
                self.mujoco.mjv_initGeom(stem, self.mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3), origin, np.eye(3).ravel(), np.array([.6, .6, .6, 1], dtype=np.float32))
                self.mujoco.mjv_connector(stem, self.mujoco.mjtGeom.mjGEOM_LINE,
                    1.0, origin, start)
                scene.ngeom += 1
                geom = scene.geoms[scene.ngeom]
                self.mujoco.mjv_initGeom(geom, self.mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3), origin, np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
                self.mujoco.mjv_connector(geom, self.mujoco.mjtGeom.mjGEOM_ARROW,
                    width, start, start + length * rotation[:, axis])
                scene.ngeom += 1

    def _apply_position_pd_control(self) -> None:
        for index, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = float(self.target_positions[index])

    def _apply_inverse_dynamics_accel_control(self) -> None:
        qpos = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addr])
        qvel = np.array([self.data.qvel[addr] for addr in self.joint_dof_addr])

        target_velocity = (
            self.target_positions - self.previous_target_positions
        ) / self.sim_period

        omega_n = 2.0 * np.pi * self.accel_natural_frequency_hz
        kp = omega_n * omega_n
        kd = 2.0 * self.accel_damping_ratio * omega_n
        qacc_des = kp * (self.target_positions - qpos) + kd * (target_velocity - qvel)
        qacc_des = np.clip(qacc_des, -self.accel_limit, self.accel_limit)

        self.data.qacc[:] = 0.0
        for index, dof_addr in enumerate(self.joint_dof_addr):
            self.data.qacc[dof_addr] = qacc_des[index]

        self.mujoco.mj_inverse(self.model, self.data)
        tau = np.array([self.data.qfrc_inverse[addr] for addr in self.joint_dof_addr])
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)

        for index, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = float(tau[index])

    def _publish_state(self) -> None:
        if not rclpy.ok():
            return

        stamp = self.get_clock().now().to_msg()

        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.header.frame_id = self.frame_id
        joint_state.name = self.joint_names
        joint_state.position = [float(self.data.qpos[addr]) for addr in self.joint_qpos_addr]
        joint_state.velocity = [float(self.data.qvel[addr]) for addr in self.joint_dof_addr]
        joint_state.effort = [float(self.data.qfrc_actuator[addr]) for addr in self.joint_dof_addr]
        self.joint_state_pub.publish(joint_state)

        force, torque = self._sum_ee_contact_wrench()
        force *= self.force_axis_sign
        wrench = WrenchStamped()
        wrench.header.stamp = stamp
        wrench.header.frame_id = self.ee_frame_id
        wrench.wrench.force.x = float(force[0])
        wrench.wrench.force.y = float(force[1])
        wrench.wrench.force.z = float(force[2])
        wrench.wrench.torque.x = float(torque[0])
        wrench.wrench.torque.y = float(torque[1])
        wrench.wrench.torque.z = float(torque[2])
        self.wrench_pub.publish(wrench)

        sensor_force, sensor_torque = self._sum_ft_sensor_wrench()
        sensor_force *= self.force_axis_sign
        sensor_wrench = WrenchStamped()
        sensor_wrench.header.stamp = stamp
        sensor_wrench.header.frame_id = self.ft_sensor_site_name
        sensor_wrench.wrench.force.x = float(sensor_force[0])
        sensor_wrench.wrench.force.y = float(sensor_force[1])
        sensor_wrench.wrench.force.z = float(sensor_force[2])
        sensor_wrench.wrench.torque.x = float(sensor_torque[0])
        sensor_wrench.wrench.torque.y = float(sensor_torque[1])
        sensor_wrench.wrench.torque.z = float(sensor_torque[2])
        self.ft_sensor_wrench_pub.publish(sensor_wrench)

        # NRS intrinsic contact sensing consumes geometry_msgs/Wrench rather
        # than WrenchStamped, so expose the same sensor-frame values directly.
        raw_wrench = Wrench()
        raw_wrench.force.x = float(sensor_force[0])
        raw_wrench.force.y = float(sensor_force[1])
        raw_wrench.force.z = float(sensor_force[2])
        raw_wrench.torque.x = float(sensor_torque[0])
        raw_wrench.torque.y = float(sensor_torque[1])
        raw_wrench.torque.z = float(sensor_torque[2])
        self.ft_sensor_wrench_raw_pub.publish(raw_wrench)

        contact_state = Bool()
        contact_state.data = bool(np.linalg.norm(force) > self.force_deadband)
        self.contact_pub.publish(contact_state)

    def _sum_ee_contact_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        if not self.contact_detection_enabled:
            return total_force, total_torque

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = self.body_name_by_geom.get(geom1)
            body2 = self.body_name_by_geom.get(geom2)

            if body1 not in self.ee_body_names and body2 not in self.ee_body_names:
                continue

            contact_force = np.zeros(6)
            self.mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
            frame = np.array(contact.frame).reshape(3, 3)
            world_force = frame.T @ contact_force[:3]
            world_torque = frame.T @ contact_force[3:]

            if body2 in self.ee_body_names:
                world_force *= -1.0
                world_torque *= -1.0

            total_force += world_force
            total_torque += world_torque

        if np.linalg.norm(total_force) < self.force_deadband:
            total_force[:] = 0.0
        if np.linalg.norm(total_torque) < self.force_deadband:
            total_torque[:] = 0.0
        return total_force, total_torque

    def _sum_ft_sensor_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the contact wrench at the configured virtual FT sensor frame.

        MuJoCo reports each contact wrench at the contact point in world
        coordinates. Shift the torque to the sensor origin with r x F, then
        rotate both vectors into the sensor site's local coordinates.
        """
        world_force = np.zeros(3)
        world_torque_at_sensor = np.zeros(3)

        if not self.contact_detection_enabled:
            return world_force, world_torque_at_sensor

        sensor_position = np.asarray(self.data.site_xpos[self.ft_sensor_site_id], dtype=float)
        sensor_rotation = np.asarray(
            self.data.site_xmat[self.ft_sensor_site_id], dtype=float
        ).reshape(3, 3)

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = self.body_name_by_geom.get(geom1)
            body2 = self.body_name_by_geom.get(geom2)

            if body1 not in self.ee_body_names and body2 not in self.ee_body_names:
                continue

            contact_wrench = np.zeros(6)
            self.mujoco.mj_contactForce(self.model, self.data, contact_index, contact_wrench)
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            contact_force_world = frame.T @ contact_wrench[:3]
            contact_torque_world = frame.T @ contact_wrench[3:]

            if body2 in self.ee_body_names:
                contact_force_world *= -1.0
                contact_torque_world *= -1.0

            contact_position = np.asarray(contact.pos, dtype=float)
            world_force += contact_force_world
            world_torque_at_sensor += contact_torque_world
            world_torque_at_sensor += np.cross(
                contact_position - sensor_position, contact_force_world
            )

        if np.linalg.norm(world_force) < self.force_deadband:
            world_force[:] = 0.0
        if np.linalg.norm(world_torque_at_sensor) < self.force_deadband:
            world_torque_at_sensor[:] = 0.0

        # Site rotation columns are the sensor axes expressed in world frame.
        return sensor_rotation.T @ world_force, sensor_rotation.T @ world_torque_at_sensor


def main(args: List[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Ur10ContactSim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.viewer is not None:
            node.viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
