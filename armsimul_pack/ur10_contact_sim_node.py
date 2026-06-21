from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import WrenchStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
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
                ("publish_rate_hz", 100.0),
                ("simulation_rate_hz", 1000.0),
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
                ("contact_state_topic", "/armsimul/contact_state"),
                ("frame_id", "base"),
                ("ee_frame_id", "attachment_site"),
                ("ee_body_names", ["wrist_3_link"]),
                ("contact_force_deadband", 0.01),
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

        self._validate_config()

        self.joint_qpos_addr = [self.model.joint(name).qposadr[0] for name in self.joint_names]
        self.joint_dof_addr = [self.model.joint(name).dofadr[0] for name in self.joint_names]
        self.actuator_ids = [self.model.actuator(name).id for name in self.actuator_names]
        self.body_name_by_geom = self._build_body_name_by_geom()

        self.target_positions = self.initial_positions.copy()
        for index, qpos_addr in enumerate(self.joint_qpos_addr):
            self.data.qpos[qpos_addr] = self.initial_positions[index]
        for index, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = self.initial_positions[index]
        mujoco.mj_forward(self.model, self.data)

        self.last_command_time = self.get_clock().now()
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.ee_frame_id = str(self.get_parameter("ee_frame_id").value)
        self.ee_body_names = set(self.get_parameter("ee_body_names").value)
        self.force_deadband = float(self.get_parameter("contact_force_deadband").value)

        self.joint_state_pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_state_topic").value), 10
        )
        self.wrench_pub = self.create_publisher(
            WrenchStamped, str(self.get_parameter("ee_wrench_topic").value), 10
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
        self.sim_timer = self.create_timer(1.0 / sim_rate, self._step_simulation)
        self.publish_timer = self.create_timer(1.0 / pub_rate, self._publish_state)

        self.viewer = None
        if bool(self.get_parameter("use_viewer").value):
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.model, self.data)

        self.get_logger().info(f"Loaded MuJoCo model: {self.model_path}")

    def _resolve_model_path(self, model_file: str) -> Path:
        path = Path(model_file).expanduser()
        if path.is_absolute():
            return path
        return Path(get_package_share_directory("armsimul_pack")) / path

    def _validate_config(self) -> None:
        count = len(self.joint_names)
        required_lengths = {
            "actuator_names": self.actuator_names,
            "initial_positions": self.initial_positions,
            "position_limits_lower": self.lower_limits,
            "position_limits_upper": self.upper_limits,
        }
        for name, values in required_lengths.items():
            if len(values) != count:
                raise ValueError(f"Parameter '{name}' must have {count} entries.")
        if count == 0:
            raise ValueError("Parameter 'joint_names' must not be empty.")

    def _build_body_name_by_geom(self) -> Dict[int, str]:
        body_names = {}
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_names[geom_id] = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id)
        return body_names

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

        for index, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = float(self.target_positions[index])

        self.mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

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

        contact_state = Bool()
        contact_state.data = bool(np.linalg.norm(force) > self.force_deadband)
        self.contact_pub.publish(contact_state)

    def _sum_ee_contact_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

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
