import argparse
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


DEFAULT_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class JointPositionSender(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("armsimul_joint_position_sender")
        self.publisher = self.create_publisher(JointState, topic, 10)

    def send(self, positions: List[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = DEFAULT_JOINT_NAMES
        msg.position = positions
        self.publisher.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one UR10e joint position command to armsimul_pack.")
    parser.add_argument("positions", nargs=6, type=float, help="Six joint positions in radians.")
    parser.add_argument("--topic", default="/armsimul/joint_position_cmd", help="JointState command topic.")
    args = parser.parse_args()

    rclpy.init()
    node = JointPositionSender(args.topic)
    try:
        for _ in range(5):
            node.send(args.positions)
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
