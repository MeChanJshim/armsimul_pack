import argparse
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class RealtimeJointPositionDemo(Node):
    def __init__(self, topic: str, rate_hz: float, amplitude: float) -> None:
        super().__init__("armsimul_realtime_joint_position_demo")
        self.publisher = self.create_publisher(Float64MultiArray, topic, 10)
        self.rate_hz = rate_hz
        self.amplitude = amplitude
        self.start_time = time.monotonic()
        self.base_position = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self.timer = self.create_timer(1.0 / self.rate_hz, self._publish_command)

    def _publish_command(self) -> None:
        elapsed = time.monotonic() - self.start_time
        command = list(self.base_position)
        command[0] += self.amplitude * math.sin(2.0 * math.pi * 0.2 * elapsed)
        command[2] += self.amplitude * math.sin(2.0 * math.pi * 0.2 * elapsed + math.pi / 2.0)

        msg = Float64MultiArray()
        msg.data = command
        self.publisher.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish continuous UR10e joint position commands.")
    parser.add_argument("--topic", default="/armsimul/joint_position_cmd_array")
    parser.add_argument("--rate", type=float, default=100.0, help="Publish rate in Hz.")
    parser.add_argument("--amplitude", type=float, default=0.15, help="Sine amplitude in radians.")
    args = parser.parse_args()

    rclpy.init()
    node = RealtimeJointPositionDemo(args.topic, args.rate, args.amplitude)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
