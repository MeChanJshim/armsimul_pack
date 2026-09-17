import json
import math
import os
import shlex
import signal
import subprocess
import threading
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import WrenchStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


class ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True


class MuJoCoGui(Node):
    def __init__(self):
        super().__init__("mujoco_simulpack_gui")
        self.declare_parameter("port", 18100)
        self.declare_parameter("simulator_node", "/ur10_contact_sim")
        self.port = int(self.get_parameter("port").value)
        self.simulator_node = str(self.get_parameter("simulator_node").value).rstrip("/")
        self.lock = threading.Lock()
        # Keep enough history for the browser to select a useful plot duration.
        self.samples = deque(maxlen=60000)
        self.last_joint = {"name": [], "position": [], "velocity": [], "effort": []}
        self.last_wrench = {"force": [0.0, 0.0, 0.0], "torque": [0.0, 0.0, 0.0]}
        self.last_contact = False
        self.parameter_client = self.create_client(
            SetParameters, f"{self.simulator_node}/set_parameters")
        self.workspace_setup = Path(get_package_prefix("mujoco_simulpack_gui")).parent / "setup.bash"
        self.process_lock = threading.RLock()
        self.processes = {
            "mujoco_simulator": {
                "name": "MuJoCo Simulator",
                "command": ["ros2", "launch", "mujoco_simulpack", "ur10_contact_sim.launch.py"],
                "process": None, "returncode": None, "log": deque(maxlen=80),
            },
            "joint_demo": {
                "name": "Joint Position Demo",
                "command": ["ros2", "run", "mujoco_simulpack", "realtime_joint_position_demo"],
                "process": None, "returncode": None, "log": deque(maxlen=80),
            },
        }
        self.create_subscription(JointState, "/armsimul/joint_states", self.on_joint, 10)
        self.create_subscription(WrenchStamped, "/armsimul/ee_wrench", self.on_wrench, 10)
        self.create_subscription(Bool, "/armsimul/contact_state", self.on_contact, 10)
        self.web_dir = Path(get_package_share_directory("mujoco_simulpack_gui")) / "web"
        self.settings_dir = Path.home() / ".ros" / "mujoco_simulpack_gui" / "settings"
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.server = None
        self.server_thread = None

    def on_joint(self, msg):
        with self.lock:
            self.last_joint = {
                "name": list(msg.name),
                "position": list(msg.position),
                "velocity": list(msg.velocity),
                "effort": list(msg.effort),
            }
            self._record_sample()

    def on_wrench(self, msg):
        with self.lock:
            self.last_wrench = {
                "force": [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
                "torque": [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
            }
            self._record_sample()

    def on_contact(self, msg):
        with self.lock:
            self.last_contact = bool(msg.data)

    def _record_sample(self):
        position = self.last_joint.get("position", [])
        force = self.last_wrench["force"]
        self.samples.append({
            "t": self.get_clock().now().nanoseconds / 1e9,
            "position": position,
            "effort": self.last_joint.get("effort", []),
            "force": force,
            "force_norm": math.sqrt(sum(value * value for value in force)),
            "contact": self.last_contact,
        })

    def start_server(self):
        node = self
        web_dir = self.web_dir

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(web_dir), **kwargs)

            def log_message(self, *_args):
                return

            def send_json(self, payload, status=200):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def read_json(self):
                length = int(self.headers.get("Content-Length", "0"))
                return json.loads(self.rfile.read(length) or b"{}")

            def do_GET(self):
                if urlparse(self.path).path == "/api/data":
                    with node.lock:
                        self.send_json({
                            "ok": True,
                            "joint": node.last_joint,
                            "wrench": node.last_wrench,
                            "contact": node.last_contact,
                            "samples": list(node.samples),
                            "sample_capacity": node.samples.maxlen,
                        })
                    return
                if urlparse(self.path).path == "/api/processes":
                    self.send_json({"ok": True, "processes": node.process_status()})
                    return
                if urlparse(self.path).path == "/api/settings":
                    self.send_json({"ok": True, "settings": node.list_settings()})
                    return
                super().do_GET()

            def do_POST(self):
                path = urlparse(self.path).path
                if path == "/api/process/start":
                    try:
                        result = node.start_process(str(self.read_json().get("id", "")))
                        self.send_json(result)
                    except Exception as exc:  # noqa: BLE001
                        self.send_json({"ok": False, "error": str(exc)}, 400)
                    return
                if path == "/api/process/stop":
                    try:
                        result = node.stop_process(str(self.read_json().get("id", "")))
                        self.send_json(result)
                    except Exception as exc:  # noqa: BLE001
                        self.send_json({"ok": False, "error": str(exc)}, 400)
                    return
                if path == "/api/settings/save":
                    try:
                        payload = self.read_json()
                        self.send_json(node.save_settings(payload.get("name", ""), payload.get("values", {})))
                    except Exception as exc:  # noqa: BLE001
                        self.send_json({"ok": False, "error": str(exc)}, 400)
                    return
                if path == "/api/settings/load":
                    try:
                        payload = self.read_json()
                        self.send_json(node.load_settings(payload.get("name", "")))
                    except Exception as exc:  # noqa: BLE001
                        self.send_json({"ok": False, "error": str(exc)}, 400)
                    return
                if path != "/api/parameters":
                    self.send_json({"ok": False, "error": "Not found"}, 404)
                    return
                try:
                    payload = self.read_json()
                    result = node.set_parameters(payload.get("parameters", {}))
                    self.send_json(result, 200 if result["ok"] else 400)
                except Exception as exc:  # noqa: BLE001
                    self.send_json({"ok": False, "error": str(exc)}, 400)

        self.server = ReusableServer(("0.0.0.0", self.port), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.get_logger().info(f"MuJoCo simulator GUI: http://127.0.0.1:{self.port}")

    def set_parameters(self, values):
        if not isinstance(values, dict) or not values:
            raise ValueError("parameters must be a non-empty object")
        request = SetParameters.Request()
        for name, value in values.items():
            parameter = Parameter()
            parameter.name = str(name)
            parameter.value = self._parameter_value(name, value)
            request.parameters.append(parameter)
        if not self.parameter_client.wait_for_service(timeout_sec=0.5):
            return {"ok": False, "error": f"Simulator parameter service unavailable: {self.simulator_node}"}
        future = self.parameter_client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(timeout=2.0):
            return {"ok": False, "error": "Parameter update timed out"}
        results = future.result().results
        errors = [result.reason for result in results if not result.successful]
        if errors:
            return {"ok": False, "error": "; ".join(errors)}
        return {"ok": True, "parameters": list(values)}

    def _settings_path(self, name):
        safe_name = str(name).strip()
        if not safe_name or safe_name in {".", ".."} or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- " for char in safe_name):
            raise ValueError("Settings name may contain only letters, numbers, spaces, _ and -")
        return self.settings_dir / f"{safe_name}.json"

    def list_settings(self):
        return sorted(path.stem for path in self.settings_dir.glob("*.json"))

    def save_settings(self, name, values):
        if not isinstance(values, dict) or not values:
            raise ValueError("No settings were supplied")
        path = self._settings_path(name)
        path.write_text(json.dumps(values, indent=2), encoding="utf-8")
        return {"ok": True, "name": path.stem, "settings": self.list_settings()}

    def load_settings(self, name):
        path = self._settings_path(name)
        if not path.exists():
            raise ValueError(f"Settings profile not found: {name}")
        return {"ok": True, "name": path.stem, "values": json.loads(path.read_text(encoding="utf-8"))}

    def process_status(self):
        with self.process_lock:
            result = []
            for process_id, spec in self.processes.items():
                process = spec["process"]
                if process is not None and process.poll() is not None:
                    spec["returncode"] = process.returncode
                    spec["process"] = None
                result.append({
                    "id": process_id,
                    "name": spec["name"],
                    "command": " ".join(spec["command"]),
                    "running": spec["process"] is not None,
                    "pid": spec["process"].pid if spec["process"] is not None else None,
                    "returncode": spec["returncode"],
                    "log": list(spec["log"])[-8:],
                })
            return result

    def start_process(self, process_id):
        with self.process_lock:
            spec = self.processes.get(process_id)
            if spec is None:
                raise ValueError(f"Unknown process: {process_id}")
            if spec["process"] is not None and spec["process"].poll() is None:
                return {"ok": True, "processes": self.process_status()}
            spec["log"].clear()
            spec["returncode"] = None
            script = (
                f"source /opt/ros/{shlex.quote(os.environ.get('ROS_DISTRO', 'humble'))}/setup.bash && "
                f"source {shlex.quote(str(self.workspace_setup))} && "
                f"exec {shlex.join(spec['command'])}"
            )
            env = os.environ.copy()
            for variable in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH"):
                env.pop(variable, None)
            process = subprocess.Popen(
                ["bash", "--noprofile", "--norc", "-c", script],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, start_new_session=True, env=env,
            )
            spec["process"] = process
            threading.Thread(target=self._capture_process, args=(process_id, process), daemon=True).start()
            return {"ok": True, "processes": self.process_status()}

    def stop_process(self, process_id):
        with self.process_lock:
            spec = self.processes.get(process_id)
            if spec is None:
                raise ValueError(f"Unknown process: {process_id}")
            process = spec["process"]
            if process is None or process.poll() is not None:
                return {"ok": True, "processes": self.process_status()}
            os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3.0)
        with self.process_lock:
            spec["returncode"] = process.returncode
            spec["process"] = None
        return {"ok": True, "processes": self.process_status()}

    def _capture_process(self, process_id, process):
        if process.stdout is not None:
            for line in process.stdout:
                text = line.rstrip()
                if text:
                    with self.process_lock:
                        self.processes[process_id]["log"].append(text)
        process.wait()
        with self.process_lock:
            spec = self.processes[process_id]
            if spec["process"] is process:
                spec["returncode"] = process.returncode
                spec["process"] = None

    @staticmethod
    def _parameter_value(name, value):
        result = ParameterValue()
        double_parameters = {
            "simulation_rate_hz", "publish_rate_hz", "contact_force_deadband"
        }
        if name in double_parameters:
            result.type = ParameterType.PARAMETER_DOUBLE
            result.double_value = float(value)
        elif isinstance(value, bool):
            result.type = ParameterType.PARAMETER_BOOL
            result.bool_value = value
        elif isinstance(value, int):
            result.type = ParameterType.PARAMETER_INTEGER
            result.integer_value = value
        elif isinstance(value, float):
            result.type = ParameterType.PARAMETER_DOUBLE
            result.double_value = value
        elif isinstance(value, str):
            result.type = ParameterType.PARAMETER_STRING
            result.string_value = value
        elif isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
            result.type = ParameterType.PARAMETER_DOUBLE_ARRAY
            result.double_array_value = [float(item) for item in value]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            result.type = ParameterType.PARAMETER_STRING_ARRAY
            result.string_array_value = value
        else:
            raise ValueError(f"Unsupported parameter value for {value!r}")
        return result

    def destroy_node(self):
        for process_id in list(self.processes):
            try:
                self.stop_process(process_id)
            except Exception:
                pass
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoGui()
    node.start_server()
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
