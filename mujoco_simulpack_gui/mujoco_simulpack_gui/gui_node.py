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
from urllib.parse import urlparse, parse_qs

import yaml
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
        self.monitor_sample_period = 0.05
        self.last_sample_time = 0.0
        self.last_joint = {"name": [], "position": [], "velocity": [], "effort": []}
        self.last_wrench = {"force": [0.0, 0.0, 0.0], "torque": [0.0, 0.0, 0.0]}
        self.sensor_wrench_seen = False
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
                "config_path": None,
            },
            "joint_demo": {
                "name": "Joint Position Demo",
                "command": ["ros2", "run", "mujoco_simulpack", "realtime_joint_position_demo"],
                "process": None, "returncode": None, "log": deque(maxlen=80),
                "config_path": None,
            },
        }
        self.create_subscription(JointState, "/armsimul/joint_states", self.on_joint, 10)
        self.create_subscription(WrenchStamped, "/armsimul/ee_wrench", self.on_wrench, 10)
        self.create_subscription(
            WrenchStamped, "/armsimul/ft_sensor_wrench", self.on_sensor_wrench, 10
        )
        self.create_subscription(Bool, "/armsimul/contact_state", self.on_contact, 10)
        self.web_dir = Path(get_package_share_directory("mujoco_simulpack_gui")) / "web"
        default_config = (Path(get_package_share_directory("mujoco_simulpack")) / "config" / "ur10_contact_sim.yaml").resolve()
        self.settings_dir = Path(self.declare_parameter("profiles_directory", str(default_config.parent / "profiles")).value).expanduser().resolve()
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        # Copy legacy profiles into the new format without deleting their originals.
        legacy_dir = Path.home() / ".ros" / "mujoco_simulpack_gui" / "settings"
        for legacy in legacy_dir.glob("*.json"):
            try:
                if not self._settings_path(legacy.stem).exists():
                    self.save_settings(legacy.stem, json.loads(legacy.read_text(encoding="utf-8")))
            except Exception as exc:
                self.get_logger().warn(f"Could not migrate {legacy.name}: {exc}")
        self.pending_config_path = None
        self.pending_profile_name = None
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
            if self.sensor_wrench_seen:
                return
            self.last_wrench = {
                "force": [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
                "torque": [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
            }

    def on_sensor_wrench(self, msg):
        with self.lock:
            self.sensor_wrench_seen = True
            self.last_wrench = {
                "force": [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
                "torque": [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
            }

    def on_contact(self, msg):
        with self.lock:
            self.last_contact = bool(msg.data)

    def _record_sample(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_sample_time < self.monitor_sample_period:
            return
        self.last_sample_time = now
        position = self.last_joint.get("position", [])
        force = self.last_wrench["force"]
        self.samples.append({
            "t": now,
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
                            "samples": list(node.samples)[-6000:],
                            "sample_capacity": node.samples.maxlen,
                        })
                    return
                if urlparse(self.path).path == "/api/processes":
                    self.send_json({"ok": True, "processes": node.process_status()})
                    return
                if urlparse(self.path).path == "/api/settings/folders":
                    try:
                        directory = parse_qs(urlparse(self.path).query).get("directory", [None])[0]
                        folder = node.profile_directory(directory)
                        children = sorted((p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name.lower())
                        self.send_json({"ok": True, "directory": str(folder), "parent": str(folder.parent), "default_directory": str(node.settings_dir), "folders": [{"name": p.name, "path": str(p)} for p in children]})
                    except Exception as exc:
                        self.send_json({"ok": False, "error": str(exc)}, 400)
                    return
                if urlparse(self.path).path == "/api/settings":
                    try:
                        directory = parse_qs(urlparse(self.path).query).get("directory", [None])[0]
                        folder = node.profile_directory(directory)
                        self.send_json({"ok": True, "settings": node.list_settings(folder), "directory": str(folder)})
                    except Exception as exc:
                        self.send_json({"ok": False, "error": str(exc)}, 400)
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
                if path == "/api/settings/import":
                    try:
                        document = yaml.safe_load(self.read_json().get("text", ""))
                        if not isinstance(document, dict) or not document:
                            raise ValueError("Expected a YAML or legacy JSON settings object")
                        values = document
                        for key in ("/ur10_contact_sim", "ur10_contact_sim", "/**"):
                            if key in document:
                                values = document[key].get("ros__parameters")
                                break
                        if not isinstance(values, dict) or not values or "control_mode" not in values:
                            raise ValueError("The file does not contain a Simulpack settings profile")
                        self.send_json({"ok": True, "values": values})
                    except Exception as exc:
                        self.send_json({"ok": False, "error": str(exc)}, 400)
                    return
                if path == "/api/settings/save":
                    try:
                        payload = self.read_json()
                        self.send_json(node.save_settings(payload.get("name", ""), payload.get("values", {}), payload.get("directory")))
                    except Exception as exc:  # noqa: BLE001
                        self.send_json({"ok": False, "error": str(exc)}, 400)
                    return
                if path == "/api/settings/load":
                    try:
                        payload = self.read_json()
                        self.send_json(node.load_settings(payload.get("name", ""), payload.get("directory")))
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
        if not completed.wait(timeout=15.0):
            return {"ok": False, "error": "Parameter update timed out after 15 seconds"}
        try:
            results = future.result().results
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Parameter update failed: {exc}"}
        errors = [result.reason for result in results if not result.successful]
        if errors:
            return {"ok": False, "error": "; ".join(errors)}
        return {"ok": True, "parameters": list(values)}

    def profile_directory(self, directory=None):
        return Path(directory).expanduser().resolve() if directory else self.settings_dir

    def _settings_path(self, name, directory=None):
        safe_name = str(name).strip()
        if safe_name.endswith((".yaml", ".yml")):
            safe_name = str(Path(safe_name).with_suffix(""))
        if not safe_name or safe_name in {".", ".."} or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- " for char in safe_name):
            raise ValueError("Use letters, numbers, spaces, _ or - for the YAML filename")
        return self.profile_directory(directory) / f"{safe_name}.yaml"

    def list_settings(self, directory=None):
        return sorted(path.name for path in self.profile_directory(directory).glob("*.yaml"))

    def save_settings(self, name, values, directory=None):
        if not isinstance(values, dict) or not values:
            raise ValueError("No settings were supplied")
        path = self._settings_path(name, directory)
        document = self._profile_document(values)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        self.pending_config_path = path
        self.pending_profile_name = path.stem
        return {"ok": True, "name": path.name, "path": str(path), "directory": str(path.parent), "settings": self.list_settings(path.parent), "staged": True}

    def load_settings(self, name, directory=None):
        path = self._settings_path(name, directory)
        if not path.exists():
            raise ValueError(f"Settings profile not found: {path}")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Expected a ROS parameter YAML profile")
        node_values = document.get("/ur10_contact_sim", document.get("ur10_contact_sim", document.get("/**", {})))
        values = node_values.get("ros__parameters") if isinstance(node_values, dict) else None
        if not isinstance(values, dict) or not values:
            raise ValueError("YAML must contain ur10_contact_sim / ros__parameters")
        self._profile_document(values)  # Validate without rewriting the selected file.
        self.pending_config_path = path
        self.pending_profile_name = path.stem
        return {"ok": True, "name": path.name, "path": str(path), "values": values, "staged": True}

    def _profile_document(self, values):
        """Return the ROS parameter document used for both saving and launching."""
        double_parameters = {
            "simulation_rate_hz", "publish_rate_hz", "contact_force_deadband",
        }
        ros_values = {
            key: float(value) if key in double_parameters else (
                [float(item) for item in value]
                if isinstance(value, list)
                and value
                and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
                else value
            )
            for key, value in values.items()
        }
        default_path = Path(get_package_share_directory("mujoco_simulpack")) / "config" / "ur10_contact_sim.yaml"
        if default_path.exists():
            default_document = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
            default_node = default_document.get("ur10_contact_sim", {})
            default_values = default_node.get("ros__parameters", {})
            known_parameters = set(default_values) | {
                "contact_detection_enabled", "solimp", "solref", "friction", "condim",
                "accel_natural_frequency_hz", "accel_damping_ratio", "accel_limit", "torque_limit",
            }
            ros_values = {key: value for key, value in ros_values.items() if key in known_parameters}
            merged_values = dict(default_values)
            merged_values.update(ros_values)
            ros_values = merged_values
        ros_values = {
            key: float(value) if key in double_parameters else (
                [float(item) for item in value]
                if isinstance(value, list)
                and value
                and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
                else value
            )
            for key, value in ros_values.items()
        }
        # A launch profile is intended to start the interactive simulator.
        # Keep the viewer enabled unless a future GUI control explicitly changes it.
        ros_values.setdefault("use_viewer", True)
        parameter_file = {"/ur10_contact_sim": {"ros__parameters": ros_values}}
        return parameter_file

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
                config_changed = (
                    process_id == "mujoco_simulator"
                    and str(spec.get("config_path")) != str(self.pending_config_path)
                )
                if not config_changed:
                    return {"ok": True, "processes": self.process_status()}
                self.stop_process(process_id)
            spec["log"].clear()
            spec["returncode"] = None
            command = list(spec["command"])
            if process_id == "mujoco_simulator" and self.pending_config_path is not None:
                command.append(f"config_file:={self.pending_config_path}")
            script = (
                f"source /opt/ros/{shlex.quote(os.environ.get('ROS_DISTRO', 'humble'))}/setup.bash && "
                f"source {shlex.quote(str(self.workspace_setup))} && "
                f"exec {shlex.join(command)}"
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
            spec["config_path"] = str(self.pending_config_path) if process_id == "mujoco_simulator" else None
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
