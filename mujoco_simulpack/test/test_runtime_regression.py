"""Run after sourcing ROS: ROS_DOMAIN_ID=216 python3 -m unittest discover -s test."""
import os
from pathlib import Path
import sys
import unittest
import numpy as np
import rclpy
from rclpy.parameter import Parameter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mujoco_simulpack.ur10_contact_sim_node import Ur10ContactSim


class Capture:
    def publish(self, msg): self.last = msg


class RuntimeRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.environ.get('ROS_DOMAIN_ID') == '216'
        args = ['--ros-args', '-p', 'use_viewer:=false']
        if os.environ.get('SIM_TEST_MODE') == 'inverse_dynamics_accel':
            args += ['-p', 'control_mode:=inverse_dynamics_accel',
                     '-p', 'model_file:=models/ur10e/ur10e_torque_contact_scene.xml']
        rclpy.init(args=args)

    @classmethod
    def tearDownClass(cls): rclpy.shutdown()

    def setUp(self): self.node = Ur10ContactSim()
    def tearDown(self): self.node.destroy_node()

    def test_invalid_batch_is_atomic(self):
        n = self.node
        old = n.force_deadband
        geom = n.model.geom_friction.copy()
        result = n.set_parameters_atomically([
            Parameter('contact_force_deadband', value=0.5),
            Parameter('friction', value=[0.4, 0.01, 0.001]),
            Parameter('force_axis_sign', value=[0., 1., 1.])])
        self.assertFalse(result.successful)
        self.assertEqual(n.force_deadband, old)
        self.assertEqual(n.get_parameter('contact_force_deadband').value, old)
        np.testing.assert_equal(n.model.geom_friction, geom)

    def test_valid_parameters_and_timestep(self):
        n = self.node
        result = n.set_parameters_atomically([
            Parameter('simulation_rate_hz', value=500.),
            Parameter('contact_force_deadband', value=0.02),
            Parameter('friction', value=[0.4, 0.01, 0.001])])
        self.assertTrue(result.successful, result.reason)
        self.assertAlmostEqual(n.model.opt.timestep, .002)
        t = n.data.time
        n._step_simulation()
        self.assertAlmostEqual(n.data.time-t, .002)
        self.assertEqual(n.force_deadband, .02)

    def test_bad_command_then_recovery(self):
        n = self.node
        old = n.target_positions.copy()
        for value in (float('nan'), float('inf')):
            n._set_target_positions([value, 0, 0, 0, 0, 0])
            np.testing.assert_equal(n.target_positions, old)
        n._set_target_positions([0.1]*6)
        np.testing.assert_allclose(n.target_positions, [.1]*6)
        for _ in range(20): n._step_simulation()
        self.assertTrue(np.isfinite(n.data.qpos).all())

    def test_force_sign_does_not_corrupt_sensor_wrench(self):
        n = self.node
        n.force_axis_sign = np.array([-1., -1., -1.])
        n._sum_ee_contact_wrench = lambda: (np.array([0., 0., 8.]), np.array([-.08, -.16, 0.]))
        n._sum_ft_sensor_wrench = n._sum_ee_contact_wrench
        n.wrench_pub = Capture(); n.ft_sensor_wrench_pub = Capture()
        n._publish_state()
        sensor = n.ft_sensor_wrench_pub.last.wrench
        self.assertEqual(n.wrench_pub.last.wrench.force.z, -8.)
        self.assertEqual(sensor.force.z, 8.)
        self.assertAlmostEqual(-sensor.torque.y / sensor.force.z, .02)
        self.assertAlmostEqual(sensor.torque.x / sensor.force.z, -.01)

if __name__ == '__main__': unittest.main()
