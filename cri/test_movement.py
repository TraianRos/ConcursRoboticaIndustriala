"""Off-board tests with a mocked Arm_Device; no hardware is opened."""
import unittest
from unittest.mock import Mock, call, patch
from cri.movement import ArmController


class MovementTests(unittest.TestCase):
    def setUp(self):
        self.device = Mock()
        self.controller = ArmController(device=self.device)
        self.sleep = patch('cri.movement.time.sleep').start()
        self.addCleanup(patch.stopall)

    def test_construct_sends_no_commands(self):
        self.assertEqual(self.device.mock_calls, [])

    def test_single_and_simultaneous_moves(self):
        self.controller.move_servo(5, 270, 700)
        self.device.Arm_serial_servo_write.assert_called_once_with(5, 270, 700)
        self.controller.move_all([90, 90, 90, 90, 270, 60], 1000)
        self.device.Arm_serial_servo_write6.assert_called_once_with(90,90,90,90,270,60,1000)
        self.assertEqual(self.sleep.call_args_list, [call(0.7), call(1.0)])

    def test_block_example_sequence_and_wrist_wait(self):
        self.controller.arm_move([90,80,50,50,270], 500)
        self.assertEqual(self.device.Arm_serial_servo_write.call_args_list,
                         [call(1,90,500),call(2,80,500),call(3,50,500),call(4,50,500),call(5,270,600)])
        self.assertEqual(self.sleep.call_args_list[-1], call(0.6))

    def test_invalid_pose_never_sends_partial_commands(self):
        for pose, duration in [([90]*4,500),([90,-1,90,90,90],500),
                               ([90,90,90,181,90],500),([90]*5,65535),([90]*5,0),([90]*5,2.5)]:
            with self.subTest(pose=pose,duration=duration), self.assertRaises(ValueError):
                self.controller.arm_move(pose,duration)
        self.assertEqual(self.device.mock_calls, [])

    def test_joint_and_angle_validation(self):
        for joint, angle in [(0,90),(7,90),(True,90),(1,181),(5,271),
                             (1,float('nan')),(1,float('inf')),(1,True)]:
            with self.subTest(joint=joint,angle=angle), self.assertRaises(ValueError):
                self.controller.move_servo(joint,angle)
        self.assertEqual(self.device.mock_calls, [])

    def test_gripper_and_lift(self):
        self.controller.arm_clamp_block(0)
        self.controller.arm_clamp_block(1)
        self.controller.arm_move_up()
        self.assertEqual(self.device.Arm_serial_servo_write.call_args_list,
                         [call(6,60,400),call(6,135,400),call(2,90,1500),call(3,90,1500),call(4,90,1500)])
        with self.assertRaises(ValueError):
            self.controller.arm_clamp_block('close')

    def test_reads_and_torque_are_explicit(self):
        self.device.Arm_serial_servo_read.side_effect = [90,None,90,90,270,60]
        self.assertEqual(self.controller.read_angles(), (90,None,90,90,270,60))
        self.assertEqual(self.device.Arm_serial_servo_read.call_args_list,
                         [call(joint) for joint in range(1,7)])
        self.controller.release()
        self.controller.hold()
        self.assertEqual(self.device.Arm_serial_set_torque.call_args_list,[call(0),call(1)])

    def test_no_wait_and_close_leave_torque_unchanged(self):
        self.controller.move_servo(1,90,wait=False)
        self.sleep.assert_not_called()
        self.controller.close()
        self.controller.close()
        self.device.bus.close.assert_not_called()
        self.device.Arm_serial_set_torque.assert_not_called()
        with self.assertRaises(RuntimeError):
            self.controller.move_servo(1,90)

    def test_owned_bus_cleanup(self):
        import sys
        fake = Mock(Arm_Device=Mock(return_value=self.device))
        with patch.dict(sys.modules, {'cri.libraries': fake}):
            with ArmController() as controller:
                self.assertIs(controller.device, self.device)
        self.device.bus.close.assert_called_once()
        self.device.Arm_serial_set_torque.assert_not_called()


if __name__ == '__main__':
    unittest.main()
