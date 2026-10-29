"""Off-board planner, gamepad event, and command-gating tests."""
import contextlib
import io
import struct
import sys
import unittest
from unittest.mock import Mock, patch

from cri import gamepad_control as control

LIMITS = [[5,175],[5,175],[5,175],[5,175],[5,265],[0,180]]


class PlannerTests(unittest.TestCase):
    def planner(self, pose=None):
        return control.JogPlanner(pose or [90]*6,LIMITS)

    def test_each_servo_moves_independently(self):
        for index in range(6):
            with self.subTest(servo=index+1):
                planner=self.planner()
                inputs=[0.]*6
                inputs[index]=1.
                target=planner.step(inputs,0.1)
                self.assertGreater(target[index],90)
                self.assertEqual([v for i,v in enumerate(target) if i != index],[90]*5)

    def test_no_wrist_compensation(self):
        target=self.planner().step([0,1,1,0,0,0],0.1)
        self.assertGreater(target[1],90)
        self.assertGreater(target[2],90)
        self.assertEqual(target[3],90)

    def test_deadzone(self):
        self.assertEqual(self.planner().step([0.1,-0.1,0.05,0,0,0],0.1),(90,)*6)

    def test_each_limit_is_independent_and_reversible(self):
        planner=self.planner([175,90,90,90,265,180])
        target=planner.step([1,1,0,0,1,1],0.1)
        self.assertEqual((target[0],target[4],target[5]),(175,265,180))
        self.assertGreater(target[1],90)
        target=planner.step([-1,0,0,0,-1,-1],0.1)
        self.assertLess(target[0],175)
        self.assertLess(target[4],265)
        self.assertLess(target[5],180)

    def test_both_limits_hold_for_long_jogs(self):
        planner=self.planner()
        for direction in (1,-1):
            for _ in range(500):
                target=planner.step([direction]*6,0.1)
                for value,(lower,upper) in zip(target,LIMITS):
                    self.assertTrue(lower <= value <= upper)
            self.assertEqual(target,tuple(pair[1 if direction==1 else 0] for pair in LIMITS))

    def test_elapsed_time_cap(self):
        planner=self.planner()
        planner.step([1,0,0,0,0,0],100)
        self.assertLessEqual(planner.target[0]-90,12*0.15+1e-9)

    def test_existing_lower_margin_violation_moves_inward_only(self):
        planner=self.planner([89,135,20,0,90,1])
        self.assertEqual(planner.step([0]*6,0.1),(89,135,20,0,90,1))
        self.assertEqual(planner.step([0,0,0,-1,0,0],0.1)[3],0)
        target=planner.step([0,0,0,1,0,0],0.1)
        self.assertEqual(target[3],1)
        self.assertLess(planner.target[3],5)
        for _ in range(10):
            target=planner.step([0,0,0,1,0,0],0.1)
        self.assertGreaterEqual(target[3],5)
        self.assertEqual(len(planner.limit_warnings([89,135,20,0,90,1])),1)

    def test_existing_upper_margin_violation_and_other_joints(self):
        planner=self.planner([180,90,90,90,90,90])
        target=planner.step([1,1,0,0,0,0],0.1)
        self.assertEqual(target[0],180)
        self.assertGreater(target[1],90)
        target=planner.step([-1,0,0,0,0,0],0.1)
        self.assertEqual(target[0],179)
        self.assertGreater(planner.target[0],175)

    def test_invalid_pose_and_inputs(self):
        for bad in (None,float('nan'),float('inf'),181):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.planner([bad,90,90,90,90,90])
        with self.assertRaises(ValueError):
            control.JogPlanner([90]*6,[[0,181]]*6)
        for bad in (float('nan'),2):
            with self.assertRaises(ValueError):
                self.planner().step([bad,0,0,0,0,0],0.1)
        with self.assertRaises(ValueError):
            self.planner().step([0]*5,0.1)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.config=control.load_config('cri/gamepad.json')
        self.bindings=self.config['bindings']
        self.axes=[0.,0.,-1.,0.,0.,-1.,0.]
        self.buttons=[False]*12

    def read(self):
        return control.servo_inputs(self.bindings,self.axes,self.buttons,0.15)

    def test_trigger_rest_and_startup_neutral(self):
        self.bindings['gripper_open']={'type':'trigger','index':2,'rest':-1,'pressed':1}
        self.bindings['gripper_close']={'type':'trigger','index':5,'rest':-1,'pressed':1}
        values,neutral=self.read()
        self.assertTrue(neutral)
        self.assertEqual(values,[0.]*6)
        self.axes[2]=self.axes[5]=1.
        values,neutral=self.read()
        self.assertFalse(neutral)
        self.assertEqual(values[5],0.)

    def test_left_trigger_opens_right_trigger_closes(self):
        self.buttons[6]=True
        self.assertEqual(self.read()[0],[0.,0.,0.,0.,0.,-1.])
        self.buttons[6]=False
        self.buttons[7]=True
        self.assertEqual(self.read()[0],[0.,0.,0.,0.,0.,1.])

    def test_dpad_only_moves_servo3(self):
        self.axes[6]=-1.
        self.assertEqual(self.read()[0],[0.,0.,1.,0.,0.,0.])

    def test_button_based_dpad_and_triggers(self):
        self.bindings['servo3']={'type':'buttons','negative':10,'positive':11,'sign':1}
        self.bindings['gripper_open']={'type':'button','index':6}
        self.bindings['gripper_close']={'type':'button','index':5}
        self.buttons[11]=self.buttons[6]=True
        self.assertEqual(self.read()[0],[0.,0.,1.,0.,0.,-1.])
        self.buttons[10]=True
        self.assertEqual(self.read()[0][2],0.)

    def test_device_mapping_and_invalid_conflicts(self):
        import json
        import tempfile
        from pathlib import Path
        control.validate_device(self.bindings,self.axes,self.buttons)
        with self.assertRaises(ValueError):
            control.validate_device(self.bindings,self.axes[:6],self.buttons)
        self.bindings['gripper_close']={'type':'button','index':self.bindings['quit_button']}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            path.write_text(json.dumps(self.config))
            with self.assertRaisesRegex(ValueError,'shared'):
                control.load_config(path)

    def test_combined_trigger_axis(self):
        import json
        import tempfile
        from pathlib import Path
        self.bindings['gripper_open']={'type':'trigger','index':2,'rest':0,'pressed':-1}
        self.bindings['gripper_close']={'type':'trigger','index':2,'rest':0,'pressed':1}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'combined.json'
            path.write_text(json.dumps(self.config))
            control.load_config(path)
        self.axes[2]=0.
        self.assertTrue(self.read()[1])
        self.axes[2]=-1.
        self.assertEqual(self.read()[0][5],-1.)


class NeutralGateTests(unittest.TestCase):
    def test_bad_trigger_rest_does_not_block_sticks(self):
        config=control.load_config('cri/gamepad.json')
        binds=config['bindings']
        binds['gripper_open']={'type':'trigger','index':2,'rest':-1,'pressed':1}
        binds['gripper_close']={'type':'trigger','index':5,'rest':-1,'pressed':1}
        axes=[0.]*7
        inputs,neutral=control.servo_channels(binds,axes,[False]*12,0.15)
        gate=control.NeutralGate()
        gate.update(neutral)
        self.assertEqual(gate.active,[True,True,True,True,True,False])
        axes[0]=1.
        inputs,_=control.servo_channels(binds,axes,[False]*12,0.15)
        self.assertEqual(gate.filter(inputs)[0],-1.)
        self.assertIn('axis 2=0.00',control.waiting_message(binds,axes,[False]*12,gate.active))

    def test_each_control_must_first_return_to_neutral(self):
        gate=control.NeutralGate()
        gate.update([False,True,True,True,True,True])
        self.assertEqual(gate.filter([1,0,0,0,0,0]),[0.]*6)
        self.assertEqual(gate.update([True]*6),[0])
        self.assertEqual(gate.filter([1,0,0,0,0,0]),[1,0,0,0,0,0])

    def test_reported_idle_state_is_ready_and_button7_closes_not_quits(self):
        config=control.load_config('cri/gamepad.json')
        buttons=[False]*12
        inputs,neutral=control.servo_channels(config['bindings'],[0.]*7,buttons,0.15)
        self.assertTrue(all(neutral))
        self.assertEqual(inputs,[0.]*6)
        buttons[7]=True
        inputs,_=control.servo_channels(config['bindings'],[0.]*7,buttons,0.15)
        self.assertEqual(inputs[-1],1.)
        self.assertFalse(buttons[config['bindings']['quit_button']])


class TrackingTests(unittest.TestCase):
    def test_transient_error_and_unchanged_joints_do_not_stop(self):
        monitor = control.TrackingMonitor()
        # Joint 6 was not commanded, so its error must not stop control.
        measured = [90,90,90,90,90,10]
        self.assertEqual(monitor.sample(measured,[90]*6,{1}),[])
        measured[0] = 60
        self.assertEqual(len(monitor.sample(measured,[90]*6,{1})),1)
        measured[0] = 89
        monitor.sample(measured,[90]*6,{1})
        self.assertEqual(monitor.failures[0],0)

    def test_sustained_error_reports_the_joint_and_angles(self):
        monitor = control.TrackingMonitor(tolerance=25,checks=5)
        for _ in range(4):
            monitor.sample([50,90,90,90,90,90],[90]*6,{1})
        with self.assertRaisesRegex(RuntimeError,'joint 1: read 50 degrees, target 90 degrees'):
            monitor.sample([50,90,90,90,90,90],[90]*6,{1})



class InputTests(unittest.TestCase):
    def test_initial_events_normalization_and_disconnect(self):
        def ioctl(fd, op, buffer, mutate):
            buffer[0]=6 if op == 0x80016a11 else 12
        with patch.object(control.os,'open',return_value=123), patch.object(control.fcntl,'ioctl',side_effect=ioctl), patch.object(control.os,'close') as close:
            pad=control.LinuxGamepad('/dev/input/js0')
            events=[struct.pack('=IhBB',0,-32768,0x82,0),
                    struct.pack('=IhBB',0,1,0x81,4), BlockingIOError()]
            with patch.object(control.os,'read',side_effect=events):
                self.assertEqual(pad.poll(),2)
            self.assertEqual(pad.axes[0],-1)
            self.assertTrue(pad.buttons[4])
            with patch.object(control.os,'read',return_value=b''), self.assertRaises(ConnectionError):
                pad.poll()
            pad.close()
            close.assert_called_once_with(123)

    def test_center_gate_then_move_without_enable_and_rotate_pincher(self):
        config=control.load_config('cri/gamepad.json')
        class Pad:
            def __init__(self):
                self.axes=[1.,0.,-1.,0.,0.,-1.,0.]
                self.buttons=[False]*12
                self.index=0
                self.closed=False
            def poll(self):
                self.index+=1
                if self.index==2:
                    self.axes[0]=0.
                if self.index==3:
                    self.axes[0]=1.
                    self.axes[3]=1.
                if self.index==4:
                    self.axes[0]=self.axes[3]=0.
                if self.index==5:
                    self.buttons[9]=True
                return 1
            def close(self):
                self.closed=True
        pad=Pad()
        arm=Mock()
        arm.read_angles.return_value=(90,)*6
        module=Mock(ArmController=Mock(return_value=arm))
        now=[0.]
        def sleep(seconds):
            now[0]+=seconds
        with patch.dict(sys.modules,{'cri.movement':module}), patch.object(control,'LinuxGamepad',return_value=pad), patch.object(control,'load_config',return_value=config), patch.object(control.time,'monotonic',side_effect=lambda:now[0]), patch.object(control.time,'sleep',side_effect=sleep), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(control.main([]),0)
        self.assertEqual(arm.move_servo.call_args_list,
                         [unittest.mock.call(1,89,s_time=100,wait=False),
                          unittest.mock.call(5,91,s_time=100,wait=False)])
        arm.close.assert_called_once()
        arm.release.assert_not_called()
        arm.hold.assert_not_called()
        self.assertTrue(pad.closed)

    def test_disconnect_stops_without_motion(self):
        config=control.load_config('cri/gamepad.json')
        pad=Mock(axes=[0.,0.,-1.,0.,0.,-1.,0.],buttons=[False]*12)
        pad.poll.side_effect=ConnectionError('Disconnected')
        arm=Mock()
        arm.read_angles.return_value=(90,)*6
        module=Mock(ArmController=Mock(return_value=arm))
        with patch.dict(sys.modules,{'cri.movement':module}), patch.object(control,'LinuxGamepad',return_value=pad), patch.object(control,'load_config',return_value=config), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(control.main([]),1)
        arm.move_servo.assert_not_called()
        arm.close.assert_called_once()
        pad.close.assert_called_once()


if __name__=='__main__':
    unittest.main()
