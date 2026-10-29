"""DOFBOT joint movement, following 3.ctrl_Arm's servo and block examples.

Degrees and milliseconds use Arm_Lib conventions. No hardware access on import.
Construction opens I2C but sends no movement or torque commands. Waits estimate
completion from requested time; the vendor API does not acknowledge writes.
"""
from math import isfinite
from numbers import Integral, Real
import time
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    if __package__:
        from .libraries import Arm_Device
    else:
        from libraries import Arm_Device


class ArmController:
    """Joint-space controls; no inverse kinematics or camera-to-arm mapping.

    Supply device= to reuse an Arm_Device; its bus remains caller-owned.
    """
    def __init__(self, device: Optional['Arm_Device'] = None,
                 open_angle: float = 60, closed_angle: float = 135):
        self.open_angle = self._angle(6, open_angle)
        self.closed_angle = self._angle(6, closed_angle)
        self._owns_device = device is None
        self._closed = False
        if device is None:
            if __package__:
                from .libraries import Arm_Device
            else:
                from libraries import Arm_Device
            device = Arm_Device()
            time.sleep(0.1)
        self.device = device

    @staticmethod
    def _joint(joint: int) -> int:
        if isinstance(joint, bool) or not isinstance(joint, Integral) or not 1 <= joint <= 6:
            raise ValueError('joint must be an integer from 1 to 6')
        return int(joint)

    @classmethod
    def _angle(cls, joint: int, angle: float) -> int:
        limit = 270 if cls._joint(joint) == 5 else 180
        if (isinstance(angle, bool) or not isinstance(angle, Real)
                or not isfinite(angle) or not 0 <= angle <= limit):
            raise ValueError('joint {} angle must be between 0 and {}'.format(joint, limit))
        return int(round(angle))

    @staticmethod
    def _duration(s_time: int) -> int:
        if (isinstance(s_time, bool) or not isinstance(s_time, Integral)
                or not 1 <= s_time <= 65535):
            raise ValueError('s_time must be an integer from 1 to 65535 milliseconds')
        return int(s_time)

    @classmethod
    def _pose(cls, pose: Sequence[float], size: int) -> Tuple[int, ...]:
        values = tuple(pose)
        if len(values) != size:
            raise ValueError('pose must contain exactly {} joint angles'.format(size))
        return tuple(cls._angle(joint, value) for joint, value in enumerate(values, 1))

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('ArmController is closed')

    def move_servo(self, joint: int, angle: float, s_time: int = 500, wait: bool = True):
        """Move one joint (6 is gripper); no broadcast through ID 0."""
        self._ensure_open()
        joint = self._joint(joint)
        angle = self._angle(joint, angle)
        duration = self._duration(s_time)
        self.device.Arm_serial_servo_write(joint, angle, duration)
        if wait:
            time.sleep(duration / 1000)

    def move_all(self, pose: Sequence[float], s_time: int = 500, wait: bool = True):
        """Move joints 1-6 simultaneously; pose includes gripper angle."""
        self._ensure_open()
        angles = self._pose(pose, 6)
        duration = self._duration(s_time)
        self.device.Arm_serial_servo_write6(*angles, duration)
        if wait:
            time.sleep(duration / 1000)

    def arm_move(self, p: Sequence[float], s_time: int = 500, wait: bool = True):
        """Block-example move of joints 1-5, leaving the gripper unchanged.

        Writes are spaced by 10 ms. Joint 5 has a 100 ms lead-in and 1.2x
        duration, as in the example. wait=False only skips the final wait.
        """
        self._ensure_open()
        angles = self._pose(p, 5)
        duration = self._duration(s_time)
        wrist_duration = self._duration(int(duration * 1.2))
        for joint, angle in enumerate(angles, 1):
            if joint == 5:
                time.sleep(0.1)
            self.device.Arm_serial_servo_write(joint, angle,
                                              wrist_duration if joint == 5 else duration)
            time.sleep(0.01)
        if wait:
            time.sleep(wrist_duration / 1000)

    def arm_clamp_block(self, enable: bool, s_time: int = 400, wait: bool = True):
        """True/1 closes to 135 degrees; False/0 opens to 60 by default."""
        if not isinstance(enable, (bool, Integral)) or enable not in (0, 1):
            raise ValueError('enable must be True/1 (close) or False/0 (open)')
        self.move_servo(6, self.closed_angle if enable else self.open_angle, s_time, wait)
        if wait:
            time.sleep(0.1)  # Example gives the gripper 100 ms settling time.

    def arm_move_up(self, s_time: int = 1500, wait: bool = True):
        """Example lift: joints 2-4 to 90; not a Cartesian vertical move."""
        self._ensure_open()
        duration = self._duration(s_time)
        for joint in (2, 3, 4):
            self.device.Arm_serial_servo_write(joint, 90, duration)
        if wait:
            time.sleep(duration / 1000)

    def read_angles(self) -> Tuple[Optional[int], ...]:
        """Read joints 1-6; unavailable readings remain None."""
        self._ensure_open()
        return tuple(self.device.Arm_serial_servo_read(joint) for joint in range(1, 7))

    def release(self):
        """Disable torque on ALL joints; support the arm before calling."""
        self._ensure_open()
        self.device.Arm_serial_set_torque(0)

    def hold(self):
        """Enable torque on ALL joints; the arm may move when engaging."""
        self._ensure_open()
        self.device.Arm_serial_set_torque(1)

    def close(self):
        """Close an owned bus, leaving torque and in-progress motion unchanged."""
        if not self._closed:
            self._closed = True
            if self._owns_device:
                self.device.bus.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
