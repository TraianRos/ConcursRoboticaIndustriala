#!/usr/bin/env python3
"""Interactive DOFBOT neutral-offset calibration. Run on the Raspberry Pi."""

import argparse
import sys
import time


def connect_arm():
    from libraries import Arm_Device

    return Arm_Device()


def show_angles(arm):
    for joint in range(1, 7):
        angle = arm.Arm_serial_servo_read(joint)
        limit = 270 if joint == 5 else 180
        reading = "unavailable" if angle is None else "{} degrees".format(angle)
        print("Joint {}: {} (range 0-{})".format(joint, reading, limit))


def calibrate_all(arm):
    messages = {
        0: "servo not found",
        1: "controller reports success",
        2: "offset out of range",
        None: "could not read status",
    }
    # ID 0 resets offsets; calibrate each actual joint instead.
    for joint in range(1, 7):
        arm.Arm_serial_servo_write_offset_switch(joint)
        time.sleep(0.5)
        status = arm.Arm_serial_servo_write_offset_state()
        result = messages.get(status, "unexpected status: {!r}".format(status))
        print("Joint {}: {}".format(joint, result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print("DOFBOT neutral-offset calibration")
    print("Stop other arm-control programs before continuing.")
    print("Support the arm before 'release'; 'hold' enables torque on all joints.")
    print("Position ALL joints at their mechanical neutral references before 'calibrate all'.")
    arm = None
    torque_request = "unchanged"
    try:
        arm = connect_arm()
        time.sleep(0.1)
        print("Commands: read, release, hold, calibrate all, quit")
        while True:
            command = input("calibration> ").strip().lower()
            if command in ("quit", "q", "exit"):
                break
            if command == "read":
                show_angles(arm)
            elif command == "release":
                arm.Arm_serial_set_torque(0)
                torque_request = "disabled"
                print("Torque-off requested.")
            elif command == "hold":
                arm.Arm_serial_set_torque(1)
                torque_request = "enabled"
                print("Torque-on requested.")
            elif command == "calibrate all":
                calibrate_all(arm)
            elif command:
                print("Commands: read, release, hold, calibrate all, quit")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
    except (ImportError, OSError) as exc:
        print("Hardware access failed: {}".format(exc), file=sys.stderr)
        print("Run on the Raspberry Pi with smbus and I2C available.", file=sys.stderr)
        return 1
    finally:
        if arm is not None:
            # Leave torque unchanged on exit; releasing can drop the arm.
            print("Last torque request: {}. Exit sends no torque command.".format(torque_request))
            print("Vendor methods may print I2C errors without raising an exception.")
            bus = getattr(arm, "bus", None)
            if bus is not None:
                bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
