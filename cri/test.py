#!/usr/bin/env python3
"""Interactive DOFBOT neutral-offset calibration. Run on the Raspberry Pi."""

import argparse
import sys
import time
from Arm_Lib import Arm_Device


def connect_arm():
    return Arm_Device()


def show_angles(arm):
    for joint in range(1, 7):
        angle = arm.Arm_serial_servo_read(joint)
        limit = 270 if joint == 5 else 180
        reading = "unavailable" if angle is None else "{} degrees".format(angle)
        print("Joint {}: {} (range 0-{})".format(joint, reading, limit))


def confirm(message, expected):
    return input("{} Type '{}' to continue: ".format(message, expected)).strip() == expected


def calibrate_joint(arm):
    value = input("Joint to calibrate (1-6, or Enter to cancel): ").strip()
    if not value:
        return
    if value not in ("1", "2", "3", "4", "5", "6"):
        print("Choose a joint from 1 to 6.")
        return
    joint = int(value)
    print("Support the arm. Use 'release' first if manual positioning is needed.")
    print("Place joint {} at its documented mechanical neutral reference.".format(joint))
    print("A reported angle alone does not establish physical alignment.")
    print("This saves a controller neutral offset; it does not command a pose.")
    if not confirm("Confirm the joint is positioned correctly.", "CALIBRATE {}".format(joint)):
        print("Cancelled.")
        return
    arm.Arm_serial_servo_write_offset_switch(joint)
    # Conservative settling interval; the vendor source specifies no minimum.
    time.sleep(0.5)
    status = arm.Arm_serial_servo_write_offset_state()
    messages = {
        0: "Servo not found. Check its connection and ID.",
        1: "Controller reports calibration success. Verify physical alignment.",
        2: "Offset out of range. Check mechanical assembly and reference position.",
        None: "Could not read calibration status (I2C error).",
    }
    print(messages.get(status, "Unexpected calibration status: {!r}".format(status)))
    print("This status report is not independent verification of the write.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print("DOFBOT neutral-offset calibration")
    print("Stop other arm-control programs before continuing.")
    print("No movement or torque change is sent at startup.")
    arm = None
    torque_request = "unchanged"
    try:
        arm = connect_arm()
        time.sleep(0.1)
        print("Commands: read, release, hold, calibrate, quit")
        while True:
            command = input("calibration> ").strip().lower()
            if command in ("quit", "q", "exit"):
                break
            if command == "read":
                show_angles(arm)
            elif command == "release":
                if confirm("Disables torque on ALL joints. Support the arm first.", "RELEASE"):
                    arm.Arm_serial_set_torque(0)
                    torque_request = "disabled"
                    print("Torque-off requested. Check release before gently positioning by hand.")
            elif command == "hold":
                if confirm("Enables torque on ALL joints; the arm may move. Keep clear.", "HOLD"):
                    arm.Arm_serial_set_torque(1)
                    torque_request = "enabled"
                    print("Torque-on requested.")
            elif command == "calibrate":
                calibrate_joint(arm)
            elif command:
                print("Commands: read, release, hold, calibrate, quit")
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
