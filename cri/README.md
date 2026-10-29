# DOFBOT-PI servo calibration investigation

This directory is the working area for calibration on the Raspberry Pi 4.
The current development machine is not connected to the arm. No hardware
commands have been run.

## What the supplied library provides

Source: `../0.py_install/Arm_Lib/Arm_Lib.py` (version 0.0.5).
Constructing `Arm_Device` immediately opens SMBus 1, at controller address
`0x15`, so even construction requires the board's I2C environment.

| Operation | Library method | Meaning in supplied source |
| --- | --- | --- |
| Read joint angle | `Arm_serial_servo_read(id)` | IDs 1–6; returns degrees or `None` on read failure/invalid position |
| Set neutral offset | `Arm_serial_servo_write_offset_switch(id)` | IDs 1–6; writes the selected ID to register `0x1c` |
| Reset offsets | `Arm_serial_servo_write_offset_switch(0)` | Source labels this as restoring initialization; treat as a reset operation |
| Read offset result | `Arm_serial_servo_write_offset_state()` | `0`: servo not found; `1`: success; `2`: out of range; `None`: I2C error |
| Switch torque | `Arm_serial_set_torque(onoff)` | `1`: enable; `0`: disable, allowing manual movement; no per-joint argument |

The offset method's comment says to power on, manually position the servo at
neutral, then send the command. The Python code does not define the physical
reference pose, permissible offset range, persistence across power cycles,
or a required delay before querying the result. Those details need confirmation
from the hardware documentation or controlled board testing before writing a
complete calibration procedure.

## Angle conventions

- Joints 1, 2, 3, 4, and 6 map 0–180 degrees to raw positions 900–3100.
- Joints 2, 3, and 4 reverse the angle (`180 - angle`) for both writes and reads.
- Joint 5 maps 0–270 degrees to raw positions 380–3700.
- The midpoint of these numerical ranges is 90 degrees for joints other than
  5, and 135 degrees for joint 5. This calculation does not establish the
  correct mechanical calibration pose.
- The `*_any` methods use a different, generic 0–180 mapping without the
  joint-specific transformations. Use the joint-aware methods when comparing
  DOFBOT joint readings with commanded angles.

## Constraints for calibration tooling

1. Keep hardware access behind an explicit command; importing a tool or asking
   for help should not instantiate `Arm_Device`.
2. Begin with readings and physical reference-pose confirmation. A servo's own
   angle report alone cannot establish mechanical alignment.
3. Support the arm before disabling torque: joints may move under gravity.
4. Keep offset writes, offset reset, torque changes, and movement as explicit,
   separate operations. Do not automatically command an all-joint pose.
5. Read and report the offset status after a calibration request. The write
   method returns no success indication and catches I2C errors internally.
6. Validate joint IDs and angle ranges in tooling. The single-joint write
   method does not enforce those ranges.

Next investigation: identify the intended mechanical neutral pose and the
observed problem (physical misalignment at a commanded angle, incorrect
readback, or offset-write failure), then build the corresponding tool here.
