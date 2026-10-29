# DOFBOT-PI servo calibration investigation

## Generic gamepad control

Bindings are edited directly in `cri/gamepad.json`. Use `--inspect` to view
raw axes/buttons; it does not open the arm. Moving a stick diagonally and seeing
multiple axes change is normal. There is no guided single-axis capture step.
At runtime all mapped servo inputs are processed together and limited independently.
If a configured axis never changes, inspect the raw inputs and update its index.
If two physical controls report the same raw input, they cannot be distinguished
in that controller mode; check the controller's analog mode if available.

Use `--trace` to print command attempts and input diagnostics. Raw input lines
are sampled once per second, so short movements may produce commands between
two neutral snapshots. Command lines are attempts, not hardware acknowledgments.

Each input controls one servo independently, matching the vendor's arm_handle
paradigm. There is no coordinated wrist compensation or directional abstraction.
Connect the gamepad to the **Pi** and run from the project root:

```bash
python3 cri/gamepad_control.py --inspect  # Check raw axis/button indices; no arm access.
python3 cri/gamepad_control.py --dry-run # Simulated angles, no arm access.
python3 cri/gamepad_control.py           # Control the real arm.
```

| Control | Servo | Default raw binding |
| --- | --- | --- |
| Left stick left/right | 1 (base) | Axis 0, sign -1 |
| Left stick up/down | 2 | Axis 1, sign +1 |
| D-pad up/down | 3 | Axis 6, sign -1 |
| Right stick up/down | 4 | Axis 4, sign -1 |
| Right stick left/right | 5 (pincher rotation) | Axis 3, sign +1 |
| Left trigger | 6 (open pincher) | Button 6 |
| Right trigger | 6 (close pincher) | Button 7 |
| Start (controller-dependent) | Exit | Button 9 |

No hold button is required. Each servo enables independently when its own
control first rests. A trigger with a mismatched rest value cannot block the
other five servos. The script prints ready servos and the raw inputs of blocked
channels. Both triggers must be released once to enable servo 6; simultaneous
equal trigger inputs then cancel. Returning inputs to rest stops new commands.
Each servo clamps independently to its limits, so another remains controllable
when one reaches a limit. The trigger buttons and quit binding match the reported
controller; stick/D-pad indices still need verification for each physical input.

**Controller indices vary:** use `--inspect` and edit `cri/gamepad.json` to match
your device. For another device pass `--device /dev/input/js1`. Use `--config
PATH` for another configuration. Change a servo binding's `sign` between -1 and
1 to reverse it. Triggers increase/decrease servo 6's angle; `gripper_sign: -1`
reverses opening/closing direction if needed.

Each `servo1` through `servo5` binding can be an axis:

```json
{"type": "axis", "index": 6, "sign": -1}
```

Or a button-based D-pad (replace indices with your inspected values):

```json
{"type": "buttons", "negative": 10, "positive": 11, "sign": 1}
```

Each `gripper_open`/`gripper_close` binding can be an analog trigger:

```json
{"type": "trigger", "index": 2, "rest": -1, "pressed": 1}
```

For triggers ranging from 0 to 1, set `rest: 0`. If both triggers share one axis,
use that index in both bindings, with `rest: 0` and opposite `pressed` endpoints
(-1 for one, 1 for the other). Digital triggers use:

```json
{"type": "button", "index": 6}
```

To diagnose missing movement, run `python3 cri/gamepad_control.py --trace`.
It prints raw input states and each actual command attempt, helping distinguish
an input-mapping problem from the arm not responding. Vendor writes still lack
success acknowledgments. Default gripper bindings use buttons 6/7, not analog
trigger axes; quit is button 9 so pressing button 7 no longer exits.

Bindings are checked against the device's input counts before arm access.
Conflicting mappings, including a trigger sharing the quit button, are rejected.
Linux joystick input needs no pygame dependency or desktop. A gamepad connected
only to the SSH client is not available to the Pi script.

The six JSON `[min, max]` pairs bound every target. Defaults are 5-175 degrees
for joints 1-4, 5-265 for joint 5, and 0-180 for joint 6. Configure limits to
suit your assembly and gripper; these margins do not guarantee collision-free
motion. Limits cannot exceed vendor ranges (0-180, or 0-270 for joint 5).
Missing, nonfinite, or hardware-out-of-range readings abort. A valid angle
outside a configured soft margin is allowed at startup: for example, servo 4
at 0 degrees with a configured minimum of 5. Only gradual jogging toward the
allowed range is accepted for that servo until it re-enters. Outward/neutral
input holds the current target; nothing snaps automatically to the boundary,
and other servos remain usable. Input speed defaults to 12 degrees/s, a 0.15 dead zone, and 10 Hz
updates. Delayed iterations cannot cause large catch-up jumps.

Feedback is checked every 0.5 seconds. Missing or hardware-invalid readings
stop control; an existing soft-margin violation can recover through inward jogging. Tracking checks apply only to commanded servos, with defaults of
`tracking_error_deg: 25` and `tracking_error_checks: 5` consecutive excessive
readings before stopping. Warnings identify the servo, read angle, and target;
a recovered reading resets that servo's counter.

Ctrl+C, the quit button, disconnect, or input-read errors stop new commands.
A previously requested 100 ms move may finish. Torque is unchanged at startup
and exit; run the arm in its normal torque-enabled state. Avoid competing
arm-control programs. Angle limits are not collision or workspace checks.

```bash
python3 -m unittest cri.test_gamepad_control -v
```

Input reference: [Linux joystick API](https://www.kernel.org/doc/html/latest/input/joydev/joystick-api.html).

## Arm movement library

`movement.py` follows the supplied `3.ctrl_Arm/3.ctrl_servo.ipynb`,
`5.ctrl_all.ipynb`, and `10.move_block.ipynb` examples, using `Arm_Device` from
our `libraries.py` module. From a script in `cri`:

```python
from movement import ArmController

with ArmController() as arm:
    print(arm.read_angles())
    arm.move_servo(1, 90, s_time=1000)
    arm.arm_clamp_block(0)  # Open gripper (60 degrees by default).
    arm.arm_move([90, 80, 50, 50, 270], s_time=1000)
    arm.arm_clamp_block(1)  # Close gripper (135 degrees by default).
```

For scripts at the workspace root, import from `cri.movement` instead. The
example pose above comes from the vendor's block demo; check its suitability
for your assembly and surroundings before running it.

- `move_servo(joint, angle, s_time=500, wait=True)` moves one joint.
- `arm_move(p, s_time=500, wait=True)` moves joints 1-5 with the block demo's
  10 ms command spacing, 100 ms delay before joint 5, and 1.2x joint-5 duration.
  Gripper position is unchanged. All inputs are validated before any writes.
- `move_all(pose, s_time=500, wait=True)` sends six joint angles together,
  including the gripper, through `Arm_serial_servo_write6`.
- `arm_clamp_block(enable, s_time=400, wait=True)` accepts 0/False to open and
  1/True to close. Customize with `ArmController(open_angle=60, closed_angle=135)`.
- `arm_move_up(s_time=1500, wait=True)` sets joints 2, 3, 4 to 90 degrees,
  matching the example. This is not a guaranteed vertical Cartesian trajectory.
- `read_angles()` returns six readings, with `None` for unavailable readings.
- `release()` disables all torque; support the arm first. `hold()` enables all
  torque. These are explicit operations, never sent automatically.
- `close()`/context-manager exit closes only a bus owned by this controller,
  leaving torque unchanged; it does not cancel motion already in progress.
  Pass `device=existing_arm_device` to reuse a caller-owned connection.

All angles are degrees; joint 5 accepts 0-270 and others 0-180. Millisecond
commands accept integers 1-65535; `arm_move` also requires its 1.2x wrist duration
within that limit. Fractional angles are rounded to integer degrees. The vendor
library handles reversed joint coordinates. Blocking movement calls wait the
requested duration (including the slower wrist in `arm_move`); `wait=False`
skips the completion wait but preserves command spacing. These waits do not
verify arrival: the supplied library can print I2C errors without raising.
The helper provides joint-space control, not collision checks, inverse
kinematics, or a mapping from detected cube/grid positions into servo angles.

Off-board verification:

```bash
python3 -m unittest cri.test_movement -v
```

## Colored cubes and 3x3-cell grid detection

### Testing on the Raspberry Pi

For a live feed over SSH, connect from your computer with port forwarding
(replace `Dofbot` with the Pi's IP address if needed):

```bash
ssh -L 8080:127.0.0.1:8080 dofbot@Dofbot
```

In that SSH session, run on the Pi:

```bash
cd ~/Dofbot
python3 cri/preview_vision.py --serve
```

Open `http://localhost:8080` in your computer's browser. The page streams the
annotated camera feed and current positions. Use **Lock grid** with the empty
board visible, then place cubes without moving the camera or board. Use
**Unlock grid** to detect the grid again. Ctrl+C in the Pi terminal stops the
server and releases the camera. This mode needs neither a graphical Pi desktop
nor image files. The server listens on Pi loopback by default.
For checkerboards add `--pattern checkerboard`. To use another port, change
both SSH forwarding ports and pass the corresponding `--port` on the Pi.

Open a terminal on the Pi desktop and run from the project root:

```bash
python3 cri/preview_vision.py --camera 0 --pattern lines
```

Keep the entire empty 3x3 grid visible, with even lighting. Press `g` in the
preview window to lock its position, then place colored cubes on it. Keep the
camera and grid fixed after locking. Press `p` to print cube positions, `r` to
unlock the grid, and `q` to exit. Use `--pattern checkerboard` for alternating
black/white square cells. Add `--cell-size-mm 20` only if the cells measure 20 mm.
The preview requires OpenCV with GUI support and a working graphical display.

For an SSH terminal without a graphical display, save one annotated image:

```bash
python3 cri/preview_vision.py --camera 0 --snapshot detection.jpg
```

This also prints positions in the terminal. Inspect `detection.jpg` afterward.
Snapshot mode detects the grid in that frame; it does not reuse a locked grid.
If the camera does not open, close other camera applications or try `--camera 1`.
The preview requests 640x480 MJPEG at 30 fps, matching the supplied color
recognition notebook. For a read timeout, check `fuser -v /dev/video0` for
competing captures, including notebook kernels. If the device is free, inspect
`v4l2-ctl -d /dev/video0 --list-formats-ext` and try `--format YUYV` or
`--format default`; `--fps` changes the requested frame rate.
No arm movement or calibration commands are sent by this preview.

`vision.py` provides an independent OpenCV library. It does not open a camera
or communicate with the arm. It needs `cv2` and `numpy` in the selected Python
environment; the supplied OS may already provide these.

From a script in `cri`:

```python
from libraries import cv2
from vision import CubeGridDetector

detector = CubeGridDetector(grid_pattern="auto", cell_size_mm=20)
frame = cv2.imread("scene.jpg")  # Or a BGR frame from your camera.
if frame is None:
    raise RuntimeError("Could not read scene.jpg")

scene = detector.detect(frame)
if scene.grid is not None:
    print("Outer corners in pixels:", scene.grid.corners_px)
    print("9 cell centers:", scene.grid.cell_centers_px)

for cube in scene.cubes:
    print(cube.color, cube.center_px, cube.cell, cube.board_xy_mm)

cv2.imwrite("detections.jpg", detector.draw(frame, scene))
```

For a script at the workspace root use `from cri.vision import CubeGridDetector`.
Replace the example's `20` with your measured cell size in millimeters, or omit
`cell_size_mm` to return pixels and grid coordinates only.

- `grid_pattern="lines"`: dark grid lines on a light background; expects four
  boundary lines along each axis, forming three cells. `"checkerboard"` detects
  a 3x3-square checkerboard with 2x2 inner corners. `"auto"` tries both.
- `scene.grid` is `None` when detection fails. Cube pixel positions are still
  returned; their grid positions are `None`.
- `grid.squares` returns nine individual `GridCell` objects in row-major order.
  Each has `row`, `column`, `center_px`, `corners_px` (TL, TR, BR, BL), and
  optional center `board_xy_mm`. Their polygons follow the regular grid's
  perspective transform, rather than independently tracing imperfect pen lines.
  The preview outlines and labels each square (`r0 c0` through `r2 c2`).
  Browser/terminal JSON includes these entries in `grid_cells` (empty when no
  grid is found).
- `grid.corners_px` contains outer corners in TL, TR, BR, BL order;
  `grid.cell_centers_px[row, column]` contains a cell's image center.
- `cube.center_px` is `(x, y)` in image pixels. `cube.cell` is `(row, column)`,
  numbered 0–2. `cube.grid_xy` is continuous `(column, row)` in cell units.
  `cube.board_xy_mm` is the corresponding planar `(x, y)` when cell size is set.
  Outside-board cube centers have no cell assignment.
- Automatic board orientation is based on the image upper-left. A symmetric
  grid cannot provide a unique physical origin after rotation. Use known outer
  corners in a consistent order when a stable physical reference is required.

If cubes obscure the pattern, detect the empty board first and reuse the result
while the camera and board stay fixed:

```python
grid = detector.detect_grid(empty_board_frame)
if grid is None:
    raise RuntimeError("Grid not found in empty-board image")
scene = detector.detect(current_frame, grid=grid)

# Alternatively provide measured OUTER corners, in TL, TR, BR, BL order:
grid = detector.grid_from_corners([(60, 60), (420, 60), (420, 420), (60, 420)])
```

Default colors are red, orange, yellow, green, blue, and purple. Configure HSV
ranges through `colors={"name": [((h0, s0, v0), (h1, s1, v1)), ...]}` and
`min_cube_area` to suit the camera and lighting. OpenCV hue values are 0–179;
saturation and brightness are 0–255. Custom color ranges should not overlap.

This is color/shape-based candidate detection, not a trained cube classifier:
similarly shaped colored objects can be reported, adjacent cubes of the same
color can merge, and dim lighting or shadows can split/miss objects. No object
tracking IDs are assigned between frames. Grid detection needs visible regular
lines or checkerboard corners; it does not recognize arbitrary printed designs.
Millimeter coordinates project the visible region center onto the board plane.
Cube height, perspective parallax, and lens distortion introduce error; these
positions are not calibrated robot coordinates or verified grasp points.

Implementation references: OpenCV's
[chessboard detection and homographies](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
and [perspective transforms](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html).

Run the synthetic, off-board tests from the workspace root:

```bash
python3 -m unittest cri.test_vision -v
```

These tests cover regular/perspective line grids, checkerboards, colors, missing
grids, outside-board positions, invalid input, noise filtering, and wrong grid
sizes. Real camera images and Raspberry Pi performance still need validation.

This directory is the working area for calibration on the Raspberry Pi 4.
The current development machine is not connected to the arm. No hardware
commands have been run.

## Running scripts with local libraries

Import from `cri.libraries` in scripts at the workspace root:

```python
from cri.libraries import Arm_Device, face_follow

arm = Arm_Device()
tracker = face_follow.face_follow()
```

For scripts inside `cri`, use `from libraries import Arm_Device` instead.
The explicit imports in `libraries.py` and workspace `.vscode/settings.json`
source paths support VS Code/Pylance completion and navigation before runtime.
`libraries.pyi` explicitly re-exports the vendor types for static analysis;
`pyrightconfig.json` supplies project-level source paths. Annotate helper
parameters as `arm: Arm_Device` so completion also works inside functions.
Each exported library is loaded only when requested; importing all exports with
`*` also loads hardware scripts and their dependencies, so prefer named imports.
Setup scripts and duplicate build outputs are not exported as libraries.

The earlier launcher remains available but is unnecessary with this module.

From the workspace root:

```bash
python3 cri/run.py test.py
```

The launcher discovers all directories containing Python source and adds them
to the child process's `PYTHONPATH`, excluding hidden directories, environments,
and build/dist copies. Existing `PYTHONPATH` entries are preserved. Script
arguments can follow the script name. Use normal imports, for example
`from Arm_Lib import Arm_Device`, rather than numbered directory names.
This makes local modules discoverable; external dependencies such as `smbus`
still need to be installed in the Python environment.

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
