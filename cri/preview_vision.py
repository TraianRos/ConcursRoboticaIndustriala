#!/usr/bin/env python3
"""Preview cube/grid detection on the Pi camera, without controlling the arm."""

import argparse
import json
import sys


def scene_positions(scene):
    return {
        "grid_found": scene.grid is not None,
        "grid_corners_px": scene.grid.corners_px.tolist() if scene.grid is not None else None,
        "grid_cells": [{
            "row": square.row,
            "column": square.column,
            "center_px": square.center_px,
            "corners_px": square.corners_px.tolist(),
            "board_xy_mm": square.board_xy_mm,
        } for square in scene.grid.squares] if scene.grid is not None else [],
        "cubes": [{
            "color": cube.color,
            "center_px": cube.center_px,
            "cell_row_col": cube.cell,
            "grid_xy": cube.grid_xy,
            "board_xy_mm": cube.board_xy_mm,
        } for cube in scene.cubes],
    }


def print_positions(scene):
    print(json.dumps(scene_positions(scene), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--format", choices=("MJPG", "YUYV", "default"), default="MJPG",
                        help="Requested camera pixel format (default: MJPG, as in vendor examples)")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS (default: 30)")
    parser.add_argument("--pattern", choices=("lines", "checkerboard", "auto"), default="lines")
    parser.add_argument("--cell-size-mm", type=float, help="Measured grid cell size, optional")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--snapshot", metavar="FILE", help="Save one annotated frame and exit without a GUI")
    mode.add_argument("--serve", action="store_true", help="Stream to a browser through an SSH tunnel")
    parser.add_argument("--host", default="127.0.0.1", help="Stream bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Stream port (default: 8080)")
    options = parser.parse_args(argv)
    if options.cell_size_mm is not None and options.cell_size_mm <= 0:
        parser.error("--cell-size-mm must be positive")
    if not 1 <= options.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if options.fps <= 0:
        parser.error("--fps must be positive")

    from libraries import cv2
    from vision import CubeGridDetector

    detector = CubeGridDetector(grid_pattern=options.pattern, cell_size_mm=options.cell_size_mm)
    capture = cv2.VideoCapture(options.camera)
    window_opened = False
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera {}. Check the camera connection and other camera apps.".format(options.camera))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, options.fps)
        if options.format != "default":
            if not capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*options.format)):
                print("Camera driver did not accept {} format; try --format default or --format YUYV."
                      .format(options.format), file=sys.stderr)
        print("Camera {}: requested 640x480, {} fps, format {}."
              .format(options.camera, options.fps, options.format), flush=True)
        # Let the camera's automatic exposure settle before using the frame.
        for _ in range(10):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    "Camera opened, but no frame was received.\n"
                    "Check for another camera user: fuser -v /dev/video{}\n"
                    "Stop its camera capture (including active notebook kernels), then retry.\n"
                    "If the device is free, inspect formats with: "
                    "v4l2-ctl -d /dev/video{} --list-formats-ext\n"
                    "You can also try --format YUYV or --format default."
                    .format(options.camera, options.camera))

        if options.serve:
            from vision_stream import serve_camera

            serve_camera(capture, detector, scene_positions, options.host, options.port)
            return 0

        fixed_grid = None
        print("Keys: g = lock empty grid, r = unlock grid, p = print positions, q = quit")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame read failed.")
            scene = detector.detect(frame, grid=fixed_grid)
            annotated = detector.draw(frame, scene)
            status = "Grid locked" if fixed_grid is not None else (
                "Grid detected" if scene.grid is not None else "Grid not found")
            cv2.putText(annotated, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 180, 0) if scene.grid is not None else (0, 0, 255), 2)
            if options.snapshot:
                if not cv2.imwrite(options.snapshot, annotated):
                    raise RuntimeError("Could not save {}".format(options.snapshot))
                print_positions(scene)
                print("Saved {}".format(options.snapshot))
                return 0

            cv2.imshow("DOFBOT vision", annotated)
            window_opened = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty("DOFBOT vision", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("g"):
                if scene.grid is None:
                    print("Grid not found. Clear the board and try again.")
                else:
                    fixed_grid = scene.grid
                    print("Grid locked. Keep the camera and board fixed; now place the cubes.")
            elif key == ord("r"):
                fixed_grid = None
                print("Grid unlocked.")
            elif key == ord("p"):
                print_positions(scene)
    except (RuntimeError, OSError, cv2.error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if window_opened:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
