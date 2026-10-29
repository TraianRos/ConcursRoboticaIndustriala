from libraries import cv2
from vision import CubeGridDetector

detector = CubeGridDetector(grid_pattern="auto", cell_size_mm=20)
frame = cv2.imread("scene.jpg")  # Or a BGR frame from your camera.
if frame is None:
    raise RuntimeError("Could not read scene.jpg")

scene = detector.detect(frame)
if scene.grid is not None:
    for square in scene.grid.squares:
        print(square)

for cube in scene.cubes:
    print(cube.color, cube.center_px, cube.cell, cube.board_xy_mm)

cv2.imwrite("detections.jpg", detector.draw(frame, scene))