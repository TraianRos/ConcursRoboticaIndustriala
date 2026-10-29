"""Synthetic off-board tests: python3 -m unittest cri.test_vision -v."""

import unittest

from cri.libraries import cv2, np

from cri.vision import CubeGridDetector


def line_board(cells=3):
    image = np.full((480, 480, 3), 255, np.uint8)
    for i in range(cells + 1):
        offset = int(round(60 + i * 360 / cells))
        cv2.line(image, (60, offset), (420, offset), (0, 0, 0), 2)
        cv2.line(image, (offset, 60), (offset, 420), (0, 0, 0), 2)
    return image


class VisionTests(unittest.TestCase):
    def test_line_grid_and_colored_cubes(self):
        image = line_board()
        # Centers (120,160) and (320,360) -> cells (0,0), (2,2).
        cv2.rectangle(image, (107, 147), (133, 173), (0, 0, 255), -1)
        cv2.rectangle(image, (307, 347), (333, 373), (255, 0, 0), -1)
        original = image.copy()
        detector = CubeGridDetector(grid_pattern="lines", cell_size_mm=20)
        scene = detector.detect(image)
        self.assertIsNotNone(scene.grid)
        self.assertEqual(scene.grid.cell_centers_px.shape, (3, 3, 2))
        by_color = {cube.color: cube for cube in scene.cubes}
        self.assertEqual(set(by_color), {"red", "blue"})
        self.assertEqual(by_color["red"].cell, (0, 0))
        self.assertEqual(by_color["blue"].cell, (2, 2))
        np.testing.assert_allclose(by_color["red"].board_xy_mm, (10, 50 / 3), atol=1)
        detector.draw(image, scene)
        np.testing.assert_array_equal(image, original)

    def test_perspective_line_grid(self):
        image = line_board()
        source = np.float32([[60, 60], [420, 60], [420, 420], [60, 420]])
        destination = np.float32([[95, 80], [405, 45], [435, 405], [45, 435]])
        matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(image, matrix, (480, 480), borderValue=(255, 255, 255))
        detector = CubeGridDetector(grid_pattern="lines")
        grid = detector.detect_grid(warped)
        self.assertIsNotNone(grid)
        pixel = cv2.perspectiveTransform(np.float32([[[200, 240]]]), matrix)[0, 0]
        np.testing.assert_allclose(grid.locate(tuple(pixel)), (7 / 6, 1.5), atol=0.1)

    def test_checkerboard_has_nine_cells(self):
        image = np.full((480, 480, 3), 255, np.uint8)
        for row in range(3):
            for col in range(3):
                if (row + col) % 2 == 0:
                    image[60 + row*120:180 + row*120, 60 + col*120:180 + col*120] = 0
        grid = CubeGridDetector(grid_pattern="checkerboard").detect_grid(image)
        self.assertIsNotNone(grid)
        np.testing.assert_allclose(grid.locate((240, 240)), (1.5, 1.5), atol=0.05)

    def test_no_grid_does_not_invent_cell_positions(self):
        image = np.full((300, 400, 3), 255, np.uint8)
        cv2.rectangle(image, (70, 80), (110, 120), (0, 255, 0), -1)
        scene = CubeGridDetector(grid_pattern="lines").detect(image)
        self.assertIsNone(scene.grid)
        self.assertEqual(len(scene.cubes), 1)
        self.assertEqual(scene.cubes[0].color, "green")
        self.assertIsNone(scene.cubes[0].cell)

    def test_grid_rejects_empty_rectangle_and_other_cell_counts(self):
        detector = CubeGridDetector(grid_pattern="lines")
        for cells in (1, 2, 4, 9):
            with self.subTest(cells=cells):
                self.assertIsNone(detector.detect_grid(line_board(cells)))

    def test_manual_grid_and_outside_cube(self):
        detector = CubeGridDetector()
        grid = detector.grid_from_corners([(60, 60), (420, 60), (420, 420), (60, 420)])
        image = np.full((480, 480, 3), 255, np.uint8)
        cv2.rectangle(image, (10, 10), (40, 40), (0, 0, 255), -1)
        scene = detector.detect(image, grid=grid)
        self.assertIs(scene.grid, grid)
        self.assertIsNone(scene.cubes[0].grid_xy)
        self.assertIsNone(grid.locate((420, 240)))

    def test_all_default_colors_including_red_hue_wrap(self):
        hsv = np.zeros((150, 600, 3), np.uint8)
        hues = [0, 179, 15, 28, 60, 110, 150]
        for index, hue in enumerate(hues):
            hsv[50:90, index*80+10:index*80+50] = (hue, 220, 220)
        cubes = CubeGridDetector().detect_cubes(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        self.assertEqual(sorted(cube.color for cube in cubes),
                         sorted(["red", "red", "orange", "yellow", "green", "blue", "purple"]))

    def test_small_noise_and_elongated_regions_are_rejected(self):
        image = np.zeros((200, 300, 3), np.uint8)
        cv2.rectangle(image, (10, 10), (13, 13), (0, 0, 255), -1)
        cv2.rectangle(image, (30, 60), (200, 75), (0, 0, 255), -1)
        self.assertEqual(CubeGridDetector().detect_cubes(image), [])

    def test_individual_grid_squares(self):
        from cri.preview_vision import scene_positions
        detector = CubeGridDetector(grid_pattern="lines", cell_size_mm=20)
        scene = detector.detect(line_board())
        squares = scene.grid.squares
        self.assertEqual([(s.row, s.column) for s in squares],
                         [(row, col) for row in range(3) for col in range(3)])
        for square in squares:
            self.assertEqual(square.corners_px.shape, (4, 2))
            np.testing.assert_allclose(scene.grid.locate(square.center_px),
                                       (square.column + 0.5, square.row + 0.5), atol=0.001)
            self.assertEqual(square.board_xy_mm,
                             ((square.column + 0.5)*20, (square.row + 0.5)*20))
        # Neighboring cells share the same boundary, even under perspective.
        np.testing.assert_allclose(squares[0].corners_px[[1, 2]], squares[1].corners_px[[0, 3]])
        data = scene_positions(scene)
        self.assertEqual(len(data["grid_cells"]), 9)
        self.assertEqual(data["grid_cells"][4]["row"], 1)
        self.assertEqual(data["grid_cells"][4]["column"], 1)
        self.assertEqual(len(data["grid_cells"][4]["corners_px"]), 4)

    def test_cell_geometry_under_perspective(self):
        detector = CubeGridDetector()
        grid = detector.grid_from_corners([(95, 80), (405, 45), (435, 405), (45, 435)])
        for square in grid.squares:
            # Verify every projected corner against its original grid coordinate.
            points = cv2.perspectiveTransform(square.corners_px.reshape(-1, 1, 2),
                                             grid.image_to_grid).reshape(-1, 2)
            row, col = square.row, square.column
            np.testing.assert_allclose(points, [(col, row), (col+1, row),
                                               (col+1, row+1), (col, row+1)], atol=0.001)

    def test_missing_grid_has_no_cell_results(self):
        from cri.preview_vision import scene_positions
        scene = CubeGridDetector(grid_pattern="lines").detect(
            np.full((200, 200, 3), 255, np.uint8))
        self.assertEqual(scene_positions(scene)["grid_cells"], [])

    def test_invalid_inputs(self):
        detector = CubeGridDetector()
        for frame in (None, np.zeros((20, 20), np.uint8), np.zeros((20, 20, 3))):
            with self.assertRaises(ValueError):
                detector.detect(frame)
        with self.assertRaises(ValueError):
            detector.grid_from_corners([(0, 0)] * 4)
        with self.assertRaises(ValueError):
            CubeGridDetector(cell_size_mm=0)
        with self.assertRaises(ValueError):
            CubeGridDetector(colors={"invalid": [((0, 0, 0), (200, 255, 255))]})


if __name__ == "__main__":
    unittest.main()
