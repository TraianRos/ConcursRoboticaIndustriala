"""Detect colored cube candidates and a 3x3-cell grid in OpenCV BGR images.

No camera, arm, or hardware is opened by this module. See README.md for usage.
Cube positions are visible color-region centers, not calibrated 3D grasp points.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from libraries import cv2, np

Point = Tuple[float, float]
HSV = Tuple[int, int, int]
ColorRanges = Mapping[str, Sequence[Tuple[HSV, HSV]]]

# OpenCV HSV: H=0..179, S/V=0..255. Tune these for the camera and lighting.
DEFAULT_COLORS = {
    "red": [((0, 100, 60), (9, 255, 255)), ((170, 100, 60), (179, 255, 255))],
    "orange": [((10, 100, 60), (21, 255, 255))],
    "yellow": [((22, 90, 60), (34, 255, 255))],
    "green": [((35, 70, 50), (85, 255, 255))],
    "blue": [((86, 90, 50), (130, 255, 255))],
    "purple": [((131, 70, 50), (169, 255, 255))],
}


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)


@dataclass
class GridCell:
    """One square of the detected regular grid, projected into image space."""
    row: int
    column: int
    center_px: Point
    corners_px: np.ndarray  # TL, TR, BR, BL
    board_xy_mm: Optional[Point] = None


@dataclass
class GridDetection:
    # Outer corners in order: top-left, top-right, bottom-right, bottom-left.
    corners_px: np.ndarray
    image_to_grid: np.ndarray
    grid_to_image: np.ndarray
    cells: int = 3
    cell_size_mm: Optional[float] = None
    method: str = "manual"

    def locate(self, pixel: Point) -> Optional[Point]:
        """Continuous (column, row), or None if outside the board."""
        x, y = _transform(np.array([pixel]), self.image_to_grid)[0]
        if 0 <= x < self.cells and 0 <= y < self.cells:
            return float(x), float(y)
        return None

    @property
    def cell_centers_px(self) -> np.ndarray:
        """Array of shape (3, 3, 2), indexed [row, column]."""
        points = [(col + 0.5, row + 0.5)
                  for row in range(self.cells) for col in range(self.cells)]
        return _transform(np.array(points), self.grid_to_image).reshape(self.cells, self.cells, 2)


    @property
    def squares(self) -> List[GridCell]:
        """Nine individual cell polygons, in row-major order.

        Geometry comes from the detected board's regular lattice and perspective
        transform; these are not independently fitted contours of drawn lines.
        """
        result = []
        for row in range(self.cells):
            for column in range(self.cells):
                corners = _transform(np.float32([
                    [column, row], [column + 1, row],
                    [column + 1, row + 1], [column, row + 1],
                ]), self.grid_to_image)
                center = _transform(np.float32([[column + 0.5, row + 0.5]]), self.grid_to_image)[0]
                millimeters = None
                if self.cell_size_mm is not None:
                    millimeters = ((column + 0.5) * self.cell_size_mm,
                                   (row + 0.5) * self.cell_size_mm)
                result.append(GridCell(row, column, (float(center[0]), float(center[1])),
                                       corners, millimeters))
        return result


@dataclass
class CubeDetection:
    color: str
    center_px: Point
    bbox_px: Tuple[int, int, int, int]  # x, y, width, height
    contour_px: np.ndarray
    area_px: float
    grid_xy: Optional[Point] = None  # continuous column, row
    cell: Optional[Tuple[int, int]] = None  # integer row, column (0..2)
    board_xy_mm: Optional[Point] = None  # x rightwards, y downwards


@dataclass
class SceneDetection:
    grid: Optional[GridDetection]
    cubes: List[CubeDetection]


class CubeGridDetector:
    def __init__(
        self,
        colors: Optional[ColorRanges] = None,
        min_cube_area: float = 150,
        min_grid_area: float = 2500,
        cell_size_mm: Optional[float] = None,
        grid_pattern: str = "auto",
    ):
        """grid_pattern: 'lines', 'checkerboard', or 'auto' (try both).

        Automatic detection requires a sufficiently visible pattern. For an
        occluded board, detect it empty once and pass that GridDetection to
        detect(frame, grid=...) while the board and camera remain fixed.
        """
        if grid_pattern not in ("auto", "lines", "checkerboard"):
            raise ValueError("grid_pattern must be auto, lines, or checkerboard")
        if min_cube_area <= 0 or min_grid_area <= 0:
            raise ValueError("minimum areas must be positive")
        if cell_size_mm is not None and (not np.isfinite(cell_size_mm) or cell_size_mm <= 0):
            raise ValueError("cell_size_mm must be positive and finite")
        selected = DEFAULT_COLORS if colors is None else colors
        self.colors: Dict[str, list] = {}
        for name, ranges in selected.items():
            self.colors[name] = []
            for low, high in ranges:
                lower, upper = np.asarray(low), np.asarray(high)
                if (lower.shape != (3,) or upper.shape != (3,)
                        or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                        or np.any(lower < 0) or np.any(lower > upper)
                        or np.any(upper > (179, 255, 255))):
                    raise ValueError("invalid HSV range for {}".format(name))
                self.colors[name].append((lower.astype(np.uint8), upper.astype(np.uint8)))
        self.min_cube_area = min_cube_area
        self.min_grid_area = min_grid_area
        self.cell_size_mm = cell_size_mm
        self.grid_pattern = grid_pattern

    @staticmethod
    def _validate_frame(frame: np.ndarray):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16):
            raise ValueError("frame must be a nonempty uint8 BGR image, at least 16x16")

    def grid_from_corners(self, corners_px: Sequence[Point]) -> GridDetection:
        """Use known OUTER board corners: TL, TR, BR, BL; no pattern check.

        Explicit ordering also establishes a stable board origin across views.
        """
        corners = np.asarray(corners_px, dtype=np.float32)
        if (corners.shape != (4, 2) or not np.isfinite(corners).all()
                or not cv2.isContourConvex(corners.reshape(-1, 1, 2))
                or abs(cv2.contourArea(corners)) < 1):
            raise ValueError("corners must be four finite, convex, ordered points")
        target = np.float32([[0, 0], [3, 0], [3, 3], [0, 3]])
        matrix = cv2.getPerspectiveTransform(corners, target)
        return GridDetection(corners.copy(), matrix, np.linalg.inv(matrix),
                             cell_size_mm=self.cell_size_mm)

    def detect_grid(self, frame: np.ndarray) -> Optional[GridDetection]:
        self._validate_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.grid_pattern in ("auto", "checkerboard"):
            grid = self._checkerboard(gray)
            if grid is not None:
                return grid
        if self.grid_pattern in ("auto", "lines"):
            return self._line_grid(gray)
        return None

    def _checkerboard(self, gray: np.ndarray) -> Optional[GridDetection]:
        # Classic OpenCV chessboard detection cannot use 2x2 inner corners.
        # Fit a candidate central dark square, then verify all nine tile interiors.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # Separate dark tiles that touch diagonally at a single corner.
        binary = cv2.erode(binary, np.ones((3, 3), np.uint8))
        contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:64]:
            if cv2.contourArea(contour) < self.min_grid_area / 20:
                continue
            quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            points = quad.reshape(4, 2).astype(np.float32)
            points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
            if cv2.contourArea(points, oriented=True) < 0:
                points = points[[0, 3, 2, 1]]
            matrix = cv2.getPerspectiveTransform(
                np.float32([[1, 1], [2, 1], [2, 2], [1, 2]]), points)
            outer = _transform(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]), matrix)
            if (not np.isfinite(outer).all() or not cv2.isContourConvex(outer)
                    or abs(cv2.contourArea(outer)) < self.min_grid_area
                    or np.any(outer < 0)
                    or np.any(outer[:, 0] >= gray.shape[1])
                    or np.any(outer[:, 1] >= gray.shape[0])):
                continue
            grid = self.grid_from_corners(outer)
            warped = cv2.warpPerspective(gray, np.diag([40., 40., 1.]) @ grid.image_to_grid, (120, 120))
            tiles = [warped[row*40+8:row*40+32, col*40+8:col*40+32]
                     for row in range(3) for col in range(3)]
            dark = float(np.mean([tile.mean() for i, tile in enumerate(tiles) if i % 2 == 0]))
            light = float(np.mean([tile.mean() for i, tile in enumerate(tiles) if i % 2 == 1]))
            if light - dark < 50:
                continue
            threshold = (dark + light) / 2
            if not all(np.mean(tile < threshold if i % 2 == 0 else tile > threshold) > 0.85
                       for i, tile in enumerate(tiles)):
                continue
            grid.method = "checkerboard"
            return grid
        return None

    def _line_grid(self, gray: np.ndarray) -> Optional[GridDetection]:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 7)
        contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < self.min_grid_area:
                break
            quad = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            points = quad.reshape(4, 2).astype(np.float32)
            # Preserve cyclic order, rotate start to the image-upper-left.
            points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
            if cv2.contourArea(points, oriented=True) < 0:
                points = points[[0, 3, 2, 1]]
            grid = self.grid_from_corners(points)
            # 40 pixels per cell; inspect internal lines to ignore outer margins.
            scale = np.diag([40., 40., 1.])
            warped = cv2.warpPerspective(binary, scale @ grid.image_to_grid, (121, 121))
            vertical = cv2.morphologyEx(warped, cv2.MORPH_OPEN, np.ones((25, 1), np.uint8))
            horizontal = cv2.morphologyEx(warped, cv2.MORPH_OPEN, np.ones((1, 25), np.uint8))
            profiles = [np.mean(vertical > 0, axis=0), np.mean(horizontal > 0, axis=1)]
            if all(self._regular_lines(profile) for profile in profiles):
                grid.method = "lines"
                return grid
        return None

    @staticmethod
    def _regular_lines(profile: np.ndarray) -> bool:
        # Two interior lines for three cells, each spanning >=45% of the board.
        for position in range(40, 120, 40):
            if np.max(profile[position - 5:position + 6]) < 0.45:
                return False
        # Reject different cell counts / dense stripes between expected lines.
        between_lines = np.ones(len(profile), dtype=bool)
        for position in range(0, 121, 40):
            between_lines[max(0, position - 6):position + 7] = False
        if np.any(profile[between_lines] > 0.35):
            return False
        return True

    def detect_cubes(self, frame: np.ndarray,
                     grid: Optional[GridDetection] = None) -> List[CubeDetection]:
        self._validate_frame(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)
        results = []
        for color, ranges in self.colors.items():
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_cube_area:
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                (_, _), (width, height), _ = cv2.minAreaRect(contour)
                # Cube silhouettes can be square or hexagonal under perspective.
                if (min(width, height) <= 0 or max(width, height) / min(width, height) > 2.0
                        or area / hull_area < 0.75 or area / (width * height) < 0.55):
                    continue
                moments = cv2.moments(contour)
                center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
                cube = CubeDetection(color, center, cv2.boundingRect(contour), contour.copy(), area)
                if grid is not None:
                    cube.grid_xy = grid.locate(center)
                    if cube.grid_xy is not None:
                        x, y = cube.grid_xy
                        cube.cell = (int(y), int(x))
                        if grid.cell_size_mm is not None:
                            cube.board_xy_mm = (x * grid.cell_size_mm, y * grid.cell_size_mm)
                results.append(cube)
        return sorted(results, key=lambda cube: (cube.color, cube.center_px))

    def detect(self, frame: np.ndarray,
               grid: Optional[GridDetection] = None) -> SceneDetection:
        """Return a new scene result; no implicit grid cache or hardware access."""
        if grid is None:
            grid = self.detect_grid(frame)
        return SceneDetection(grid, self.detect_cubes(frame, grid))

    @staticmethod
    def draw(frame: np.ndarray, scene: SceneDetection) -> np.ndarray:
        """Return an annotated copy, leaving the input unchanged."""
        output = frame.copy()
        if scene.grid is not None:
            cv2.polylines(output, [np.rint(scene.grid.corners_px).astype(np.int32)],
                          True, (0, 200, 255), 2)
            for square in scene.grid.squares:
                polygon = np.rint(square.corners_px).astype(np.int32)
                cv2.polylines(output, [polygon], True, (0, 200, 255), 2)
                x, y = (int(round(value)) for value in square.center_px)
                cv2.circle(output, (x, y), 3, (0, 200, 255), -1)
                cv2.putText(output, "r{} c{}".format(square.row, square.column),
                            (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 140, 220), 1)
        for cube in scene.cubes:
            x, y, w, h = cube.bbox_px
            label = cube.color
            if cube.cell is not None:
                label += " r{} c{}".format(*cube.cell)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(output, tuple(int(round(v)) for v in cube.center_px), 4, (0, 255, 0), -1)
            cv2.putText(output, label, (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return output
