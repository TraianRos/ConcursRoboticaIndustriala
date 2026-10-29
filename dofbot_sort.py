#!/usr/bin/env python3
"""
DOFBOT (Raspberry Pi 4) - color cube sorting onto matching line-art outlines.

Modes:
  python3 dofbot_sort.py hsv        live camera view + detections (tune colors/sizes)
  python3 dofbot_sort.py calibrate  teach pixel -> arm angles (do this once)
  python3 dofbot_sort.py grid       live view of detected grid cells (r1c1 ...)
  python3 dofbot_sort.py teach      memorize the color pattern currently shown in the grid
  python3 dofbot_sort.py plan       (no arm) preview which loose cube goes to which cell
  python3 dofbot_sort.py build      arm rebuilds the memorized pattern from loose cubes
  python3 dofbot_sort.py run        pick every cube and place it in the outline
                                    of the SAME color
Add --headless to any mode if you have no display (prints instead of windows).

Setup assumptions (edit the CONFIG block if yours differs):
  * Cubes are solid colored (red / green / blue / yellow).
  * Targets are hollow outlines (squares drawn with a marker) in the same colors,
    on a white/light sheet.
  * The camera is the DOFBOT camera (index 0) and the arm is moved out of
    the camera's view (HOME pose) while looking.
"""
import sys, json, time, os
import numpy as np
import cv2

# ----------------------------------------------------------------- CONFIG
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")
PATTERN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pattern.json")
N_CALIB_POINTS = 6          # >= 4. 6 is better. Spread them over the whole work area.

HOME = [90, 135, 20, 25, 90, 30]       # looks at table, arm out of the way
GRIP_OPEN = 30                          # servo 6 angle, open  (tune)
GRIP_CLOSED = 135                       # servo 6 angle, closed on cube (tune)
MOVE_MS = 1200                          # time of each arm move

# HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255). Tune with `hsv` mode.
COLORS = {
    "red":    [((0, 120, 70), (8, 255, 255)), ((172, 120, 70), (179, 255, 255))],
    "yellow": [((20, 110, 90), (35, 255, 255))],
    "green":  [((40, 80, 50), (85, 255, 255))],
    "blue":   [((95, 110, 50), (130, 255, 255))],
}
DRAW = {"red": (0, 0, 255), "yellow": (0, 255, 255),
        "green": (0, 255, 0), "blue": (255, 0, 0)}

MIN_SIDE, MAX_SIDE = 22, 140   # accepted bounding-box side length in pixels
CUBE_FILL_MIN = 0.60           # solid blob  -> cube
OUTLINE_FILL_MAX = 0.40        # hollow shape -> outline (line art)
# ------------------------------------------------------------------------

KERNEL = np.ones((3, 3), np.uint8)


def color_mask(hsv, name):
    m = None
    for lo, hi in COLORS[name]:
        part = cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = part if m is None else cv2.bitwise_or(m, part)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, KERNEL)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL)
    return m


def detect(frame):
    """Returns {'cubes': [(color,cx,cy,w,h)], 'outlines': [...]}"""
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    cubes, outlines = [], []
    for name in COLORS:
        mask = color_mask(hsv, name)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
                continue
            if not (0.55 < w / h < 1.8):
                continue
            fill = cv2.countNonZero(mask[y:y + h, x:x + w]) / float(w * h)
            item = (name, x + w // 2, y + h // 2, w, h)
            if fill >= CUBE_FILL_MIN:
                cubes.append(item)
            elif fill <= OUTLINE_FILL_MAX:
                outlines.append(item)
    # a cube already sitting inside an outline of its color is "done"
    todo = []
    for cb in cubes:
        done = any(o[0] == cb[0] and abs(o[1] - cb[1]) < o[3] / 2 and
                   abs(o[2] - cb[2]) < o[4] / 2 for o in outlines)
        if not done:
            todo.append(cb)
    return {"cubes": todo, "outlines": outlines}


def annotate(frame, res):
    for kind, items in res.items():
        for name, cx, cy, w, h in items:
            col = DRAW[name]
            if kind == "cubes":
                cv2.rectangle(frame, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), col, 2)
                cv2.putText(frame, f"cube {name}", (cx - w // 2, cy - h // 2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            else:
                cv2.circle(frame, (cx, cy), max(w, h) // 2, col, 2)
                cv2.putText(frame, f"target {name}", (cx - w // 2, cy - h // 2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return frame


# ----------------------------------------------------------------- GRID
# Grid = thin dark lines (black marker / print) on a light sheet, drawn as a FULL
# rectangle (outer border included) and roughly straight in the image.
# Method: keep only THIN dark structures (black-hat), ignore colored pixels, keep only
# long horizontal / vertical runs, then read the line positions from projections.
# So cubes inside cells, soft shadows and partly hidden lines are tolerated.
GRID_THICK_MAX = 21       # odd; must be bigger than the line thickness in pixels
GRID_CONTRAST = 35        # a line must be this much darker than its surroundings (0-255)
GRID_MIN_LINE = 40        # shortest straight run (pixels) that counts as a line piece
CELL_MIN_SIDE, CELL_MAX_SIDE = 30, 220
GRID_ROWS, GRID_COLS = 3, 3   # known grid size -> fits a perfectly regular grid; 0, 0 = auto-detect


GRID_DIAG = ""            # short text explaining what the last detection saw


def thin_dark_mask(frame):
    """Mask of thin, dark, uncolored structures (= grid lines). Shadows/cubes are excluded."""
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (GRID_THICK_MAX, GRID_THICK_MAX))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)       # thin dark things only:
    lines = (bh > GRID_CONTRAST).astype(np.uint8) * 255       # broad shadows are ignored
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    colored = cv2.inRange(hsv, np.array((0, 90, 40)), np.array((179, 255, 255)))
    lines[cv2.dilate(colored, np.ones((5, 5), np.uint8)) > 0] = 0   # cubes are not lines
    return lines


def _segments(mask):
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, 25,
                           minLineLength=GRID_MIN_LINE, maxLineGap=12)
    if segs is None:
        return []
    return [tuple(float(v) for v in s) for s in np.asarray(segs).reshape(-1, 4)]   # OpenCV 4 and 5


def _dominant_angle(segs):
    """Grid rotation in radians, in (-45deg, 45deg]; length-weighted, 90-degree periodic."""
    z = 0j
    for x1, y1, x2, y2 in segs:
        z += np.hypot(x2 - x1, y2 - y1) * np.exp(4j * np.arctan2(y2 - y1, x2 - x1))
    return float(np.angle(z) / 4)


def _families(segs, th):
    """Split segments into near-horizontal (offsets along n) and near-vertical (along u)."""
    u = np.array((np.cos(th), np.sin(th)))
    n = np.array((-np.sin(th), np.cos(th)))
    A, B = [], []
    for x1, y1, x2, y2 in segs:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1) - th)
        ang = (ang + 90) % 180 - 90                       # [-90, 90)
        length = float(np.hypot(x2 - x1, y2 - y1))
        mid = np.array(((x1 + x2) / 2, (y1 + y2) / 2))
        p1, p2 = np.array((x1, y1)), np.array((x2, y2))
        if abs(ang) <= 12:                  # item = (offset, length, start, end along the line)
            a, b = sorted((float(u @ p1), float(u @ p2)))
            A.append((float(n @ mid), length, a, b))
        elif abs(abs(ang) - 90) <= 12:
            a, b = sorted((float(n @ p1), float(n @ p2)))
            B.append((float(u @ mid), length, a, b))
    return A, B


def _cluster(items):
    """Merge nearby segments into lines. Returns [(position, weight, extent)], weak ones dropped.
    extent = how far the line stretches; table/paper edges are much longer than the grid."""
    if not items:
        return []
    items = sorted(items)
    groups = [[items[0]]]
    for it in items[1:]:
        if it[0] - groups[-1][-1][0] <= min(12, CELL_MIN_SIDE / 2):
            groups[-1].append(it)
        else:
            groups.append([it])
    lines = []
    for g in groups:
        w = sum(l for _, l, _, _ in g)
        lines.append((sum(o * l for o, l, _, _ in g) / w, w,
                      max(hi for _, _, _, hi in g) - min(lo for _, _, lo, _ in g)))
    top = max(w for _, w, _ in lines)
    return [ln for ln in lines if ln[1] >= max(0.12 * top, 1.5 * GRID_MIN_LINE)]


def _best_run(pos):
    """Longest run of roughly equally spaced lines = the grid itself."""
    if len(pos) < 2:
        return []
    d = np.diff(pos)
    valid = d[(d >= CELL_MIN_SIDE) & (d <= CELL_MAX_SIDE)]
    if len(valid) == 0:
        return []
    m = float(np.median(valid))
    runs, cur = [], [pos[0]]
    for a, b in zip(pos[:-1], pos[1:]):
        if abs((b - a) - m) <= 0.35 * m:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    best = max(runs, key=len)
    return best if len(best) >= 2 else []


def _fit_regular(pairs, n):
    """Fit n equally spaced lines to observed (position, weight, extent) lines.
    Needs at least n-1 of them to agree; a hidden line is filled in, stray lines are ignored,
    and lines far longer than the grid itself (table / paper edges) are rejected.
    Returns n positions, or [] if no good fit."""
    if len(pairs) < n - 1:
        return []
    pos = np.array([p for p, _, _ in pairs])
    wts = np.array([w for _, w, _ in pairs])
    ext = np.array([e for _, _, e in pairs])
    best_score, best = 0.0, None
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            for k in range(1, n):                       # observed line j is k steps after i
                pitch = (pos[j] - pos[i]) / k
                if not CELL_MIN_SIDE <= pitch <= CELL_MAX_SIDE:
                    continue
                for m in range(0, n - k):               # line i sits at lattice index m
                    lat = pos[i] - m * pitch + pitch * np.arange(n)
                    d = np.abs(lat[:, None] - pos[None, :])
                    nearest = d.argmin(axis=1)
                    dist = d[np.arange(n), nearest]
                    tol = 0.12 * pitch
                    ok = (dist <= tol) & (ext[nearest] <= 1.25 * n * pitch)  # not longer than the grid
                    if ok.sum() >= n - 1:
                        # fit quality (how exactly the lines sit on the lattice); line length
                        # only breaks ties, so long stray edges can't beat the real grid
                        score = float(np.sum(1.0 - dist[ok] / tol)) + 1e-3 * float(wts[nearest][ok].sum() / wts.max())
                        if score > best_score:
                            best_score, best = score, (nearest, ok)
    if best is None:
        return []
    nearest, ok = best
    idx = np.where(ok)[0]
    A = np.vstack([np.ones(len(idx)), idx]).T
    t0, p = np.linalg.lstsq(A, pos[nearest[idx]], rcond=None)[0]   # refine with least squares
    return [float(t0 + p * k) for k in range(n)]


def detect_grid(frame):
    """Returns (cells, n_rows, n_cols). cell = dict(row,col,cx,cy,w,h), 1-based, row 1 = top."""
    global GRID_DIAG
    segs = _segments(thin_dark_mask(frame))
    if len(segs) < 4:
        GRID_DIAG = f"only {len(segs)} line pieces found (lower GRID_CONTRAST / GRID_MIN_LINE?)"
        return [], 0, 0
    th = _dominant_angle(segs)
    A, B = _families(segs, th)
    pa, pb = _cluster(A), _cluster(B)
    if GRID_ROWS > 0 and GRID_COLS > 0:             # known size: fit a regular grid
        ts, ss = _fit_regular(pa, GRID_ROWS + 1), _fit_regular(pb, GRID_COLS + 1)
        GRID_DIAG = (f"tilt {np.degrees(th):.1f} deg, saw {len(pa)} horizontal + {len(pb)} vertical "
                     f"lines, need {GRID_ROWS + 1} + {GRID_COLS + 1} (at least one fewer is OK)")
    else:                                           # unknown size: longest evenly spaced run
        ts, ss = _best_run([p for p, _, _ in pa]), _best_run([p for p, _, _ in pb])
        GRID_DIAG = (f"tilt {np.degrees(th):.1f} deg, horizontal-ish lines: {len(ts)}, "
                     f"vertical-ish lines: {len(ss)}")
    if len(ts) < 2 or len(ss) < 2:
        return [], 0, 0
    u = np.array((np.cos(th), np.sin(th)))
    n = np.array((-np.sin(th), np.cos(th)))
    cells = []
    for ri in range(len(ts) - 1):
        for ci in range(len(ss) - 1):
            corners = [u * ss[ci] + n * ts[ri], u * ss[ci + 1] + n * ts[ri],
                       u * ss[ci + 1] + n * ts[ri + 1], u * ss[ci] + n * ts[ri + 1]]
            ctr = np.mean(corners, axis=0)
            cells.append({"row": ri + 1, "col": ci + 1, "cx": int(ctr[0]), "cy": int(ctr[1]),
                          "w": int(ss[ci + 1] - ss[ci]), "h": int(ts[ri + 1] - ts[ri]),
                          "poly": [(int(x), int(y)) for x, y in corners]})
    return cells, len(ts) - 1, len(ss) - 1


def cell_of(cells, px, py):
    for c in cells:
        poly = c.get("poly")
        if poly is not None:
            if cv2.pointPolygonTest(np.array(poly, np.int32), (float(px), float(py)), False) >= 0:
                return c
        elif abs(px - c["cx"]) <= c["w"] / 2 and abs(py - c["cy"]) <= c["h"] / 2:
            return c
    return None


def annotate_grid(frame, cells, n_rows, n_cols):
    MAG = (255, 0, 255)
    for c in cells:
        poly = np.array(c["poly"], np.int32)
        cv2.polylines(frame, [poly], True, MAG, 2)
        cv2.putText(frame, f"r{c['row']}c{c['col']}", (int(poly[0][0]) + 4, int(poly[0][1]) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAG, 1)
    cv2.putText(frame, f"grid {n_rows}x{n_cols} ({len(cells)} cells)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAG, 2)
    return frame


def grid_view(cap, headless):
    global GRID_CONTRAST, GRID_THICK_MAX, GRID_MIN_LINE
    print("Grid view. q = quit. Window 'grid lines (debug)' shows what counts as a line (white)")
    print("and has SLIDERS: move them until the grid lines are clean and complete.")
    win = "grid lines (debug)"
    if not headless:
        cv2.namedWindow(win)
        nothing = lambda v: None
        cv2.createTrackbar("contrast", win, GRID_CONTRAST, 100, nothing)
        cv2.createTrackbar("thick/2", win, GRID_THICK_MAX // 2, 20, nothing)
        cv2.createTrackbar("min line", win, GRID_MIN_LINE, 150, nothing)
    last = None
    while True:
        if not headless:                        # read sliders
            GRID_CONTRAST = max(1, cv2.getTrackbarPos("contrast", win))
            GRID_THICK_MAX = 2 * max(1, cv2.getTrackbarPos("thick/2", win)) + 1
            GRID_MIN_LINE = max(10, cv2.getTrackbarPos("min line", win))
        frame = grab(cap, 1)
        cells, nr, nc = detect_grid(frame)
        res = detect(frame)
        summary = [f"grid {nr}x{nc} ({len(cells)} cells)"]
        if not cells:
            summary.append(GRID_DIAG)
        for name, cx, cy, _, _ in res["cubes"]:
            c = cell_of(cells, cx, cy)
            summary.append(f"{name} cube in " + (f"r{c['row']}c{c['col']}" if c else "no cell"))
        if summary != last:                     # print only when something changes
            print(" | ".join(summary))
            last = summary
        if headless:
            time.sleep(0.5)
            continue
        dbg = cv2.cvtColor(thin_dark_mask(frame), cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, GRID_DIAG[:70], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow(win, dbg)
        out = annotate(annotate_grid(frame, cells, nr, nc), res)
        cv2.imshow("dofbot grid", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    print("\nIf the sliders gave a good result, put these at the top of the file:")
    print(f"GRID_CONTRAST = {GRID_CONTRAST}\nGRID_THICK_MAX = {GRID_THICK_MAX}\nGRID_MIN_LINE = {GRID_MIN_LINE}")


def open_cam(source=None):
    """source: None -> CAM_INDEX, '1' -> device index, 'http://...' -> network stream."""
    if source is None:
        source = CAM_INDEX
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    t0 = time.time()
    print(f"Opening camera {source} ... (can take up to 30 s on Windows)", flush=True)
    if isinstance(source, int) and sys.platform.startswith("win"):
        cap = None
        for be_name, be in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)):
            c = cv2.VideoCapture(source, be)
            ok = c.isOpened() and c.read()[0]
            print(f"  backend {be_name}: {'works' if ok else 'no picture'}", flush=True)
            if ok:
                cap = c
                break
            c.release()
        if cap is None:
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    print(("Camera opened" if cap.isOpened() else "Camera FAILED to open") +
          f" in {time.time() - t0:.1f} s", flush=True)
    if not sys.platform.startswith("win"):   # on Windows this renegotiation can take seconds;
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)   # grab() rescales frames anyway
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit("Cannot open camera. Try --cam 0 / 1 / 2, or --cam http://PHONE_IP:PORT/video")
    return cap


def grab(cap, flush=5):
    for _ in range(flush):          # drop stale buffered frames
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        sys.exit("Camera read failed.")
    if frame.shape[1] != FRAME_W:   # phones often send 720p/1080p: scale to 640 wide, keep aspect
        s = FRAME_W / frame.shape[1]
        frame = cv2.resize(frame, (FRAME_W, int(round(frame.shape[0] * s))),
                           interpolation=cv2.INTER_AREA)
    return frame


# ----------------------------------------------------------------- ARM
class Arm:
    def __init__(self):
        from Arm_Lib import Arm_Device      # Yahboom library on the DOFBOT image
        self.dev = Arm_Device()
        time.sleep(0.1)

    def move(self, angles, ms=MOVE_MS):
        a = [int(max(0, min(180, v))) for v in angles]
        self.dev.Arm_serial_servo_write6(a[0], a[1], a[2], a[3], a[4], a[5], ms)
        time.sleep(ms / 1000.0 + 0.15)

    def grip(self, angle, ms=500):
        self.dev.Arm_serial_servo_write(6, int(angle), ms)
        time.sleep(ms / 1000.0 + 0.1)

    def read(self):
        out = []
        for i in range(1, 7):
            v = None
            for _ in range(5):
                v = self.dev.Arm_serial_servo_read(i)
                if v is not None:
                    break
                time.sleep(0.02)
            out.append(v if v is not None else 90)
        return out

    def torque(self, on):
        try:
            self.dev.Arm_serial_set_torque(1 if on else 0)
        except AttributeError:
            print("(!) Your Arm_Lib has no Arm_serial_set_torque; "
                  "update Arm_Lib or move the arm by hand with torque already off.")

    def home(self):
        self.move(HOME)


# ----------------------------------------------------------------- CALIBRATION
def feat(px, py):
    x, y = px / FRAME_W, py / FRAME_H
    return np.array([x, y, x * y, 1.0])


def predict(model, px, py):
    return (feat(px, py) @ np.array(model)).tolist()   # 5 joint angles


def calibrate(cap, arm):
    pts, down, up = [], [], []
    print(f"\nCalibration with {N_CALIB_POINTS} points. Put ONE cube in view each time.")
    for i in range(N_CALIB_POINTS):
        input(f"\n[{i+1}/{N_CALIB_POINTS}] Arm goes HOME. Place ONE cube at a new spot "
              f"(spread over the work area), then press Enter...")
        arm.home()
        res = detect(grab(cap))
        if len(res["cubes"]) != 1:
            print(f"  Saw {len(res['cubes'])} cubes, need exactly 1. Skipping, retry this point.")
            return calibrate(cap, arm)
        _, cx, cy, _, _ = res["cubes"][0]
        print(f"  Cube found at pixel ({cx},{cy}). Torque OFF: move the gripper by hand")
        arm.torque(False)
        input("  Put the open gripper right AROUND the cube (ready to grab), press Enter...")
        d = arm.read()
        input("  Now lift the gripper ~4-5 cm straight UP (same position above), press Enter...")
        u = arm.read()
        arm.torque(True)
        arm.move(HOME)
        pts.append((cx, cy)); down.append(d[:5]); up.append(u[:5])
        print(f"  down={d[:5]} up={u[:5]}")

    A = np.array([feat(x, y) for x, y in pts])
    m_down = np.linalg.lstsq(A, np.array(down, float), rcond=None)[0]
    m_up = np.linalg.lstsq(A, np.array(up, float), rcond=None)[0]
    json.dump({"down": m_down.tolist(), "up": m_up.tolist()}, open(CALIB_FILE, "w"))
    print(f"\nSaved {CALIB_FILE}")
    err = np.abs(A @ m_down - np.array(down, float)).max()
    print(f"Max fit error on teach points: {err:.1f} deg (should be a few degrees)")


# ----------------------------------------------------------------- PICK & PLACE
def go(arm, model, px, py):
    return predict(model, px, py)


def pick(arm, cal, px, py):
    up = go(arm, cal["up"], px, py)
    down = go(arm, cal["down"], px, py)
    arm.grip(GRIP_OPEN)
    arm.move(up + [GRIP_OPEN])
    arm.move(down + [GRIP_OPEN], 900)
    arm.grip(GRIP_CLOSED)
    arm.move(up + [GRIP_CLOSED], 900)


def place(arm, cal, px, py):
    up = go(arm, cal["up"], px, py)
    down = go(arm, cal["down"], px, py)
    arm.move(up + [GRIP_CLOSED])
    arm.move(down + [GRIP_CLOSED], 900)
    arm.grip(GRIP_OPEN)
    arm.move(up + [GRIP_OPEN], 900)


# ----------------------------------------------------------------- PATTERN (teach / plan / build)
def load_pattern():
    if not os.path.exists(PATTERN_FILE):
        sys.exit("No pattern.json - run: python dofbot_sort.py teach")
    p = json.load(open(PATTERN_FILE))
    target = {tuple(int(v) for v in k.split(",")): c for k, c in p["cells"].items()}
    return p["rows"], p["cols"], target


def show_pattern(rows, cols, target):
    for r in range(1, rows + 1):
        print("  " + " ".join(target.get((r, c), ".")[0].upper() for c in range(1, cols + 1)))
    print("  (R=red G=green B=blue Y=yellow .=empty)")


def teach(cap, headless):
    print("Put the cubes in the grid as the robot should copy them. Keep still, hands out of view...")
    votes, n = {}, 8
    cells, nr, nc, frame = [], 0, 0, None
    for _ in range(n):                       # several frames -> majority vote, ignores flicker
        frame = grab(cap, 2)
        cells, nr, nc = detect_grid(frame)
        for name, cx, cy, _w, _h in detect(frame)["cubes"]:
            c = cell_of(cells, cx, cy)
            if c:
                d = votes.setdefault((c["row"], c["col"]), {})
                d[name] = d.get(name, 0) + 1
        time.sleep(0.1)
    if not cells:
        sys.exit("No grid found. Run `grid` mode first and fix the detection.")
    target = {}
    for key, d in votes.items():
        name, v = max(d.items(), key=lambda kv: kv[1])
        if v >= n / 2:
            target[key] = name
    print(f"Grid {nr}x{nc}, {len(target)} cubes recorded:")
    show_pattern(nr, nc, target)
    json.dump({"rows": nr, "cols": nc,
               "cells": {f"{r},{c}": v for (r, c), v in target.items()}}, open(PATTERN_FILE, "w"))
    cv2.imwrite("pattern.jpg", annotate(annotate_grid(frame.copy(), cells, nr, nc), detect(frame)))
    print(f"Saved {PATTERN_FILE} and pattern.jpg. Now take the cubes OUT of the grid.")


def plan_jobs(cells, res, target):
    """Match loose cubes (outside the grid) to cells that still need a cube."""
    occupant, pile = {}, []
    for cb in res["cubes"]:
        c = cell_of(cells, cb[1], cb[2])
        if c:
            occupant[(c["row"], c["col"])] = cb[0]
        else:
            pile.append(cb)
    jobs, problems = [], []
    for (r, c), color in sorted(target.items()):
        occ = occupant.get((r, c))
        if occ == color:
            continue
        if occ is not None:
            problems.append(f"r{r}c{c} holds a {occ} cube but needs {color} - remove it by hand")
            continue
        cell = next((x for x in cells if x["row"] == r and x["col"] == c), None)
        if cell is None:
            problems.append(f"cell r{r}c{c} not detected")
            continue
        src = next((p for p in pile if p[0] == color), None)
        if src is None:
            problems.append(f"no free {color} cube for r{r}c{c}")
            continue
        pile.remove(src)
        jobs.append((src, cell))
    return jobs, problems


def plan_view(cap, headless, once=False):
    """Live preview of which loose cube would go to which cell (no arm). q = quit.
    --once or --headless: analyse a single frame, save plan.jpg and exit."""
    rows, cols, target = load_pattern()
    print("Pattern to build:")
    show_pattern(rows, cols, target)
    if not (headless or once):
        print("Live preview - move cubes around and the plan updates. q = quit.")
    t0 = time.time()
    for _ in range(5):                       # let the camera's auto exposure settle
        grab(cap, 1)
    print(f"Camera warm-up took {time.time() - t0:.1f} s", flush=True)
    last = None
    while True:
        t1 = time.time()
        frame = grab(cap, 1)
        t_grab = time.time() - t1
        t1 = time.time()
        cells, nr, nc = detect_grid(frame)
        res = detect(frame)
        jobs, problems = plan_jobs(cells, res, target)
        t_det = time.time() - t1
        lines = []
        if (nr, nc) != (rows, cols):
            lines.append(f"WARNING: grid seen is {nr}x{nc}, pattern is {rows}x{cols}.")
        lines += [f"{cb[0]} cube -> r{cell['row']}c{cell['col']}" for cb, cell in jobs]
        lines += ["PROBLEM: " + p for p in problems]
        if not jobs and not problems:
            lines.append("Pattern already complete.")
        if lines != last:                    # print only when the plan changes
            print("---")
            print("\n".join(lines))
            print(f"(camera frame {t_grab * 1000:.0f} ms, detection {t_det * 1000:.0f} ms)")
            last = lines
        out = annotate(annotate_grid(frame.copy(), cells, nr, nc), res)
        for cb, cell in jobs:
            cv2.arrowedLine(out, (cb[1], cb[2]), (cell["cx"], cell["cy"]), DRAW[cb[0]], 2, tipLength=0.1)
        if headless or once:
            cv2.imwrite("plan.jpg", out)
            print("Saved plan.jpg")
            break
        cv2.imshow("plan (q = quit)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def build(cap, arm):
    rows, cols, target = load_pattern()
    if not os.path.exists(CALIB_FILE):
        sys.exit("No calib.json - run: python3 dofbot_sort.py calibrate")
    cal = json.load(open(CALIB_FILE))
    arm.home()
    arm.grip(GRIP_OPEN)
    tries = {}
    for _ in range(60):
        time.sleep(0.4)
        frame = grab(cap, 5)                 # arm is at HOME, out of the camera's view
        cells, nr, nc = detect_grid(frame)
        if (nr, nc) != (rows, cols):
            print(f"Grid seen is {nr}x{nc}, pattern is {rows}x{cols}. Stopping.")
            break
        jobs, problems = plan_jobs(cells, detect(frame), target)
        if not jobs:
            for p in problems:
                print("PROBLEM:", p)
            print("Done - pattern complete." if not problems else "Stopped: fix the problems above.")
            break
        cb, cell = jobs[0]
        key = (cell["row"], cell["col"])
        tries[key] = tries.get(key, 0) + 1
        if tries[key] > 3:
            print(f"Giving up on r{key[0]}c{key[1]} after 3 tries.")
            break
        print(f"Placing {cb[0]} cube ({cb[1]},{cb[2]}) -> r{key[0]}c{key[1]}")
        pick(arm, cal, cb[1], cb[2])
        place(arm, cal, cell["cx"], cell["cy"])
        arm.home()
    arm.home()


def dry_run(cap, headless):
    """No arm: show which cube would go to which outline."""
    frame = grab(cap, 3)
    res = detect(frame)
    out = annotate(frame.copy(), res)
    used = set()
    for cb in res["cubes"]:
        tgt = [o for i, o in enumerate(res["outlines"]) if o[0] == cb[0] and i not in used]
        if not tgt:
            print(f"{cb[0]} cube at ({cb[1]},{cb[2]}): NO matching outline found")
            continue
        used.add(res["outlines"].index(tgt[0]))
        t = tgt[0]
        print(f"{cb[0]} cube ({cb[1]},{cb[2]}) -> {t[0]} outline ({t[1]},{t[2]})")
        cv2.arrowedLine(out, (cb[1], cb[2]), (t[1], t[2]), DRAW[cb[0]], 2, tipLength=0.1)
    if not headless:
        cv2.imshow("dry run (any key to close)", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite("dry_run.jpg", out)
        print("Saved dry_run.jpg")


def run(cap, arm):
    if not os.path.exists(CALIB_FILE):
        sys.exit("No calib.json - run: python3 dofbot_sort.py calibrate")
    cal = json.load(open(CALIB_FILE))
    arm.home()
    arm.grip(GRIP_OPEN)
    fails = 0
    while True:
        time.sleep(0.4)
        res = detect(grab(cap))
        job = None
        for cb in res["cubes"]:
            tgt = [o for o in res["outlines"] if o[0] == cb[0]]
            if tgt:
                job = (cb, tgt[0])
                break
        if job is None:
            left = [c[0] for c in res["cubes"]]
            print("Nothing left to sort." if not left else
                  f"No matching outline visible for cubes: {left}")
            break
        cb, tg = job
        print(f"Moving {cb[0]} cube ({cb[1]},{cb[2]}) -> outline ({tg[1]},{tg[2]})")
        pick(arm, cal, cb[1], cb[2])
        place(arm, cal, tg[1], tg[2])
        arm.home()
        fails += 1
        if fails > 30:
            print("Safety stop after 30 moves.")
            break
    arm.home()


def hsv_view(cap, headless):
    print("HSV view. q = quit. Edit COLORS / sizes at the top of the file.")
    while True:
        frame = grab(cap, 1)
        res = detect(frame)
        if headless:
            print(res)
            time.sleep(0.5)
            continue
        cv2.imshow("dofbot", annotate(frame, res))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    headless = "--headless" in sys.argv
    cam = sys.argv[sys.argv.index("--cam") + 1] if "--cam" in sys.argv else None
    if mode == "list":
        print("Scanning cameras (close the Windows Camera app and other camera apps first)...")
        backends = [("default", None)]
        if sys.platform.startswith("win"):
            backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
        for i in range(10):
            for name, be in backends:
                c = cv2.VideoCapture(i) if be is None else cv2.VideoCapture(i, be)
                ok, fr = c.read() if c.isOpened() else (False, None)
                if ok:
                    print(f"  index {i} [{name}]: picture OK, size {fr.shape[1]}x{fr.shape[0]}  "
                          f"-> use --cam {i}")
                c.release()
        print("Done. Indexes not listed gave no picture.")
        sys.exit(0)
    cap = open_cam(cam)
    try:
        if mode == "hsv":
            hsv_view(cap, headless)
        elif mode == "grid":
            grid_view(cap, headless)
        elif mode == "teach":
            teach(cap, headless)
        elif mode == "plan":
            plan_view(cap, headless, once="--once" in sys.argv)
        elif mode == "build":
            build(cap, Arm())
        elif mode == "dry":
            dry_run(cap, headless)
        elif mode == "calibrate":
            calibrate(cap, Arm())
        elif mode == "run":
            run(cap, Arm())
        else:
            print(__doc__)
    finally:
        cap.release()
