"""Browser MJPEG transport for preview_vision.py (Python standard library)."""

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

if __package__:
    from .libraries import cv2
else:
    from libraries import cv2


_PAGE = b"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>DOFBOT camera</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font:16px sans-serif;max-width:900px;margin:24px auto;padding:0 16px}
img{max-width:100%;background:#222}button{padding:10px;margin:8px 8px 8px 0}
pre{white-space:pre-wrap}</style>
<h1>DOFBOT camera</h1><img src="/stream.mjpg" alt="Camera feed">
<p>Lock the empty grid, then place cubes. Keep the camera and board fixed.</p>
<button onclick="act('lock')">Lock grid</button>
<button onclick="act('unlock')">Unlock grid</button>
<p id="status">Waiting for camera...</p><pre id="positions"></pre>
<script>
async function act(action){
  try { const r=await fetch('/'+action,{method:'POST'});
    if(!r.ok) throw Error('Control request failed');
  } catch(e){document.getElementById('status').textContent=e.message;}
}
async function update(){
  try { const r=await fetch('/positions'); if(!r.ok) throw Error('Disconnected');
    const data=await r.json();
    document.getElementById('status').textContent=data.status;
    document.getElementById('positions').textContent=JSON.stringify(data,null,2);
  } catch(e){document.getElementById('status').textContent='Camera disconnected';}
  setTimeout(update,500);
} update();
</script></html>"""


class StreamState:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = None
        self.sequence = 0
        self.positions = {"status": "Waiting for camera"}
        self.commands = deque(maxlen=8)
        self.running = True

    def publish(self, jpeg, positions):
        with self.condition:
            self.jpeg = jpeg
            self.positions = positions
            self.sequence += 1
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()


def make_server(host, port, state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, content_type, payload):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            try:
                if self.path == "/":
                    self.reply(200, "text/html; charset=utf-8", _PAGE)
                elif self.path == "/positions":
                    with state.condition:
                        payload = json.dumps(state.positions).encode("utf-8")
                    self.reply(200, "application/json", payload)
                elif self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    previous = -1
                    while True:
                        with state.condition:
                            state.condition.wait_for(
                                lambda: not state.running or state.sequence != previous, timeout=5)
                            if not state.running:
                                break
                            if state.jpeg is None or state.sequence == previous:
                                continue
                            jpeg, previous = state.jpeg, state.sequence
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                else:
                    self.reply(404, "text/plain", b"Not found")
            except (ConnectionError, OSError):
                pass  # A browser tab disconnected; keep the capture loop running.

        def do_POST(self):
            # Only accept same-origin button requests, including over SSH tunnels.
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.reply(403, "text/plain", b"Origin rejected")
                return
            if self.path not in ("/lock", "/unlock"):
                self.reply(404, "text/plain", b"Not found")
                return
            with state.condition:
                state.commands.append(self.path[1:])
            self.reply(202, "text/plain", b"Queued")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve_camera(capture, detector, positions_for_scene, host="127.0.0.1", port=8080):
    state = StreamState()
    server = make_server(host, port, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("Camera stream: http://{}:{} (Ctrl+C to stop)".format(host, port), flush=True)
    fixed_grid = None
    message = ""
    try:
        while True:
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame read failed.")
            with state.condition:
                commands = list(state.commands)
                state.commands.clear()
            # Commands operate on this frame, before publishing positions.
            for command in commands:
                if command == "unlock":
                    fixed_grid = None
                    message = ""
                else:
                    candidate = detector.detect_grid(frame)
                    if candidate is not None:
                        fixed_grid = candidate
                        message = ""
                    else:
                        message = "Lock failed: clear the board and try again."
            scene = detector.detect(frame, grid=fixed_grid)
            annotated = detector.draw(frame, scene)
            status = message or ("Grid locked" if fixed_grid is not None else (
                "Grid detected" if scene.grid is not None else "Grid not found"))
            cv2.putText(annotated, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 180, 0) if scene.grid is not None else (0, 0, 255), 2)
            ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError("Could not encode camera frame.")
            positions = positions_for_scene(scene)
            positions["status"] = status
            positions["grid_locked"] = fixed_grid is not None
            state.publish(jpeg.tobytes(), positions)
            time.sleep(max(0, 0.1 - (time.monotonic() - started)))
    finally:
        state.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
