"""Off-board HTTP tests using a synthetic JPEG on an ephemeral local port."""

import http.client
import json
import threading
import unittest

from cri.libraries import cv2, np
from cri.vision_stream import StreamState, make_server


class StreamTests(unittest.TestCase):
    def test_browser_stream_positions_and_controls(self):
        state = StreamState()
        ok, encoded = cv2.imencode('.jpg', np.zeros((32, 32, 3), np.uint8))
        self.assertTrue(ok)
        state.publish(encoded.tobytes(), {'status': 'Grid detected', 'cubes': []})
        server = make_server('127.0.0.1', 0, state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        try:
            client.request('GET', '/')
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn(b'/stream.mjpg', response.read())
            client.request('GET', '/positions')
            response = client.getresponse()
            self.assertEqual(json.loads(response.read())['status'], 'Grid detected')
            for action in ('lock', 'unlock'):
                client.request('POST', '/' + action)
                response = client.getresponse()
                self.assertEqual(response.status, 202)
                response.read()
            self.assertEqual(list(state.commands), ['lock', 'unlock'])
            client.request('POST', '/unlock', headers={'Origin': 'http://unrelated.example'})
            response = client.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
            client.request('GET', '/stream.mjpg')
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.readline(), b'--frame\r\n')
            self.assertEqual(response.readline(), b'Content-Type: image/jpeg\r\n')
            length = int(response.readline().split(b':')[1])
            self.assertEqual(response.readline(), b'\r\n')
            image = cv2.imdecode(np.frombuffer(response.read(length), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape, (32, 32, 3))
        finally:
            state.stop()
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
