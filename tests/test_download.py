import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from scripts.download_models import download_ranges


class DownloadTests(unittest.TestCase):
    def test_bounded_ranges_reassemble_exact_file(self):
        payload = bytes(range(91))

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                first, last = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                body = payload[first:last + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(payload)}")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / "test-model.bin"
                download_ranges(f"http://127.0.0.1:{server.server_port}/weights", target,
                                len(payload), hashlib.sha256(payload).hexdigest(), chunk_size=16)
                self.assertEqual(target.read_bytes(), payload)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
