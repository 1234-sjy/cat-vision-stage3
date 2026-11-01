import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from scripts.download_models import download_ranges


class DownloadTests(unittest.TestCase):
    def test_wrong_content_range_stops_after_one_request(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(self.headers.get("Range"))
                self.send_response(206)
                self.send_header("Content-Length", "16")
                self.send_header("Content-Range", "bytes 16-31/32")
                self.end_headers()
                self.wfile.write(b"x" * 16)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaisesRegex(RuntimeError, "Content-Range"):
                    download_ranges(f"http://127.0.0.1:{server.server_port}/weights",
                                    Path(folder) / "test-model.bin", 32,
                                    hashlib.sha256(b"x" * 32).hexdigest(), chunk_size=16)
            self.assertEqual(len(calls), 1)
        finally:
            server.shutdown()
            server.server_close()

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
