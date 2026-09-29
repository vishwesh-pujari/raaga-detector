import functools
import http.server
import re
import threading
import zipfile

import pytest

from raaga.data import remote


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server that understands ``Range: bytes=N-`` (like Zenodo does)."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        m = re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        with open(path, "rb") as f:
            data = f.read()
        start = int(m.group(1)) if m else 0
        if start >= len(data):
            self.send_response(416)
            self.end_headers()
            return
        self.send_response(206 if m else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


@pytest.fixture
def server(tmp_path):
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    with zipfile.ZipFile(srv_dir / "data.zip", "w") as z:
        z.writestr("Root/Hindustani/a.pitch", "x" * 1000)
        z.writestr("Root/Carnatic/b.pitch", "y" * 1000)
        z.writestr("__MACOSX/Root/Hindustani/._a.pitch", "junk")
    handler = functools.partial(RangeHandler, directory=str(srv_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/data.zip", srv_dir / "data.zip"
    httpd.shutdown()


def _want(name):
    return "__MACOSX" not in name and "/Hindustani/" in name


def test_fetch_extracts_only_wanted_and_cleans_up(server, tmp_path):
    url, zpath = server
    dest = tmp_path / "out"
    out = remote.fetch_members(url, dest, _want, md5=remote.md5sum(zpath), verbose=False)
    assert [p.name for p in out] == ["a.pitch"]
    assert not (dest / "Root" / "Carnatic").exists()
    assert not (dest / "_download.zip").exists()
    # second call is a no-op (marker file), even with the server gone
    assert [p.name for p in remote.fetch_members("http://127.0.0.1:1/none.zip", dest, _want, verbose=False)] == ["a.pitch"]


def test_bad_checksum_is_rejected(server, tmp_path):
    url, _ = server
    with pytest.raises(IOError):
        remote.fetch_members(url, tmp_path / "out", _want, md5="0" * 32, verbose=False)
    assert not (tmp_path / "out" / "_download.zip").exists()


def test_resume_partial_download(server, tmp_path):
    url, zpath = server
    dest = tmp_path / "f.zip"
    dest.with_name("f.zip.part").write_bytes(zpath.read_bytes()[:100])  # an interrupted earlier attempt
    remote.download_file(url, dest, verbose=False)
    assert dest.read_bytes() == zpath.read_bytes()
