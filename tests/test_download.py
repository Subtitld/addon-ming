"""The weights download: pinned files, resumed after an interruption, refused when they do not check out.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ming_addon import engine

FILES = {'config.json': b'{"architecture": "ming"}', 'model.safetensors': bytes(range(256)) * 400}


class _Hub(BaseHTTPRequestHandler):
    served: dict[str, bytes] = {}
    ignore_range = False
    requests: list[tuple[str, str | None]] = []

    def do_GET(self):  # noqa: N802
        name = self.path.rsplit('/', 1)[-1]
        data = self.served.get(name)
        self.requests.append((name, self.headers.get('Range')))
        if data is None or f'/resolve/{engine.MODEL_REVISION}/' not in self.path:
            self.send_error(404)
            return
        start = 0
        wanted = self.headers.get('Range')
        if wanted and not self.ignore_range:
            start = int(wanted.split('=', 1)[1].split('-', 1)[0])
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{len(data) - 1}/{len(data)}')
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def hub(monkeypatch, tmp_path):
    _Hub.served = dict(FILES)
    _Hub.ignore_range = False
    _Hub.requests = []
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Hub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv('HF_ENDPOINT', f'http://127.0.0.1:{server.server_address[1]}')
    monkeypatch.delenv('MING_MODEL_DIR', raising=False)
    monkeypatch.setattr(engine, 'MODEL_FILES', tuple(
        (name, len(data), hashlib.sha256(data).hexdigest()) for name, data in FILES.items()))
    yield tmp_path / 'Ming-3.0-tts-0.5B'
    server.shutdown()


def _never():
    return False


def test_downloads_every_file_and_remembers_it_checked_out(hub):
    seen = []
    engine.ensure_weights(hub, lambda done, total: seen.append((done, total)), _never)
    for name, data in FILES.items():
        assert (hub / name).read_bytes() == data
    assert engine.weights_present(hub)
    total = sum(len(d) for d in FILES.values())
    assert seen[-1] == (total, total)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)

    _Hub.requests.clear()
    engine.ensure_weights(hub, lambda *a: None, _never)
    assert _Hub.requests == []  # nothing fetched again


def test_an_interrupted_download_resumes(hub):
    hub.mkdir(parents=True)
    data = FILES['model.safetensors']
    (hub / 'model.safetensors.part').write_bytes(data[:30000])
    assert engine.missing_bytes(hub) == len(FILES['config.json']) + len(data) - 30000
    engine.ensure_weights(hub, lambda *a: None, _never)
    assert (hub / 'model.safetensors').read_bytes() == data
    assert ('model.safetensors', 'bytes=30000-') in _Hub.requests


def test_a_server_ignoring_the_range_still_gives_the_whole_file(hub):
    hub.mkdir(parents=True)
    (hub / 'model.safetensors.part').write_bytes(FILES['model.safetensors'][:500])
    _Hub.ignore_range = True
    engine.ensure_weights(hub, lambda *a: None, _never)
    assert (hub / 'model.safetensors').read_bytes() == FILES['model.safetensors']


def test_a_file_that_does_not_match_its_sha256_is_discarded(hub):
    _Hub.served['model.safetensors'] = bytes(len(FILES['model.safetensors']))
    with pytest.raises(engine.DownloadError, match='sha256'):
        engine.ensure_weights(hub, lambda *a: None, _never)
    assert not (hub / 'model.safetensors').exists()
    assert not (hub / 'model.safetensors.part').exists()
    assert not engine.weights_present(hub)


def test_a_missing_file_on_the_server_is_an_error(hub):
    del _Hub.served['config.json']
    with pytest.raises(engine.DownloadError, match='HTTP 404'):
        engine.ensure_weights(hub, lambda *a: None, _never)


def test_cancel_keeps_what_was_downloaded(hub):
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(engine.Cancelled):
        engine.ensure_weights(hub, lambda *a: None, cancel)
    assert not engine.weights_present(hub)


def test_a_folder_given_by_the_user_is_used_as_it_is(monkeypatch, tmp_path):
    monkeypatch.setenv('MING_MODEL_DIR', str(tmp_path))
    assert engine.model_dir() == tmp_path
    with pytest.raises(engine.DownloadError, match='lacks'):
        engine.ensure_weights(tmp_path, lambda *a: None, _never)
    for name, _size, _sha in engine.MODEL_FILES:
        (tmp_path / name).write_bytes(b'x')
    assert engine.weights_present(tmp_path)


def test_the_weights_go_to_the_hosts_model_cache(monkeypatch, tmp_path):
    monkeypatch.delenv('MING_MODEL_DIR', raising=False)
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    assert engine.model_dir() == tmp_path / 'ming' / 'Ming-3.0-tts-0.5B'
