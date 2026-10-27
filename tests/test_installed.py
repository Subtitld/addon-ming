"""The frozen binary, unpacked from its release archive the way Subtitld installs it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

$MING_TEST_BINARY names the binary; without it these tests skip. The
release workflow sets it on every platform. The weights are 2.99 GB, too
big for CI, so CI proves the bundle with --self-test (the whole inference
stack imported, the speaker-encoder features and the vocoder run) and the protocol. The tests
marked `weights` also synthesize and cancel for real; they run when
$MING_TEST_MODELS_DIR holds the weights (as $SUBTITLD_MODELS_DIR would):

    MING_TEST_BINARY=... MING_TEST_MODELS_DIR=~/.cache/subtitld/models pytest tests/test_installed.py
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import wave
from pathlib import Path

import pytest

BINARY = os.environ.get('MING_TEST_BINARY', '')
MODELS = os.environ.get('MING_TEST_MODELS_DIR', '')
pytestmark = pytest.mark.skipif(not BINARY, reason='MING_TEST_BINARY is not set')
needs_weights = pytest.mark.skipif(not MODELS, reason='MING_TEST_MODELS_DIR is not set')


class Addon:
    """The binary on the protocol, with a watchdog so a hang fails instead of stalling CI."""

    def __init__(self, tmp_path: Path, timeout: float):
        env = dict(os.environ, SUBTITLD_TEMP_DIR=str(tmp_path),
                   SUBTITLD_MODELS_DIR=MODELS or str(tmp_path / 'models'))
        self.proc = subprocess.Popen([BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env)
        self.timer = threading.Timer(timeout, self.proc.kill)
        self.timer.start()

    def send(self, frame: dict) -> None:
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, 'the add-on closed its output (crashed or timed out)'
        return json.loads(line)

    def until_terminal(self, req_id: str, on_progress=None) -> dict:
        while True:
            frame = self.read()
            if frame.get('id') != req_id:
                continue
            if frame['type'] in ('result', 'error'):
                return frame
            if on_progress is not None:
                on_progress(frame)

    def close(self) -> int:
        try:
            self.send({'type': 'shutdown'})
        except (BrokenPipeError, OSError):
            pass
        try:
            return self.proc.wait(timeout=15)
        finally:
            self.timer.cancel()
            self.proc.kill()


def test_version():
    out = subprocess.run([BINARY, '--version'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.startswith('ming-addon ')


def test_self_test_runs_the_inference_stack():
    out = subprocess.run([BINARY, '--self-test'], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report['ok'] is True and report['vocoder_samples'] > 0 and report['fbank'][1] == 80


def test_handshake_and_refusals(tmp_path):
    addon = Addon(tmp_path, timeout=120)
    try:
        hello = addon.read()
        assert hello['type'] == 'hello' and hello['addon'] == 'ming'
        addon.send({'type': 'ready', 'host': 'test'})
        addon.send({'id': 'a', 'type': 'tts.synthesize',
                    'params': {'text': 'Hi', 'voice': 'nope', 'output_path': str(tmp_path / 'a.wav')}})
        assert addon.until_terminal('a')['code'] == 'unsupported_voice'
        addon.send({'id': 'b', 'type': 'tts.synthesize',
                    'params': {'text': 'Hi', 'voice': 'ming-clone', 'output_path': str(tmp_path / 'b.wav')}})
        assert addon.until_terminal('b')['code'] == 'bad_params'
    finally:
        assert addon.close() == 0


@pytest.mark.weights
@needs_weights
def test_synthesizes_a_described_voice_then_clones_it(tmp_path):
    addon = Addon(tmp_path, timeout=1800)
    try:
        addon.read()
        out = tmp_path / 'line.wav'
        addon.send({'id': 'a', 'type': 'tts.synthesize', 'params': {
            'text': 'Bonjour. On commence.', 'voice': 'ming-warm-woman', 'language': 'fr',
            'seed': 5, 'output_path': str(out)}})
        result = addon.until_terminal('a')
        assert result['type'] == 'result', result
        assert result['data']['sample_rate'] == 44100
        with wave.open(str(out)) as clip:
            assert clip.getframerate() == 44100 and clip.getnframes() > 44100 // 2

        clone = tmp_path / 'clone.wav'
        addon.send({'id': 'b', 'type': 'tts.synthesize', 'params': {
            'text': 'À bientôt.', 'voice': 'ming-clone', 'language': 'fr', 'seed': 5,
            'output_path': str(clone), 'voice_ref_audio': str(out), 'voice_ref_text': 'Bonjour. On commence.'}})
        assert addon.until_terminal('b')['type'] == 'result'
        assert clone.stat().st_size > 1000
    finally:
        assert addon.close() == 0


@pytest.mark.weights
@needs_weights
def test_cancel_stops_a_line_while_it_is_spoken(tmp_path):
    addon = Addon(tmp_path, timeout=1800)
    try:
        addon.read()
        cancelled = []

        def on_progress(frame):
            if frame.get('message') == 'Speaking...' and not cancelled:
                cancelled.append(frame['value'])
                addon.send({'type': 'cancel', 'target': 'a'})

        addon.send({'id': 'a', 'type': 'tts.synthesize', 'params': {
            'text': 'This sentence is long enough to be stopped well before its end, ' * 3,
            'voice': 'ming-uk-george', 'language': 'en', 'output_path': str(tmp_path / 'a.wav')}})
        error = addon.until_terminal('a', on_progress)
        assert cancelled and error['type'] == 'error' and error['code'] == 'cancelled'
        assert not (tmp_path / 'a.wav').exists()
    finally:
        assert addon.close() == 0
