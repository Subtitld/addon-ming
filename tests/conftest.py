"""Shared fixtures: the request handler with a stand-in for the model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ming_addon import __main__ as addon  # noqa: E402
from ming_addon import audio, engine  # noqa: E402

SETTINGS = ('VOICE_DESCRIPTION', 'VOICE_REF_AUDIO', 'VOICE_REF_TEXT', 'MODEL_DIR', 'THREADS',
            'EMOTION', 'SPEED', 'PITCH', 'VOLUME', 'DEVICE')


class FakeModel:
    """Records each synthesis request and returns a short tone; runs the step callback like the real loop."""

    def __init__(self):
        self.calls: list[dict] = []
        self.loads = 0
        self.steps = 8

    def load(self, folder, device, threads=0, lean=True):
        self.loads += 1
        return engine.Loaded(tts=None, device=device, sample_rate=44100)

    def synthesize(self, loaded, *, seed=None, on_step=None, **request):
        self.calls.append(dict(request, seed=seed))
        for _ in range(self.steps):
            if on_step is not None:
                on_step()
        return (0.25 * np.sin(np.arange(4410) / 10.0)).astype(np.float32)


@pytest.fixture
def run(monkeypatch, tmp_path):
    """run(params) -> (frames, fake): one tts.synthesize request through the handler."""
    frames: list[dict] = []
    fake = FakeModel()
    monkeypatch.setattr(addon, '_emit', frames.append)
    monkeypatch.setattr(addon, '_loaded', None)
    monkeypatch.setattr(engine, 'weights_present', lambda folder: True)
    monkeypatch.setattr(engine, 'load', fake.load)
    monkeypatch.setattr(engine, 'synthesize', fake.synthesize)
    monkeypatch.setattr(engine, 'resolve_device', lambda choice: 'cpu')
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path / 'models'))
    monkeypatch.setenv('SUBTITLD_TEMP_DIR', str(tmp_path))
    for name in SETTINGS:
        monkeypatch.delenv(f'MING_{name}', raising=False)

    def _run(params: dict, cancel_at_step: int | None = None):
        state = addon._Request('req-1')
        if cancel_at_step is not None:
            original = fake.synthesize

            def cancelling(loaded, *, on_step=None, **kwargs):
                count = [0]

                def step():
                    if count[0] == cancel_at_step:
                        state.cancel.set()
                    count[0] += 1
                    on_step()
                return original(loaded, on_step=step, **kwargs)
            monkeypatch.setattr(engine, 'synthesize', cancelling)
        frames.clear()
        addon._handle_synthesize(state, params)
        return list(frames), fake
    return _run


@pytest.fixture
def reference(tmp_path):
    """A one-second WAV standing in for the speaker's clip."""
    path = tmp_path / 'speaker.wav'
    audio.write_wav(path, np.zeros(16000, dtype=np.float32), 16000)
    return str(path)
