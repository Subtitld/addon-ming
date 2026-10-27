"""Reading and writing WAV with the standard library.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Upstream reads its reference clip with `torchaudio.load(backend="soundfile")`,
which needs soundfile and libsndfile, and recent torchaudio routes it through
torchcodec and FFmpeg's libraries instead. Subtitld's clips are PCM WAV; the
add-on converts anything else with ffmpeg first (see __main__.py), so a PCM
reader is all synthesis needs.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class UnreadableAudio(Exception):
    """Not a PCM WAV file this reader takes."""


def read_wav(path) -> tuple[np.ndarray, int]:
    """(samples as float32 in [-1, 1], shape (frames, channels); sample rate)."""
    try:
        with wave.open(str(path), 'rb') as clip:
            width, channels, rate = clip.getsampwidth(), clip.getnchannels(), clip.getframerate()
            data = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError) as exc:
        raise UnreadableAudio(f'{path}: {exc}') from exc
    if width == 1:
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = (np.where(value >= 1 << 23, value - (1 << 24), value) / float(1 << 23)).astype(np.float32)
    elif width == 4:
        samples = (np.frombuffer(data, dtype='<i4').astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        raise UnreadableAudio(f'{path}: {8 * width}-bit samples')
    return samples.reshape(-1, channels), rate


def load_tensor(path):
    """torchaudio.load for PCM WAV: a float32 (channels, frames) tensor and the rate."""
    import torch
    samples, rate = read_wav(path)
    return torch.from_numpy(np.ascontiguousarray(samples.T)), rate


def write_wav(path, wav, sample_rate: int) -> float:
    """16-bit mono WAV, which the host's dubbing pipeline expects. Returns seconds."""
    samples = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (samples * 32767.0).round().astype('<i2')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(sample_rate))
        out.writeframes(pcm.tobytes())
    return len(samples) / float(sample_rate)
