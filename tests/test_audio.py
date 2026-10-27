"""The standard-library WAV reader and writer.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from ming_addon import audio

SR = 16000


def _tone(n, freq, amplitude):
    return (amplitude * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


def _write(path, samples: np.ndarray, width: int, channels: int, rate: int) -> None:
    scaled = np.clip(samples, -1, 1)
    if width == 1:
        data = ((scaled * 127).round() + 128).astype(np.uint8).tobytes()
    elif width == 2:
        data = (scaled * 32767).round().astype('<i2').tobytes()
    elif width == 3:
        ints = (scaled * (2 ** 23 - 1)).round().astype('<i4')
        data = b''.join(int(v).to_bytes(4, 'little', signed=True)[:3] for v in ints.reshape(-1))
    else:
        data = (scaled * (2 ** 31 - 1)).round().astype('<i4').tobytes()
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(data)


@pytest.mark.parametrize('width', [1, 2, 3, 4])
def test_reads_pcm_wav_of_every_width(tmp_path, width):
    left, right = _tone(800, 200, 0.5), _tone(800, 300, -0.25)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / f'{width}.wav'
    _write(path, stereo, width, 2, SR)
    samples, rate = audio.read_wav(path)
    assert rate == SR and samples.shape == (800, 2) and samples.dtype == np.float32
    tolerance = 1.5 / 128 if width == 1 else 1e-4
    assert np.max(np.abs(samples - stereo)) < tolerance
    tensor, rate = audio.load_tensor(path)
    assert rate == SR and tuple(tensor.shape) == (2, 800)
    assert np.max(np.abs(tensor.numpy().T - stereo)) < tolerance


def test_writes_16_bit_mono_and_clips(tmp_path):
    path = tmp_path / 'out' / 'line.wav'
    seconds = audio.write_wav(path, np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32), 44100)
    assert abs(seconds - 5 / 44100) < 1e-9
    with wave.open(str(path)) as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 44100)
    samples, rate = audio.read_wav(path)
    assert rate == 44100
    assert np.allclose(samples[:, 0], [0.0, 0.5, -0.5, 1.0, -1.0], atol=1e-4)


def test_what_it_cannot_read_is_reported(tmp_path):
    path = tmp_path / 'not.wav'
    path.write_bytes(b'ID3 an mp3, not a wav')
    with pytest.raises(audio.UnreadableAudio):
        audio.read_wav(path)
