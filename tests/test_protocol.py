"""The tts.synthesize request: which task each voice runs, and the frames it answers with.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

from ming_addon import voices

REPO = Path(__file__).resolve().parents[1]


def _terminal(frames):
    terminal = [f for f in frames if f.get('type') in ('result', 'error')]
    assert len(terminal) == 1, frames
    return terminal[0]


def test_clone_with_transcript_continues_from_the_clip(run, reference, tmp_path):
    out = tmp_path / 'line.wav'
    frames, fake = run({'text': '  Hola,   mundo ', 'voice': voices.CLONE, 'output_path': str(out),
                        'language': 'es-MX', 'voice_ref_audio': reference,
                        'voice_ref_text': 'Hello there', 'seed': 7})
    result = _terminal(frames)
    assert result['type'] == 'result', result
    assert fake.calls == [{'task': 'TTS', 'text': 'Hola, mundo', 'language_id': 'es',
                           'prompt_wav_path': reference, 'prompt_text': 'Hello there',
                           'use_spk_emb': True, 'seed': 7}]
    data = result['data']
    assert data['path'] == str(out) and data['sample_rate'] == 44100 and data['channels'] == 1
    with wave.open(str(out)) as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 44100)
        assert clip.getnframes() == 4410
    assert abs(data['duration_sec'] - 0.1) < 1e-6
    values = [f['value'] for f in frames if f['type'] == 'progress']
    assert values == sorted(values) and values[-1] == 1.0


def test_clone_without_transcript_clones_the_timbre_only(run, reference, tmp_path):
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav'),
                        'voice_ref_audio': reference})
    assert _terminal(frames)['type'] == 'result'
    call = fake.calls[0]
    assert call['prompt_wav_path'] == reference and call['use_spk_emb'] is True
    assert 'prompt_text' not in call


def test_english_numbers_are_spelled_out(run, reference, tmp_path):
    frames, fake = run({'text': 'It costs $12, 3 times.', 'voice': voices.CLONE, 'language': 'en',
                        'output_path': str(tmp_path / 'a.wav'), 'voice_ref_audio': reference})
    assert _terminal(frames)['type'] == 'result'
    assert fake.calls[0]['text'] == 'It costs twelve dollars, three times.'


def test_a_language_the_model_lacks_falls_back_to_english(run, reference, tmp_path):
    frames, fake = run({'text': 'Olá', 'voice': voices.CLONE, 'language': 'pt-BR',
                        'output_path': str(tmp_path / 'a.wav'), 'voice_ref_audio': reference})
    assert _terminal(frames)['type'] == 'result'
    assert fake.calls[0]['language_id'] == 'en'


def test_clone_falls_back_to_the_configured_clip(run, reference, tmp_path, monkeypatch):
    monkeypatch.setenv('MING_VOICE_REF_AUDIO', reference)
    monkeypatch.setenv('MING_VOICE_REF_TEXT', 'configured words')
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav')})
    assert _terminal(frames)['type'] == 'result'
    assert fake.calls[0]['prompt_text'] == 'configured words'


def test_the_configured_transcript_never_pairs_with_the_hosts_clip(run, reference, tmp_path, monkeypatch):
    monkeypatch.setenv('MING_VOICE_REF_TEXT', 'words of another clip')
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav'),
                        'voice_ref_audio': reference})
    assert _terminal(frames)['type'] == 'result'
    assert 'prompt_text' not in fake.calls[0]


def test_clone_without_any_clip_is_refused(run, tmp_path):
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav')})
    error = _terminal(frames)
    assert error['type'] == 'error' and error['code'] == 'bad_params'
    assert not fake.calls


def test_an_emotion_conditions_on_the_timbre_and_wins_over_speed(run, reference, tmp_path, monkeypatch):
    monkeypatch.setenv('MING_EMOTION', 'sad')
    monkeypatch.setenv('MING_SPEED', 'fast')
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav'),
                        'voice_ref_audio': reference, 'voice_ref_text': 'Hello there'})
    assert _terminal(frames)['type'] == 'result'
    call = fake.calls[0]
    assert call['task'] == 'EMOTION' and call['controls'] == {'emotion': '悲伤'}
    assert call['use_spk_emb'] is True and 'prompt_text' not in call


def test_speed_pitch_and_volume_are_one_basic_request(run, reference, tmp_path, monkeypatch):
    monkeypatch.setenv('MING_EMOTION', 'none')
    monkeypatch.setenv('MING_PITCH', 'low')
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav'),
                        'voice_ref_audio': reference})
    assert _terminal(frames)['type'] == 'result'
    call = fake.calls[0]
    assert call['task'] == 'BASIC' and call['controls'] == {'speed': '中速', 'pitch': '低', 'volume': '中'}


def test_a_built_in_voice_speaks_the_lines_language(run, tmp_path, monkeypatch):
    monkeypatch.setenv('MING_EMOTION', 'happy')  # built-in voices ignore it
    frames, fake = run({'text': 'Guten Tag', 'voice': 'ming-uk-george', 'language': 'de',
                        'output_path': str(tmp_path / 'a.wav')})
    assert _terminal(frames)['type'] == 'result'
    assert fake.calls == [{'task': 'IP', 'text': 'Guten Tag', 'language_id': 'de',
                           'controls': {'ip': 'UK_George'}, 'use_zero_spk_emb': True, 'seed': 0}]


def test_described_voice_makes_its_anchor_once_per_speaker(run, tmp_path):
    params = {'text': 'Bonjour', 'voice': 'ming-calm-man', 'language': 'fr-CA', 'seed': 11,
              'output_path': str(tmp_path / 'a.wav')}
    frames, fake = run(params)
    assert _terminal(frames)['type'] == 'result'
    anchor, line = fake.calls
    assert anchor == {'task': 'VOICE_DESIGN', 'text': voices.ANCHORS['fr'], 'language_id': 'fr',
                      'controls': {'style': voices.BY_ID['ming-calm-man'].description},
                      'use_zero_spk_emb': True, 'seed': 11}
    assert line['task'] == 'TTS' and line['text'] == 'Bonjour'
    assert line['prompt_text'] == voices.ANCHORS['fr']
    assert Path(line['prompt_wav_path']).is_file()

    frames, fake = run(dict(params, text='Bonsoir'))
    assert _terminal(frames)['type'] == 'result'
    assert [call['text'] for call in fake.calls[-1:]] == ['Bonsoir']
    assert len(fake.calls) == 3  # the anchor was reused

    frames, fake = run(dict(params, seed=12))  # another speaker, another voice
    assert len(fake.calls) == 5


def test_the_voice_you_describe_reads_the_setting(run, tmp_path, monkeypatch):
    frames, fake = run({'text': 'Hi', 'voice': voices.DESCRIBED, 'output_path': str(tmp_path / 'a.wav')})
    error = _terminal(frames)
    assert error['type'] == 'error' and error['code'] == 'bad_params'

    monkeypatch.setenv('MING_VOICE_DESCRIPTION', 'A child, cheerful voice')
    frames, fake = run({'text': 'Hi', 'voice': voices.DESCRIBED, 'output_path': str(tmp_path / 'a.wav')})
    assert _terminal(frames)['type'] == 'result'
    assert fake.calls[0]['controls'] == {'style': 'A child, cheerful voice'}


def test_unknown_voice_and_missing_fields(run, tmp_path):
    error = _terminal(run({'text': 'Hi', 'voice': 'qwen3-ryan', 'output_path': str(tmp_path / 'a.wav')})[0])
    assert error['code'] == 'unsupported_voice'
    assert _terminal(run({'text': '   ', 'voice': voices.CLONE, 'output_path': 'x.wav'})[0])['code'] == 'bad_params'
    assert _terminal(run({'text': 'Hi', 'voice': voices.CLONE})[0])['code'] == 'bad_params'


def test_cancel_stops_the_line_between_steps(run, reference, tmp_path):
    out = tmp_path / 'a.wav'
    frames, _fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(out),
                         'voice_ref_audio': reference}, cancel_at_step=3)
    error = _terminal(frames)
    assert error['type'] == 'error' and error['code'] == 'cancelled'
    assert not out.exists()


def test_the_process_says_hello_and_leaves_on_shutdown():
    env = dict(os.environ, PYTHONPATH=str(REPO))
    proc = subprocess.Popen([sys.executable, '-m', 'ming_addon'], cwd=REPO, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        hello = json.loads(proc.stdout.readline())
        proc.stdin.write(b'{"type":"ready","host":"test"}\n')
        proc.stdin.write(b'{"id":"x","type":"asr.transcribe","params":{}}\n')
        proc.stdin.flush()
        error = json.loads(proc.stdout.readline())
        proc.stdin.write(b'{"type":"shutdown"}\n')
        proc.stdin.flush()
        assert proc.wait(timeout=20) == 0
    finally:
        proc.kill()
    assert hello['type'] == 'hello' and hello['addon'] == 'ming'
    capability = hello['capabilities'][0]
    assert capability['task'] == 'tts.synthesize' and capability['voice_clone'] is True
    assert {v['id'] for v in capability['voices']} == set(voices.BY_ID)
    assert error == {'id': 'x', 'type': 'error', 'code': 'bad_params',
                     'message': "unsupported frame type 'asr.transcribe'", 'retryable': False}
