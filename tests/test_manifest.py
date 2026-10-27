"""manifest.json agrees with the code: voices, languages, settings, version, model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ming_addon import __version__, engine, voices
from ming_addon import __main__ as addon
from ming_addon.ming import tts_tasks

REPO = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO / 'manifest.json').read_text(encoding='utf-8'))
FIELDS = {field['key']: field for field in MANIFEST['config_schema']['fields']}


def test_identity_and_version():
    assert MANIFEST['id'] == 'ming'
    assert MANIFEST['executable'] == 'bin/ming-addon'
    assert MANIFEST['tasks'] == ['tts.synthesize']
    pyproject = (REPO / 'pyproject.toml').read_text(encoding='utf-8')
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == __version__
    assert MANIFEST['version'] == __version__ == addon.ADDON_VERSION
    assert re.fullmatch(r'\d\d\.\d\d?\.\d+', __version__)  # YY.MM.N, like Subtitld


def test_voices_and_languages_match_the_code():
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    clone = [v for v in MANIFEST['voices'] if v.get('requires')]
    assert [v['id'] for v in clone] == [voices.CLONE]
    for voice in voices.VOICES:
        if voice.id not in (voices.CLONE, voices.DESCRIBED):
            assert bool(voice.ip) != bool(voice.description), voice.id


def test_languages_and_controls_are_the_models_own():
    assert set(voices.LANGUAGES) == {'zh', 'en'} | set(tts_tasks.MULTILINGUAL_LANGUAGE_IDS)
    assert set(voices.ANCHORS) == set(voices.LANGUAGES)
    assert set(voices.EMOTIONS.values()) == set(tts_tasks.EMOTION_VALUES)
    assert set(voices.SPEEDS.values()) == set(tts_tasks.BASIC_ATTRIBUTE_VALUES['speed'])
    assert set(voices.LEVELS.values()) == set(tts_tasks.BASIC_ATTRIBUTE_VALUES['pitch'])
    assert voices.target_language('ko-KR') == 'ko'
    assert voices.target_language('pt') == 'en'
    assert voices.target_language('') == 'en'


def test_every_setting_the_code_reads_is_declared():
    read = set(addon._options())
    assert read <= set(FIELDS)
    assert 'skip_ref_text' in FIELDS  # read by the host, not the add-on
    assert FIELDS['skip_ref_text']['type'] == 'bool'


def test_select_settings_offer_what_the_code_maps():
    assert {o['value'] for o in FIELDS['emotion']['options']} == {'none'} | set(voices.EMOTIONS)
    assert {o['value'] for o in FIELDS['speed']['options']} == set(voices.SPEEDS)
    for key in ('pitch', 'volume'):
        assert {o['value'] for o in FIELDS[key]['options']} == set(voices.LEVELS)
    defaults = {key: FIELDS[key]['default'] for key in ('emotion', 'speed', 'pitch', 'volume')}
    assert voices.controls_for(defaults) == ('', {})


def test_setting_defaults_are_valid():
    for key, field in FIELDS.items():
        if field['type'] == 'select':
            assert field['default'] in [o['value'] for o in field['options']], key
        if field['type'] in ('int', 'number'):
            assert field['min'] <= field['default'] <= field['max'], key


def test_the_code_clamps_threads_to_the_declared_range(monkeypatch):
    field = FIELDS['threads']
    monkeypatch.setenv('MING_THREADS', str(field['max'] * 10 + 1))
    assert addon._options()['threads'] == field['max']
    monkeypatch.setenv('MING_THREADS', str(field['min'] - 1))
    assert addon._options()['threads'] == field['min']


def test_the_model_entry_is_the_pinned_revision():
    model = MANIFEST['models'][0]
    assert model['repo'] == engine.MODEL_REPO
    assert model['revision'] == engine.MODEL_REVISION
    assert model['size_mb'] == round(engine.MODEL_BYTES / 2**20)
    assert model['bundled'] is False
    assert MANIFEST['disk_required_mb'] > model['size_mb']


def test_cpu_lists_parse():
    assert engine._cpu_list('0-3,8,10-11\n') == [0, 1, 2, 3, 8, 10, 11]
