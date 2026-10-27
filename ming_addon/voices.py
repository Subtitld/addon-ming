"""The voices the add-on offers, the languages Ming-3.0-tts speaks, and its controls.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Three kinds of voice:

- `ming-clone` clones the speaker from the video: the host hands over a clip
  of their speech, and its transcript when it has one.
- Built-in voices are speakers the model was trained on (upstream's "IP"
  voices). They are the same on every line, and speak every language.
- The others are made from a description. Ming-3.0-tts makes a voice up from
  a few words, but a different one each time, and a dub needs the same voice
  on every line. So a described voice is made once per speaker, as a fixed
  sentence in the line's language (an "anchor"), cached, and every line
  clones the anchor.

Emotion and speed, pitch and volume are upstream's EMOTION and BASIC tasks.
They apply to cloned and described voices, which they condition on the
speaker's timbre alone, as upstream's examples do; and one task at a time,
so an emotion, when set, wins over the other three.
"""

from __future__ import annotations

from dataclasses import dataclass

CLONE = 'ming-clone'
DESCRIBED = 'ming-described'


@dataclass(frozen=True)
class Voice:
    id: str
    display_name: str
    ip: str = ''            # a built-in voice's name in the model
    description: str = ''   # a preset described voice


VOICES = (
    Voice(CLONE, "Speaker's own voice (cloned from the video)"),
    Voice('ming-us-emma', 'Emma (American woman)', ip='US_Emma'),
    Voice('ming-uk-george', 'George (British man)', ip='UK_George'),
    Voice('ming-uk-charlotte', 'Charlotte (British woman)', ip='UK_Charlotte'),
    Voice('ming-ling-xiaorou', 'Ling Xiaorou (Chinese woman, clear and patient)', ip='灵小柔'),
    Voice('ming-ling-kai', 'Ling Kai (Chinese man, warm late-night radio)', ip='灵楷'),
    Voice('ming-warm-woman', 'Warm woman',
          description='An adult woman, warm and friendly voice, clear and natural.'),
    Voice('ming-older-woman', 'Gentle older woman',
          description='An elderly woman, soft and gentle voice, slightly husky, unhurried.'),
    Voice('ming-calm-man', 'Calm man',
          description='A middle-aged man, deep and calm voice, steady pace.'),
    Voice('ming-young-man', 'Energetic young man',
          description='A young man, clear and energetic voice.'),
    Voice('ming-narrator', 'Narrator',
          description='A professional documentary narrator, rich and articulate male voice, measured pace.'),
    Voice(DESCRIBED, 'Voice you describe (in Configure)'),
)
BY_ID = {voice.id: voice for voice in VOICES}

# The seven languages of the model card, as BCP-47 primary subtags. Each is
# passed to the model as its target language.
LANGUAGES = ('zh', 'en', 'ja', 'ko', 'de', 'fr', 'es')

# Anchor sentences: about six seconds of plain speech, worded so that no form
# depends on the speaker's gender.
ANCHORS = {
    'en': 'Good morning. Today we have a long road ahead of us, so it would be wise to set off early, '
          'before the rain begins.',
    'es': 'Buenos días. Hoy tenemos un largo camino por delante, así que conviene salir temprano, '
          'antes de que empiece a llover.',
    'fr': "Bonjour. Aujourd'hui, nous avons une longue route devant nous, alors mieux vaut partir tôt, "
          'avant que la pluie ne commence.',
    'de': 'Guten Morgen. Heute liegt ein langer Weg vor uns, also brechen wir besser früh auf, '
          'bevor der Regen beginnt.',
    'zh': '早上好。今天我们还有很长的路要走，所以最好早点出发，趁着还没下雨。',
    'ja': 'おはようございます。今日は長い道のりになりそうなので、雨が降り出す前に早めに出発しましょう。',
    'ko': '좋은 아침이에요. 오늘은 갈 길이 멀어서, 비가 오기 전에 일찍 출발하는 게 좋겠어요.',
}

# Settings value -> the model's own control value (upstream's tts_tasks).
EMOTIONS = {
    'happy': '高兴', 'sad': '悲伤', 'angry': '愤怒', 'fearful': '恐惧',
    'disgusted': '厌恶', 'surprised': '惊讶', 'neutral': '中性',
}
SPEEDS = {'slow': '慢速', 'normal': '中速', 'fast': '快速'}
LEVELS = {'low': '低', 'normal': '中', 'high': '高'}


def primary_language(tag) -> str:
    """`pt-BR` -> `pt`; '' when there is none."""
    return str(tag or '').strip().lower().replace('_', '-').split('-', 1)[0]


def target_language(tag) -> str:
    """The line's language if the model speaks it, else English."""
    language = primary_language(tag)
    return language if language in LANGUAGES else 'en'


def controls_for(options: dict) -> tuple[str, dict]:
    """(task, controls) for the emotion/speed settings; ('', {}) when all are default."""
    emotion = EMOTIONS.get(str(options.get('emotion') or '').lower())
    if emotion:
        return 'EMOTION', {'emotion': emotion}
    speed = SPEEDS.get(str(options.get('speed') or '').lower(), SPEEDS['normal'])
    pitch = LEVELS.get(str(options.get('pitch') or '').lower(), LEVELS['normal'])
    volume = LEVELS.get(str(options.get('volume') or '').lower(), LEVELS['normal'])
    if (speed, pitch, volume) == (SPEEDS['normal'], LEVELS['normal'], LEVELS['normal']):
        return '', {}
    return 'BASIC', {'speed': speed, 'pitch': pitch, 'volume': volume}


def manifest_voices() -> list[dict]:
    """The voice list as manifest.json and the hello frame carry it."""
    out = []
    for voice in VOICES:
        entry = {'id': voice.id, 'language': '*', 'display_name': voice.display_name}
        if voice.id == CLONE:
            entry['requires'] = ['voice_ref_audio']
        out.append(entry)
    return out
