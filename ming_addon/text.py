"""The line's text as the model takes it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Upstream normalises Chinese and English text with FST grammars from pynini,
which has wheels for Linux only. Without them the model still reads Chinese
digits well; English numbers, symbols and units are spelled out with
upstream's own inflect-based rules, which upstream runs before the FSTs.
Upstream's sentence splitting and punctuation fixes run in MingTTS itself.
"""

from __future__ import annotations

import re

from ming_addon.ming.frontend.number_en import normalize_numbers


def prepare(text, language: str) -> str:
    """Whitespace collapsed; for English, numbers spelled out. '' when nothing is left."""
    text = ' '.join(str(text or '').split())
    if text and language == 'en':
        # The rules leave a space before the punctuation that followed a number.
        text = re.sub(r'\s+([,.;:!?])', r'\1', ' '.join(normalize_numbers(text).split()))
    return text
