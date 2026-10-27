# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld Ming-3.0-tts add-on.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Produces `dist/ming-addon/` (launcher + `_internal/`): CPU PyTorch, the
vendored Ming-3.0-tts inference code and what it imports. The weights are not
in it; they download on the first synthesis. The release step copies the folder to `bin/` inside
the archive, next to manifest.json.

Run from the add-on root, with the dependencies installed as the README says:

    pyinstaller pyinstaller.spec --noconfirm
    dist/ming-addon/ming-addon --self-test
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

SPEC_ROOT = Path(SPECPATH).resolve()

REQUIRED = ('torch', 'torchaudio', 'transformers', 'tokenizers', 'safetensors', 'onnxruntime',
            'inflect', 'certifi')
_missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
if _missing:
    raise SystemExit('pyinstaller.spec: not installed in the build env: ' + ', '.join(_missing))

hiddenimports = collect_submodules('ming_addon')
# transformers resolves Qwen2's model and tokenizer through its lazy module
# table, which PyInstaller cannot follow. Take its core and Qwen2 only.
hiddenimports += collect_submodules(
    'transformers',
    filter=lambda name: ((not name.startswith('transformers.models.')
                          or name.startswith('transformers.models.qwen2.'))
                         and 'testing_utils' not in name),
    on_error='ignore',
)

# transformers checks its dependencies' versions through their metadata.
datas = copy_metadata('transformers', recursive=True)
for name in ('torch', 'torchaudio', 'safetensors', 'tokenizers', 'onnxruntime', 'inflect'):
    datas += copy_metadata(name)


# ---------------------------------------------------------------------------
# libexpat sanity check (from addon-whispercpp): a stale expat on the build
# host makes the frozen binary die before Python starts.
# ---------------------------------------------------------------------------
import os
import subprocess
import sys

_EXPAT_SYMBOL = 'XML_SetReparseDeferralEnabled'


def _exports_symbol(path: str, symbol: str) -> bool:
    try:
        out = subprocess.run(['nm', '-D', '--defined-only', path],
                             capture_output=True, text=True, timeout=30)
        return symbol in out.stdout
    except Exception:
        return True  # no nm (macOS/Windows): assume fine rather than break the build


def _fix_libexpat(entries):
    if not sys.platform.startswith('linux'):
        return entries
    out = []
    for entry in entries:
        name = entry[0] if entry else ''
        src = entry[1] if len(entry) > 1 else ''
        if not os.path.basename(name).startswith('libexpat.so') or _exports_symbol(src, _EXPAT_SYMBOL):
            out.append(entry)
            continue
        for cand in ('/usr/lib/x86_64-linux-gnu/libexpat.so.1', '/usr/lib64/libexpat.so.1',
                     '/lib/x86_64-linux-gnu/libexpat.so.1'):
            if os.path.exists(cand) and _exports_symbol(cand, _EXPAT_SYMBOL):
                print(f'WARNING: {src} lacks {_EXPAT_SYMBOL}; substituting {cand}')
                out.append((name, cand) + tuple(entry[2:]))
                break
        else:
            print(f'ERROR: bundled libexpat lacks {_EXPAT_SYMBOL}; the frozen binary will not start.')
            out.append(entry)
    return out


a = Analysis(
    [str(SPEC_ROOT / 'ming_addon' / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # inflect checks its arguments with typeguard, which reads the functions'
    # source: ship inflect's .py files next to the compiled ones.
    module_collection_mode={'inflect': 'pyz+py'},
    excludes=[
        # What upstream's full requirements bring and the vendored copy does
        # without. Left out even when installed, so none of it (nor
        # soundfile's LGPL libsndfile) can slip back into the bundle.
        'diffusers', 'x_transformers', 'accelerate', 'loguru', 'pynini', 'torchcodec',
        'gradio', 'datasets',
        'librosa', 'soundfile', 'soxr', 'audioread', 'scipy', 'sklearn', 'numba', 'llvmlite',
        'pooch', 'lazy_loader', 'joblib', 'threadpoolctl',
        'matplotlib', 'pandas', 'IPython', 'jupyter', 'notebook',
        'torch.utils.tensorboard', 'tensorboard', 'triton',
        'tensorflow', 'jax', 'flax', 'onnx', 'onnxruntime.quantization',
        'PySide6', 'PyQt6', 'PyQt5', 'tkinter', 'Tkinter', '_tkinter',
        'pytest', '_pytest', 'pluggy', 'iniconfig', 'setuptools', 'pkg_resources',
        # GPL (readline) and ncurses: the standard modules that link them are
        # for interactive terminals, which the add-on never has.
        'readline', 'curses', '_curses', '_curses_panel',
    ],
    noarchive=False,
)

a.binaries = _fix_libexpat(a.binaries)
# glibc's vector maths library, which PyTorch links, is part of every glibc
# system and must come from it, like libc and libm (which PyInstaller already
# leaves out): a copy from the build machine would also bring glibc's LGPL.
a.binaries = [entry for entry in a.binaries if not os.path.basename(entry[0]).startswith('libmvec.so')]
# Intel's Level Zero GPU loader: torch names it through ctypes for its XPU
# backend, so PyInstaller copies it from the build machine when it has one.
# The CPU build never loads it.
a.binaries = [entry for entry in a.binaries if not os.path.basename(entry[0]).startswith('libze_loader.so')]

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ming-addon',
    debug=False,
    bootloader_ignore_signals=False,
    # strip=False: stripping the native libraries saves little and breaks
    # macOS code signatures.
    strip=False,
    upx=False,
    console=True,  # stdio is the protocol channel
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ming-addon',
)
