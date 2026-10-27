"""The licence audit's rules, on native file names seen in the three platforms' bundles.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('collect_licences', REPO / 'tools' / 'collect_licences.py')
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def _rule(name: str):
    for pattern, notice, component, source in collect._COMPILED:
        if pattern.search(name.lower()):
            return notice, component, source
    return None


@pytest.mark.parametrize('name, notice', [
    ('ming-addon', 'pyinstaller/COPYING.txt'),
    ('ming-addon.exe', 'pyinstaller/COPYING.txt'),
    ('libpython3.12.so.1.0', 'python/LICENSE.txt'),
    ('python312.dll', 'python/LICENSE.txt'),
    ('python3.dll', 'python/LICENSE.txt'),
    ('Python', 'python/LICENSE.txt'),
    ('_ssl.pyd', 'python/LICENSE.txt'),
    ('_uuid.cpython-312-x86_64-linux-gnu.so', 'python/LICENSE.txt'),
    ('libgcc_s.so.1', 'gcc-runtime/COPYING.RUNTIME'),
    ('libstdc++.so.6', 'gcc-runtime/COPYING.RUNTIME'),
    ('libgomp.so.1.0.0', 'gcc-runtime/COPYING.RUNTIME'),
    ('libquadmath-2284e583.so.0.0.0', 'libquadmath/COPYING.LIB'),
    ('libcrypto-3.dll', 'python-runtime-libs/openssl-LICENSE.txt'),
    ('libssl.3.dylib', 'python-runtime-libs/openssl-LICENSE.txt'),
    ('libffi-8.dll', 'python-runtime-libs/libffi-LICENSE'),
    ('libz.so.1', 'python-runtime-libs/zlib-LICENSE'),
    ('sqlite3.dll', 'python-runtime-libs/sqlite-NOTICE.txt'),
    ('libuuid.so.1', 'python-runtime-libs/libuuid-COPYING.BSD-3-Clause'),
    ('libc++.1.0.dylib', 'llvm-runtime/LICENSE.TXT'),
    ('VCRUNTIME140_THREADS.dll', 'msvc-runtime/NOTICE.txt'),
    ('MSVCP140_ATOMIC_WAIT.dll', 'msvc-runtime/NOTICE.txt'),
    ('VCOMP140.DLL', 'msvc-runtime/NOTICE.txt'),
    ('api-ms-win-crt-math-l1-1-0.dll', 'msvc-runtime/NOTICE.txt'),
])
def test_known_libraries_have_a_committed_notice(name, notice):
    found = _rule(name)
    assert found is not None and found[0] == notice, (name, found)
    assert (REPO / 'THIRD_PARTY_LICENSES' / notice).is_file() or notice in (
        'pyinstaller/COPYING.txt', 'python/LICENSE.txt'), notice  # those two are harvested at build time


@pytest.mark.parametrize('name', ['libmvec.so.1', 'libc.so.6', 'libreadline.so.8', 'libgdbm.so.6'])
def test_glibc_and_gpl_libraries_are_refused(name):
    found = _rule(name)
    assert found is not None and found[0] is None, (name, found)


def test_libquadmath_needs_its_source():
    assert _rule('libquadmath-2284e583-a9307bba.so.0.0.0')[2] == 'libquadmath'
    assert any(pattern.search('libquadmath-2284e583.so.0.0.0') for pattern, _ in collect.LGPL_BY_NAME)


def test_the_vendored_code_and_mpl_packages_are_accounted_for():
    notices = REPO / 'THIRD_PARTY_LICENSES'
    for name in ('LICENSE', 'x-transformers-LICENSE', 'diffusers-LICENSE', 'README.md'):
        assert (notices / 'ming-3.0-tts' / name).is_file(), name
    assert (REPO / 'ming_addon' / 'ming' / 'LICENSE').read_bytes() == (notices / 'ming-3.0-tts' / 'LICENSE').read_bytes()
    assert set(collect.COPYLEFT_OK) == {'certifi', 'tqdm'}
    assert (notices / 'MPL-SOURCES.txt').is_file()


def test_copyleft_detection():
    assert collect._COPYLEFT.search('LGPL-2.1-or-later')
    assert collect._COPYLEFT.search('MPL-2.0')
    assert not collect._COPYLEFT.search('BSD-3-Clause')
    assert collect._COPYLEFT_EXCEPTION.search('GPL-2.0-or-later WITH Bootloader-exception')
