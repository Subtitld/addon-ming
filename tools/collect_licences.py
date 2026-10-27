#!/usr/bin/env python3
"""Gather THIRD_PARTY_LICENSES/ for a release archive, and refuse native code without a notice.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The frozen bundle carries CPython, the PyInstaller bootloader and the Python
distributions it was frozen from (PyTorch, transformers, ONNX Runtime, NumPy...):
those whose modules are in the executable's archive or whose files are in
_internal/. Each one's licence files, as its wheel ships them, are copied to
THIRD_PARTY_LICENSES/<distribution>/; ONNX Runtime's include the notices
of the libraries it bundles. A distribution that ships no licence file gets
its declared licence recorded instead, and the build fails if it declares
none either. Tools used only to build or test are left out.

The notices committed in THIRD_PARTY_LICENSES/ (the GCC runtime, libquadmath,
the libraries CPython links, Microsoft's runtime) are copied as they are.

Then every native file in the bundle (ELF, Mach-O or PE, whatever its name)
must belong to an installed distribution, by its RECORD or, for the copies
PyInstaller puts next to the executable, by name; or match a rule in RULES,
whose notice must be in the output. Anything else fails the build: a library
new to the bundle has to come with its licence before it ships. A distribution
under a copyleft licence fails it too, unless it is in COPYLEFT_OK. Each LGPL
library that needs its source next to the release is written to --needs-source,
one name per line, for the release workflow to build.

    python tools/collect_licences.py --bundle dist/ming-addon \
        --out release-staging/THIRD_PARTY_LICENSES --needs-source sources.txt
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, distribution, distributions, packages_distributions
from pathlib import Path

BUILD_ONLY = {
    'pip', 'setuptools', 'wheel', 'uv', 'pyinstaller', 'pyinstaller-hooks-contrib', 'altgraph',
    'macholib', 'pefile', 'pywin32-ctypes', 'pytest', 'pluggy', 'iniconfig', 'pygments',
    'subtitld-addon-ming',
}
REPO_ROOT = Path(__file__).resolve().parents[1]

# (pattern on the lower-cased file name, notice inside THIRD_PARTY_LICENSES/,
# component, LGPL source to attach or None) for native files no installed
# distribution owns. A None notice refuses the file.
RULES = [
    (r'^ming-addon(\.exe)?$', 'pyinstaller/COPYING.txt', 'PyInstaller bootloader', None),
    (r'^(libpython3[\d.]*\.(so|dylib)[\d.]*|python3\d*\.dll|python)$', 'python/LICENSE.txt', 'CPython', None),
    (r'(\.cpython-3\d+[^/]*\.(so|dylib)|\.pyd)$', 'python/LICENSE.txt', 'CPython extension module', None),
    (r'^libquadmath[-.]', 'libquadmath/COPYING.LIB', 'libquadmath (LGPL)', 'libquadmath'),
    (r'^(libgcc_s|libstdc\+\+|libgfortran|libgomp)[-.]', 'gcc-runtime/COPYING.RUNTIME', 'GCC runtime', None),
    (r'^(libssl|libcrypto)[-.\d]', 'python-runtime-libs/openssl-LICENSE.txt', 'OpenSSL', None),
    (r'^libffi[-.\d]', 'python-runtime-libs/libffi-LICENSE', 'libffi', None),
    (r'^libexpat[-.\d]', 'python-runtime-libs/expat-COPYING', 'Expat', None),
    (r'^(libz\.(so|dylib)|zlib\d*\.dll)', 'python-runtime-libs/zlib-LICENSE', 'zlib', None),
    (r'^libbz2', 'python-runtime-libs/bzip2-LICENSE', 'bzip2', None),
    (r'^liblzma', 'python-runtime-libs/xz-COPYING', 'XZ Utils', None),
    (r'^(libsqlite3|sqlite3\.dll)', 'python-runtime-libs/sqlite-NOTICE.txt', 'SQLite (public domain)', None),
    (r'^libmpdec', 'python-runtime-libs/mpdecimal-LICENSE.txt', 'mpdecimal', None),
    (r'^libuuid', 'python-runtime-libs/libuuid-COPYING.BSD-3-Clause', 'libuuid', None),
    (r'^libzstd', 'python-runtime-libs/zstd-LICENSE', 'zstd', None),
    (r'^(libncursesw?|libtinfo|libpanelw?|libformw?|libmenuw?)[-.]', 'python-runtime-libs/ncurses-COPYING', 'ncurses', None),
    (r'^(libc\+\+|libc\+\+abi|libunwind|libomp)[-.]', 'llvm-runtime/LICENSE.TXT', 'LLVM runtime', None),
    (r'^(vcruntime140(_1|_threads)?|msvcp140(_\w+)?|vcomp140|concrt140)\.dll$', 'msvc-runtime/NOTICE.txt',
     'Microsoft C runtime', None),
    (r'^(ucrtbase|api-ms-win-[\w-]+)\.dll$', 'msvc-runtime/NOTICE.txt', 'Universal C runtime', None),
    # Refused: glibc belongs to the system the add-on runs on, and these are GPL.
    (r'^(libc|libm|libmvec|libpthread|libdl|librt|ld-linux[\w-]*)\.so', None, 'glibc', None),
    (r'^(libreadline|libhistory|libgdbm)', None, 'GPL library', None),
]
_COMPILED = [(re.compile(pattern), notice, component, source) for pattern, notice, component, source in RULES]

# LGPL libraries matched by name wherever they are, owned by a distribution
# or not: their source has to travel with the release.
LGPL_BY_NAME = [(re.compile(r'^libquadmath[-.]'), 'libquadmath')]

# Copyleft distributions that may ship, and why that is fine.
COPYLEFT_OK = {
    'certifi': 'MPL-2.0, unmodified; source on PyPI (MPL-SOURCES.txt)',
    'tqdm': 'MPL-2.0 AND MIT, unmodified; source on PyPI (MPL-SOURCES.txt)',
}
_COPYLEFT = re.compile(r'\b(L?GPL|AGPL|MPL|EPL|CDDL|EUPL)|General Public|Mozilla Public', re.I)
_COPYLEFT_EXCEPTION = re.compile(r'exception', re.I)

_MACHO_MAGIC = {b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', b'\xce\xfa\xed\xfe',
                b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'}

_LICENCE_FILE = re.compile(r'(^|/)(licen[cs]e|copying|notice|authors|third[-_]?party)[^/]*$', re.I)


def _normal(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def _licence_files(dist) -> list[tuple[Path, str]]:
    """(file, where it goes) for each licence file the wheel ships.

    In the .dist-info folder, or at the top of a package: ONNX Runtime keeps
    its LICENSE and ThirdPartyNotices.txt (the notices of the libraries
    compiled into it) in onnxruntime/.
    """
    found = []
    for entry in dist.files or []:
        text = str(entry).replace('\\', '/')
        if '.dist-info/' in text and _LICENCE_FILE.search(text):
            rel = text.split('.dist-info/', 1)[1]
        elif text.count('/') == 1 and _LICENCE_FILE.search(text):
            rel = text.split('/', 1)[1]
        else:
            continue
        path = Path(dist.locate_file(entry))
        if path.is_file():
            found.append((path, rel))
    return found


def _declared(dist) -> str:
    meta = dist.metadata
    expression = meta.get('License-Expression') or ''
    if expression:
        return expression
    classifiers = [c.split('::')[-1].strip() for c in meta.get_all('Classifier') or []
                   if c.startswith('License ::')]
    text = (meta.get('License') or '').strip()
    if text and len(text) < 200:
        classifiers.insert(0, text)
    return '; '.join(classifiers)


def is_native(path: Path) -> bool:
    with path.open('rb') as handle:
        head = handle.read(4)
    return head == b'\x7fELF' or head[:2] == b'MZ' or head in _MACHO_MAGIC


def _owners() -> tuple[dict[str, str], dict[str, set[str]]]:
    """(path inside site-packages -> distribution, file name -> distributions)."""
    by_path: dict[str, str] = {}
    by_name: dict[str, set[str]] = {}
    for dist in distributions():
        name = _normal(dist.metadata['Name'] or '')
        if not name or name in BUILD_ONLY:
            continue
        for entry in dist.files or []:
            text = str(entry).replace('\\', '/')
            if text.startswith('..'):
                continue
            by_path[text] = name
            by_name.setdefault(text.rsplit('/', 1)[-1], set()).add(name)
    return by_path, by_name


def bundled_distributions(bundle: Path) -> set[str]:
    """The distributions the bundle carries, from its top-level names.

    Modules frozen into the executable's archive (PYZ) and the folders and
    extension modules in _internal/, mapped to distributions by their RECORD
    or top_level.txt.
    """
    from PyInstaller.archive.readers import CArchiveReader
    names: set[str] = set()
    for entry in (bundle / '_internal').iterdir():
        names.add(entry.name)
        names.add(entry.name.split('.', 1)[0])
    executable = next(p for p in bundle.iterdir() if p.is_file() and p.name.startswith('ming-addon'))
    archive = CArchiveReader(str(executable))
    pyz = next(name for name in archive.toc if name.endswith('.pyz'))
    names |= {module.split('.', 1)[0] for module in archive.open_embedded_archive(pyz).toc}
    owners = packages_distributions()
    return {_normal(dist) for name in names for dist in owners.get(name, ())}


def audit(bundle: Path, out: Path) -> tuple[list[tuple[str, str]], set[str], list[str]]:
    """([(file, what covers it)], LGPL sources needed, problems) for every native file."""
    by_path, by_name = _owners()
    internal = bundle / '_internal'
    covered, sources, problems = [], set(), []
    for path in sorted(p for p in bundle.rglob('*') if p.is_file() and not p.is_symlink()):
        if not is_native(path):
            continue
        rel = path.relative_to(internal if internal in path.parents else bundle).as_posix()
        name = path.name.lower()
        for pattern, source in LGPL_BY_NAME:
            if pattern.search(name):
                sources.add(source)
        owner = by_path.get(rel) or ', '.join(sorted(by_name.get(path.name, ())))
        if owner:
            covered.append((rel, owner))
            continue
        for pattern, notice, component, source in _COMPILED:
            if pattern.search(name):
                if notice is None:
                    problems.append(f'{rel}: {component}, which must not be bundled')
                elif not (out / notice).is_file():
                    problems.append(f'{rel}: {component}, but THIRD_PARTY_LICENSES/{notice} is missing')
                else:
                    covered.append((rel, f'{component} ({notice})'))
                    if source:
                        sources.add(source)
                break
        else:
            problems.append(f'{rel}: no distribution owns it and no rule covers it')
    return covered, sources, problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path, help='the PyInstaller onedir folder')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--needs-source', type=Path, help='write the LGPL sources to attach here')
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    committed = REPO_ROOT / 'THIRD_PARTY_LICENSES'
    if committed.is_dir():
        shutil.copytree(committed, out, dirs_exist_ok=True)

    rows = []
    problems = []
    seen = set()
    bundled = bundled_distributions(args.bundle)
    for dist in sorted(distributions(), key=lambda d: _normal(d.metadata['Name'] or '')):
        name = _normal(dist.metadata['Name'] or '')
        if not name or name in BUILD_ONLY or name in seen or name not in bundled:
            continue
        seen.add(name)
        target = out / name
        files = _licence_files(dist)
        declared = _declared(dist)
        target.mkdir(exist_ok=True)
        for path, relative in files:
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target / relative)
        if not files:
            if not declared:
                problems.append(name)
                continue
            (target / 'DECLARED-LICENSE.txt').write_text(
                f'{dist.metadata["Name"]} {dist.version} ships no licence file; its metadata declares: '
                f'{declared}\n', encoding='utf-8')
        rows.append((dist.metadata['Name'], dist.version, declared or 'see files', name))
        if _COPYLEFT.search(declared) and not _COPYLEFT_EXCEPTION.search(declared) and name not in COPYLEFT_OK:
            problems.append(f'{name} ({declared}): copyleft, and not in COPYLEFT_OK')

    # CPython, which the bundle embeds, and the PyInstaller bootloader.
    stdlib = Path(sysconfig.get_paths()['stdlib'])
    for candidate in (Path(sys.base_prefix) / 'LICENSE.txt', stdlib / 'LICENSE.txt', Path(sys.base_prefix) / 'LICENSE'):
        if candidate.is_file():
            (out / 'python').mkdir(exist_ok=True)
            shutil.copyfile(candidate, out / 'python' / 'LICENSE.txt')
            rows.append(('CPython', sys.version.split()[0], 'PSF-2.0', 'python'))
            break
    else:
        problems.append('CPython (no LICENSE.txt next to the interpreter)')
    try:
        pyinstaller = [p for p, _rel in _licence_files(distribution('pyinstaller')) if p.name.upper().startswith('COPYING')]
    except PackageNotFoundError:
        pyinstaller = []
    if pyinstaller:
        (out / 'pyinstaller').mkdir(exist_ok=True)
        shutil.copyfile(pyinstaller[0], out / 'pyinstaller' / 'COPYING.txt')
        rows.append(('PyInstaller bootloader', distribution('pyinstaller').version,
                     'GPL-2.0-or-later WITH Bootloader-exception', 'pyinstaller'))
    else:
        problems.append('PyInstaller (no COPYING.txt)')

    covered, sources, native_problems = audit(args.bundle, out)
    problems += native_problems
    if problems:
        print('FATAL:\n  ' + '\n  '.join(problems), file=sys.stderr)
        return 1
    if args.needs_source:
        args.needs_source.write_text(''.join(f'{s}\n' for s in sorted(sources)), encoding='utf-8')

    lines = ['# Third-party licences', '',
             'What this archive carries besides the add-on itself (Apache-2.0, LICENSE at the root).',
             'The Ming-3.0-tts-0.5B weights are not in the archive; they download on first use and are MIT. '
             'The add-on carries a modified copy of the Ming-3.0-tts inference code (MIT, `ming-3.0-tts/`), '
             'with a few lines from x-transformers (MIT) and diffusers (Apache-2.0).', '',
             '| Component | Version | Licence | Files |', '|---|---|---|---|']
    lines += [f'| {name} | {version} | {declared} | `{folder}/` |' for name, version, declared, folder in rows]
    extra = sorted(p.name for p in committed.iterdir() if p.is_dir()) if committed.is_dir() else []
    if extra:
        lines += ['', 'Native libraries that are not Python distributions have their notices in:',
                  ', '.join(f'`{name}/`' for name in extra) + '.']
    if sources:
        lines += ['', 'LGPL libraries whose complete source is attached to the same GitHub release: '
                  + ', '.join(sorted(sources)) + '.']
    lines += ['', '## Native files', '', '| File | Covered by |', '|---|---|']
    lines += [f'| `{rel}` | {what} |' for rel, what in covered]
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{out}: {len(rows)} components, {len(covered)} native files covered'
          + (f'; LGPL sources to attach: {", ".join(sorted(sources))}' if sources else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
