#!/usr/bin/env python3
"""Build the libquadmath source tarball attached to every release.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Adapted from addon-phonon2's tool of the same name (itself from addon-whispercpp),
for the libquadmath build this bundle carries.

Why: the Linux archive ships libquadmath (LGPL-2.1-or-later), which the GCC
Fortran runtime of numpy's OpenBLAS links. LGPL-2.1 section 4 lets
object code be distributed by offering the complete corresponding source "from a
designated place" alongside it, with no written-offer shortcut. The release
attaches this tarball to the same GitHub release as the binaries.

Which source: one unmodified distribution build, identified by GNU build-id:
AlmaLinux 8's libquadmath-8.5.0-28.el8_10.alma.1 (libquadmath-2284e583*.so.0.0.0),
from numpy's wheel, built in a manylinux_2_28 image. Its source is AlmaLinux's
gcc-8.5.0-28.el8_10.alma.1.src.rpm. (scipy, which brought a CentOS 7 build too,
is not in this bundle.)

From each source package the tarball keeps libquadmath/, the top-level build
machinery its configure and Makefile use, and the distribution's spec and patches
(a few MB), not the rest of the compiler. Each package stays linked in the README.

    python tools/build_libquadmath_source.py --out ming-26.10.1-libquadmath-source.tar.gz
    python tools/build_libquadmath_source.py --check-bundle dist/ming-addon

`--check-bundle` fails when a bundled libquadmath is not one of these builds (a
numpy or OpenBLAS upgrade that changes it must update BUILDS, the tarball
and ../THIRD_PARTY_LICENSES/libquadmath/).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import lzma
import bz2
import os
import posixpath
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Build:
    key: str              # directory name inside the tarball
    distro: str
    rpm_release: str
    build_id: str         # GNU build-id of the shipped library
    lib_sha256: str       # sha256 of the original file (auditwheel only renames it)
    srpm_name: str
    srpm_sha256: str
    srpm_urls: tuple[str, ...]
    gcc_tarball: str
    gcc_tree: str
    binary_rpm_url: str
    mtime: int            # the GCC snapshot date, for reproducible output
    shipped: str          # where the archive carries it


# Must match THIRD_PARTY_LICENSES/libquadmath/SOURCE.txt.
BUILDS = (
    Build(
        key='almalinux8-gcc-8.5.0-28.el8_10.alma.1',
        distro='AlmaLinux 8',
        rpm_release='8.5.0-28.el8_10.alma.1',
        build_id='c1c5f8d61f17e5579c4e85718845b74e4b94116c',
        lib_sha256='2284e5833f5ea62b141499d32373fc207878c1778fe004b4867bc41d75086be2',
        srpm_name='gcc-8.5.0-28.el8_10.alma.1.src.rpm',
        srpm_sha256='c94dbbd2cd6d5d01a5a6822ea18f8ce34942c25a80e17b4682d547d5e652dbd0',
        srpm_urls=(
            'https://vault.almalinux.org/8.10/BaseOS/Source/Packages/gcc-8.5.0-28.el8_10.alma.1.src.rpm',
            'https://repo.almalinux.org/vault/8.10/BaseOS/Source/Packages/gcc-8.5.0-28.el8_10.alma.1.src.rpm',
        ),
        gcc_tarball='gcc-8.5.0-20210514.tar.xz',
        gcc_tree='gcc-8.5.0-20210514',
        binary_rpm_url=('https://repo.almalinux.org/almalinux/8/BaseOS/x86_64/os/Packages/'
                        'libquadmath-8.5.0-28.el8_10.alma.1.x86_64.rpm'),
        mtime=1620950400,  # 2021-05-14
        shipped=('`libquadmath-2284e583*.so.0.0.0`, in `bin/_internal/numpy.libs/` and '
                 '`bin/_internal/`, linked by the GCC Fortran runtime of numpy\'s OpenBLAS'),
    ),
)

# Inside each GCC tree: what building libquadmath reads besides libquadmath/
# (its configure sources ../config-ml.in, ../ltmain.sh, ../config/*.m4; its
# Makefile reads ../gcc/BASE-VER, ../contrib/make_sunver.pl and the texinfo
# includes; printf/gmp-impl.h includes ../../libgcc/longlong.h, whose macros are
# compiled into the library). build() refuses to write a tarball in which any
# kept source includes a `../` file left out.
_KEEP_DIRS = ('libquadmath/', 'config/', 'gcc/doc/include/')
_KEEP_FILES = ('gcc/BASE-VER', 'gcc/DATESTAMP', 'gcc/DEV-PHASE', 'contrib/make_sunver.pl',
               'libgcc/longlong.h')
_INCLUDE_RE = re.compile(rb'^\s*#\s*include\s+"(\.\./[^"]+)"', re.M)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# --check-bundle
# ---------------------------------------------------------------------------
def elf_build_id(path: Path) -> str | None:
    """The NT_GNU_BUILD_ID of a 64-bit little-endian ELF file, or None."""
    with open(path, 'rb') as f:
        ident = f.read(16)
        if ident[:4] != b'\x7fELF' or ident[4] != 2 or ident[5] != 1:
            return None
        header = f.read(48)
        e_phoff = struct.unpack_from('<Q', header, 16)[0]
        e_phentsize, e_phnum = struct.unpack_from('<HH', header, 38)
        for i in range(e_phnum):
            f.seek(e_phoff + i * e_phentsize)
            p_type, _flags, p_offset, _vaddr, _paddr, p_filesz = struct.unpack('<IIQQQQ', f.read(40))
            if p_type != 4:  # PT_NOTE
                continue
            f.seek(p_offset)
            notes = f.read(p_filesz)
            pos = 0
            while pos + 12 <= len(notes):
                namesz, descsz, ntype = struct.unpack_from('<III', notes, pos)
                pos += 12
                name = notes[pos:pos + namesz]
                pos += (namesz + 3) & ~3
                desc = notes[pos:pos + descsz]
                pos += (descsz + 3) & ~3
                if ntype == 3 and name.rstrip(b'\0') == b'GNU':  # NT_GNU_BUILD_ID
                    return desc.hex()
    return None


def check_bundle(root: Path) -> int:
    known = {b.build_id: b for b in BUILDS}
    found = sorted(p for p in root.rglob('libquadmath*.so*')
                   if p.is_file() and not p.is_symlink())
    if not found:
        _log(f'no libquadmath in {root}; nothing to check')
        return 0
    bad = 0
    for path in found:
        build_id = elf_build_id(path)
        build = known.get(build_id)
        bad += build is None
        _log(f'{"ok " if build else "BAD"} {path.relative_to(root)} build-id {build_id}'
             f'{f" ({build.distro} libquadmath-{build.rpm_release})" if build else ""}')
    if bad:
        _log(f'\nFATAL: {bad} bundled libquadmath file(s) are not one of the builds whose source '
             f'the release attaches. Find the new one\'s origin and add it to BUILDS in '
             f'tools/build_libquadmath_source.py and to THIRD_PARTY_LICENSES/libquadmath/.')
        return 1
    return 0


# ---------------------------------------------------------------------------
# RPM / cpio reading (no rpm2cpio needed)
# ---------------------------------------------------------------------------
def _rpm_payload(raw) -> io.BufferedIOBase:
    """The decompressed cpio stream of an RPM file object."""
    if raw.read(4) != b'\xed\xab\xee\xdb':
        raise SystemExit('FATAL: not an RPM file')
    raw.seek(96)  # the lead
    for padded in (True, False):  # signature header, then the main header
        intro = raw.read(16)
        if intro[:3] != b'\x8e\xad\xe8':
            raise SystemExit('FATAL: malformed RPM header')
        nindex, hsize = struct.unpack('>II', intro[8:16])
        size = 16 * nindex + hsize
        raw.seek(size + ((8 - size % 8) % 8 if padded else 0), os.SEEK_CUR)
    start = raw.tell()
    magic = raw.read(6)
    raw.seek(start)
    if magic.startswith(b'\x1f\x8b'):
        return gzip.GzipFile(fileobj=raw)
    if magic.startswith(b'\xfd7zXZ'):
        return lzma.LZMAFile(raw)
    if magic.startswith(b'BZh'):
        return bz2.BZ2File(raw)
    raise SystemExit(f'FATAL: unsupported RPM payload compression {magic!r}')


def _read_exact(stream, n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise SystemExit('FATAL: truncated RPM payload')
    return data


def _extract_rpm(rpm: Path, dest: Path, wanted) -> list[str]:
    """Extract the regular files whose name satisfies `wanted` (newc cpio)."""
    extracted: list[str] = []
    with open(rpm, 'rb') as raw, _rpm_payload(raw) as stream:
        while True:
            header = _read_exact(stream, 110)
            if header[:6] not in (b'070701', b'070702'):
                raise SystemExit(f'FATAL: unexpected cpio header {header[:6]!r}')
            fields = [int(header[6 + 8 * i:14 + 8 * i], 16) for i in range(13)]
            mode, filesize, namesize = fields[1], fields[6], fields[11]
            name = _read_exact(stream, namesize)[:-1].decode('utf-8')
            _read_exact(stream, (4 - (110 + namesize) % 4) % 4)
            if name == 'TRAILER!!!':
                return extracted
            name = name.removeprefix('./')
            keep = (mode & 0o170000) == 0o100000 and wanted(name)
            target = dest / Path(name).name if keep else None
            out = open(target, 'wb') if target else None
            try:
                left = filesize
                while left:
                    chunk = stream.read(min(left, 1 << 20))
                    if not chunk:
                        raise SystemExit('FATAL: truncated RPM payload')
                    left -= len(chunk)
                    if out:
                        out.write(chunk)
            finally:
                if out:
                    out.close()
            if keep:
                extracted.append(Path(name).name)
            _read_exact(stream, (4 - filesize % 4) % 4)


# ---------------------------------------------------------------------------
# --out
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _fetch_srpm(build: Build, dest: Path, srpm_dir: Path | None) -> Path:
    if srpm_dir is not None:
        local = srpm_dir / build.srpm_name
        if local.is_file():
            if _sha256(local) != build.srpm_sha256:
                raise SystemExit(f'FATAL: {local} is not the pinned {build.srpm_name}')
            return local
    # With a --srpm-dir, a download is kept there for the next build.
    target = (srpm_dir if srpm_dir is not None else dest) / build.srpm_name
    target.parent.mkdir(parents=True, exist_ok=True)
    for url in build.srpm_urls:
        _log(f'fetching {url}')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'subtitld-addon-ming'})
            with urllib.request.urlopen(req, timeout=300) as resp, open(target, 'wb') as out:
                shutil.copyfileobj(resp, out, 1 << 20)
        except OSError as exc:
            _log(f'  failed: {exc}')
            continue
        digest = _sha256(target)
        if digest == build.srpm_sha256:
            return target
        _log(f'  sha256 {digest} does not match the pinned {build.srpm_sha256}')
    raise SystemExit(f'FATAL: could not fetch a verified {build.srpm_name}')


def _keep_member(rel: str, is_dir: bool) -> bool:
    if '/' not in rel:
        return not is_dir  # top-level build machinery: configure, Makefile.in, ltmain.sh, ...
    return rel.startswith(_KEEP_DIRS) or rel in _KEEP_FILES


def missing_includes(sources: dict[str, bytes], kept: set[str]) -> list[str]:
    """`from -> to` for every `#include "../..."` in `sources` (tree-relative
    path -> content) that resolves to a file not in `kept`."""
    missing = []
    for rel, data in sorted(sources.items()):
        for target in _INCLUDE_RE.findall(data):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target.decode()))
            if resolved not in kept:
                missing.append(f'{rel} -> {resolved}')
    return missing


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mtime: int) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = mtime
    tar.addfile(info, io.BytesIO(data))


def _add_build(tar: tarfile.TarFile, prefix: str, build: Build, srpm: Path, work: Path) -> dict:
    """Write one build's subset into the open tarball. Returns what the README states."""
    _log(f'extracting {build.srpm_name}')
    files = _extract_rpm(srpm, work,
                         lambda n: n in (build.gcc_tarball, 'gcc.spec') or n.endswith('.patch'))
    patches = sorted(f for f in files if f.endswith('.patch'))
    if build.gcc_tarball not in files or 'gcc.spec' not in files or not patches:
        raise SystemExit(f'FATAL: {build.srpm_name} lacks {build.gcc_tarball}, gcc.spec or its patches')
    touching = [p for p in patches
                if re.search(rb'^(\+\+\+|---) \S*libquadmath/', (work / p).read_bytes(), re.M)]
    base = f'{prefix}/{build.key}'
    kept = 0
    kept_names: set[str] = set()
    quadmath_sources: dict[str, bytes] = {}
    with tarfile.open(work / build.gcc_tarball, 'r:*') as src:
        for member in src:
            rel = member.name.removeprefix(build.gcc_tree + '/')
            if not member.name.startswith(build.gcc_tree + '/') or not rel:
                continue
            if not _keep_member(rel, member.isdir()):
                continue
            if member.issym() or member.islnk():
                continue  # never follow a link out of the tree
            member.name = f'{base}/{build.gcc_tree}/{rel}'
            member.uid = member.gid = 0
            member.uname = member.gname = ''
            if member.isfile():
                data = src.extractfile(member).read()
                kept_names.add(rel)
                if rel.startswith('libquadmath/') and rel.endswith(('.c', '.h')):
                    quadmath_sources[rel] = data
                tar.addfile(member, io.BytesIO(data))
                kept += 1
            else:
                tar.addfile(member)
    missing = missing_includes(quadmath_sources, kept_names)
    if missing:
        raise SystemExit(f'FATAL: the {build.key} subset would not build libquadmath; these '
                         f'includes point at files left out (add them to _KEEP_FILES):\n  '
                         + '\n  '.join(missing))
    for name in ['gcc.spec'] + patches:
        sub = 'distribution' if name == 'gcc.spec' else 'distribution/patches'
        _add_bytes(tar, f'{base}/{sub}/{name}', (work / name).read_bytes(), build.mtime)
    for name in ['gcc.spec', build.gcc_tarball] + patches:
        (work / name).unlink()
    _log(f'  {kept} files from {build.gcc_tree}, {len(patches)} patches; '
         f'patches touching libquadmath/: {touching or "none"}')
    return {'kept': kept, 'patches': len(patches), 'touching': touching}


def build(out: Path, srpm_dir: Path | None, version: str) -> int:
    prefix = out.name.removesuffix('.tar.gz')
    out.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            with open(out, 'wb') as raw, \
                    gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz, \
                    tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as tar:
                for b in BUILDS:
                    srpm = _fetch_srpm(b, work, srpm_dir)
                    facts = _add_build(tar, prefix, b, srpm, work)
                    sections.append(_SECTION.format(
                        key=b.key, distro=b.distro, rpm=b.rpm_release, srpm=b.srpm_name,
                        srpm_sha=b.srpm_sha256, srpm_url=b.srpm_urls[0], binary_rpm=b.binary_rpm_url,
                        build_id=b.build_id, lib_sha=b.lib_sha256, tree=b.gcc_tree,
                        tarball=b.gcc_tarball, shipped=b.shipped, kept=facts['kept'],
                        patch_count=facts['patches'],
                        rebuild=_REBUILD_CLEAN if not facts['touching'] else _REBUILD_PATCH.format(
                            patches=', '.join(f'`{p}`' for p in facts['touching'])),
                    ).replace('{tree}', b.gcc_tree))
                    if srpm.parent == work:
                        srpm.unlink()
                readme = _README.format(version=version, sections='\n'.join(sections))
                _add_bytes(tar, f'{prefix}/README.md', readme.encode('utf-8'), BUILDS[-1].mtime)
        except BaseException:
            out.unlink(missing_ok=True)
            raise
    _log(f'wrote {out} ({out.stat().st_size / 2**20:.1f} MiB)')
    return 0


_README = """# libquadmath source for the Subtitld Ming-3.0-tts add-on {version}

The Linux archive of this add-on release ships libquadmath, licensed under the
GNU Lesser General Public License 2.1 or later
(`THIRD_PARTY_LICENSES/libquadmath/COPYING.LIB` in the archive). This tarball
is the complete corresponding source of the build it carries, offered from the
same place as the binaries (LGPL-2.1 section 4), in a directory of its own.

The release refuses to publish a bundle whose libquadmath files are not these
builds (`tools/build_libquadmath_source.py --check-bundle`).

{sections}"""

_SECTION = """## `{key}/`: {distro}'s libquadmath-{rpm}

Shipped as {shipped}. It is the unmodified `/usr/lib64/libquadmath.so.0.0.0` of
{distro}'s `libquadmath-{rpm}.x86_64` ({binary_rpm}), with only its SONAME
renamed by auditwheel:

| | |
|---|---|
| GNU build-id | `{build_id}` |
| sha256 of the original file | `{lib_sha}` |

| Path | What |
|---|---|
| `{tree}/libquadmath/` | the library's source |
| `{tree}/*`, `config/`, `gcc/BASE-VER` ..., `contrib/make_sunver.pl`, `gcc/doc/include/` | the GCC build machinery that configures and builds it |
| `{tree}/libgcc/longlong.h` | a header the library's printf code includes |
| `distribution/gcc.spec` | the distribution's build recipe (libquadmath is built as a GCC target library) |
| `distribution/patches/` | all {patch_count} patches of the source package |

The files come from `{tarball}` inside the source package `{srpm}`:

    {srpm_url}
    sha256 {srpm_sha}

That package also holds the rest of the compiler, which is not part of the
library and is left out here ({kept} files kept).

{rebuild}
"""

_REBUILD_CLEAN = """None of the package's patches change `libquadmath/`, so nothing needs applying.
libquadmath configures and builds on its own, out of tree, with the system C
compiler:

    mkdir build && cd build
    ../{tree}/libquadmath/configure --disable-multilib
    make

The library is `build/.libs/libquadmath.so.0.0.0`. Copy it over the shipped file,
keeping the shipped file name. The distribution built it as a target library of
a full GCC build (`distribution/gcc.spec`, `%build`); for that, use the whole
source package above."""

_REBUILD_PATCH = """Apply the patches that change `libquadmath/` first ({patches}), from the tree's
top directory with `patch -p0` or `-p1` as the spec's `%prep` does. Then
libquadmath configures and builds on its own, out of tree, with the system C
compiler:

    mkdir build && cd build
    ../{{tree}}/libquadmath/configure --disable-multilib
    make

The library is `build/.libs/libquadmath.so.0.0.0`. Copy it over the shipped file,
keeping the shipped file name. The distribution built it as a target library of
a full GCC build (`distribution/gcc.spec`, `%build`); for that, use the whole
source package above."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path, help='tarball to write (*.tar.gz)')
    mode.add_argument('--check-bundle', type=Path, metavar='DIR',
                      help='verify every libquadmath under DIR is one of the pinned builds')
    ap.add_argument('--srpm-dir', type=Path,
                    help='source packages folder: used when they are there, downloaded into it when not')
    ap.add_argument('--version', help='add-on version for the README (default: manifest.json)')
    args = ap.parse_args(argv)
    if args.check_bundle:
        return check_bundle(args.check_bundle)
    if not args.out.name.endswith('.tar.gz'):
        ap.error('--out must end in .tar.gz')
    version = args.version or json.loads(
        (Path(__file__).resolve().parent.parent / 'manifest.json').read_text(encoding='utf-8'))['version']
    return build(args.out, args.srpm_dir, version)


if __name__ == '__main__':
    raise SystemExit(main())
