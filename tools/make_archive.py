#!/usr/bin/env python3
"""Zip a staged add-on directory the way Subtitld's installer can unpack it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Subtitld installs add-ons with `zipfile.ZipFile.extractall`, which cannot
recreate symbolic links. PyInstaller's onedir output contains some (on Linux,
top-level aliases of the relocated native libraries; on macOS, framework
aliases), so every link is stored as the file it points to. That also
sidesteps `shutil.make_archive`, which follows file links but writes linked
DIRECTORIES as empty entries.

    python tools/make_archive.py release-staging ming-26.10.1-linux-x86_64.zip
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path


def make_archive(root: Path, out: Path) -> int:
    root = root.resolve()
    count = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            here = os.path.realpath(dirpath)
            keep = []
            for name in sorted(dirnames):
                path = os.path.join(dirpath, name)
                if os.path.islink(path):
                    target = os.path.realpath(path)
                    if here == target or here.startswith(target + os.sep):
                        continue  # a link to an ancestor would recurse forever
                keep.append(name)
            dirnames[:] = keep
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                if not os.path.exists(full):
                    raise SystemExit(f'FATAL: broken symlink in the bundle: {full}')
                arcname = os.path.relpath(full, root).replace(os.sep, '/')
                zf.write(full, arcname)
                count += 1
    return count


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root, out = Path(argv[0]), Path(argv[1])
    count = make_archive(root, out)
    print(f'{out}: {count} files, {out.stat().st_size:,} bytes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
