#!/usr/bin/env python
"""hprc contest-compliant inflate runtime.

Reads archive_dir/0.bin via the packaged substrate parser, then for
each video in file_list writes contest raw bytes under output_dir/*.raw.
No scorer-network imports (strict-scorer-rule contract).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))
from tac.substrates.hprc.inflate import inflate_one_video

def main() -> int:
    if len(sys.argv) != 4:
        print('usage: inflate.py <archive_dir> <output_dir> <file_list>',
              file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / '0.bin').read_bytes()
    for line in file_list_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rel = Path(line).with_suffix('.raw')
        if rel.is_absolute() or any(part in {'', '..'} for part in rel.parts):
            raise ValueError(f'unsafe file_list entry: {line!r}')
        inflate_one_video(archive_bytes, output_dir / rel, device='cpu')
    return 0

if __name__ == '__main__':
    sys.exit(main())
