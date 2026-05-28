#!/usr/bin/env python
"""time_traveler_l5_z7_mamba2 contest-compliant inflate runtime.

Reads archive_dir/0.bin via the packaged substrate parser, then for
each base in file_list writes per-frame .png under output_dir/<base>/.
No scorer-network imports (strict-scorer-rule contract).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))
from tac.substrates.time_traveler_l5_z7_mamba2.inflate import inflate_one_video

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
        base = line.rsplit('.', 1)[0]
        inflate_one_video(archive_bytes, output_dir / base, device='cpu')
    return 0

if __name__ == '__main__':
    sys.exit(main())
