# Source Generated with Decompyle++
# File: profile_endgame_archive_decision.cpython-312.pyc (Python 3.12)

'''CLI wrapper for PR85-family endgame archive decision profiling.'''
from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from tac.endgame_archive_decision import main
if __name__ == '__main__':
    raise SystemExit(main())
