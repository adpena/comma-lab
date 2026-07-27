#!/usr/bin/env python3
"""Run the retrospective-only G57 codec diagnostic linter.

This CLI never emits a live admission.  Use
``audit_taskspace_codec_adversarial_gate_v2.py`` for the chained G59 gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.witness_control.taskspace_codec_adversarial_gate_v1 import (  # noqa: E402
    canonical_json,
    review_request,
    write_once_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_bytes())
    receipt = review_request(request)
    write_once_receipt(args.output, receipt)
    sys.stdout.buffer.write(canonical_json(receipt))
    return 20


if __name__ == "__main__":
    raise SystemExit(main())
