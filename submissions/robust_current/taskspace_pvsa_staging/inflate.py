#!/usr/bin/env python3
"""Repository-bound launcher for the G85 PVSA1 staging receiver.

This is intentionally not the final self-contained public runtime.  The
decoder-only tree-shake named in the G85 receipt must land before promotion.
"""

from __future__ import annotations

from tac.witness_dsl.taskspace_g85_pvsa_public_receiver_v1 import main

if __name__ == "__main__":
    raise SystemExit(main())
