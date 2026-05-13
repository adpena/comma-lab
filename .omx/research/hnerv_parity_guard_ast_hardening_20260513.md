# HNeRV Parity Guard AST Hardening - 2026-05-13

research_only=true

## Finding

REVIEW-OMNI found that `tac.hnerv_training_parity_guard` still accepted some
HNeRV parity contracts through raw substrings. In particular, emitted runtime
templates could satisfy checks through comments, docstrings, or dead strings
rather than executable `inflate.sh` / `inflate.py` behavior.

## Landing

- Replaced runtime-template substring checks with structural checks:
  - tokenized executable-line checks for `inflate.sh` 3-arg forwarding,
    `set -e*`, and no passthrough `$@`;
  - AST checks for emitted `inflate.py` `main()` enforcing the exact rejection
    guard `len(sys.argv) != 4`, consuming `sys.argv[1:4]`, reading `archive_dir / "0.bin"`, and iterating
    file-list `read_text(...).splitlines()`;
  - AST-level scorer/runtime forbidden import/name detection so comments and
    dead strings do not count.
- Preserved PR95/HNeRV parity requirements in `_full_main`: differentiable YUV6
  patch before scorer load, executable `apply_eval_roundtrip=True`, EMA
  update/apply/state export, archive packing, runtime emission, and archive ZIP
  build.
- Added negative tests for `inflate.py` docstring/dead-string spoofing,
  inverted `len(sys.argv) == 4` arity checks, `inflate.sh` comment-only
  signature spoofing, and executable scorer imports.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_training_parity_guard.py -q
# 11 passed
```
