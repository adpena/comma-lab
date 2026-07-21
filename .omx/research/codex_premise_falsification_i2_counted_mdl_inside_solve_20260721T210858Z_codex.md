# Codex premise falsification — I2 counted MDL is not inside the solve

**UTC:** 2026-07-21T21:08:58Z
**Scope:** FORMULATION
**Pointer:** 0.1910828242 `[contest-CPU]` UNMOVED
**Result:** `FALSIFIED`

The delegated premise said the I2 polytope-member arm optimized “code length INSIDE the solve.”
It does not optimize the counted PPCS/archive length. In
`src/tac/optimization/mdl_polytope_member.py`, the bounded member search prices zlib-9 over
decoded uint8 camera arrays. `tools/measure_mdl_polytope_member.py` measures the seed/PPCS
construction only after all member solves.

Consequences:

- The n64 selected member is byte-identical to canonical at 64/64 and cuts zero diagnostic
  bytes (`77,651,017 -> 77,651,017`).
- The raw 884,872-byte PPCS seed parse/serializes byte-identically; `78,969 B` is its zlib
  encoding, but neither is a complete `archive.zip`. `79,385 B` is zlib over seed bytes
  concatenated with calibration JSON and is not parser-valid PPCS.
- Those byte figures are not coupled to the selected member's
  `d_seg=0.00012636184692382812`, `d_pose=0.000060022091887905524` and cannot form a score tuple.
- The correct conclusion is `NO_ADMISSION_IDENTITY_ZERO_CUT` for this bounded formulation.
  Joint counted-packet MDL inside the receiver/generator solve remains OPEN.

No workaround or inferred score is substituted. The #366 successor must place actual counted
archive-description length in the same objective that renders through integer/uint8 and both
frozen scorers.

Primary evidence: `.omx/research/mdl_polytope_member_solve_receipt_20260721.json` and
`/Volumes/VertigoDataTier/pact/evidence/mdl_member_20260721/receipt_n64.json`
(file SHA-256 `b71ad6ab036b62fca640c4dc1a76fa37eec404901f6854239d35eb5c90d803a3`).

MAIN landing review is required.
