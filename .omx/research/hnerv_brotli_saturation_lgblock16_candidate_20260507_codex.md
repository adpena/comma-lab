# HNeRV Brotli Saturation And lgblock16 Candidate - 2026-05-07 Codex

## Scope

Bounded local work on the current exact frontier decoder entropy target only:
`PR106x-lowlevel-brotli` and its `decoder_packed_brotli` section. This pass did
not dispatch remote work, did not claim a lane, did not run GPU eval, and did
not make a score claim.

## Source Anchor

[empirical:experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip]

- source archive bytes: `186080`
- source archive SHA-256:
  `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- source payload bytes: `185980`
- source payload SHA-256:
  `b6ba493aa37446143b003235eaeb3a49c0748a6d64392fb9a666a6c872629171`
- source decoder section bytes: `170127`
- source decoder section SHA-256:
  `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`
- measured decoder section entropy: `7.998223575625` b/B

## Brotli Grid Audit

[empirical:experiments/results/hnerv_brotli_saturation_pr106x_20260507_codex/audit_lgblock_grid.json]

The extended local grid covered qualities `0..11`, modes `generic/text/font`,
default plus `lgwin=10..24`, and default plus `lgblock=16..24`:

- attempts: `5760`
- rate-positive attempts: `18`
- best section bytes: `170126`
- best section SHA-256:
  `a812f1e837afd0e463a7f133b680ea6c027339ff8816db7012dd41253435afbf`
- best params: `quality=10`, `lgwin=default`, `lgblock=16`, `mode=generic`
- best section delta: `-1` byte
- raw decoder roundtrip: `true`
- audit blockers: none

The default-lgblock-only grid had no rate-positive attempt, so the byte win is
specifically an explicit `lgblock=16` Brotli parameter improvement rather than
a generic quality/lgwin/mode-only recode.

## Candidate Archive

[empirical:experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex/result.json]

- candidate archive:
  `experiments/results/hnerv_lowlevel_repack_pr106x_lgblock16_20260507_codex/pr106x_lowlevel_brotli_hnerv_brotli_repack_candidate.zip`
- candidate archive bytes: `186079`
- candidate archive SHA-256:
  `866dc135e9168d61fab02b6b1c218c4b1d6eed779154a6dc3095fd05e48024f2`
- candidate payload bytes: `185979`
- candidate payload SHA-256:
  `0a83096defc59120ee551c45e73f69e089165df78ae706fbbe2be3e9bc284765`
- changed section: `decoder_packed_brotli`
- source section bytes: `170127`
- candidate section bytes: `170126`
- section byte delta: `-1`
- raw decoder SHA-256 before/after:
  `f22eb6be56499fa5785f47f85d2bef7f71246f29674691fd3e06af733c8c0703`
- latents raw SHA-256 before/after:
  `a38778c6304bacba39705cd9c45af337d73bc90c6b7b4ccf2563febfc312328e`
- `candidate_diff_audit.total_byte_delta`: `-1`
- `candidate_diff_audit.ready_for_archive_preflight`: `true`
- `candidate_diff_audit.ready_for_exact_eval_dispatch`: `false`
- `candidate_diff_audit.blockers`: none

This is a byte-different local archive candidate with charged-byte proof only.
It remains non-promotable until normal archive/pre-submission checks, lane
claim, and exact CUDA auth eval are explicitly authorized.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_lowlevel_packer.py src/tac/tests/test_hnerv_brotli_saturation.py -q
.venv/bin/ruff check src/tac/hnerv_lowlevel_packer.py src/tac/hnerv_brotli_saturation.py src/tac/tests/test_hnerv_lowlevel_packer.py src/tac/tests/test_hnerv_brotli_saturation.py tools/build_hnerv_lowlevel_repack_candidate.py tools/audit_hnerv_brotli_saturation.py
.venv/bin/python -m py_compile src/tac/hnerv_lowlevel_packer.py src/tac/hnerv_brotli_saturation.py tools/build_hnerv_lowlevel_repack_candidate.py tools/audit_hnerv_brotli_saturation.py
```

Results:

- focused pytest: `9 passed`
- ruff: `All checks passed`
- py_compile: passed
