# Codex HNeRV Decoder Section Guard - 2026-05-14

## Scope

Follow-up to the post-xmember byte-target review. The highest byte-mass target
is the HNeRV decoder section (`inner_decoder_packed_brotli`, `169990` bytes in
the active xmember exact-CUDA packet). This pass does not create a new archive
or score claim; it hardens the guard that any future decoder-section successor
must clear before exact-eval dispatch.

## Landed engineering

- Extended `tac.hnerv_section_repack.audit_candidate_section_diff()` with three
  opt-in fail-closed requirements:
  - `require_raw_equivalence`
  - `require_byte_reduction`
  - `require_same_runtime_full_frame_parity`
- Expanded raw-equivalence auditing to cover active sidecar-wrapper section
  names, including `inner_decoder_packed_brotli` and
  `inner_latents_and_sidecar_brotli`.
- Added same-runtime full-frame parity validation bound to source archive SHA,
  candidate archive SHA, streaming output digest equality, total-byte equality,
  all-pair coverage, and non-claiming parity proof metadata.
- Added CLI switches to `tools/audit_hnerv_section_candidate_diff.py` so normal
  operator flows can require the strict decoder-candidate proof packet before
  archive preflight.
- Added regressions proving a strict inner-decoder candidate passes only when
  raw equivalence, byte reduction, and same-runtime full-frame parity are all
  present. Missing parity, missing raw-equivalence proof, or section growth
  fails closed.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_hnerv_section_repack.py`
  - `11 passed`
- `.venv/bin/python -m pytest -q src/tac/tests/test_hnerv_decoder_recode.py src/tac/tests/test_hnerv_section_repack.py src/tac/tests/test_hnerv_entropy_candidate_packet.py`
  - `44 passed`
- `git diff --check`
  - clean

## Interpretation

This is an enabling guard, not a byte win. The next decoder-section candidate
can now be audited with one command and will be refused unless it proves:

1. the changed decoder section is byte-smaller;
2. the recoded decoder raw stream is equal to source;
3. same-runtime full-frame streaming output is identical across all 600 pairs;
4. the proof remains non-promotional until exact CUDA auth eval lands.

Only after that proof packet should a lane claim and exact CUDA dispatch be
considered.
