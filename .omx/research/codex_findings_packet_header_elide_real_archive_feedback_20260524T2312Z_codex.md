# Codex Findings: Packet Header Elide Real-Archive Feedback

UTC: 2026-05-24T23:12:00Z
Lane: `codex_packet_header_elide_real_archive_feedback_20260524`

## Finding

The first executable `packet_member_zip_header_elide_v1` implementation was
byte-closed on fixture archives, but a real deflated archive exposed a
contaminating effect: using Python `zipfile` to rewrite the archive stripped
24 bytes of selected header metadata but recompressed `renderer.bin`, growing
`submissions/robust_current/archive_correct.zip` by 916 bytes.

That was not a method negative; it was an implementation bug. Header elision
must preserve the compressed member stream unless a separate recompression
operation is explicitly selected.

## Landing

Replaced the header-elide writer with a low-level ZIP wire rewrite:

- parses local headers directly;
- copies each selected member's compressed payload bytes unchanged;
- rewrites local and central headers with selected extra/comment fields elided;
- rewrites EOCD offsets/sizes deterministically;
- refuses data-descriptor and ZIP64 cases fail-closed.

Added `tools/run_family_agnostic_materializer_sweep.py` to turn proof-chain
results into typed `family_agnostic_materializer_empirical_observation.v1`
rows plus planner feedback. This keeps real materializer observations out of
chat and gives the acquisition layer a small, structured feedback surface.

Wired those observations into
`tac.optimization.inverse_steganalysis_acquisition` by preserving materializer
`target_kind`, `materializer_id`, `receiver_contract_kind`, deterministic rate
gain, runtime identity, and cache identity during normalization. Operation-set
compiler hints now match feedback rows by target/materializer identity, not only
by exact candidate id, so grouped action sets can reuse real materializer
evidence.

## Empirical Anchor

Command:

```bash
.venv/bin/python tools/run_family_agnostic_materializer_sweep.py \
  --archive robust_current_archive_correct=submissions/robust_current/archive_correct.zip \
  --member-name renderer.bin \
  --output-dir experiments/results/packet_header_elide_sweep_20260524T2324Z \
  --output-json experiments/results/packet_header_elide_sweep_20260524T2324Z/sweep.json \
  --observation-jsonl experiments/results/packet_header_elide_sweep_20260524T2324Z/observations.jsonl
```

Result:

- source archive: `345802` bytes, SHA-256 `4dd46fed78ed064bc97c9b3205088e82838c03667394f7936c8ae8d20f9837ab`
- candidate archive: `345750` bytes, SHA-256 `a5b58ade4d9cfdbcaed2a3236f8b815226a352072972de10056347db8912b770`
- saved bytes: `52`
- selected member payload SHA-256 preserved
- selected member compressed-byte count preserved
- receiver proof passed
- observation axis: `[local-materializer-proof]`
- observed deterministic rate gain: `3.462466556235291e-05`
- score authority remains false

This is a local proof-chain and real-archive rate observation, not an auth-eval
score claim.

## Verification

```bash
.venv/bin/ruff check src/tac/optimization/family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializers.py tools/run_family_agnostic_materializer_sweep.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_acquisition.py
.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_inverse_steganalysis_acquisition.py -q
```

Result: ruff clean; `53 passed`.

## Remaining Work

The same raw ZIP writer should be reused for all-member header elision and
packet-member reorder so the final-byte optimizer can collect the remaining
per-member extra-field bytes without recompression contamination.
