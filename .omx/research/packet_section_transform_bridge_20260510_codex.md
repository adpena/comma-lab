# Packet-section transform bridge (2026-05-10)

Generated: `2026-05-10T18:05:00Z`

`score_claim=false`; `dispatch_attempted=false`; `ready_for_exact_eval_dispatch=false`.

## Why this exists

The public HNeRV PR deconstruction, PR103 destructive negative, PR106 q10
rate-only positive, and PR101 CUDA/proxy drift all point to the same missing
abstraction: score-lowering transforms must operate on parser-proven archive
sections, not on loose blobs or prose-level codec ideas.

This landing starts that bridge in `src/tac/packet_section_transform.py`.

## Implemented

- `PacketIR`: typed archive/member/section identity built from
  `tac.analysis.hnerv_packet_sections`.
- `SectionIR`: typed section name, offset, length, SHA-256, and role.
- `PacketSectionTransform`: protocol for deterministic byte-section transforms.
- `TransformOutput`: transformed bytes plus metadata and blockers.
- `BrotliRecodeSectionTransform`: reusable wrapper around the existing
  HNeRV Brotli recode search.
- `CompositePacketSectionTransform`: one-transform-per-section wrapper for
  deterministic multi-section compiler calls.
- `compile_hnerv_pr106_section_transform_candidate`: deterministic PR106
  single-member compiler that updates the `0xff + len24` header, writes a
  stored ZIP, emits source/candidate IR, records old/new section hashes, and
  refuses exact-eval authority.
- `tools/build_hnerv_packet_section_transform_candidate.py`: operator-facing
  CLI for PR106 Brotli section recode candidates. It is intentionally
  `ready_for_archive_preflight` only and always refuses exact-eval dispatch
  authority.

## Important correctness detail

The compiler distinguishes:

- `content_changed`: section bytes changed.
- `length_changed`: section length changed.
- `offset_changed`: section boundary moved even if bytes are identical.

This matters for PR106: shrinking `decoder_packed_brotli` changes the len24
header and moves `latents_and_sidecar_brotli` even when latent bytes are
unchanged. Treating offset motion as invisible would weaken runtime-consumption
and no-op proofs.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_packet_section_transform.py \
  src/tac/tests/test_hnerv_lowlevel_packer.py \
  src/tac/tests/test_hnerv_packet_sections.py -q
# 22 passed

.venv/bin/python tools/build_hnerv_packet_section_transform_candidate.py --help
# CLI surface prints the PR106 recode contract.
```

## Limits

This is not a score claim and not a dispatch candidate. It supports PR106
`0xff + len24` HNeRV packet compilation first. PR101/A1 and PR103 arithmetic
composition still need transform implementations that prove runtime adapter
closure on their grammars.

## Next implementation steps

1. Add a PR103 arithmetic section-transform wrapper that can run through this
   bridge and emit old/new AC section hashes.
2. Add a PR101/A1 transform path using `tac.analysis.hnerv_packet_sections`
   parser manifests and PR101 split-Brotli codec constants.
3. Teach the exact-eval readiness queue to consume this bridge output directly
   instead of bespoke low-level repack manifests.
4. Keep exact CUDA dispatch gated on archive SHA, runtime tree SHA, no-op proof,
   raw-equivalence or scorer-visible-change proof, and terminal lane claim.
