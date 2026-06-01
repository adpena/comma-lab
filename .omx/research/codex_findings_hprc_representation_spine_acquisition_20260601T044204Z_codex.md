# HPRC Representation Spine Acquisition Landed

UTC: 2026-06-01T04:42:04Z

## Verdict

Compact learned/programmatic renderers now enter acquisition through one HPRC
charged-section spine instead of family-specific readiness readers.  PR95
u32-HNeRV, packed HNeRV, PACT-NeRV-VQ, and generic RNeRV/PACT-NeRV/Tree/Hi/SR/
VQ-NeRV/SIREN/FINER/C3-Cool-Chic/procedural-prior blobs can all emit the same
false-authority packet projection:

- decoder/program weights -> `decoder_qw`
- latents -> `latents_rc`
- codebooks/atoms -> `codebooks_q`
- selectors/temporal policy -> `selectors_rc`
- scorer-priced residuals -> `residual_rc`
- allocation hints -> `rdo_plan`
- charged constants/headers -> `receiver_state`

Acquisition now spends from that shared spine plus hard byte ceilings.  Residual
sidecars are blocked unless measured full-video non-rate improvement beats the
contest rate cost, i.e. `delta_nonrate + 25*delta_archive_bytes/N < 0`.

## Live Projection Rows

Live projection and acquisition reports are on the SSD tier, not local scratch:

- PR95/HNeRV projection:
  `/Volumes/VertigoDataTier/pact/hprc_representation_spine_live_pr95_20260601T043416Z/hprc_representation_spine_manifest.json`
  -> source archive `178,417` bytes, rate term `0.11880055683919845`.
- Packed HNeRV projection:
  `/Volumes/VertigoDataTier/pact/hprc_representation_spine_live_hnerv_20260601T043416Z/hprc_representation_spine_manifest.json`
  -> source archive `178,258` bytes, rate term `0.11869468526565202`.
- PACT-NeRV-VQ projection:
  `/Volumes/VertigoDataTier/pact/hprc_representation_spine_live_pvq_20260601T043452Z/hprc_representation_spine_manifest.json`
  -> source archive `135,960` bytes, rate term `0.09053018326649041`,
  but declares `32` pairs and is therefore blocked from full-600 base comparison.
- Acquisition report:
  `/Volumes/VertigoDataTier/pact/hprc_spine_acquisition_live_20260601T044157Z/hprc_spine_acquisition_queue.json`
  -> no full-coverage row fits `178,000` bytes; packed HNeRV is the best
  full-coverage row under `216,000` and `285,000`.

## Discipline

The PACT-NeRV-VQ row is intentionally not allowed to win the byte race until it
scales to the contest `600` pair coverage.  This prevents the exact failure mode
where a short-render or proxy representation looks superior by omitting video
coverage.  All rows remain non-promotable and non-score-authoritative until a
byte-closed archive, receiver proof, and exact CPU/CUDA eval exist.

## Next Actuator

The next highest-EV actuator is a full-coverage compact-base sweep under hard
ceilings, emitting this spine by default, followed by scorer-priced residual
admission only where replayed value-per-byte beats the rate term.
