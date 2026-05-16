# TT5L side-info consumption proof

Date: 2026-05-16

Scope: local no-GPU L5-v2 gate proof for `time_traveler_l5_autonomy`.

Artifact:

- Proof JSON: `.omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.json`
- Proof SHA-256: `8bb68ba5e14f0bbb0511812cbb7b7465e58ef639997e300558c04c3cdae98605`
- Inflated-output manifest: `.omx/research/tt5l_sideinfo_consumption_manifest_20260516_codex.json`
- Producer: `tools/prove_tt5l_sideinfo_consumption.py`

Verdict:

- `PER_PAIR_SIDE_INFO_BLOB` is parser-consumed and changes inflated raw output under byte mutation.
- `AC_STATE_BLOB` is parser-consumed and changes inflated raw output under byte mutation.
- The L5-v2 local parser/inflate consumption proof is bound by path and SHA, but it no longer satisfies the full `byte_closed_temporal_sideinfo_consumption` gate by itself.
- This is not a score claim and not promotion evidence.

Gate boundary update:

As of 2026-05-16, this two-frame local proof is classified as `local_consumption_proof`.
The full L5-v2 gate requires contest-scale full-frame custody: 600 pairs / 1200
frames, file-list SHA-256, and distinct source/candidate raw-output aggregate
SHA-256s in both the proof and inflated-output manifest. This artifact remains
useful parser-consumption evidence, but not a dispatch-unlocking proof.

Important limitation:

`AC_STATE_BLOB` is consumed today as residual calibration, not as a real range/ANS arithmetic decoder. The next score-lowering task is to replace calibration with a byte-closed entropy decode path and rerun this proof plus paired CPU/CUDA exact anchors.

Propagation:

- `TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT.runtime_overlay_consumed=True`
- `TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT.score_improvement_mechanism_status="OPERATIONAL"`
- `canonical_substrate_inventory().time_traveler_l5_autonomy.sideinfo_consumed=True`
- The composition row no longer carries `requires_byte_closed_temporal_sideinfo_consumption_proof`; remaining blockers are paired CPU/CUDA axis plan and C1/Z5/TT5L probe disambiguation.
