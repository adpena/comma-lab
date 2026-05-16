# TT5L side-info consumption proof

Date: 2026-05-16

Scope: local no-GPU L5-v2 gate proof for `time_traveler_l5_autonomy`.

Artifact:

- Proof JSON: `.omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.json`
- Proof SHA-256: `8d3a2285c8b6b2804b78c01b50d857973fe0f553db3546a71a2a2959f3332c76`
- Inflated-output manifest: `.omx/research/tt5l_sideinfo_consumption_manifest_20260516_codex.json`
- Producer: `tools/prove_tt5l_sideinfo_consumption.py`

Verdict:

- `PER_PAIR_SIDE_INFO_BLOB` is parser-consumed and changes inflated raw output under byte mutation.
- `AC_STATE_BLOB` is parser-consumed and changes inflated raw output under byte mutation.
- The L5-v2 `byte_closed_temporal_sideinfo_consumption` gate accepts the proof artifact when bound by path and SHA.
- This is not a score claim and not promotion evidence.

Important limitation:

`AC_STATE_BLOB` is consumed today as residual calibration, not as a real range/ANS arithmetic decoder. The next score-lowering task is to replace calibration with a byte-closed entropy decode path and rerun this proof plus paired CPU/CUDA exact anchors.

Propagation:

- `TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT.runtime_overlay_consumed=True`
- `TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT.score_improvement_mechanism_status="OPERATIONAL"`
- `canonical_substrate_inventory().time_traveler_l5_autonomy.sideinfo_consumed=True`
- The composition row no longer carries `requires_byte_closed_temporal_sideinfo_consumption_proof`; remaining blockers are paired CPU/CUDA axis plan and C1/Z5/TT5L probe disambiguation.
