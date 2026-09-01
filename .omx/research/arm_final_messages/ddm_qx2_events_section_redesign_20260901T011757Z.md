QX2 is complete with verdict `OVER`.

The address-free conditional section reconstructs all 17,926 events exactly:

- Payload: 22,661 B, Brotli q11
- Complete envelope: 136,553 B
- Conditional headroom: 1,432 B below the largest legal archive
- Archive SHA-256: `da63908d…4923d`
- Deterministic repeat: byte-identical

This does not clear QX1 overall: decoding depends on an external 117,964,800-byte GT/S2-derived baseline that no counted QX1 receiver produces. No scorer, contest evaluation, Metal, Modal, or pointer movement occurred.

Artifacts:

- [QX2 memo](/Users/adpena/Projects/pact/.omx/research/ddm_qx2_events_section_redesign_20260831.md)
- [Runner](/Users/adpena/Projects/pact/experiments/ddm_qx2_events_section_redesign.py)
- [Result](/Volumes/APDataStore/pact/ddm_qx2/RESULT.json)
- [Conditional archive](/Volumes/APDataStore/pact/ddm_qx2/retained/candidates/boundary_enumerative_r0/archive.zip)

Validation passed: exact independent parse-back, ruff, `py_compile`, zero payload-retention findings, and two review-tracker passes over 44 entities.

The shared branch remains at `117abd7964f5bd9a53ec0651431520cf0c407fc8`. Managed Git writes were denied, so the serializer returned rc=17 and retained fallback commit `25a4ae3069ef0db0a94b05cec1fd6a70f1ca4767` in the verified [bundle](/Volumes/VertigoDataTier/pact/ddm_qx2/receipts/commit_serializer_fallbacks/20260901T011645.488337Z-39064/intended-commit.bundle).

## NEXT_IF_RESUMED

- **QUEUED-WITH-FIRE-ORDER** — owner: MAIN-assigned QX1 receiver-binding arm; consumer store: `/Volumes/APDataStore/pact/ddm_qx2/RESULT.json`; fire trigger: a counted QX1 receiver produces baseline SHA-256 `02a2a3f5…ccf34` without GT/S2 tables. Then bind QXC1, rebuild, and promote only if exact parse-back and `<137,986 B` both survive.

## LIVE-HYPOTHESES

- A QX1-native continuous latent may generate the required radius-0 boundary field. This is plausible because 98.6891% of events lie exactly on it and the conditional archive has 1,432 B headroom.
- A different receiver baseline may work if it preserves deterministic candidate ordering and the event subset is re-ranked and re-priced against it.

## DEAD-ENDS

- Temporal persistence and bounded ego-shifts cover only 0.4128% and 3.7766% of events.
- Smooth curves and scanline/component grammars are contradicted by 15,397 singleton components among 16,369 total.
- Boundary radii 1/2/4, dense radius-0 occupancy, and distance ranks all remain over the gate.
- The 136,553-byte archive must not be promoted or scored while its external baseline and QX1 pose cap remain unresolved.

**Frontier line:** canonical pointer remains **S = 0.14797617125559104 @ 180,002 B `[contest-CUDA T4 n600]`**. QX2 moved no score or pointer.