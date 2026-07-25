---
schema: codex_findings.v1
date_utc: 2026-07-25T13:22:47Z
lane_id: lane_ddm_e5a_midcampaign_e5_adapter_20260725
research_only: true
score_claim: false
pointer_moved: false
main_landing_review_required: true
---

# Codex findings — DDM E5A mid-campaign adapter

## Verdict

`PASS_E5A_MIDCAMP_CHECKPOINT_TO_PACKET`.

The premise was verified before editing: E5 accepted receiver-closed WS1 bytes,
while the campaign checkpoint carried canonical optimizer/cursor state plus a
realized-archive identity but no archive payload. The missing type transition
was real. A typed adapter now restores the checkpoint through the canonical
resume registry, compiles the live `theta` shadow, and refuses any identity,
shadow, lane-materialization, stage, or ticket mismatch.

The adapter is not a parallel exporter. Its output enters the existing E5
compiler and existing two-stream grammar. E5 now optionally stores the
irreducible WS1 semantic components as canonical LA1-selected frames; the
generic carrier, coupled-margin, preuint8, and warm-start compilers remain free
receiver code in the embedded `inflate.py`.

## Measured result

- Checkpoint: 19,723 bytes, SHA `043c2a8b3c89688510cc0ff002f37a375a974205a5f8760d93133c47b7cec7c1`.
- Materialized state: 138,813 bytes, SHA `2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241`.
- LA1 selected frames: 127,951 bytes.
- Canonical semantic bundle: 128,001 bytes.
- Complete E5 packet: 130,101 bytes, SHA `fb69964da2649c310b7694416ff9863e13f54594af215cd771dcd50f5898a85d`.
- Exact packet custody: `/Volumes/VertigoDataTier/pact/ddm_ct1_campaign_telemetry_encode_20260725/e5a_runtime/candidate_packet/archive.zip`, certified by `packet_cold_store_receipt.json`; the gitignored archive bytes are intentionally excluded from the commit.
- Rebase: the prior 128,254-byte figure is not a closed packet; complete cost is
  +1,847 bytes, while the exact bundle itself is -253 bytes.
- Exact advisory n600: `d_seg=0.06974277072482639`,
  `d_pose=35.499820809591`, objective `25.902302117302376`.
- Axis: `[macOS-CPU frozen-scorer advisory]`; no contest score or promotion.

## Adversarial closure

- Repository parse-back and the emitted standalone runtime both reconstruct the
  source state byte-identically.
- Packet compilation repeats byte-identically.
- N600 uses batch32, four CPU threads, deterministic algorithms, exact frozen
  scorer/model hashes, and 19 preserved batch receipts.
- The psutil gate observed 90,005,356,544 available bytes against a
  21,474,836,480-byte minimum before n600.
- Fresh-process resume proof is green.
- The live campaign directory fingerprint is identical before and after R6.
- The old blocker `R6_BLOCKED_E5_MIDCAMP_CHECKPOINT_ADAPTER_ABSENT` is dissolved
  for this typed checkpoint class.

Two implementation defects failed closed during rehearsal and were fixed:

1. R6 initially omitted E5's mandatory `minimum_free_bytes`; the generated
   config now always supplies the explicit 8 GiB floor.
2. Runtime cleanliness initially lacked the four new generic bundled-module
   imports; the allowlist now names only LA1 and the three exact generic wrapper
   compilers. Forbidden tokens and long literal hashes remain rejected.

## Remaining authority

G4 is locally green only. G1-G3 and G5-G6 are unchanged; no fire is authorized.
The canonical start card is absent from this older isolated branch and hot on
MAIN, so the merge-ready G4 delta is a separate artifact rather than an add/add
card reconstruction. MAIN must review source hashes, hot-file overlap, packet
bytes, exact axis labels, and the card-row application before landing.
