# Codex Findings: HFV Sidecar Frontier Decision Packet

- timestamp_utc: 2026-05-21T20:17:00Z
- lane: hfv_sidecar_frontier_decision_packet
- status: LANDED_HFV_POLICY_SELECTION_SURFACE
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New planner:

- `tools/plan_hfv_sidecar_frontier_decision.py`

The planner compares the current parity-proven HFV7/HFV8/HFV9 sidecar
artifacts, verifies local custody, and emits an operator-routable exact-eval
recommendation. It does not score and does not dispatch Modal.

## Decision packet artifact

- Output directory: `experiments/results/hfv_sidecar_frontier_decision_packet_20260521T201618Z`
- JSON: `experiments/results/hfv_sidecar_frontier_decision_packet_20260521T201618Z/hfv_sidecar_frontier_decision_packet.json`
- Markdown: `experiments/results/hfv_sidecar_frontier_decision_packet_20260521T201618Z/hfv_sidecar_frontier_decision_packet.md`

Hashes:

```text
fe8e5864303664ef40bfb85f01994e03b0d24ecab9927f088719bf3883b7ccee  hfv_sidecar_frontier_decision_packet.json
ca690bdeec9b47e54973e49a76cfd3bbd344e46c70418c60cce6a0dcb4e35b89  hfv_sidecar_frontier_decision_packet.md
```

## Result

All three current HFV candidates pass local audit:

```text
hfv7_exp_golomb             audit_errors: none
hfv8_explicit_row           audit_errors: none
hfv9_magic_explicit_row     audit_errors: none
```

Policy ranking:

```text
1. hfv9_magic_explicit_row  178553 bytes  row_archive_contained_magic_identified
2. hfv8_explicit_row        178549 bytes  row_archive_contained_length_discriminated
3. hfv7_exp_golomb          178529 bytes  rate_minimal_profile_row_runtime_profile
```

Byte ranking:

```text
1. hfv7_exp_golomb          178529 bytes
2. hfv8_explicit_row        178549 bytes
3. hfv9_magic_explicit_row  178553 bytes
```

The recommended exact-eval candidate after the claim surface clears is
`hfv9_magic_explicit_row`, because it is the most compliance-defensible
byte-closed HFV packet: active row and format magic are both charged inside
`archive.zip`.

HFV7 remains the rate-minimal alternative only if runtime-profile
interpretation is explicitly accepted. HFV8 remains the middle policy if
archive-contained active-row bytes are required but a length-only format
discriminator is accepted.

## Current blocker

No paired Modal exact eval was executed. The decision packet records
`dispatch_blocked: true` because `tools/claim_lane_dispatch.py summary`
reported `active=13`.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/plan_hfv_sidecar_frontier_decision.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/plan_hfv_sidecar_frontier_decision.py \
  --active-claim-count 13 \
  --output-dir experiments/results/hfv_sidecar_frontier_decision_packet_20260521T201618Z
```
