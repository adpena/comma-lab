---
title: DDM E1 runtime exporter Build 636 DAG feed
date_utc: 2026-07-23
lane_id: lane_ddm_e1_runtime_exporter_20260723
task: 636
research_only: true
execution_allowed: false
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
main_landing_review_required: true
---

# Verdict

`PASS_EXACT_N600_RUNTIME_EXPORT_ADVISORY_ONLY`

The corrected 368-DOF `v15_j2_lane_seed_theta0` state now exports to one
deterministic counted archive plus a standalone generic receiver. A clean
single-thread CPU inflate reproduced every source camera byte and finished
well inside the 1,800-second gate.

# Exact identities

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `packet/archive.zip` | 339,094 | `05775433089d6aa2ae6800f2f8551358252d91288dcc1f1dbbfcc0d5517f26c1` |
| `packet/inflate.py` | 28,108 | `453a9b5b6aaf133662f57e63105c3828d55ab7ef206e1722f8dc6e7c1a36b4e3` |
| `packet/inflate.sh` | 264 | `4bea5bf1f31dca1feadad4b928e66230094f68668a652d8d929bd754695f0dc5` |
| source receiver raw | 3,662,409,600 | `5936308b2a37221ed33f743463889c66f0f59863045cb753104922ec295ac838` |
| packaged receiver raw | 3,662,409,600 | `5936308b2a37221ed33f743463889c66f0f59863045cb753104922ec295ac838` |

Raw identity is exact, not perceptual: source and packaged bytes have the
same length and SHA-256 over the complete canonical 1,200-frame stream.

# Receiver and timing proof

- Clean-from-empty inflate total: **237.560918 seconds**.
- Render portion: **51.412196 seconds**.
- Stage policy: 38 preserved atomic checkpoints, 16 pairs per full stage.
- Runtime dependencies: exactly `torch` and `brotli`.
- Runtime stays on CPU even when the host exposes CUDA; no host-capability
  refusal or device-dependent output branch remains.
- Counted ZIP members: canonical manifest, chart state, and composed semantic
  sufficient statistic. All 339,094 ZIP bytes have exactly one member or
  container home.
- The manifest exposes independently framed typed slots for amplitude,
  per-element tolerance duals, texture-quotient residual statistics,
  coder/probability parameters, and realization-map metadata. They are
  explicitly inactive here, so no no-op section bytes are counted.
- Language version `ddm_composed_language.v1` and all L/D1-D6 version stamps
  were checked against live section hashes at receiver consumption time.
- Runtime cleanliness scan: no weights, target labels, target poses, cache
  keys, sealed source SHA literal, video-derived literal, or undeclared
  dependency.

# Advisory S row

The frozen scorer was re-run on the exact seeded state; the unseeded v15 row
was not borrowed.

| Term | Measured value | S contribution |
|---|---:|---:|
| d_seg | `0.028614807129` | `2.861480712900` |
| d_pose | `163.052947489859` | `40.37981519148633` |
| archive bytes | `339094` | `0.22578877585000956` |
| total S | — | **`43.46708468023633`** |

This row is `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`. It is
not contest-CPU, contest-CUDA, promotion, or pointer authority.

# Receiver-effective dimension

- G1 island translations: 326 DOFs.
- Lane counted programs: 24 DOFs.
- Shared one-cell template RGB: 18 DOFs.
- Total: **368 DOFs**.

The stale 706 count is not propagated.

# Fixed-budget joint iteration curve

Cycle 0 is measured and receiver-closed; it is not a fixed-point claim.

| Cycle | L bytes | D2 | D1 | D4 | D6 | D5 | Argmax agreement proxy | d_seg | d_pose | S |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 333,571 | 0 | 0 | 346 | 5,177 | 0 | `0.971385192871` | `0.028614807129` | `163.052947489859` | `43.46708468023633` |

The byte columns sum exactly to 339,094. The measured camera-paint described
fraction is `0.501102493834`. The registered order is
`L -> D2 -> D1 -> D4 -> D6 -> D5`; global reinvestment is enabled and
the stop law is a full joint cycle with no net gain at constant bytes.
Status is `OPEN_ONE_MEASURED_EXPORT_CYCLE_ONLY`; successor cycles remain owed.

# Semantic-paint J_paint

The exact `semantic label -> floor-index camera gather -> role RGB overwrite`
map was derived and measured on the named state:

- equation: `ddm_semantic_paint_camera_uint8_jacobian_v1`;
- painted camera pixels across n600 and both frames: **611,747,528**;
- per-label-cell camera support across both frames: **8 to 18 pixels**;
- a legal unit palette coefficient perturbation changes exactly one output
  uint8 byte per active camera pixel in that role;
- the receipt binds the semantic member and state archive hashes and requires
  re-derivation on either mismatch.

# Frozen upstream harness rehearsal

The actual frozen upstream shell and Python evaluator passed against an
SSD-only copy of the exact packet:

- upstream git: `11ad728f563d8970929e8947a1cf6124ee6303e4`;
- end-to-end wall clock: **620.655014 seconds**;
- archive bytes observed: **339,094**;
- raw identity: 3,662,409,600 bytes,
  `5936308b2a37221ed33f743463889c66f0f59863045cb753104922ec295ac838`;
- upstream CPU-advisory report: `d_seg=0.02861482`,
  `d_pose=163.05291748`, rounded S `43.47`.

First contact failed twice before bulk scoring: once because the harness did
not receive the configured Brotli-capable interpreter, then because resolving
the venv symlink selected its base interpreter and lost site packages. Both
failure receipts are preserved. The final wrapper binds the literal venv path
in argv; this is the concrete harness-interface fix. Its PASS additionally
requires `failure_reasons=[]`, exact archive bytes, exact raw bytes/SHA,
successful report parsing, zero exit status, and the wall-clock guard.

# Durable proof

- Export receipt:
  `.omx/research/ddm_e1_runtime_exporter_n600_20260723/ddm_e1_runtime_export_receipt.json`
- Verification receipt:
  `.omx/research/ddm_e1_runtime_exporter_n600_20260723/ddm_e1_runtime_verification_receipt.json`
- Frozen upstream harness receipt:
  `.omx/research/ddm_e1_runtime_exporter_n600_20260723/ddm_e1_upstream_harness_receipt.json`
- SSD proof root:
  `/Volumes/VertigoDataTier/pact/evidence/ddm_e1_runtime_exporter_20260723`
- Scorer batch digest:
  `065097592b42b99f43fa4987a8c937db5a662529f60783ac57ec4bd73a6003dc`
- Superseded prototypes were certified by size/SHA and losslessly moved to
  the SSD path recorded in `superseded_packet_manifest.json`.

# Exact re-derive

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/export_ddm_runtime.py \
  --config .omx/research/configs/ddm_e1_runtime_exporter_n600_20260723.json

unzip -n \
  .omx/research/ddm_e1_runtime_exporter_n600_20260723/packet/archive.zip \
  -d /Volumes/VertigoDataTier/pact/evidence/ddm_e1_runtime_exporter_20260723/extracted

PYTHON=/Users/adpena/Projects/pact/.venv/bin/python \
  bash .omx/research/ddm_e1_runtime_exporter_n600_20260723/packet/inflate.sh \
  /Volumes/VertigoDataTier/pact/evidence/ddm_e1_runtime_exporter_20260723/extracted \
  /Volumes/VertigoDataTier/pact/evidence/ddm_e1_runtime_exporter_20260723/fresh/inflated \
  /Users/adpena/Projects/pact/upstream/public_test_video_names.txt

/Users/adpena/Projects/pact/.venv/bin/python \
  tools/rehearse_ddm_runtime_upstream.py \
  --config .omx/research/configs/ddm_e1_upstream_harness_20260723.json

/Users/adpena/Projects/pact/.venv/bin/python \
  tools/verify_ddm_runtime_export.py \
  --export-config .omx/research/configs/ddm_e1_runtime_exporter_n600_20260723.json \
  --scorer-config .omx/research/configs/ddm_v15_scorer_solved_templates_n600_20260722.json
```

# DAG delta

`sealed v15 archive -> complete J2 Lane seed -> corrected 368-DOF state ->
materialized semantic sufficient statistic + chart state -> deterministic
stored ZIP -> strict standalone receiver -> 38 stage checkpoints -> atomic
camera raw -> exact source/package identity -> fixed-budget cycle-0 receipt ->
advisory frozen-scorer S row`

No frontier pointer moved. MAIN review is required before any landing,
contest-axis replay, or promotion use.

# STORES CONSULTED

Build #636 authority prompt; `CLAUDE.md`; `AGENTS.md`;
`docs/operating_manual_craft_handoff.md`; v15 archive and receipt; J2 receipt
and corrected consumer source; v10 production receiver #402 patterns; target
cache custody; canonical lane/subagent state; per-arm and broadcast inboxes.
