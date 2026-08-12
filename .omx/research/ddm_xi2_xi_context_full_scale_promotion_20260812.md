# DDM XI2 full-scale xi-context promotion build

Date: 2026-08-12

Status: **READY_TO_FIRE**. This unit built and CPU-verified the full-n600
xi-context treatment, retained its inputs and deterministic context payloads,
and queued the governed MAIN Metal run. It did not run Metal, a scorer, an
evaluator, or an archive build. It did not produce a full-scale XI2 Range byte
count and did not move either frontier.

## Result boundary

| Row | Range token bytes | Evidence | Disposition |
|---|---:|---|---|
| Banked CL1 lambda-1.0 unwarped-previous-partition control | 116,716 | MEASURED real Range, full n600, exact decode | Fixed comparator; not retrained |
| XI1 xi selected-n120 screen projected to n600 | 121,100 | PROJECTION only, 5x selected-token extrapolation from a weak baseline | Excluded from promotion |
| XI2 xi-warped-previous-decoded-partition treatment | pending | MAIN Metal full n600, 60 epochs, lambda 1.0 | Must be at most 114,381 B |

The preregistered promotion rule is strict: `xi/control < 0.98`. Therefore
114,381 bytes passes and 114,382 bytes fails. Failure is
`FORMULATION_CLOSED_FULL_SCALE`; XI1's 14.6x n120 ratio is not transferred.

The banked comparator is under
`/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_uninterrupted_twin/training/`:

- Range: 116,716 B, SHA-256
  `ac2c549c1f48756ad33c6c99af8563f2170db1de61cd50d0615d4c1a0cdd7b87`.
- Decoded tokens: 117,964,800 B, SHA-256
  `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.
- Packed model: 15,088 B, SHA-256
  `b74be4d5f4c8f7f1aec37577f2277d43ca44ef6d53e5b0138a8ce5e7d7e02325`.

All three artifacts, the four safe-run receipts, the CL1 trainer report, the
CL1/XI1 sources, PR130 coder sources, xi geometry sources, EMA sources,
canonical cache, initializer, pose rows, and calibration are SHA-pinned by the
XI2 runner. The pinned CL1 report's complete training config must equal the XI2
config except for the context mode.

## Built mechanism

The owned runner is `tools/run_ddm_xi2_xi_context_full_scale.py`, SHA-256
`df8bd3ad6374509744283670601a32c8ae1d0a2db984b1d4976ac648e4b66e2e`.
It leaves the receipt-pinned CL1 and XI1 sources unchanged. Its treatment uses
the banked CL1 topology, capacity, seed, optimizer, EMA law, 60-epoch schedule,
lambda 1.0 rate term, epoch-31 QAT transition, terminal EMA deployment, real
PR130 self-compression, sparse integer HPAC, and `constriction` Range coder.
The only treatment coordinate is the context plane.

The named receiver derivation is:

`causal_context_for_frame -> derive_xi_context -> XI1 warp_previous_partition -> tac.lie SE(3) class composite`

Frame 0 is zero. At frame `t > 0`, both encoder and decoder use the previous
exact decoded partition and the already-counted pose row for `t`. Ground-plane
classes 0/1/3 use the SE(3) ground homography, class 2 uses rotation only, and
class 4 remains image-fixed. Lossless token decode makes the training
teacher-forced plane exact. The retained full context is training and debug
evidence only; the codec never reads it as a receiver sidecar. Generic
derivation code is rule-118 free, while no video-derived content is hidden in
code.

The runner exposes only declared `--leg` values, uses `--resume-from auto`,
stores an immutable checkpoint after every epoch plus both stage ends, deploys
the terminal EMA shadow, and verifies the checkpoint's complete causal-state
digest before resume or pack. Checkpoints include live weights, EMA, optimizer,
scheduler, MPS/CPU/Python/NumPy RNG state, source/input/config identity, and
resume lineage. Every packed, Range, repeat, and decoded payload is retained
with bytes and SHA-256.

## CPU build evidence

- Targeted pytest: 5/5 passed. It covered deterministic zero-screw context,
  frame-0 zero, supplied-previous-decode causality, real PR130 pack/unpack exact
  logits, parser legs/auto-resume, and the strict 114,381-byte boundary.
- Ruff lint and format check: passed on both owned Python files.
- Python compile: passed on both owned Python files.
- Strict payload-retention preflight: 2 Python files, 0 findings.
- Governed CPU prepare: exit 0 under `safe_run`; no MPS or scorer work.
- Storage preflight: 727,973,756,928 B free versus 4,294,967,296 B required at
  the final prepare replay.
- Retained primary and repeat n600 contexts: 117,964,928 B each, identical file
  SHA-256 `4280603ab16cd3d1b7c34ce0eee291f6adb0e8974670b8c712424ff48bb1da44`
  and identical raw-array SHA-256
  `435458bd9ae7109a0088cad81b2fe167d193962410137d93797b9f86e16410b0`.
- Build receipt:
  `/Volumes/APDataStore/pact/ddm_xi2_20260812/BUILD_RECEIPT.json`, SHA-256
  `e91eb294416d7781f6a631411c791abcab1bd922de895c543f713de283bfec1c`.
- Fire receipt:
  `/Volumes/APDataStore/pact/ddm_xi2_20260812/READY_TO_FIRE.json`, SHA-256
  `9462243fd9796dba3f0d72023824ef5ef8d2fe3d5e933c4eab956ab12ab31a93`.

The 4 GiB run projection is derived, not measured on XI2: banked matched CL1
peaked at 1,673.391 MiB; each full `long` target/context tensor is 943,718,400
B and the retained uint8 context is 117,964,800 B. The expected 82.4104-minute
pipeline time is also derived from banked CL1 receipts: 2,894.155 s training,
2.225 s pack, two 679.825 s encodes, and 688.595 s decode. Xi warp overhead is
unmeasured.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus, arm receipts, tools, source, prompt
tree, canonical equation listing, research indexes, DAG FEED blocks, and live
state by content. Query families included `xi`, `screw`, `warp previous`,
`previous partition`, `temporal context`, `HPAC`, `transport amortization`,
`jitter`, `CL1`, and `event coordinates`.

Beyond the charter seeds, recall found:

- `partition_temporal_transport_amortization_jitter_bound_v1` records an n600
  negative for raster/zlib transport, but its domain is formulation-scoped and
  explicitly names a learned boundary-context coder as a reactivation trigger.
  This prevented a false family-level kill of XI2.
- The XI1 memo's `spatial` zero-plane arm is not the banked CL1 full-scale
  control. The actual CL1 control consumes the unwarped previous decoded
  partition through `conv_past`. This changed the comparator to the pinned
  116,716-byte banked row and removed the zero-plane baseline from promotion.
- The concurrent EC1 evidence closes a whole adjacent-partition SP1 event
  grammar at n600 while leaving local sparse event coordinates plausible for
  HPAC. That is a separate successor hypothesis, not a change to XI2's single
  treatment coordinate.
- The banked CL1 receipts supplied the exact 60-epoch config, real
  48.2359167 s/epoch timing, 1,673.391 MiB peak, and pack/encode/decode timing.
  These replaced estimates with receipt-derived launch bounds.

## Fire order

Owner: MAIN Metal executor. Consumer store:
`/Volumes/APDataStore/pact/ddm_xi2_20260812/FULL_SCALE_RESULT.json`. Fire only
when live MPS is available, the local-Metal lane is free, all SHA pins still
match, and storage plus system-memory admission pass:

```bash
PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python tools/safe_run.py --rss-mb 4096 --projected-gib 4 --timeout 7200 --label ddm_xi2_xi_context_n600 --status-receipt /Volumes/APDataStore/pact/ddm_xi2_20260812/run/main.safe_run.json -- .venv/bin/python tools/run_ddm_xi2_xi_context_full_scale.py --leg all --resume-from auto
```

The effective frontier remains CP135 `S = 0.16195513827824176 @ 186,252 B`
`[contest-CUDA T4, n600]`. The own-vehicle frontier remains LC2
`S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`.
