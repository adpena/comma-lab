# Full-Stack Compression Audit Worker - 2026-05-03

Scope: local archive-size and full-pipeline byte audit only. No remote GPU,
Lightning, Modal, Vast.ai, training, or exact-eval dispatch was performed by
this worker. No score claim is made here; exact CUDA/T4 JSONs are referenced
only as existing custody inputs.

Write scope:

- Results: `experiments/results/full_stack_compression_audit_worker_20260503/`
- Ledger: `.omx/research/full_stack_compression_audit_worker_20260503.md`

## Inputs And Generated Artifacts

Primary generated artifacts:

- `experiments/results/full_stack_compression_audit_worker_20260503/exact_t4_archive_byte_accounting_collection.json`
- `experiments/results/full_stack_compression_audit_worker_20260503/exact_t4_archive_byte_accounting_collection.md`
- `experiments/results/full_stack_compression_audit_worker_20260503/exact_t4_atom_table.csv`
- `experiments/results/full_stack_compression_audit_worker_20260503/current_frontier_top40_p6_with_actions_profile.json`
- `experiments/results/full_stack_compression_audit_worker_20260503/current_frontier_top40_p6_atom_table.csv`
- `experiments/results/full_stack_compression_audit_worker_20260503/c082_276333_lossless_profile.json`
- `experiments/results/full_stack_compression_audit_worker_20260503/c082_276333_lossless_atom_table.csv`
- `experiments/results/full_stack_compression_audit_worker_20260503/local_renderer_blocker_archive_byte_accounting_collection.json`
- `experiments/results/full_stack_compression_audit_worker_20260503/unsupported_payload_zip_profiles.json`

Existing exact T4 archives profiled:

| label | archive bytes | SHA-256 | existing exact score source |
|---|---:|---|---|
| C088/top40 P3 predecessor | 276386 | `9feef7ffaa254f9e5408996a122682757a054144a3000539553786c5292b7d0a` | `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/contest_auth_eval.json` |
| current read: top40 P6 | 276342 | `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8` | `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/contest_auth_eval.json` |
| C082 276333 lossless repack | 276333 | `30932c684c6b09a7bd2bd248e17fd66a4bc448ed2fe5747f92e74bcceec66681` | `experiments/results/lightning_batch/exact_eval_c082_p6_stream_resweep_276333_t4_20260503T0705Z/contest_auth_eval.json` |
| C082 276394 P6 repack | 276394 | `9b78333dd39c12c986ca7fc02a4bb35a6a61ef5a1cc0c9c9c840820eb840058a` | `experiments/results/lightning_batch/exact_eval_c082_qp1_p6_delta_varint_actions_stream_resweep_t4_20260503T0626Z/contest_auth_eval.json` |
| lag_eval top67 P6 | 276352 | `d73f0957cf2d8da0526c1786613443331ccb913f5648131cfaaac5e6f7eae972` | `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_lag_eval_top67_p6_t4_20260503T0608Z/contest_auth_eval.json` |
| lag_eval pose2 top67 P6 | 276338 | `af7a34cb1c051b1accebe2768245a44f55280e2596b315f8e4809a73a23926cd` | `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_lag_eval_pose2_top67_p6_t4_20260503T0608Z/contest_auth_eval.json` |
| lag_eval pose4 top67 P6 | 276338 | `de7d549ec4437b3ed9508c1f62f8a79bef88440ba3db0f4a7de5a2c83fe160ef` | `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_lag_eval_pose4_top67_p6_t4_20260503T0626Z/contest_auth_eval.json` |
| pose-safe positive ampminus1 P6 | 276317 | `6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796` | `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_pose_safe_positive_ampminus1_p6_t4_20260503T0632Z/contest_auth_eval.json` |
| public PR75 QP1 replay | 276741 | `cc1dfa346811dade15044910fa0143e73a7a049b98aeec39b96a53481a0788bd` | `experiments/results/lightning_batch/exact_eval_pr75_qp1_public_replay_t4_20260503T0608Z/contest_auth_eval.json` |

## Current Byte Anatomy

Current exact frontier read for this audit is the already-harvested
`c067_pr75_qp1_top40_p6` artifact, not the older C067/C088 notes.

For `archive.zip` SHA
`0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`:

| layer | bytes | notes |
|---|---:|---|
| ZIP wrapper | 100 overhead | single stored member `p`; no wrapper-scale opportunity |
| payload header | 12 overhead | `public_pr75_qzs3_qp1_segactions_p6_delta_varint` |
| `masks.mkv` | 219472 | `brotli_av1_obu`, decoded SHA `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb` |
| `renderer.bin` | 55965 | `brotli_qzs3`, decoded SHA `5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb` |
| `optimized_poses.qp1` | 677 | `public_qp1_brotli`, too small for byte-only target |
| `seg_tile_actions.bin` | 116 | `brotli_seg_tile_actions_delta_varint_v1`, too small for byte-only target |

Target pressure from existing exact score for this archive:

- Sub-`0.314` at unchanged components needs `2209` bytes removed, target archive
  `274133` bytes.
- Sub-`0.300` at unchanged components needs `23235` bytes removed, target archive
  `253107` bytes.
- The sub-`0.314` byte gap is about `1.01%` of the mask stream or `3.95%` of the
  renderer stream.
- The sub-`0.300` byte gap is about `10.59%` of the mask stream or `41.52%` of
  the renderer stream.

Profiler probes found `0` deployable nested recompression savings for
`masks.mkv`, `renderer.bin`, `optimized_poses.qp1`, and `seg_tile_actions.bin`.
The attack surface is representation, prediction, quantization, hyperprior,
arithmetic, and pack format, not generic recompression.

## Active-Candidate Read

Exact T4 replay already shows the PR75 P6/action atom family is real but small.
The best exact artifact in this local read is `top40_p6`; the C082 276333
lossless repack is 9 bytes smaller but scored worse in the existing exact T4
artifact because components drifted. That means action/packer micro-work should
be kept as additive polish, not as the main wall-clock path.

Stream deltas of note:

- `top40_p6` versus C088/top40 P3: action stream `162 -> 116` bytes, archive
  `276386 -> 276342`.
- C082 276333 lossless repack: mask stream `219472 -> 219465`, pose `677 -> 676`,
  action `116 -> 115`, archive `276342 -> 276333`; decoded streams are reported
  preserved in the local builder, but exact replay did not beat `top40_p6`.
- Public PR75 QP1 replay carries a larger renderer/pose/action budget
  (`56034`, `899`, `236`) and is not a byte target despite useful component
  signal.

## Ranked Candidates And Blockers

1. `zero_fp4_frame1_head_0.1` renderer shrink: dispatch-worthy only because it
   is already locally pose-safe and could move more than `0.0001` by rate if
   components hold. It was not dispatched by this worker and already has an
   active claim in `.omx/state/active_lane_dispatch_claims.md`.
   - Archive:
     `experiments/results/renderer_parity_shrink_search_20260503_worker/zero_fp4_frame1_head_0.1/archive.zip`
   - Bytes/SHA: `275900`,
     `bb8d1835303478183465e15b28fd9af8776b4ea1c07cdd6cfe0895032a267b64`
   - Delta vs `top40_p6`: `-442` bytes, formula-only rate delta
     `-0.00029430965728`
   - Local pose-safety sample: safe in the source ledger
     `renderer_parity_shrink_search_20260503_worker.md`; still requires exact
     CUDA before any score language.

2. `zero_fp4_shared_trunk_0.1` renderer shrink: byte-feasible for sub-`0.314`
   by rate, but blocked by local render-output parity.
   - Archive:
     `experiments/results/renderer_parity_shrink_search_20260503_worker/zero_fp4_shared_trunk_0.1/archive.zip`
   - Bytes/SHA: `273951`,
     `9cf86ac92f3d7a97190a0abbb86b7d65277e3333b6f168ff547cd934c38c7ce9`
   - Delta vs `top40_p6`: `-2391` bytes, formula-only rate delta
     `-0.0015920687569151118`
   - Blocker: local pose-safety failed (`mean_abs=6.8192`, `rms=11.3568`,
     `max_abs=224.216`). Do not exact-eval until the transform is narrowed or
     trained.

3. `zero_fp4_all_fp4_0.1` renderer shrink: larger byte win, worse local parity.
   - Archive:
     `experiments/results/renderer_parity_shrink_search_20260503_worker/zero_fp4_all_fp4_0.1/archive.zip`
   - Bytes/SHA: `272069`,
     `623e764ae1feeb11aba31d3be8c825ecf4e74948c8a5f5bb756ac92e8f91905a`
   - Delta vs `top40_p6`: `-4273` bytes, formula-only rate delta
     `-0.0028452153066910383`
   - Blocker: local pose-safety failed (`mean_abs=8.5851`, `rms=13.4757`,
     `max_abs=231.695`). This is a boundary probe, not dispatchable.

4. Existing Q-FAITHFUL QZS3/QP1 byte artifact: has enough archive bytes for
   formula-only sub-`0.314`, but is not a safe renderer transplant.
   - Archive:
     `experiments/results/lane_q_faithful_retrain_20260501/remote_artifacts/postprocess_fixed_snapshot_20260501T2146Z_fix1/qzs3_rp2_qp1/archive.zip`
   - Bytes/SHA: `272986`,
     `d90a937da2127086f28b66f7df58a027c8c565488eb8e765e468808361602128`
   - Delta vs `top40_p6`: `-3356` bytes, formula-only rate delta
     `-0.002234622646678007`
   - Blocker: changed mask and pose payloads plus render-output parity failure;
     deep byte parser rejects this older payload magic, so this audit records
     ZIP-level profile only.

5. Existing trained QBF1/Block-FP transplant: not a current candidate.
   - Archive:
     `experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146/trained_qbf1_b0512/archive.zip`
   - Bytes/SHA: `283432`,
     `6174424516a93aa35852cd4dc22b9132516fc38f824b77562b3f91b3f55f58dc`
   - Delta vs `top40_p6`: `+7090` bytes, formula-only rate penalty
     `+0.0047209399776361955`
   - Blocker: byte-regressive and exact-negative in the existing ledger.

6. C082 276333 lossless repack: safe packaging base, not a standalone next
   spend.
   - Archive:
     `experiments/results/lightning_batch/exact_eval_c082_p6_stream_resweep_276333_t4_20260503T0705Z/archive.zip`
   - Bytes/SHA: `276333`,
     `30932c684c6b09a7bd2bd248e17fd66a4bc448ed2fe5747f92e74bcceec66681`
   - Delta vs `top40_p6`: `-9` bytes, formula-only rate delta
     `-0.0000059927305781`
   - Exact T4 already exists and did not beat `top40_p6`; keep its stream
     choices as a packaging component only.

## Exact Eval Command For Dispatch-Worthy Candidate

Do not duplicate the existing active claim for `renderer_zero_fp4_frame1_head_010_t4`.
If an operator needs to rerun it after closing or superseding that claim, use a
new claim first:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id renderer_zero_fp4_frame1_head_010_t4 \
  --platform lightning \
  --instance-job-id exact_eval_renderer_zero_fp4_frame1_head_010_t4_<STAMP> \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc <UTC_ISO8601> \
  --status eval \
  --notes "T4 exact eval for locally pose-safe renderer shrink; archive 275900B sha bb8d1835303478183465e15b28fd9af8776b4ea1c07cdd6cfe0895032a267b64"
```

Canonical CUDA auth-eval command on a CUDA/T4-equivalent runner:

```bash
.venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/renderer_parity_shrink_search_20260503_worker/zero_fp4_frame1_head_0.1/archive.zip \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir experiments/results/full_stack_compression_audit_worker_20260503/exact_eval_work_zero_fp4_frame1_head_010_t4
```

## Recommendation

Near-term score-down path:

1. Harvest the already-claimed `zero_fp4_frame1_head_0.1` T4 eval; if it fails
   components, use the exact component trace to train/narrow renderer FP4
   shrink rather than dispatching broader zeroing.
2. Continue fixed-mask/fixed-pose renderer self-compression burns as the main
   representation-scale route. Existing byte-only Q-FAITHFUL artifacts prove
   the byte target is plausible, but current artifacts are blocked by
   transplant parity.
3. Treat PR75 P6/action repacks as polish only. They are exact-transferable but
   current wins are tens of bytes and cannot close sub-`0.314` or sub-`0.300`
   alone.
4. Do not spend T4 on mask-geometry replacement without a stronger
   geometry-preservation proof. The mask stream is the largest byte lever, but
   recent CMG/PMG/micro-mask exact diagnostics show PoseNet cliffs.

## Verification

Commands run locally:

```bash
.venv/bin/python experiments/profile_archive_byte_accounting.py --target-score 0.314 ... exact T4 collection
.venv/bin/python experiments/profile_archive_byte_accounting.py --target-score 0.314 ... local renderer blocker collection
.venv/bin/python experiments/profile_archive_bytes.py ... unsupported older payload ZIP profiles --continue-on-error
.venv/bin/python -m pytest src/tac/tests/test_profile_archive_byte_accounting.py src/tac/tests/test_archive_byte_profile.py src/tac/tests/test_unpack_renderer_payload_fixedslice.py -q
```

Verification result:

```text
30 passed, 1 warning
```

Code changes made by this worker: none.
