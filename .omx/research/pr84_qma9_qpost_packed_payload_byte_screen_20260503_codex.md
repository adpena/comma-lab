# PR84 QMA9 QPost Packed-Payload Byte Screen - 2026-05-03

## Scope

Local-only PR84-native archive-packing screen for the PR84+PR82 Henosis stack.
No remote GPU job, exact eval, training, scorer invocation, or lane dispatch
was performed.

The running PR84+PR82 T4 jobs use the existing expanded-member builder layout.
This screen keeps the exact PR84 public member `p` intact and adds only
`qpost.bin` as a charged sidecar. The robust inflate path already expands `p`
through `submissions/robust_current/unpack_renderer_payload.py`, then applies
`qpost.bin` after renderer inflate.

## Source Assumptions

- PR84 source archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr84/archive.zip`
- PR84 archive bytes/SHA-256:
  `215735`,
  `a607a6c3ae9b610e6edfb546c3206004ae40fc348ecaef2446b7134a19b8e07f`
- PR84 stored member `p`: `215635` bytes, SHA-256
  `35b14e37ee34eeb2197117d52d805afb7aff8b33378b367685476dc4097e324d`
- PR84 fixed slices inside `p`:
  - `range_mask.qma9`: `159011` bytes, SHA-256
    `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
  - `split_model_reordered.br_bundle`: `55725` bytes, SHA-256
    `b649b0dacb1dcc93fd7da2e7f5c6d398fa933d2fe3087520359612afc8e4832d`
  - `optimized_poses.qp1.br`: `899` bytes, SHA-256
    `83767bbd10ae72c3237e468351eb9c465a954e25a6af04a0e0cb84d1f7af9b51`
  - router tail: absent, `0` bytes.
- PR82 source archive:
  `experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/archive.zip`
- PR82 archive bytes/SHA-256:
  `296789`,
  `a0e07c360223c1dd3d3b92263225d38d542e218e83d095ad9b91bf872f94c6e4`

## Runtime And Raw-Parity Guard

Builder mode:

```bash
.venv/bin/python experiments/build_pr84_pr82_henosis_stack_candidate.py \
  --archive-layout public_payload_plus_qpost \
  --output-dir experiments/results/pr84_qma9_qpost_packed_payload_20260503_codex
```

The builder preflighted the PR84 public payload parser:

- unpacker: `submissions/robust_current/unpack_renderer_payload.py`
- unpacker SHA-256:
  `abef476e2bff31cc410dcac42d4c3c62123c10d26e6c34b718d65014fadd9d5c`
- payload format:
  `public_qma9_reordered_qzs3_qp1_no_router_fixed_slices`
- unpacked members:
  - `masks.qma9`: `159011` bytes
  - `renderer.bin`: `55729` bytes
  - `optimized_poses.qp1`: `1140` bytes

For every emitted archive, member `p` is byte-identical to the PR84 source
payload: `215635` bytes, SHA-256
`35b14e37ee34eeb2197117d52d805afb7aff8b33378b367685476dc4097e324d`.
Each candidate also carries the existing deterministic synthetic raw-output
delta proof for the exact `qpost.bin` bytes. This is a no-op guard only; it is
not component or score evidence.

## Local Candidate Set

All rows are `empirical_local_archive_build_and_runtime_preflight_only`.
Dispatch requires a lane claim and exact CUDA auth eval after the running
PR84+PR82 jobs are harvested.

| candidate | archive bytes | archive SHA-256 | qpost bytes | delta vs expanded PR84+PR82 | planning score if PR82 components carry | dispatch gate |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `pr84_qma9_pr82_qps1_controls_all600_packedp` | `218396` | `9fd8be8ba707e13eb5db06daff51ae8381e4e0f67f8a461883d4cd653ca7b82a` | `2567` | `-477` | `0.2461260422200476` | ready after lane claim |
| `pr84_qma9_pr82_qps1_qrm1_all072_randmulti_packedp` | `232355` | `d82bc774de93c073b31c1acc9dd24c57b28d14b554cee53c7b7be3404c8f2e2c` | `16526` | `-477` | `0.25542076734668` | ready after lane claim |
| `pr84_qma9_pr82_qps1_controls_qrm1_all072_packedp` | `234886` | `3ccea52612fa038bd0d2ce9d0b6389e97ad03a41be5562610e55f54ce912786a` | `19057` | `-477` | `0.2571060563570322` | ready after lane claim |

Artifact root:
`experiments/results/pr84_qma9_qpost_packed_payload_20260503_codex/`.
The root `candidate_summary.json` and each candidate `manifest.json` record
member sizes, SHA-256s, source payload parity, qpost SHA-256, raw delta proof,
and `remote_dispatch_performed=false`.

## Interpretation

This is a deterministic packing win over the expanded-member PR84+PR82 local
candidates, not a new component claim. It saves `477` bytes per PR84+PR82
candidate by preserving PR84's charged public payload and avoiding expanded
runtime member archive overhead. If PR82 component behavior carries, the full
controls+QRM1 packed candidate has a planning score of
`0.2571060563570322`, but only exact CUDA auth eval can validate that
interaction.

The packed candidates intentionally do not duplicate the running T4 jobs. They
are the next local archive set to consider after the current PR84+PR82 exact
eval harvest.

## Verification

```bash
.venv/bin/python -m py_compile \
  experiments/build_pr84_pr82_henosis_stack_candidate.py \
  src/tac/tests/test_build_pr84_pr82_henosis_stack_candidate.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_pr84_pr82_henosis_stack_candidate.py -q
```

Result: `4 passed`.
