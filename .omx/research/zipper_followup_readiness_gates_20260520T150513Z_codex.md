# Zipper Follow-Up Readiness Gates

**UTC:** 2026-05-20T15:05:13Z
**Owner:** Codex
**Source package:** `.omx/research/inbox_zipper_20260520T144021Z_codex/research_package.zip`
**Lane:** `lane_zipper_package_intake_20260520`

## Verdict

Zipper follow-up is now wired into Pact-native gates rather than package
pseudocode. No GPU dispatch was attempted and no score claim was made.

The immediate actionable path is:

1. SIREN is locally ready for a first-anchor training run, but remains not
   remote-dispatch-ready without operator authorization and a lane claim.
2. VQ-VAE local renderer/mask codec tests pass; its next meaningful step is
   a byte-closed export/provenance gate, not the package scaffold.
3. Foveation/RAFT local tests pass; next step is payload or dependency-gated
   candidate construction, not package pseudocode.
4. Cool-Chic/C3 empty sidecar materializers produce byte-closed research
   artifacts with explicit `score_claim=false` and
   `ready_for_exact_eval_dispatch=false`.
5. A small readiness bug-class fix landed: the shared auth-eval gate now
   returns `auth_eval_exact_cuda_complete=true` explicitly for valid CUDA
   claims, and SIREN consumes that canonical field.

## Gate Results

| Gate | Command / artifact | Result | Authority |
|---|---|---|---|
| SIREN readiness audit | `.venv/bin/python tools/audit_siren_substrate_readiness.py --json --output experiments/results/zipper_siren_readiness_20260520_codex/readiness.json` | `local_contract_ready=true`; `ready_for_first_anchor_training=true`; no local blockers; dispatch blockers are operator authorization, lane claim, and no-GPU-spend gate. | `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false` |
| SIREN focused tests | `.venv/bin/python -m pytest src/tac/substrates/siren/tests/test_siren_roundtrip.py src/tac/tests/test_siren_substrate_readiness.py -q` | Initially found stale/deleted guard expectation; after patch: `22 passed`. | Local readiness only |
| Shared auth-eval gate tests | `.venv/bin/python -m pytest src/tac/tests/test_smoke_auth_eval_gate.py -q` | `27 passed`. | Guard wiring |
| VQ-VAE tests | `.venv/bin/python -m pytest src/tac/tests/test_train_vqvae_as_renderer.py src/tac/tests/test_vqvae_mask_codec.py -q` | `46 passed`. | Local renderer/mask codec readiness only |
| Foveation/RAFT tests | `.venv/bin/python -m pytest src/tac/tests/test_build_lapose_foveation_payload_archive.py src/tac/tests/test_codec_pipeline_raft_pose.py src/tac/tests/test_raft_pose_stream.py -q` | `40 passed`. | Local payload/runtime readiness only |
| Cool-Chic empty materializer | `tools/materialize_cool_chic_residual_pr106_sidecar.py --output-dir experiments/results/zipper_c3_coolchic_smoke_20260520_codex/coolchic_empty --residual-mode empty --skip-no-op-smoke` | Built `cool_chic_pr106_residual_sidecar_archive.zip`, size `186724`, SHA-256 `e1f2d35518b286b00d0e5da575a86c661b553f33fddacf63f8f70ccb6349d018`, residual bytes `0`. | Research signal only |
| C3 empty materializer | `tools/materialize_c3_residual_pr106_sidecar.py --output-dir experiments/results/zipper_c3_coolchic_smoke_20260520_codex/c3_empty --residual-mode empty --skip-no-op-smoke` | Built `c3_pr106_residual_sidecar_archive.zip`, size `186724`, SHA-256 `e35c6bede6543397c839cd61bf40c547dc3df7b2f9374c7441c8baf7d9db5498`, residual bytes `0`. | Research signal only |

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `experiments/results/zipper_siren_readiness_20260520_codex/readiness.json` | `1c78d07d00ff9a666f41263091f080817ac5de5de943b7015f1a46559cab6a59` |
| `experiments/results/zipper_c3_coolchic_smoke_20260520_codex/coolchic_empty/materialization_manifest.json` | `199a4a51fa785291628b65a5e2bba782e2440bf3d080010801ee69652786e574` |
| `experiments/results/zipper_c3_coolchic_smoke_20260520_codex/c3_empty/materialization_manifest.json` | `aede95fe4738757f3cd4489771133724366974a55e0e7a31bc7c4f375283ec19` |

## Code Wiring Patch

Files touched:

- `src/tac/substrates/_shared/smoke_auth_eval_gate.py`
- `experiments/train_substrate_siren.py`
- `src/tac/tests/test_siren_substrate_readiness.py`

Change:

- CUDA auth-eval success now returns `auth_eval_exact_cuda_complete=true`
  from the canonical shared gate, matching the non-CUDA result schema.
- SIREN trainer consumes this explicit field instead of relying on an implied
  success.
- SIREN readiness test now asserts the canonical `_canon_gate_auth_eval_call`
  path rather than a deleted older helper name.

This is additive and preserves the fail-closed behavior: smoke, CPU-advisory,
missing lane claim, and missing operator authorization still do not produce
score authority.

## Next Frontier-Moving Artifact Path

The best next path from Zipper is **SIREN first-anchor local CPU smoke**, not
more package-roadmap work:

```bash
.venv/bin/python experiments/train_substrate_siren.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/siren_smoke_<utc> \
  --epochs 3 \
  --device cpu --smoke \
  --skip-archive-build --skip-auth-eval
```

Reason: SIREN now has real local contract readiness, archive/inflate/runtime
surfaces, score-aware-loss wiring, and a canonical remote-dispatch blocker
list. This smoke produces the cheapest next empirical signal without risking
PR #110 or spending GPU.

## Non-Authority Flags

- `score_claim = false`
- `promotion_eligible = false`
- `ready_for_exact_eval_dispatch = false`
- `gpu_dispatch_attempted = false`
- `pr110_branch_or_release_asset_changed = false`
