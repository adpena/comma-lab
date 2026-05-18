# Codex Findings: RATE-OP-1 Authority Xray

Date: 2026-05-18T23:06:53Z  
Author: Codex  
Task: `rate_attack_op_1_stable_orbit_packet_diet`  
Lane: `lane_rate_attack_op1_stable_orbit_packet_diet_20260518`

## Verdict

Authority for RATE-OP-1 should be established as a staged evidence chain:

1. `planning_xray_not_score_evidence`
2. packet-valid probe with `CandidateModificationSpec`
3. byte-different archive plus byte-consumption/no-op proof
4. full-frame inflate parity
5. exact `[contest-CUDA]` before/after artifact
6. paired Linux `[contest-CPU]` public-axis replay

The new OP1 xray builder lands stage 1 only. It is intentionally useful to
Cathedral autopilot for ranking/inspection, but it carries no score,
promotion, rank/kill, or dispatch authority.

## Landed Artifacts

Code:

- `src/tac/contest_exploits/stable_orbit_packet_diet.py`
- `tools/build_rate_attack_op1_stable_orbit_packet_diet_xray.py`
- `src/tac/contest_exploits/tests/test_stable_orbit_packet_diet.py`

Generated local evidence:

- `experiments/results/rate_attack_op1_stable_orbit_packet_diet_20260518T230502Z/xray_manifest.json`
- `experiments/results/rate_attack_op1_stable_orbit_packet_diet_20260518T230502Z/byte_cost_table.csv`
- `experiments/results/rate_attack_op1_stable_orbit_packet_diet_20260518T230502Z/overlay_inputs.json`
- `experiments/results/rate_attack_op1_stable_orbit_packet_diet_20260518T230502Z/allocation_candidates.jsonl`
- `experiments/results/rate_attack_op1_stable_orbit_packet_diet_20260518T230502Z/cathedral_autopilot_candidates.jsonl`

The generated directory is an experiment artifact, not committed durable state.
This ledger records the durable pointer and result summary.

## Real-Archive Xray Result

Inputs:

| label | archive | SHA-256 prefix | axis tag | parsed grammar | gradient status |
|---|---|---|---|---|---|
| `a1_control` | `submissions/a1/archive.zip` | `87ec7ca5f2f3` | `[contest-CPU GHA Linux x86_64]` | `a1_prefixed_hnerv_microcodec` | `unregistered_sidecar` |
| `fec6_cpu_frontier` | `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip` | `6bae0201fb08` | `[contest-CPU GHA Linux x86_64]` | `pr101_fixed_offset_hnerv_microcodec` | `unregistered_sidecar` |

Important authority outcome:

- A1 is parsed as `a1_prefixed_hnerv_microcodec`; PR101 fixed-offset slicing is
  not assumed for A1.
- The current `.omx/state/master_gradient_anchors.jsonl` entries did not match
  these exact archive ZIP SHAs in a grammar-aware coordinate system, so
  gradient-weighted allocation remains blocked with
  `matching_canonical_gradient_anchor_missing` and
  `fec6_gradient_extrapolation_forbidden`.
- Uniform ZIP/header/Brotli xray found only `200` proxy bytes of directly
  observable uniform slack across A1 plus fec6, below the `4506` byte
  materiality floor for a `-0.003` rate-score move.

## Cathedral Autopilot Authority Contract

The emitted `cathedral_autopilot_candidates.jsonl` loads through
`tools.cathedral_autopilot_autonomous_loop.load_candidates_from_jsonl`.
All rows carry:

- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_packet_ready=false`
- `score_affecting_payload_changed=false`
- `charged_bits_changed=false`
- `proxy_row=true`
- `included_in_aggregate=false`
- `target_modes=["contest_one_video_replay"]`

`contest_one_video_replay` is preserved as semantic intent, not exact dispatch
authority. A later exact-ready packet must separately add the dispatch action
target `contest_exact_eval` after byte-closed readiness is proven.

## Subagent Findings Absorbed

1. `contest_one_video_replay` is sanctioned, but exact-dispatch gates currently
   recognize `contest_exact_eval`; rows therefore need semantic replay mode and
   later exact-eval dispatch mode rather than either mode erasing the other.
2. Exact-readiness runtime proof is over-specialized to PR101; deterministic
   compiler consumption proofs need a future adapter.
3. A1 specialized inverter non-authority is correct for phase 0 but must not be
   treated as method-negative. Its status is `SANCTIONED_FEASIBILITY`.
4. Gradient-weighted OP1 must not inherit fec6 or advisory gradient rows unless
   archive SHA, byte domain, tensor kind, hardware, axis, and full pair
   coverage match.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/contest_exploits/tests/test_stable_orbit_packet_diet.py \
  src/tac/contest_exploits/tests/test_a1_specialized_inverter.py -q

.venv/bin/python -m ruff check \
  src/tac/contest_exploits/stable_orbit_packet_diet.py \
  src/tac/contest_exploits/__init__.py \
  tools/build_rate_attack_op1_stable_orbit_packet_diet_xray.py \
  src/tac/contest_exploits/tests/test_stable_orbit_packet_diet.py
```

Both passed.

## Next Authority Upgrade

The next concrete OP1 step is not exact eval. It is a packet-valid mutation
builder for A1 latent/sidecar sections that emits:

- `CandidateModificationSpec`
- grammar-aware repack
- ZIP header/CRC proof
- inflate success
- byte-consumption/no-op proof
- exact before/after score artifact request only after those proofs exist
