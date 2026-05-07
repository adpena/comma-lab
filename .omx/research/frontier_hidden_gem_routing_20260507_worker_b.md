# Frontier Hidden-Gem Routing - 2026-05-07 Worker B

## Scope

Worker B routing pass under the active PR103-on-PR106 rate-only floor. This
pass produced local CPU-prep/routing artifacts only. It did not dispatch GPU
work, did not claim a new score, did not edit code, did not stage files, and
did not touch dispatch state.

## Active Floor

The active rate-only floor for this pass is the PR103-on-PR106 A++ archive:

- archive bytes: `185578`
- archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- exact eval artifact:
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`
- recomputed score in the existing exact eval artifact:
  `0.20898105277982337`
- routing rule: rate-only exact-eval spend requires archive bytes below
  `185578`, unless the candidate declares a scorer-changing stack path.

This ledger is not a new score claim. It uses the existing exact-eval anchor
only as a routing floor.

## Generated Artifacts

Primary summary:

- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/routing_summary.json`

Tool outputs:

- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hidden_gems.json`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hidden_gems.md`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hidden_gem_readiness.json`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hidden_gem_readiness.md`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hnerv_entropy_selection.json`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hnerv_entropy_selection.md`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/frontier_entropy_gap_ranking.json`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/frontier_entropy_gap_ranking.md`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/field_meta_selection.json`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/roadmap_status.json`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/roadmap_status.md`
- `experiments/results/frontier_hidden_gem_routing_20260507_worker_b/pr101_codecop_lgwin18_pre_submission_compliance.json`

## Routing Results

| candidate | bytes / SHA | class | policy outcome | CPU blockers | next CPU-prep action |
|---|---:|---|---|---|---|
| `pr101_codecop_lgwin18` | `178258` / `c95c59933f95746f6b8dd5fb7b4450419a25c01b2c9f8dac6e586cd4b3582933` | rate-only, below floor | byte policy allows future spend, but selector blocks as `static_exact_eval_packet_not_ready` | strict compliance failed on missing `report.txt` and public-hygiene local absolute path; runtime-tree/inflate-output parity not promoted; lane claim and exact CUDA missing | repair release surface without changing archive bytes; rerun strict compliance; bind runtime-tree and inflate-output parity |
| `pr101_codecop_sweep_auto_select_true` | predicted `178252`, no archive SHA | rate-only predicted, below floor if materialized | not exact-evaluable because archive is missing | archive substitution surgery pending; no candidate SHA; no runtime parity; no lane claim; no exact CUDA | materialize the predicted smallest codecop archive and rerun local decoder parity plus strict release-surface checks |
| `categorical_hpm1_refresh` | `179979` / `9bfea530158ab498a55ec626804c5e9eb1bb80da14a2f2d21d7262c1841bc2fe` | scorer-changing byte-closed archive | not blocked by rate-only floor, but not dispatchable | HPM1 decode/reencode parity not passed; full decode and byte-exact reencode not proven; semantic runtime-output parity not proven | debug the HPM1 entropy decode contract mismatch at frame 0 group 10 symbol 191, then prove full decode/reencode and runtime-output parity |
| `wr01_apply_pr106x_half` | `186222` / `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628` | scorer-changing/output-changing candidate | field-meta does not rate-floor block it, but strict candidate preflight refuses it | strict candidate preflight not ready; component-response/adversarial evidence missing; lane claim and exact CUDA missing | refresh static preflight and produce component-response evidence before any exact-eval lane claim |
| `hdm3` | `186066` / `5b5619628b54ccec44d51360ecb258dfe61742a581c7605c74d1ddaa5c025771` | rate-only | blocked, `+488` bytes over floor | rate-only floor blocker; lane claim and exact CUDA missing | do not dispatch as-is; rebase below `185578` or pair with a scorer-changing stack path |
| `pr106_q10_151byte_brotli` | `186088` / `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7` | rate-only | blocked, `+510` bytes over floor | rate-only floor blocker; terminal/stale claim state; KKT/Pareto gates not ready | retire from exact-eval spend until below-floor or scorer-changing |
| `pr106x_lgblock16_1byte_brotli` | `186079` / `866dc135e9168d61fab02b6b1c218c4b1d6eed779154a6dc3095fd05e48024f2` | rate-only | blocked, `+501` bytes over floor | rate-only floor blocker; no active claim; KKT/Pareto gates not ready | keep as Brotli saturation evidence only |
| `pr106_surgery_identity_smoke` | `186239` / `cb9976bd33468475aac54a98c3baff996101c144b00e8d7e2c5107c86cda6182` | no-op control | not eligible | `charged_bits_changed=false`; `score_affecting_payload_changed=false`; above floor | keep as surgery-control provenance; do not route |

## Top CPU-Prep Targets

1. `pr101_codecop_lgwin18`: closest exact-eval prep path under the new policy
   because it is byte-closed and below the rate-only floor. The next artifact
   is a passing strict pre-submission compliance report plus runtime-tree and
   inflate-output parity for the existing archive SHA.
2. `categorical_hpm1_refresh`: highest scorer-changing byte-closed route. It
   needs HPM1 full decode/reencode and semantic runtime-output parity before it
   can be considered for any exact-eval claim.
3. `pr101_codecop_sweep_auto_select_true`: smallest codecop predicted row, but
   no archive exists yet. It should be materialized and validated before it can
   enter selector/field-meta routing.

## Tool Outcomes

- HNeRV entropy selector: selected no exact-evaluable next candidate. It
  accepted `pr101_codecop_lgwin18` as below the byte floor but blocked it on
  static packet readiness; all PR106/PR106x rate-only candidates above
  `185578` were blocked by policy.
- Field-meta selector: no row became exact-eval ready. `pr106_q10` and
  `pr106x_lgblock16` were blocked by the active floor; `wr01` was blocked by
  strict candidate preflight.
- Roadmap status: next unblocked workstreams start with
  `categorical_qma9_clade_spade_openpilot`, followed by cross-paradigm and
  entropy-hardening rows. No candidate packet was ready for exact-eval dispatch.
- Frontier entropy-gap ranking: next rate-only research action remains building
  an old/new section SHA and charged-byte diff manifest for the largest current
  frontier section, `decoder_packed_brotli`; this is not a dispatch artifact.
- Hidden-gem readiness audit: 15 registry rows, 0 ready for exact-eval dispatch,
  no missing evidence paths or missing integration targets.

## Verification

Commands run. Long selector/ranking argument lists are abbreviated below; the
exact argv is preserved in the generated tool-run manifests and JSON outputs.

```bash
.venv/bin/python tools/list_hidden_gems.py --format json --output experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hidden_gems.json
.venv/bin/python tools/audit_hidden_gem_readiness.py --format json --output experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hidden_gem_readiness.json
.venv/bin/python tools/select_hnerv_entropy_frontier_candidate.py --active-candidate active_pr103_pr106=experiments/results/pr103_repack_pr106_standalone_20260507/final_runtime_packet_proof.json ... --json-out experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hnerv_entropy_selection.json --md-out experiments/results/frontier_hidden_gem_routing_20260507_worker_b/hnerv_entropy_selection.md
.venv/bin/python tools/rank_hnerv_frontier_entropy_gaps.py ... --json-out experiments/results/frontier_hidden_gem_routing_20260507_worker_b/frontier_entropy_gap_ranking.json --md-out experiments/results/frontier_hidden_gem_routing_20260507_worker_b/frontier_entropy_gap_ranking.md
.venv/bin/python tools/build_field_meta_dispatch_selection.py ... --claims-path .omx/state/active_lane_dispatch_claims.md --now-utc 2026-05-07T22:26:35Z --json-out experiments/results/frontier_hidden_gem_routing_20260507_worker_b/field_meta_selection.json
.venv/bin/python tools/build_frontier_roadmap_status.py ... --claims-path .omx/state/active_lane_dispatch_claims.md --now-utc 2026-05-07T22:26:35Z --json-out experiments/results/frontier_hidden_gem_routing_20260507_worker_b/roadmap_status.json --md-out experiments/results/frontier_hidden_gem_routing_20260507_worker_b/roadmap_status.md
.venv/bin/python -m pytest src/tac/tests/test_hnerv_entropy_frontier_selector.py src/tac/tests/test_build_field_meta_dispatch_selection.py src/tac/tests/test_build_frontier_roadmap_status.py -q
.venv/bin/python scripts/pre_submission_compliance_check.py --submission-dir experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/full_runtime_packet --archive experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/full_runtime_packet/archive.zip --expect-single-member x --expected-archive-sha256 c95c59933f95746f6b8dd5fb7b4450419a25c01b2c9f8dac6e586cd4b3582933 --expected-archive-size-bytes 178258 --public-scan-path experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/full_runtime_packet --json-out experiments/results/frontier_hidden_gem_routing_20260507_worker_b/pr101_codecop_lgwin18_pre_submission_compliance.json --strict
```

Focused tests: `48 passed`.

The strict pre-submission compliance command intentionally exited nonzero and
wrote the blocker artifact. It found 19 passing checks and 3 failing checks:
missing `report.txt`, missing report presence, and public-hygiene failure from
a local absolute operator path in `runtime_custody_manifest.json`.
