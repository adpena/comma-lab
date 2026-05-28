# Codex Findings: Archive-Bound Candidate Contract Pipeline

Date: 2026-05-28T19:59:13Z
Agent: Codex

## Scope

Consolidated archive/entropy materializer outputs around one reusable `tac`
contract instead of adding more tool-specific row shapes.

## Landed

- Added `tac_archive_bound_candidate_contract.v1` and
  `tac_archive_bound_candidate_contract_surface.v1` in
  `tac.optimization.archive_bound_candidate_contract`.
- FEC, selector, Huffman, range/ANS prototype, ZIP repack, packet/member, and
  receiver-adapter variants now emit common archive-bound contracts with file
  custody, receiver proof, substrate tags, byte deltas, acquisition penalties,
  exact-axis blockers, and false-authority fields.
- Stack search now consumes those contracts directly, preserving full variant
  substrate coverage while using selected-contract and surface-level penalties
  in acquisition/budget routing.
- Exact-ready bridge, entropy-stage chain execution, and the bounded autonomous
  runner now carry the same contract through summaries and handoff rows.

## Evidence

- `.venv/bin/ruff check` on touched queue/materializer/contract files: passed.
- Focused pytest for queue fleet, archive-bound contracts, byte-transform
  materializers, exact-ready bridge, and real archive floor-loop: `14 passed`.
- Live repaired materialization queue
  `repair_multi_archive_live_psv3_fec6_materialization`: `90/90` succeeded,
  healthy, zero blockers, telemetry-only/no score authority.
- Queue fleet scan after classifier fixes: `INVALID_QUEUE=0`, `NEEDS_INIT=0`,
  `READY_TO_SUPERVISE=0`, `TERMINAL=49`, `NEEDS_RECOVERY=11`.
- `tools/review_tracker.py mark-file ... --status reviewed` was run on touched
  Python files.

## Remaining Blocker

The unified contract is still an acquisition and exact-handoff planning surface.
It deliberately does not grant score, promotion, dispatch, or budget-spend
authority until contest CPU/CUDA custody signs the candidate.
