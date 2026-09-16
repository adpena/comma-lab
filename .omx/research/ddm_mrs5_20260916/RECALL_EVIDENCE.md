# mrs5 recall and implementation scope

## RECALL EVIDENCE

Before implementing, searched the full `.omx/research/` corpus by CONTENT with
`receiver.rewrite|receiver.only|cost fraction|wide.integer|rlc1_geometry`.
A second content search covered `.omx/research/` (including canonical index and
sub015 DAG FEED files) and `docs/` (including design/SPEC files) using
`same.host|timing.risk|geometry.c`. The canonical task ledger was searched with
`mrs[1-5]|receiver.*timing|prefire`. Exact queries, exit codes, scopes and matches
are retained in `/Volumes/APDataStore/pact/ddm_mrs5/recall/query_{0,1,2}.{txt,json}`.
The complete equation registry was exported with
`.venv/bin/python tools/list_canonical_equations.py --json`: 490 rows, 176 matching
receiver/wall.clock/geometry/native retained in `recall/equations.json`. Matching
is a recall denominator, not a claim that every result applies. The Codex memory
registry query `mrs5|prefire|minimal.packet|common_contract` had no matches.
No recent directive-named file was found in the 20260915–20260916 filename scope.

Beyond the charter seeds:

- `ddm_pr19_risk_gate_identity_class_and_chain_inheritance_20260911.md` documents
  2.83x variation in the local cold instrument and retains receiver-change
  refusals. This led to reading CURRENT executable contract rules rather than
  assuming the original pr12 title permits a receiver-only intent.
- `ddm_rr8_t4_wallclock_verdict_20260820.md:89-101` shows why an unmatched local
  component split cannot transfer to T4. Our ratio remains a conditional
  projection, never a measured T4 time or a statistical confidence bound.
- mrs2's source boundary audit uses standard-C11 split arithmetic instead of
  compiler extensions. The sealed geometry uses `__int128`; the mrs5 port uses
  two unsigned 64-bit limbs with explicit signed floor division, retaining the
  original ordered integer operations rather than narrowing to int64.
- The task ledger already owns the mrs1 T4 successor and mrs4 timing decision.
  New follow-ons will name MAIN and concrete consumers; no competing fire or
  scorer slot is claimed here.

## OPTIMAL FORM

Reference forms: sealed move53 native set (range, corrector, geometry; Python
HPAC), mrs4 sources at e621fd5a9, the current frozen first-measurement contract.
Delta: geometry.c plus ctypes with the Python geometry fallback, third shell
compile, README explanation. Existing range/corrector sources and archive are
unchanged. This is source-derived receiver implementation, not novel compression.
All video-derived parameters still arrive from the counted archive.

The six consumer hooks are scoped explicitly: sensitivity = sampled timing
receipt; Pareto = seven-file/150-C-line/1260-second charter constraints; bit
allocation = N/A, archive unchanged; autopilot = N/A, dispatch prohibited;
posterior = parity/raw/timing receipts with false score authority; disambiguation
= real geometry-bin, token and raw comparisons plus actual contract refusal.
The work is research_only=true. No frontier movement is claimed.

<!-- # FORMALIZATION_PENDING: source equivalence and instrument evidence, not a new score law -->
