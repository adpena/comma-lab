# Recall and scope

## RECALL EVIDENCE

Searched the full `.omx/research/` Markdown and JSON corpus by content with
`corrector.{0,50}(native|float64|timing)|corrector_coding_row|Lane.{0,20}mixer.{0,30}(timing|native)`.
Searched `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, and `docs/` with
`corrector|rlc1_geometry|Lane.mixer`, including design and SPEC surfaces.
Searched `.omx/state/canonical_task_status.jsonl` with
`corrector|Lane.mixer|mrs[1234]`. Exported the complete equation registry with
`.venv/bin/python tools/list_canonical_equations.py --json` and searched it for
`corrector|float64|dyadic|decode|runtime`. Outputs are retained under
`/Volumes/APDataStore/pact/ddm_mrs4/recall/`; these are search matches, not a
claim that every matching equation applies. The Codex memory registry search
`mrs4|minimal.packet|adaptive.corrector|common_contract` returned no matches.

Beyond the charter seeds:

- `ddm_fcd1_field_for_coder_diagonal_20260829.md:83-88` records a native/Python
  family-generation mismatch. This port retains the current 23-family order,
  adds a Python configuration/constant check before binding, and compares all
  79 persistent native state arrays after each sampled frame.
- `ddm_rr8_t4_wallclock_verdict_20260820.md:89-101` withdraws transfer of an
  unmatched local component split to T4. Consequently the two requested ratios
  remain one assumption in rounded and full-precision forms, and the stage
  projection uses an explicit conservative GPU bound rather than inventing a
  measured prior-network time.
- `ddm_mrs3_20260916/RECALL_EVIDENCE.md` points to the persistent-state inventory
  in `ddm_tc3_20260910/native_trace_state_review.md`. The external harness reads
  the exact C record layout and uses the retained pre-frame states. It verifies
  the seeded 48-frame selection independently of receipt reuse.
- The canonical native-runtime rule requires a payload-cleanliness audit and
  Python oracle. Those live beside this memo; neither belongs in the six-file
  public tree. No new compression law or originality claim is made.

The source-derived correction remains the sealed float64 adaptive corrector.
Two Lane-mixer arithmetic kernels, fixed log2 and probability normalization,
share `corrector.c`. They preserve the Python sequence of IEEE operations.
The Lane geometry remains the original Python integer implementation; no
wide-integer approximation or third native file is introduced.

## OPTIMAL FORM

Reference: mrs2 landing `2171aaf9b`, mrs3 landing `34bbf7842`, sealed native source
SHA-256 `3e2705f5505036121d85329958a4f23b5ea95e6d20d45ecf92901f2b65cca92a`.
The declared changes are the native corrector, the two shared mixer kernels,
ctypes selection with a Python fallback, two compile commands, and README prose.
The archive and range decoder are unchanged. This is a receiver runtime port,
`research_only=true`, `score_claim=false`, `promotable=false` until MAIN performs
the required new authority measurement. A local identity proof is not that row.

## Integration dispositions

Sensitivity-map consumer: the stage attribution in `HANDOFF.md` and its linked
JSON receipts. Pareto consumer: the 700-line, six-file, 10x and 1,200-second
charter gates. Bit allocation: N/A, the archive is unchanged. Autopilot dispatch:
N/A, prohibited here; MAIN owns the reader and any new T4 row. Posterior update:
the source-bound local parity and timing receipts, with their stated sample and
axis. Probe disambiguator: ordered probability, state, token and raw comparison
distinguishes arithmetic errors from timing failure. Future work is routed only
through `MAIN_FIRE_ORDER.json`; no new production gate or equation is claimed.

<!-- # FORMALIZATION_PENDING: implementation equivalence and timing evidence, not a new score law -->
