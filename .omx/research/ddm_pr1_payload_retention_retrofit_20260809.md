<!-- # FORMALIZATION_PENDING:static-detector-census-and-retention-custody-only-no-new-predictive-scientific-equation -->
# DDM-PR1 payload-retention structural retrofit

Date: 2026-08-09  
Owner: codex arm `ddm_pr1`  
Disposition: **PARTIAL STRUCTURAL CURE LANDED; STRICT FLIP NOT EARNED**  
Axis: `[static AST census; macOS-CPU byte-only scorer-free synthetic controls]`  
`score_claim=false`; pointer movement: none

## Outcome

The payload-retention gate is now called by `preflight_all()` and reports an explicit
denominator. It remains WARN-ONLY because the corrected live-source population is nonzero. A normal
commit still does **not** execute this call: `tools/preflight_hook.py::_preflight_command` appends
`--no-codebase` unless `PREFLIGHT_FULL=1`, and the gate lives in the skipped `check_codebase` block.
This landing therefore establishes full-preflight coverage, not normal-hook coverage.

Two high-value live coder races now retain every materialized candidate they compare, not only the
winner. Each retained row records durable SSD path, exact byte count, and SHA-256. No scorer,
training, Modal job, archive evaluation, or frontier measurement ran. An append-only progress row
in `.omx/state/operator_p0_ledger.jsonl` keeps task #1001 `in_progress` with the fire order below.

## Corrected static population

These are **detector findings**, not an adjudicated count of confirmed runtime violations. The
denominator is every `.py` file examined in the declared scope; the candidate count is the subset
AST-parsed after a lossless text prefilter.

| snapshot scope | findings | files with findings | `.py` examined | AST candidates | unreadable |
|---|---:|---:|---:|---:|---:|
| repository: `experiments`, `tools`, `src/tac`, `scripts` | 1,819 | 885 | 51,009 | 4,248 | 0 |
| `preflight_all` live-source scope, excluding `results/tests/fixtures` | 1,072 | 515 | 6,390 | 1,292 | 0 |
| SSD: `/Volumes/VertigoDataTier/pact`, excluding `.git/.venv/node_modules/__pycache__/upstream/*_intake_*` | 9,854 | 4,222 | 206,192 | 14,099 | 0 |

The machine-readable census is retained at
`/Volumes/VertigoDataTier/pact/ddm_pr1_20260809/controls/census_result.json` (1,531 bytes,
SHA-256 `eb14cb04da2746b61672d2ca8b1eec660aec3d0a74499f2abb630dfb65724c6e`).

The charter's `427 / ~51,000` repository number is **not** the corrected live population and its
assumption that it was an upper bound did not survive. At task start, the pre-edit implementation
reported 1,633 findings in 935 files. The final call-site population is larger, even though the
charter's named nested expression now correctly contributes exactly one finding. The reason is a
second, opposite counting error: keying on `(root, file)` collapsed distinct materializations that
reused common names such as `blob`, so restoring one identity per materializing call exposed more
distinct sites than the nested-producer fix removed.

The same correction applies to the charter's SSD seed: `1,289 / 184,639` is neither the current
denominator nor an upper bound on call-site findings. The bounded live snapshot is `9,854 / 206,192`.
Both figures are preserved as static-detector populations; neither is promoted to a claim that every
site will execute or that every flagged byte object is ultimately discarded.

The final identity is `(materialization line, column, producer)` within a lexical scope. It:

- treats `zlib.compress(cq.tobytes() + np.float32(s).tobytes(), 9)` as one outer compressed payload;
- counts repeated `len(blob)` measurements of one materialization once;
- resolves `len(blob)` to the latest preceding assignment in the same lexical scope;
- keeps same-named locals in unrelated functions separate; and
- clears only the materialization actually passed to a byte persister.

An intermediate module-wide binding implementation was rejected before landing because it joined
same-named locals across functions and inflated the SSD count. Its numbers are not evidence and are
not banked.

## Wire-in and positive control

`src/tac/preflight.py::preflight_all` schedules the live-source census in its existing parallel
runner and prints findings, examined/discovered files, AST candidates, unreadable files, and the
WARN-ONLY state. Strict mode is deliberately not enabled while 1,072 findings remain in that scope.

The strict positive-control fixture is durable at
`/Volumes/VertigoDataTier/pact/ddm_pr1_20260809/controls/strict_positive_control/experiments/anchor.py`
(222 bytes, SHA-256 `03ac669c2b30e965fd0de4769dcd7a5e51ee1606b610af27177909334aa77733`).
Running `check_no_measure_and_discard_payload(..., strict=True)` against that isolated scope exited
with rc=1 by raising `RuntimeError` and reported exactly the two discarded Range/ANS payloads.

## Retrofitted live rows

| harness | pre-edit findings -> post-edit | retained candidates | future default custody |
|---|---:|---|---|
| `experiments/ddm_hp1_learned_ar_prior_race.py` | 1 -> 0 | raw token frame; shipped IX2 Brotli; forced IX2 LZMA; raw LZMA; raw Brotli; every non-skipped learned-context frame | `/Volumes/VertigoDataTier/pact/ddm_hp1_20260806/retained/` |
| `experiments/ddm_tk1_semantic_stream_race.py` | 3 -> 0 | raw labels; LZMA, zlib, bz2, and available Brotli output; every non-skipped learned model; full learned frame or estimate-only subset-control frame | `/Volumes/VertigoDataTier/pact/ddm_tk1_20260806/retained/` |

Both harnesses write atomically, re-read the retained file, recompute its SHA-256 and byte count,
and fail closed on disagreement before emitting the scalar result row. Per-candidate files are kept;
selection no longer destroys losing candidates.

The executed synthetic end-to-end control is
`/Volumes/VertigoDataTier/pact/ddm_pr1_20260809/controls/control_result.json` (7,751 bytes, SHA-256
`153cd44caf45c9eb31abbfd8cbcea882ff3408cdccee4591efa35c7683c4592c`, seed 20260809). It retained:

- HP1: five baseline objects at 180, 232, 133, 170, and 180 bytes;
- TK1 generic: raw plus four coded objects at 192, 18, 16, 48, and 13 bytes; and
- TK1 learned: both 30-byte and 1,080-byte model candidates plus the 112-byte subset-control frame.

Those byte counts validate retention mechanics only. They are synthetic, scorer-free, and make no
compression-quality or score claim.

## Deliberately queued remainder

- **QUEUED-WITH-A-FIRE-ORDER (1):** owner `ddm_tz1` current source owner, then MAIN's task-#1001
  successor; consumer store `.omx/state/operator_p0_ledger.jsonl`; fire trigger is the current dirty
  `experiments/ddm_tz1_token_sweep_rate_attack.py` landing clean with no active writer. Retrofit every
  per-candidate token-sweep payload before the next run.
- **QUEUED-WITH-A-FIRE-ORDER (2):** owner `ddm_pk2` current source owner, then MAIN's task-#1001
  successor; consumer store `.omx/state/operator_p0_ledger.jsonl`; fire trigger is `ddm_pk2` harvest
  or owner handoff. Its live pose-representation race takes priority over older dormant findings.
- **QUEUED-WITH-A-FIRE-ORDER (3):** owner MAIN's task-#1001 successor; consumer store
  `.omx/state/operator_p0_ledger.jsonl`; fire trigger is completion of orders 1-2. Adjudicate
  `experiments/ddm_lv1_token_coder_race.py` next, then the remaining 1,072 live-source findings by
  current execution value; retain real payloads or add a substantive same-line `MEASURE_ONLY_OK`
  waiver. Do not mechanically classify all static findings as confirmed violations.

## RECALL EVIDENCE

Full-corpus searches, beyond the charter seeds, used:

- `rg -n -i 'ans_real_n600|always.keep.*payload|retained.*payload|payload.*retention|measure.and.discard'`
  across `.omx/research`, `.omx/state`, `experiments`, `tools`, and `src/tac`;
- `rg -n -i 'payload retention|always keep|measure.and.discard|ans_real'` across
  `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, and task/state stores;
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  `payload|entropy|coder|compression|archive`; and
- exact task/ownership terms `#1001|ddm_pr1|ddm_ap1|ddm_dt1|ddm_tm1|ddm_hp1|ddm_tk1` across the P0
  ledger, arm receipts, spawn log, task bridge, and hot state.

Findings beyond the seeds changed the plan:

1. DT1/AP1/TM1 already established the reference retention shape: preserve every candidate, record
   bytes plus hash, and consume the same retained object byte-identically. The live retrofit copied
   that manifest discipline instead of inventing a winner-only store.
2. AP1/DT1 proved the ANS and Range payload identities and TM1 retained all six candidate families;
   this made coder races the highest-value retrofit order.
3. HP1 and TK1 were clean, landed source surfaces with current coder-race semantics. TZ1 was higher
   value but already dirty under another owner, so it was queued rather than overwritten.
4. The canonical equation registry contained nearby coder/archive arithmetic but no payload-custody
   equation. No canonical equation or scientific DAG edge changed; this is a static hygiene and
   custody landing, not a predictive scientific result.

## Validation

- focused tests: `37 passed`;
- Ruff: all six changed Python files passed;
- compileall: all six changed Python files passed;
- dogfood gate: `0 findings / 6 changed .py`;
- strict anchor control: rc=1, two findings;
- synthetic HP1/TK1 control: every recorded path, byte count, and SHA-256 re-read successfully;
- review tracker: two reviewed passes completed for each of the six changed `.py` files.

## Measurement boundary

Measured here: bounded static-detector populations with explicit denominators; strict-control refusal;
synthetic retained-file bytes and hashes; and before/after detector findings on HP1/TK1. Not measured:
SegNet, PoseNet, source-video quality, `upstream/evaluate.py`, contest CPU/CUDA, a composed archive, or
any score. The PR130 baseline remains `S=0.172141297491896447` at 191,052 bytes
`[contest-CUDA, DALI GT, n600]`; this arm did not move it.
