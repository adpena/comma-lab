# ddm_t1r1 T1 whole-container build rehearsal

Tags: [no-triality] [p0-ledger-ok]  
Axis: [macOS-CPU scorer-free byte/custody apparatus]  
Score claim: false

## RESULT

The pass-4 rehearsal is receiver-closed. It is **not a candidate**: the pose
section is the stale pass-4 CPR1 stand-in, no scorer ran, and no distortion or
score transfers to this object. Prediction m38 was confirmed. CP135 had the
pieces needed to export a new HP3 probability object and encode tokens, but it
did not have a wired whole-container path for a different semantic plane. The
adapter added by this arm closes that build gap.

| measured object or check | result |
|---|---:|
| rehearsal `archive.zip` | **187,046 B**, SHA-256 `12a5b181fef4e15ad8a752161c744347beca0b5a1224c5d3d542ab148f6ece80` |
| repeat archive | **187,046 B**, same SHA, byte-identical |
| delta from CP135 base | **+794 B** versus 186,252 B |
| real C1-on-HP3 RC64 token stream | **115,237 B**, SHA-256 `b9071d5251af23c72fc922737f802a825fd215a451613d7a39f0cf193fdcd69b` |
| token delta from CP135 HP3 stream | **+6 B** versus 115,231 B |
| C1 source plane | 117,964,800 tokens, SHA-256 `2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5` |
| event-order plane | 117,964,800 symbols, SHA-256 `76e6fba6c7c1bca12c69200a95059c2be9129bcef09544c799fd5b696b7b9f61` |
| HP3 export wall clock | 582.704 s |
| RC64 encode / independent decode | 8.542 s / 11.640 s |
| shipped receiver RC64 decode | **11.844 s**, 0.658% of the 30-minute budget |
| complete container parse-back | **14.883 s** |
| receiver acceptance | **PASS** |
| positive inflate assertions | **PASS** |
| corrupt-archive / mismatched-member controls | **REFUSED / REFUSED** |
| full CUDA render or scorer | **NOT RUN** |

The real HP3 result supersedes the cross-state `+11 B` F26 proxy for this build.
It does not supersede that proxy on its own F26 probability state. The rehearsal
archive is 177 B smaller than the stale pass-4 archive, but that is a cross-state
container comparison and is not a rate credit or a candidate claim.

## INPUT PINS AND PARENT HONESTY

- CP135 base: `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip`,
  186,252 B, SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
- C1 solved plane: `/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained/c1_solved_tokens_n600.u8`,
  117,964,800 B, SHA-256 `2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5`.
- Stale pose stand-in: pass-4 archive SHA-256
  `e269d1ffbe0bf56ec8471a6869b7ec081f3de07e852b193aa251a963c543becb`,
  CPR1 23,051 B SHA-256
  `4c1a65c7f3a9bfa1b0f7677494ddbfdad87881fe0f4b78613893bd555f725ef2`,
  and direct int16 coefficient array SHA-256
  `da9bba74fdaadc8110b9eb0614decb6d3a5caa076a03b01eee5647d32c37590e`.
- The pass-4 CX2/TM1 parser restored the same 600 x 12 coefficient codes as the
  direct selected file before composition. The live solve tree was read only.

The probability-object receipt is stable and source-bound at
`PROBABILITY_IDENTITY.json`, SHA-256
`d6f58542808ea94d640510fe403eac2ce10a9ab59e380048d99a088c3a964cce`.
It excludes volatile wall time while retaining the complete per-frame export
records and source pins.

## BUILD RECIPE AS EXECUTED

The complete entry point is:

```bash
.venv/bin/python experiments/ddm_t1r1_container_build_rehearsal.py all \
  --output /Volumes/APDataStore/pact/ddm_t1r1/retained
```

Its resumable stages, also used separately while closing adapter defects, are:

```bash
.venv/bin/python experiments/ddm_t1r1_container_build_rehearsal.py prepare \
  --output /Volumes/APDataStore/pact/ddm_t1r1/retained
.venv/bin/python experiments/ddm_t1r1_container_build_rehearsal.py reencode \
  --output /Volumes/APDataStore/pact/ddm_t1r1/retained
.venv/bin/python experiments/ddm_t1r1_container_build_rehearsal.py build \
  --output /Volumes/APDataStore/pact/ddm_t1r1/retained
.venv/bin/python experiments/ddm_t1r1_container_build_rehearsal.py parseback \
  --output /Volumes/APDataStore/pact/ddm_t1r1/retained
```

`prepare` pins the source objects, proves the source pass-4 CX2/TM1 path, and
materializes the C1 event-order plane with an exact inverse and 24-frame
checkpoints. `reencode` calls the CP135 probability exporter with explicit C1
event-order and spatial SHA pins, then calls RC64 and independently restores all
117,964,800 symbols. `build` converts CPR1 to lossless dynamic CAP1, races all
12 Brotli qualities independently for each of HPAC, semantic, and carrier
sections, retains all 36 payloads, writes the complete member and deterministic
ZIP twice, and installs the bounded receiver adapter into a copied CP135
runtime. Selected qualities were HPAC q10 at 13,910 B, semantic q11 at 34,763 B,
and carrier q0 at 22,934 B; their packed model body is 71,613 B.

`parseback` uses the adapted runtime's own section parser and its shipped
decoder-only RC64 C source, compiled from retained source into a retained shared
object. It proves CP135 HP3 probability-object identity, semantic-state identity,
residual identity, pass-4 CPR1 identity, exact CAP1 coefficient/basis recovery,
and exact C1 event-order and spatial-token identity. The independent
ExperimentBook RC64 backend is behavior-identical but not byte-identical to the
receiver's decoder-only backend, so receiver acceptance rests on the actual
shipped backend's full n600 decode, not a source-file proxy.

Two fail-closed controls were exercised and retained. A one-byte-corrupt archive
was refused as `archive.zip does not match the promoted F26 artifact`; a valid
archive paired with a mismatched extracted member was refused as `extracted
payload does not match archive.zip`.

## ADAPTER GAP CLOSED

`experiments/ddm_cp135_rate_compose.py` now accepts explicit expected event-order
and spatial-token digests, records encode/decode timing, binds checkpoint resume
to a stable probability identity, and imports the actual HPAC optimizer from
`runtime.hpac_inference`. Existing defaults retain the original CP135 pins.

`experiments/ddm_t1r1_container_build_rehearsal.py` adds the missing end-to-end
adapter. It fails closed on source pins and storage, retains every materialized
payload, uses source-specific resumable HP3 export state, supports dynamic CAP1
inside the F26 model grammar, emits deterministic archives, and performs the
complete receiver receipt and negative controls. Focused tests cover pin
compatibility, resume identity, dynamic CAP1 parse domains, payload retention,
receiver compilation, and manifest exclusions.

The build did not run clean on its first attempt, so m38 was not falsified. The
rehearsal found and closed four concrete adapter defects before terminal day:
the direct-script namespace bootstrap was missing; CP135's exporter imported the
HPAC optimizer from the wrong runtime module; the generalized CAP1 receiver
incorrectly treated the high bits of the first two natural u16 section lengths
as reserved; and the acceptance check initially compared the ExperimentBook
encoder/decoder C file byte-for-byte with the smaller shipped decoder-only C
file instead of proving the shipped decoder's behavior on all symbols.

## RECEIPTS AND RETENTION

All materialized payloads remain below
`/Volumes/APDataStore/pact/ddm_t1r1/retained/`. The deterministic custody
manifest records 1,398 files totaling 1,775,757,636 B, with tree SHA-256
`5e4ff4961d1997075601b711022b438e740722a822ed62e2bc3e0cce608384e0`.
The manifest itself is 643,182 B with SHA-256
`4852cfcffa7a67d22bcd3b792dd58e6bd12d4e3523d362395ebab7248c926bf7`.
It explicitly excludes itself, macOS `._*` metadata, `__pycache__`, and `.pyc`
files; no payload was deleted. Storage preflight passed with 753,204,330,496 B
free against a 2 GiB requirement.

Primary machine-readable receipts are `20_PREPARE_RESULT.json`,
`50_HP3_REENCODE_RESULT.json`, `60_BUILD_RESULT.json`,
`70_PARSEBACK_RESULT.json`, and `receiver_state/RC64_COMPILE_RESULT.json` under
that retained root.

## VERIFICATION

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_ddm_cp135_rate_compose.py \
  src/tac/tests/test_ddm_t1r1_container_build_rehearsal.py
18 passed in 2.88s

.venv/bin/ruff check experiments/ddm_cp135_rate_compose.py \
  experiments/ddm_t1r1_container_build_rehearsal.py \
  src/tac/tests/test_ddm_cp135_rate_compose.py \
  src/tac/tests/test_ddm_t1r1_container_build_rehearsal.py
PASS

.venv/bin/python -m py_compile experiments/ddm_cp135_rate_compose.py \
  experiments/ddm_t1r1_container_build_rehearsal.py \
  src/tac/tests/test_ddm_cp135_rate_compose.py \
  src/tac/tests/test_ddm_t1r1_container_build_rehearsal.py
PASS

git diff --check
PASS
```

Two recorded review passes cover each changed Python file. The broader developer
preflight was also run:

```bash
.venv/bin/python -m tac.preflight --scope dev --timeout-s 30
```

It failed 8 of 25 repo-wide gates already present outside this change: strict
state-writer loading, authoritative-tag custody metadata, codebase drift,
dispatch-helper coverage, landing-solver wire-in, lane preregistration,
score-aware scorer-contract coverage, and trainer pose defaults. None of the
focused files was named as the violating source. This is a reported repository
boundary, not a waiver or a claim that the full preflight passed.

## RECALL EVIDENCE

The full local corpus search covered `.omx/research/` memos and receipts,
`.omx/state/main_hot_state.md`, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`,
design/SPEC and task-ledger surfaces, and all 431 canonical equations returned by
`.venv/bin/python tools/list_canonical_equations.py --json`. Content queries
included `CP135|HP3|F26|C1|RC64|ANS|CAP1|CX2|TM1|whole-container|probability
object|115231|114717|same-parent|terminal|split Brotli`.

Beyond the charter seeds, the search found the live HY1/js1 reseal amendment and
consumer store, LP135's settled same-state result that RC64 beats ANS by 9 B on
the exact HP3-step2 state, and the canonical
`brotli_cascade_bounded_per_stream_v1` equation. These changed the build in three
ways: it targets the HY1/js1 terminal consumer, does not reopen the settled ANS
race, and performs a complete 12-quality race separately on all three bounded
Brotli sections. No equation replaced the required real HP3 re-encode or
whole-container measurement. T0R1 and the E4/e5a chain supplied the CX2/TM1
parse-back shape, but no searched path supplied the missing C1-on-CP135 build
adapter.

## TERMINAL DIFF LIST

Exactly one step changes when the terminal pose artifact lands:

1. Replace `pass4_pose.cpr1` with the terminal same-parent pose carrier, then
   rerun `all`; every source pin, section race, archive hash, receiver receipt,
   and repeat receipt is recomputed and retained.

No scorer or GPU/evaluator lane was claimed or used. The sole ps135b scorer lane
and live solve tree were left untouched.

The effective floor remains CP135 composed at **S = 0.16195513827824176 @
186,252 B [contest-CUDA T4, n600], ours**. The own-vehicle frontier remains LC2
at **S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**. This
scorer-free rehearsal moved neither frontier.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: HY1/js1 whole-container builder. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/`, with new build receipts retained under `/Volumes/APDataStore/pact/ddm_t1r1/retained/`. Fire trigger: ps135 emits its terminal safe-run receipt and terminal same-parent pose carrier; replace the one stale CPR1 input and rerun `all` before any scorer request.**

## LIVE-HYPOTHESES

- The terminal same-parent pose carrier may fit at or below the stale carrier's
  22,934-B selected section because later pose passes optimize the same compact
  coefficient family; this is plausible but untested on the terminal bytes.
- The complete terminal object may clear the HY1 realization gate if the C1
  renderer preserves at least 82.824% of the solved-plane Seg gain. Exact token
  carriage is now proved, but renderer/R/scorer survival remains unmeasured.
- Full inflate should remain within 30 minutes because the shipped RC64 decode
  consumed only 11.844 s and the receiver change is a bounded parse adapter, but
  the exact terminal archive still needs its governed full render.

## DEAD-ENDS

- Adding HY1's `+11 B` F26 proxy to CP135 is closed. The real HP3/C1 stream is
  115,237 B, or `+6 B` against the CP135 HP3 stream.
- Gluing the C1 stream without re-encoding is closed. CP135 and F26 use different
  probability objects.
- Treating the ExperimentBook RC64 source as the shipped backend is closed. The
  files differ; the actual shipped backend now independently restores all
  117,964,800 symbols.
- Reserving high bits in the first two model-section lengths is closed. Natural
  compressed lengths can set those bits; only the explicitly tagged CAP1 field
  may carry the dynamic-format tag.
- Re-running ANS on this exact HP3 state is closed at INSTANCE scope by LP135's
  retained 9-B loss to RC64.
- Calling this rehearsal a candidate or assigning it a score is closed. It has a
  stale pose stand-in and no scorer, evaluator, or full-render receipt.
