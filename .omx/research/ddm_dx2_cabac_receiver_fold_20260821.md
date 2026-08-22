# ddm_dx2 CABAC receiver fold — exact −18 B candidate, raw gate queued

- **arm:** `ddm_dx2_cabac_receiver_fold` · **date:** 2026-08-21 · **fire authority:** MAIN only
- **base:** fx5_e1, **S 0.14823186109359 @ 180,386 B `[contest-CUDA T4, n600]`**, archive
  `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`
- **axis of this arm:** `[macOS-CPU scorer-free exact byte and receiver parse-back]`
- **verdict:** `BYTE_CLOSED_PARSEBACK_IDENTICAL__BLOCKED_RAW_IDENTITY_SLOT`
- **score claim:** false · **promotion eligible:** false · **Modal calls:** none

## Answer first

The measured DX1 adaptive-context Rice/CABAC-prefix winner is implemented in the real fx5_e1
receiver and folded into a retained candidate. The archive is **180,368 B**, exactly **−18 B**
from fx5_e1, with sha256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`. A separately
materialized repeat is byte-identical. The 9,811-byte CABAC payload re-encoded from the retained
600×12 symbols is byte-identical to the DX1 winner, and the candidate receiver reconstructs every
consumed fx5 field identically.

The charter's required fresh-process full `0.raw` decode was **not run**. The live local claim
registry still assigns the governed heavy local slot to `ddm_jo2_joint_objective_fx5_train_r7`
(`jo4_r7_fire`, state `eval`, 21–35 h solve). The charter explicitly forbids touching that slot.
Receiver parse-back is not a substitute for the 3,662,409,600-byte raw-output gate, so no seal was
written, no `SEAL_VALID` or `READY` claim is made, and MAIN must not fire the CUDA row yet.

| quantity | fx5_e1 base | DX2 candidate | delta / verdict |
|---|---:|---:|---:|
| Rice / CABAC coefficient payload | 9,829 B | **9,811 B** | **−18 B** |
| carrier body | 22,026 B | **22,008 B** | **−18 B** |
| archive | 180,386 B | **180,368 B** | **−18 B** |
| archive member `p` | — | 180,268 B, sha `365f1b8d7046…` | measured |
| deterministic archive repeat | — | same 180,368 B / same sha | **identical** |
| real receiver parse-back | base vs candidate | 9 decoded fields | **identical** |
| full raw identity | 3,662,409,600 B expected | not launched | **BLOCKED by occupied slot** |
| rate-only ΔS | — | −1.1985461156199085e−05 | derived from exact bytes |
| candidate S | 0.14823186109359 | 0.1482198756324338 | **projection only** until identity + T4 |

## The shipped mechanism

`src/tac/dx2_cabac_coefficients.py` is the encoder reference and receiver implementation. It
implements the exact measured cap=8 family: per-dimension, per-unary-prefix adaptive binary
contexts; Rice remainders are equiprobable bypass bins. All range state, frequencies and updates
are integers. The module imports NumPy only for fixed byte/int arrays; there is no torch, MLX,
CUDA, MPS or device-dependent branch.

The candidate retains RR5's disjoint basis rider (`0x08`) and adds DX2's coefficient rider
(`0x10`), changing the packed reserved value from `0x0A` to `0x1A`. The receiver restores RR5,
then DX2 reconstructs the original fixed-`k` Rice stream before CAP1 framing consumes its bit
count. `inflate.py`'s archive bytes and sha were derived from the archive written to disk.

The build is reproducible through:

```text
.venv/bin/python experiments/ddm_dx2_cabac_receiver_fold.py \
  --out-dir /Volumes/APDataStore/pact/ddm_dx2/<fresh-run>
```

Canonical retained run: `/Volumes/APDataStore/pact/ddm_dx2/r7`.

## Controls and boundaries

All of these controls passed on the real fx5_e1 body:

1. The unedited fx5 container re-emits byte-identically under its incumbent Brotli q9/lgwin16
   settings.
2. Re-encoding the real 600×12 coefficient object yields the exact retained DX1 winner:
   9,811 B, sha `b93131a52674abb4ada677e1b6cf08eebc6afb94381136d23d010e70a287e210`.
3. CABAC decode returns the retained symbols exactly; CABAC-to-Rice restoration returns the exact
   original carrier body.
4. The archive delta is exactly −18 B and the independently materialized repeat is identical.
5. The real shipping receiver parser compares ten fields, including nine decoded/consumed fields,
   with no mismatch. Only the compressed-model container provenance changes because Brotli sees a
   different 18-byte-shorter body; a read-site census proves that provenance field inert.
6. A persisted one-bit-corrupted CABAC payload is refused by canonical decode/re-encode. The
   negative payload is 9,811 B, sha `d5e54fcb244de9efd7eb8195bdc491b1bb71cfa3f73f25238fea4ea0dfb7bcd5`.
7. The staged runtime contains the reviewed coder bytes exactly and no `__pycache__` or `.pyc`.

The following did **not** happen: no scorer ran; no `upstream/evaluate.py` ran; no full raw render
ran; no Modal job was dispatched; no score or frontier promotion was recorded. The contest-CUDA
raw anchor remains `6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883`;
the local macOS-CPU raw anchor is `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`.

## Custody

- Result receipt: `/Volumes/APDataStore/pact/ddm_dx2/r7/RESULT.json`, sha
  `45324ac02fadfa490795b0f37d67e9ac2ec4f9a97a22c6d38785dadc2f80a68c`
- Retention manifest: `/Volumes/APDataStore/pact/ddm_dx2/r7/RETENTION_MANIFEST.json`, sha
  `3bcaa040e00f08a795c99c223d836ed8f75f2c7a8df7a49602db4e80158fe44d`
- Candidate: `/Volumes/APDataStore/pact/ddm_dx2/r7/retained/candidate_dx2_cabac.zip`,
  180,368 B, sha `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`
- Repeat: `/Volumes/APDataStore/pact/ddm_dx2/r7/retained/candidate_dx2_cabac.repeat.zip`,
  same bytes and sha
- Re-encoded CABAC payload:
  `/Volumes/APDataStore/pact/ddm_dx2/r7/retained/dx2_payload_adaptive_ctx_rice_cap8.bin`,
  9,811 B, sha `b93131a52674abb4ada677e1b6cf08eebc6afb94381136d23d010e70a287e210`
- Corrupted negative payload:
  `/Volumes/APDataStore/pact/ddm_dx2/r7/retained/negative_control_corrupt_cabac.bin`,
  9,811 B, sha `d5e54fcb244de9efd7eb8195bdc491b1bb71cfa3f73f25238fea4ea0dfb7bcd5`
- Candidate runtime: `/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2`

Runs `r1`–`r6` are retained as superseded build evidence. `r3` caught host bytecode residue created
by the test import; `r5` exposed AppleDouble metadata sidecars created by the SSD bridge; `r6`
removed those sidecars; `r7` additionally fails closed on the exact fx5 T4 receipt and its authority
fields and is canonical. No materialized codec payload was discarded.

One custody-label correction matters: the charter's `0bfe31cf…` is the sha256 of the contiguous
`int32[600,12]` **array contents**. The `.npy` container file itself hashes to
`8fc44020c3d5cb8ebe7d4adfabe7d1b0e05ad321f85bed03cb7086f04f201d95`. Both objects are pinned
and verified; this is a label distinction, not a data mismatch.

## RECALL EVIDENCE

I searched the full `.omx/research/` corpus and receipts by content for `CABAC`, `adaptive-ctx`,
`Rice`, `dxi`, `coefficient stream`, `CAP1`, `RR5`, `receiver fold` and `rate ceiling`; queried
the canonical-equations JSON; checked `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG FEED blocks,
the design/SPEC surfaces, the canonical task ledger and the live lane registry.

Beyond the charter seeds, this found:

- `ddm_fx5_composed_rate_candidate_20260821.md` had already measured on the live lineage that RR5
  codes the basis symbols while DX1 codes the coefficient stream. This changed the receiver plan
  from a possible collision to two explicitly disjoint reserved bits.
- `ddm_r012_rate_representation_20260821.md` independently decomposed the live body's whole
  composable ceiling as fx5's 70 B plus DX1's 18 B. It also closes the larger interpretation:
  harvesting DX2 does not make the body capable of sub-0.12; a new rate representation is still
  required.
- The task ledger row `fx5_dx1_cabac_carrier_leg` was pending/unassigned and named the CAP1 signal
  plus receiver decoder as the remaining blocker. This arm consumes that exact ownerless unit;
  it does not re-open the coder race.
- The live lane registry, newer than the charter's `jo1` wording, names JO4's governed local solve
  as the current slot owner. This changed the close from fresh raw decode/seal to an explicit
  queued gate rather than an unauthorized concurrent launch.
- No canonical equation added a different codec or changed the measured −18 B decision in the
  searched scope.

## Queued raw gate, seal and MAIN command

When the heavy local slot is terminal and released, run the fresh-process raw gate into the named
consumer store:

```text
.venv/bin/python tools/launch_detached_process.py \
  --output-dir /Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/launch \
  --cwd /Users/adpena/Projects/pact \
  --purpose "ddm_dx2 full n600 raw decode-identity gate; scorer-free" \
  --authority "[macOS-CPU advisory] n600 raw-only identity; no scorer; no score claim" \
  --derive-resource-budgets --measured-peak-rss-gib 10.0 \
  --measured-thread-need 4 --walltime-cap-s 5400 \
  --done-receipt ddm_dx2_decode_r1.done -- \
  bash /Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2/inflate.sh \
  /Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/extracted \
  /Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/inflated \
  /Volumes/APDataStore/pact/upstream_eval_mirror_20260815/public_test_video_names.txt
```

Admit only if `inflated/0.raw` is exactly 3,662,409,600 B and hashes to the local fx5 anchor
`7246a4ff…f5f2de7`; preserve the raw receipt and payload. Then make the seal:

```text
.venv/bin/python tools/make_candidate_seal.py \
  --candidate-id ddm_dx2_fx5_e1_cabac_coefficients \
  --runtime-dir /Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2 \
  --axis contest_cuda \
  --out /Volumes/APDataStore/pact/ddm_dx2/r7/CANDIDATE_SEAL_dx2_fx5_cabac.json \
  --receiver inflate.py --receiver inflate.sh \
  --receiver runtime/residual_archive.py \
  --receiver runtime/dx2_cabac_coefficients.py \
  --receiver runtime/rr5_arith_basis.py \
  --receiver runtime/f26_corrector_native.c \
  --receiver runtime/native_free_corrector.py \
  --receiver runtime/fx2_model_axis_corrector.py \
  --receiver runtime/free_corrector.py \
  --archive-member p \
  --retained-path /Volumes/APDataStore/pact/ddm_dx2/r7/retained \
  --retained-path /Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1 \
  --falsifier "INSTANCE: local raw must be 3662409600 B sha 7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7 and contest-CUDA raw must be sha 6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883; any mismatch refuses" \
  --falsifier "INSTANCE: archive must remain 180368 B sha 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674 and d_seg/d_pose must equal fx5 at report precision" \
  --admit-bar-net-ds -0.0000035 --pointer-axis contest_cuda \
  --verify-archive-sha 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674 \
  --bound-base-receipt /Volumes/APDataStore/pact/ddm_fx5/t4_row_r1/MODAL_REMOTE_RESULT.json \
  --sealed-by ddm_dx2 \
  --notes "single-axis waiver: lossless rate-only fold; fresh full raw decode must match the retained fx5 object before fire"
```

Only after the seal command reports `SEAL_VALID`, MAIN's exact fire command is:

```text
.venv/bin/python tools/fire_modal_auth_eval.py \
  --seal /Volumes/APDataStore/pact/ddm_dx2/r7/CANDIDATE_SEAL_dx2_fx5_cabac.json \
  --output-dir /Volumes/APDataStore/pact/ddm_dx2/r7/t4_row_r1 \
  --lane-id lane_ddm_dx2_fx5_cabac_cuda_20260822 \
  --instance-job-id ddm_dx2_fx5_cabac_t4_r1 \
  --axis cuda
```

**Own-vehicle frontier: S 0.14823186109359 @ 180,386 B `[contest-CUDA T4, n600]` — UNMOVED.**
This arm produced an exact byte-closed candidate, not an authority row.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — raw identity:** owner `MAIN/local decode custodian`; consumer store
  `/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1`; fire trigger: the latest
  `ddm_jo2_joint_objective_fx5_train_r7` local claim is terminal/released and storage preflight
  admits the 3,662,409,600-byte output. Run the exact detached command above and retain its raw,
  hash and launch receipts.
- **QUEUED-BEHIND-RAW — seal:** owner `MAIN`; consumer store
  `/Volumes/APDataStore/pact/ddm_dx2/r7/CANDIDATE_SEAL_dx2_fx5_cabac.json`; fire trigger: the fresh
  local `0.raw` is exactly 3,662,409,600 B with sha `7246a4ff…f5f2de7`. Run the exact seal command
  above; any raw, archive or pointer mismatch refuses.
- **QUEUED-BEHIND-SEAL — contest row:** owner `MAIN`; consumer store
  `/Volumes/APDataStore/pact/ddm_dx2/r7/t4_row_r1`; fire trigger: the named seal exists and validates
  as `SEAL_VALID`, and the CUDA lane is claimed without duplication. Run the exact MAIN command
  above; the expected contest-CUDA raw sha is `6bf8acf8…e79883`.

## LIVE-HYPOTHESES

- The candidate will reproduce fx5's raw output exactly because the real receiver has already
  reconstructed every consumed field identically, including the original Rice carrier bytes; the
  untested part is the required full fresh-process render, not a known semantic gap.
- The T4 row will differ from fx5 only by the exact rate term because the transform is lossless and
  receiver-closed. This remains a hypothesis until contest-CUDA independently reproduces the raw
  anchor and components.
- A new rate representation remains the only plausible route from the post-DX2 body toward sub-0.12:
  DX2 exhausts the last measured 18 B of the old body's 88 B composable rate ceiling.

## DEAD-ENDS

- Re-running the DX1 fruit/coder search is closed for this unit: the 16-coder optimal-form race
  already selected the exact 9,811-byte winner, and the charter forbids approximating or re-racing it.
- Treating RR5 and DX2 as a carrier collision is closed: real parsing shows RR5 recodes basis symbols
  while DX2 recodes the disjoint Rice coefficient stream; both flags and both inverses compose.
- Using the withdrawn OV1 −0.001190 ceiling is closed: it priced the wrong object and was 99% too large.
- A device-dependent probability path is closed: the receiver's probabilities and range updates are
  integer-only and have no CPU/CUDA/MPS branch, avoiding the prior RR2 desynchronization class.
- Sealing on parse-back alone or launching the heavy raw decode concurrently is closed: the charter
  requires fresh full raw identity and explicitly reserves the occupied local slot.
