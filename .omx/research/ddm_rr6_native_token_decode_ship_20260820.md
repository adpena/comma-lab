# ddm_rr6 — the native token decoder, proven in the shipping runtime and actually switched on

Date: 2026-08-20 · Owner: ddm_rr6 (Opus build arm) · Status: **IDENTITY PROVEN full n600 in the real
runtime · 1.790x MEASURED `[macOS-CPU advisory]` · shipping-axis UNMEASURED, one T4 row owed**

Charter: the native token-decode port on the submission critical path. `ddm_wc2` established the
target (jg5 token decode **1,341.540 s = 94.5%** of a **1,419.900 s** inflate `[contest-CUDA T4]`,
against a corrected CUDA residual of **[890.6, 1430.6] s** — WARN, fits by 10.7 s on a warm cache
only). `ddm_wc2c` built the split path and proved it on a bench driver.

---

## HEADLINE

1. **The port was proven, but it could never have fired in the contest.** `upstream/evaluate.sh:47`
   runs `bash inflate.sh <dir> <out> <list>` **with no environment**, and the staged tree read
   `export F26_TOKEN_DECODER="${F26_TOKEN_DECODER:-python}"`. The accelerator was reachable only by
   a human exporting a variable that the contest never exports. Lifting the refusal moved the
   stranding down one level rather than removing it. **Fixed:** the default is now `native-hpac`.
2. **A compiler failure would have cost the whole submission, not the speedup.** `inflate.sh` runs
   under `set -euo pipefail`, so a `cc` failure on unknown contest silicon aborted the inflate —
   turning a wall-clock WARN into a zero. **Fixed:** both build attempts moved inside an `if`
   condition (which errexit does not apply to) and failure now degrades to the proven Python
   decoder. **Verified by execution**, with a compiler that fails only on the native translation
   unit: `f26 native build unavailable; using the python token decoder`, **rc=0**.
3. **Identity holds on the full field, in the real runtime, not a bench driver.** A local
   macOS-arm64 CPU decode of the jg5 archive reproduces every `[contest-CUDA T4]` token anchor.
4. **MEASURED 1.790x on a matched instrument** — same host, same runtime, same archive, same thread
   counts, the two runs differing only in the token decoder: **589.456 s → 329.370 s**.
5. **The shipped binary now carries no unexecuted instruction.** `-DF26_FORCE_SCALAR=1` compiles out
   the hand-written AVX2/NEON kernels. They are integer-only and carry no FP hazard, but the AVX2
   kernels have **never been executed** and the contest box is x86 — and `ddm_wc2c` MEASURED the
   intrinsic-free twin at 324.779 s against the dispatched NEON build's 326.160 s (both at 4
   threads), i.e. **0.4% faster**. Getting to those two numbers required correcting a mislabelled
   retained receipt, which is §2.2 and is worth reading before anyone quotes a scalar-vs-SIMD figure.
   The speed comes from the compiler's auto-vectoriser, not the hand kernels, so pinning
   the twin costs nothing measured and removes the risk of a wrong-fast decode.

---

## 1. The identity gate, and why it has two halves

The T4 receipt's `0.raw` (`6bf8acf8…`) and a local CPU `0.raw` do **not** match, and must not be
expected to: the neural render is a torch forward pass and CUDA/CPU are separate numeric regimes.
Comparing them would be the cross-regime error. So identity is gated on two separate surfaces, each
scoped to what it can actually prove:

| surface | compares | proves |
|---|---|---|
| **cross-axis, token stage** | rr6 local native vs the jg5 `[contest-CUDA T4]` receipt | the token decode is axis-invariant AND the native port reproduces it exactly |
| **same-axis, full pipeline** | rr6 local native vs the jg5 local python advisory (`advisory_final`) | swapping the decoder changes nothing downstream, all the way to the scored bytes |

### 1.1 Cross-axis token anchors — MATCH, all four

jg5 archive `f3bce5d259a0…`, 180,625 B, n600, `token_decoder: native-hpac-split`, `decode_path: neon`,
4 threads.

| anchor | jg5 `[contest-CUDA T4]` | rr6 local native | verdict |
|---|---|---|---|
| `decoded_token_sha256` | `cc10a7b0…636efb` | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | **MATCH** |
| `corrected_quantized_logit_sha256` | `8269fe1a…eec4dd` | `8269fe1aad031620b18051ad784d877bc9e6e9a4a71e775e78681955c4eec4dd` | **MATCH** |
| `corrected_cdf_input_sha256` | `370a5e2a…e46000` | `370a5e2a85ccbb1e598c84333cc851f0a8c352091fde272160826b4b04e46000` | **MATCH** |
| `decoder_bit_position` | `910837` | `910837` | **MATCH** |

The 117,964,800-byte token field was re-hashed independently of the receipt and agrees:
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.

### 1.2 Same-axis full pipeline — IDENTICAL to the last byte of the scored output

| run | token decoder | `0.raw` bytes | `0.raw` sha256 |
|---|---|---:|---|
| `ddm_jg5/advisory_final` | `python` | 3,662,409,600 | `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7` |
| `ddm_rr6/advisory_native_r1` | `native-hpac-split` | 3,662,409,600 | `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7` |

**Byte-for-byte identical across all 3.66 GB** (`aggregate_sha256 23fc14a6…` likewise). Both runs
then score identically through `evaluate.py`, to the last digit — `canonical_score`
**0.19335265651220337**, `avg_posenet_dist` 0.00014701, `avg_segnet_dist` 0.0003474 — which is
forced by the byte identity and is recorded as the arithmetic check that it is, not as new evidence.
(That is the `[macOS-CPU advisory]` score for this body; the `[contest-CUDA T4]` score is
0.14839100138338618. The axis gap is the known GT-lineage/CUDA difference and is not this arm's
subject — the load-bearing fact is that r1 and the baseline agree EXACTLY on one axis.)

**The consequence for MAIN is the one that matters: this change cannot move the score.** `evaluate.py`
reads only `0.raw` and `archive.zip`; the archive is untouched (`f3bce5d259a0…`, 180,625 B) and the
raw is bit-identical. So a T4 fire on the staged tree is a **wall-clock measurement, not a score
row** — the score is `0.14839100138338618` by construction, and a T4 row that disagrees would
falsify the identity proof rather than reprice the candidate.

## 2. The speedup, on a matched instrument

Both rows: Apple M5 Max, `torch 2.12.1`, `cpu_num_threads 4`, `cpu_num_interop_threads 1`, the same
`experiments/contest_auth_eval.py` path via `tools/fire_local_advisory.py`, the same archive.

| token decoder | run | token stage s | **whole inflate s** | token ratio |
|---|---|---:|---:|---:|
| `python` | `ddm_jg5/advisory_final` | **589.456** | **978.873** | 1.000x |
| `native-hpac-split` | `ddm_rr6/advisory_native_r1` | **329.370** | **707.0** | **1.790x** |

The whole inflate falls **978.873 s → 707.0 s = 1.384x**, and the 271.9 s saved reconciles with the
260.1 s token delta to within 12 s of prelude and I/O noise — an arithmetic consistency check that
the saving is where the receipt says it is and not somewhere else.

`ddm_wc2c`'s bench driver measured 326.160 s for the same work; the real runtime lands within 1% of
it, which is the corroboration that its bench was representative.

**Against the charter's bars — WARN ≥ 1.096x, PASS ≥ 1.804x — this secures WARN and lands 0.8%
short of PASS.** That is the same straddle `ddm_wc2c` reported, now on the runtime rather than the
bench, and I will not round it up: 1.790x is below 1.804x.

### 2.1 NEW FINDING — the speedup is THREADS, not the lowering

Reading `ddm_wc2c`'s own retained receipts rather than its table:

| build | threads | n600 token s | vs python 589.456 |
|---|---:|---:|---:|
| scalar twin `97c25602` | **1** | **585.238** | **1.007x** |
| scalar twin `97c25602` | 4 | 324.779 | 1.815x |
| dispatched NEON `93ab636d` | 4 | 326.160 | 1.807x |

(Ratios recomputed against THIS arm's runtime python baseline of 589.456 s so all rows share one
denominator; `ddm_wc2c` quoted the same seconds against its bench baseline of 578.716 s, which is
why its table reads 1.782x / 1.774x. Same measurements, different denominator — not a disagreement.)

**At one thread the C is 1.007x — no win at all.** Essentially the entire 1.79x comes from the
pthread pool spreading patches across cores, not from lowering numpy to C. `ddm_wc2c` measured the
DECODED FIELD to be thread-independent (correct and load-bearing) but never reported that the SPEED
is thread-borne, so the port has been discussed as if C were the mechanism. It is not.

Consequences, none of them fatal but all of them worth holding:

* The contest CPU instance has **4 vCPU** (`upstream/README.md:114`) and `F26_HPAC_THREADS` defaults
  to a hard-coded 4 (`f26_hpac_native.c:135`, `long value = 4` — not `nproc`, not `min(4, cores)`).
  That happens to match both the contest CPU spec and the measured configuration, so it is left
  alone: raising it or making it adaptive would be an unmeasured change to the one knob that
  carries the entire win.
* Any contest environment that gives the process fewer usable cores collapses the win toward 1.0x
  **without changing a byte** — the fail-closed design degrades to slow-and-correct, which is the
  intended failure mode, but the budget predicate must read `decode_threads` from the receipt or it
  will grade a 1-thread run against a 4-thread expectation.
* It also reprices the corrector port: if the remaining Python side is likewise parallelisable, that
  is a cheaper lever than 2,121 lines of C. **UNMEASURED — a hypothesis this arm did not test.**

### 2.2 A receipt filename that lies about its contents

`/Volumes/APDataStore/pact/ddm_wc2/retained/wc2c_scalar_twin_n600_t4.json` carries
`decode_seconds 585.2376984581351` — the float is byte-identical to
`wc2c_thread_independence_t1.json`. The file named `_t4` contains the **t1** run. The 4-thread scalar
row lives in `wc2c_thread_independence_scalar_t4.json` (324.779 s).

`ddm_wc2c`'s memo number is CORRECT; only the filename is wrong. Recording it because it nearly
inverted a decision here: read at face value it says the intrinsic-free build is 1.79x SLOWER than
NEON, which would have made pinning `-DF26_FORCE_SCALAR=1` a catastrophic choice instead of a free
one. This is the phantom-directory genus at receipt scale — the name must not lie about the content.

**This is not a shipping-axis number.** The T4 baseline evaluates the sparse model on CUDA with two
host↔device round trips per group; this evaluates it on CPU torch. Transferring the ratio across
those regimes is the cross-regime constant genus. What the split path does do on T4 is remove the
114,000 round trips entirely, because it never touches the GPU — a directional reason to expect the
shipping ratio to be *better*, which is a hypothesis and not a measurement.

### 2.3 What the T4 arithmetic would become, labelled as PROJECTION

Non-token inflate on the jg5 T4 row is fixed at `1419.900 − 1341.540 = 78.360 s`.

| if the shipping ratio is | token s | inflate s | vs residual [890.6, 1430.6] |
|---|---:|---:|---|
| 1.000x (today, python) | 1341.5 | 1419.9 | WARN — inside the wide end by 10.7 s |
| 1.096x (the WARN bar) | 1223.6 | 1302.0 | WARN with 128.6 s of margin |
| **1.790x (local measured)** | **749.5** | **827.8** | **PASS — inside the NARROW end by 62.8 s** |

All four rows are PROJECTIONS. **One T4 row replaces the whole table.**

### 2.4 Why the T4 number genuinely cannot be inferred, in both directions

The split path moves the sparse model **off the GPU onto the CPU**. That makes the T4 result depend
on the T4 instance's vCPU count and per-core speed, which `upstream/README.md:114` does not state
(it specifies CPU count only for the CPU instance: 4).

Two measured facts pull in opposite directions and neither settles it:

* **For:** the T4's CUDA token stage (1,341.540 s) is **2.28x SLOWER** than this laptop's *CPU*
  python token stage (589.456 s) on the same archive. A GPU losing to a CPU by that margin is
  strong corroboration of `ddm_wc2`'s latency-bound reading — 114,000 iterations, each two
  host↔device round trips around a tiny kernel. The split path deletes every one of them.
* **Against:** an M5 Max core is not a cloud vCPU, and §2.1 says the win is thread-borne. A T4 box
  with few or slow vCPUs shrinks the win; the CPU-side baseline it must beat also gets slower, so
  even the sign of the *ratio* change is not determined by argument.

Quoting the 1.790x as a shipping figure would be the cross-regime constant-transfer genus with the
regimes swapped end for end. It is a local advisory number and nothing more.

---

## 3. What changed, and the discipline used

The sealed jg5 tree was not touched. Every change is a **recorded textual transformation with an
asserted match count** in `experiments/ddm_wc2c_stage_native_split_runtime.py`, so a drifted base
refuses to stage instead of silently patching text that no longer means what it did. `ddm_wc2c`
contributed rewrites 1–4; this arm added 5 and 6:

| # | rewrite | why |
|---|---|---|
| 5 | `F26_TOKEN_DECODER` default `python` → `native-hpac` | upstream sets no env; the default IS the shipped configuration |
| 6a | both `cc` attempts moved into the `if` condition, `else` exports `F26_TOKEN_DECODER=python` | a hostile toolchain must cost speed, never the submission |
| 6b | `-DF26_FORCE_SCALAR=1` on both attempts | ship no instruction that has never been executed |

An explicitly-named-but-missing `F26_HPAC_NATIVE_LIBRARY` still `exit 69`s. Operator
misconfiguration and a hostile toolchain are different classes and are treated differently.

### 3.1 The canonical local firer could not express this run

`tools/fire_local_advisory.py` carries exactly `PATH` and `PYTHONDONTWRITEBYTECODE`, because
omitting either produced a measured launch failure (ck1 rc=2 at t=0.0s; V7 refused at t=5s). It had
no way to pass a runtime knob, so selecting the decoder meant hand-assembling the launcher argv —
the error factory the tool exists to replace. Added `--env KEY=VALUE` (repeatable), recorded
separately in the launch manifest because the env IS the instrument.

**The passthrough may not set a carried key.** A collision is a REFUSAL, not a precedence rule:
letting `--env PATH=…` through would reopen exactly the two classes the tool closes. Duplicate keys
also refuse, because an argv-order-dependent effective value is a confound.

---

## 4. Fire order for MAIN (this arm does not touch the paid axis)

**Ask:** one `[contest-CUDA T4]` row on the staged tree, same archive, via
`tools/fire_modal_auth_eval.py`.

| field | value |
|---|---|
| runtime tree | `/Volumes/APDataStore/pact/ddm_rr6/candidate_runtime_jg5_ship` |
| archive | `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`, 180,625 B — **UNCHANGED from jg5** |
| env | **none** — the whole point is that the default now selects `native-hpac` |
| harvest | `stage_seconds.token_decode_or_checkpoint_load`, `inflate_elapsed_seconds`, `evaluate_elapsed_seconds`, and `token_decoder.{decode_path, decode_threads}` |

**It is a wall-clock row, not a score row.** `0.raw` is byte-identical to the Python path (§1.2), so
the score must come back `0.14839100138338618`. Read the three verdicts this way:

* score unchanged **and** inflate well under 1,302 s → the sub-0.15 archive is CI-shippable with
  margin instead of by 10.7 s;
* score unchanged, inflate still near 1,420 s → the T4's CPU cannot carry the token stage; the
  identity work stands, the wall does not move, and §2.1 says look at thread count first;
* **score changed → the identity proof is falsified.** Stop and treat it as a defect, not a
  re-pricing. Nothing in this change is permitted to move a byte.

Also worth reading off the same row: `decode_path` should be `scalar` (the FORCE_SCALAR pin) and
`decode_threads` should be `4`. A `token_decoder: python` in the receipt means the native build
failed on the runner and fell back — the intended failure mode, and exactly the case §2.1 says the
budget predicate must be able to see.

**Re-pin owed before the fire:** the harness enforces the runtime tree SHA in three places
(`contest_auth_eval.py` `_validate_expected_runtime_tree` :1554, `_validate_expected_runtime_files`
:1566, `inflate_script_sha256` :1627). The staged tree changes all of them, so any stored fire
manifest carrying jg5's expected SHAs will refuse until re-pinned. That refusal is the validator
working.

## 5. Owed — named, not buried

1. **A T4 row on the staged tree.** The only number that decides the CI wall. **This arm does not
   touch `tools/fire_*` for the paid axis; the fire order is in §5.**
2. **x86 is still UNVERIFIED**, and now deliberately unreachable: `-DF26_FORCE_SCALAR=1` means the
   AVX2 kernels are not in the shipped binary at all. Re-enabling them is deleting one `-D`, and it
   should happen only after someone executes them on an x86 host.
3. **The corrector port is the remaining large lever, and this arm did not take it.** After the
   integer half is lowered, the Python side is 62.4% of the split run (203.6 s of 326.2 s), and the
   float64 corrector alone is **59.2%** of it (`coding_row` 111.479 + `observe` 55.997 +
   `group_state` 25.639 = 193.115 s, from `ddm_wc2c`'s profile). Stating the corrector at 62.8%
   would fold in the probability table, the digests, the RC64 ctypes calls and the transfers, which
   the port would not touch. It is 2,121 lines
   across four modules (`rr4 → fx1 → fx2 → ma1`) including a mixer family, an SSE stage with
   `dyadic_power` over `np.sqrt` radicals, and a hand-fixed summation order whose measured failure
   mode is **S = 27.83** — a desynchronised decoder, not a rounding wobble. `ddm_wc2c` priced it at
   ~3.7x total for a 5x corrector. It should be argued on its own measured merits and given its own
   arm, not inherited from this one's momentum.
4. **Per-frame resume still does not exist** (`rc64_backend.c` exports no state save/restore).
   Resume granularity is the whole token stage — unchanged from the Python path.
