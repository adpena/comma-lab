# ddm_wc2c — `native-hpac` re-enabled behind a full-field identity proof: ~1.8x on the token stage

Date: 2026-08-20 · Owner: ddm_wc2c (Opus build arm) · Status: **BUILT + IDENTITY-PROVEN
(full n600), speedup MEASURED `[macOS-CPU advisory]`, shipping-axis UNMEASURED**

Charter: `.omx/research/ddm_wc2_wall_clock_pass_20260820.md` GO section. The jg5 pointer body
(S 0.14839100138338618 @ 180,625 B) does not fit the CI wall: inflate **1,419.900 s** against a CUDA
residual of **[822, 1302] s** = REFUSE. Token decode is **1,341.540 s = 94.5% of it** (95.72% is the
share of the 1,401.58 s instrumented-stage sum, a different denominator — corrected 2026-08-20 by
`ddm_pq8` per `ddm_nv1`; this line was a fifth instance of the wrong-referent beyond the four the
`ddm_pq8` charter named). The accelerator
that would fix this was hard-refused at `runtime/f26_inflate.py:435-441`.

---

## HEADLINE

1. **The refusal is lifted, and the lift is proven, not asserted.** The split native path reproduces the
   jg5 `[contest-CUDA T4]` receipt **bit for bit on the full 600-frame field** — a local macOS-arm64
   decode reproducing a T4 receipt exactly, across all four anchors and the retained 117,964,800-byte
   token payload.
2. **MEASURED 1.774x–1.834x** on the token stage, n600, `[macOS-CPU advisory]`: **578.716 s →
   326.160 s / 315.614 s** across two dispatched runs of identical work. The charter's derived PASS bar
   is **1.804x**. **The measurement STRADDLES it.** I will not quote the favourable run: two runs of
   the same binary on the same host differ by 3.3%, which is larger than the distance to the bar, so
   the local proxy cannot decide PASS-vs-WARN. Only a shipping-axis row can.
3. **The charter's port target was wrong, and measurement is what found it.** The charter says "port the
   ddm_rr2 `FreeCorrector`". The module the shipping tree actually loads as
   `runtime/free_corrector.py` is **`ddm_ma1`'s `Ma1WithinMissCorrector`**, the tip of a four-level
   chain `rr4 → fx1 → fx2 → ma1` totalling **2,121 lines of stateful float64**. The named one-function
   port does not exist.
4. **So the split is on the axis with no FP hazard.** MEASURED per-step, the loop is 63.8% integer model
   and 33.4% float64 corrector. Lowering the integer half needs no reduction-order reasoning at all;
   lowering the float half puts 2,121 lines under the hazard whose measured failure is `S = 27.83`.
   The integer half alone clears WARN outright and reaches the PASS boundary.
5. **CUSTODY FINDING, unprompted and material: the shipping decoder is not in version control.** A
   34-file census of the jg5 candidate tree found **24 files with no repo source of any kind**,
   including `runtime/f26_inflate.py`, `runtime/residual_archive.py`, `runtime/free_corrector.py`, all
   of `cpr1/`, and `inflate.py` / `inflate.sh` themselves. The thing we would submit exists only on an
   external SSD at 93% capacity. Six more are import-rewritten derivatives of `experiments/` modules.
6. **Two decode-time dependencies removed.** OpenMP is gone from the C. It was both a correctness hazard
   (PyTorch already loads an OpenMP runtime; on macOS that is an immediate abort, and the documented
   escape hatch is labelled unsafe) and a build-time dependency inside the 30-minute wall
   (`brew --prefix libomp` on Darwin, `-fopenmp` on Linux). The built library now links libc only.

---

## 1. The measurement that chose the design

`experiments/ddm_wc2c_token_stage_profile.py` is the shipping loop with a `perf_counter` around each of
its fourteen steps. MEASURED, n600, real archive, 114,000 iterations, `[macOS-CPU advisory]`:

| step | seconds | share |
|---|---:|---:|
| `sparse_selected_logits` | 351.076 | 60.66% |
| `corrector_coding_row` | 111.479 | 19.26% |
| `corrector_observe` | 55.997 | 9.68% |
| `corrector_group_state` | 25.639 | 4.43% |
| `frame_context` | 18.216 | 3.15% |
| `probability_table` | 6.368 | 1.10% |
| everything else (digests, RC64, transfers, argmax) | ~9.9 | ~1.7% |
| **total** | **578.716** | |

**Integer model = 63.81%. float64 corrector = 33.37%.** RC64 is 0.4%; the two in-loop sha256 digests
are 0.2% combined, so ranked-head item 4 (trimming them) is worth ~1 second and is not worth its risk.

The n=12 prefix gave 64.37% / 32.34% — within 0.6 points of the n600 shares. Recorded because I
expected the opposite: the corrector's context tables warm up, so I predicted the prefix would
UNDER-state it. It did not. The prefix was representative *for this ratio*; that is a fact about this
quantity, not a licence to trust prefixes generally.

## 2. What was built

`runtime-rs/native/f26-hpac/f26_hpac_native.c` gains a **split entry surface** beside the existing
fused one:

```
f26_hpac_frame_begin(decoder, model, previous, frame, current)
f26_hpac_group_logits(decoder, model, boundary, group, corrected_out, predicted_out, flat_out)
f26_hpac_group_commit(decoder, model, group, symbols, current)
```

It stops at the corrected float rows and hands them back. The caller keeps the probability table, the
corrector, and the RC64 coder in the exact numpy and ctypes code the shipping receipt was produced
with. Both surfaces call the same static kernels (`f26_group_model`, `f26_row_corrected`,
`f26_group_apply`), so agreement between them is structural — there is one implementation of the
arithmetic, not two that happen to agree.

**Threading.** A pthread pool replaces OpenMP. Every parallel region partitions patches and each patch
writes only its own slice; no cross-thread reduction exists anywhere, so the decoded field is
independent of `F26_HPAC_THREADS`. That is checked, not assumed.

**ISA gating.** `AVX2 → scalar` on x86 chosen at runtime by `__builtin_cpu_supports`, so a prebuilt
library shipped to unknown silicon still selects a legal path. NEON on arm64 is architectural — the
target IS the gate. `-DF26_FORCE_SCALAR=1` builds an intrinsic-free twin as the identity oracle.
`f26_hpac_dispatch_path()` reports the selection and the caller records it as `decode_path`, so a
fallback appears in the receipt instead of being silent.

## 3. Identity — the only thing that makes any of it admissible

Full n600 field, jg5 archive `f3bce5d2…` (180,625 B), against the `[contest-CUDA T4]` receipt at
`/Volumes/APDataStore/pact/ddm_jg5/t4_row_r1/harvested_artifacts/`:

| anchor | expected | split native | verdict |
|---|---|---|---|
| `decoded_token_sha256` | `cc10a7b0…636efb` | same | **MATCH** |
| `corrected_quantized_logit_sha256` | `8269fe1a…eec4dd` | same | **MATCH** |
| `corrected_cdf_input_sha256` | `370a5e2a…e46000` | same | **MATCH** |
| `decoder_bit_position` | `910837` | same | **MATCH** |
| retained 117,964,800-byte token payload sha256 | — | `cc10a7b0…636efb` | **MATCH** |

The local pure-Python baseline reproduces the same four anchors, which independently confirms the T4
receipt is reproducible off-axis at all.

Also proven: dispatched ≡ intrinsic-free scalar twin; thread counts 1 / 4 / 8 agree (n12 prefix) and
1 / 4 agree (n600). Re-checkable with
`experiments/ddm_wc2c_python_reference_equivalence_test.py`, which re-evaluates the receipts rather
than trusting a prior PASS marker, and reports a missing receipt as a failure rather than a skip.

## 4. The speedup, and what it is NOT

| row | build | threads | n600 seconds | ratio vs Python |
|---|---|---:|---:|---:|
| pure-Python shipping loop | — | 4 | **578.716** | 1.000x |
| split native, dispatched (NEON) | pre-review `5d74dfa1` | 4 | 315.614 | **1.834x** |
| split native, dispatched (NEON) | **final `93ab636d`** | 4 | 326.160 | **1.774x** |
| split native, scalar twin | final `97c25602` | 4 | 324.779 | 1.782x |

All `[macOS-CPU advisory]`. Inside the final dispatched run: native 122.513 s (37.6%), Python side
203.6 s (62.4%). The integer model alone went **369.292 s → 109.372 s = 3.376x**.

**Read the spread honestly.** The two dispatched rows differ by 3.3% on work that is byte-identical;
the only source-level difference between the builds is 190 integer comparisons per frame and a removed
clamp with no effect at `patch_count = 48`. That spread is host noise, and it is **wider than the
distance to the 1.804x bar**. So the local proxy places the split path somewhere around the bar and
cannot say which side. Anyone who quotes 1.834x as "PASS" has picked a run.

The scalar twin at 4 threads (324.779 s) lands within 0.4% of the dispatched NEON build (326.160 s),
which confirms at n600 what the n12 prefix already showed: **the hand-written NEON kernels buy
essentially nothing against clang's auto-vectoriser on this host.** They are retained for the runtime
x86 dispatch and as a floor where auto-vectorisation is unavailable — not because they were measured
faster.

**This is NOT a shipping-axis number and must not be quoted as one.** The T4 baseline evaluates the
sparse model on CUDA with two host↔device round trips per group; my baseline evaluates it on CPU torch.
Those are different regimes, and transferring a ratio across them is the cross-regime constant-transfer
genus this lab has already been bitten by. The honest statement: **the shipping ratio is UNMEASURED and
needs one T4 row.** A directional hypothesis worth testing, not asserting: the T4's 11.77 ms per
iteration is 2.3x my local 5.08 ms for the same work, consistent with the charter's latency-bound
reading, and the split path removes the device round trips entirely because it never touches the GPU.

Against the charter's derived bars — **WARN ≥ 1.096x, PASS ≥ 1.804x** — the local proxy clears WARN
comfortably and sits at the PASS boundary inside its own noise. The honest verdict is: **WARN is
secured, PASS is unresolved, and one T4 row resolves it.**

## 4a. Compile-at-decode cost — MEASURED, not waved through

The native library is compiled inside the 30-minute wall, so its build time is a budget item and is
priced rather than assumed. MEASURED, three repeats, Apple M5 Max / clang 21, `-O3 -mcpu=native`:

| translation unit | seconds |
|---|---:|
| `f26_hpac_native.c` (added by this path) | **0.203 / 0.204 / 0.207** |
| `rc64_backend.c` (already in the prelude) | 0.058 / 0.059 |

**~0.2 s added** against a token stage measured in hundreds of seconds. PROJECTION: a contest vCPU is
slower, so call it 1-2 s there; still trivially favourable, and still priced rather than waved through.

The change is plausibly net-NEGATIVE on prelude time: the removed OpenMP branch shelled out to
`brew --prefix libomp` on Darwin, which costs more than the compile it was guarding — and could fail
outright, which is the real reason to be rid of it.

## 5. Headroom, priced

The corrector is now **62.8%** of the split run. Porting it is the only remaining large lever:

| scenario | token seconds | ratio vs Python |
|---|---:|---:|
| today (integer model lowered) | 326.2 | 1.774x |
| + corrector at 5x | ~157 | ~3.7x |
| + corrector at 10x | ~137 | ~4.2x |

A C port of the corrector **is** provably exact if written faithfully: numpy's element-wise float64 ops
are IEEE per-element, the only order-sensitive steps are reductions, and the Python already fixes those
orders by hand (`free_corrector.py:266-279` refuses `sum(axis=1)`; `_PHAT_SCALE` makes the expectation
accumulator fixed-point so `np.add.at` cannot depend on order). The cost is 2,121 lines across four
modules under a hazard whose measured failure is catastrophic, and it should be argued on its own
measured merits rather than inherited from this arm's momentum.

## 6. Landed

| artifact | what it is |
|---|---|
| `runtime-rs/native/f26-hpac/f26_hpac_native.c` | split surface, pthread pool, runtime ISA dispatch |
| `runtime-rs/native/f26-hpac/{README,binary_source_audit,embedded_constants_audit,rebuild_instructions}` | audit bundle, updated to the post-review build |
| `experiments/ddm_wc2c_token_stage_profile.py` | the 14-step profile that chose the design |
| `experiments/ddm_wc2c_native_split_decode.py` | identity driver, full-field and prefix |
| `experiments/ddm_wc2c_split_token_decoder.py` | the shipping split decoder (staged as `runtime/f26_split_token_decoder.py`) |
| `experiments/ddm_wc2c_python_reference_equivalence_test.py` | re-runnable four-check equivalence test |
| `experiments/ddm_wc2c_stage_native_split_runtime.py` | stager: lifts the refusal by recorded, asserted rewrites |

Retained under `/Volumes/APDataStore/pact/ddm_wc2/retained/` (payload kept, not just its length):
both n600 token fields (117,964,800 B each) with sha256, every receipt JSON, and the profile.

**Two defects I found in my own C during review, and fixed:**
- Cached group geometry was never invalidated. A second model with a larger `target_count` would size
  `decoded_workspace` from the stale maximum and let `group_commit` write past its end. Unreachable in
  the one-model-per-decoder flow — which is exactly why it would have survived review and detonated
  later. Now recomputed per call; the scan is 190 integer comparisons.
- The worker pool let surplus workers from a short job increment the *next* job's completion counter,
  so a later `parallel_for` could return while its workers were still writing. Unreachable only because
  `patch_count` (48) exceeds the pool width. Now every worker always participates and short jobs get
  empty ranges.

## 7. Owed — named, not buried

1. **A T4 row for the split path.** The only number that decides the CI wall. Owned by whichever arm
   holds the fire path; this arm does not touch `tools/fire_*`.
2. **The staged tree is not submittable yet.** Staging changes the runtime-tree SHA (three validators
   enforce it) and the token-decoder fingerprint. A staged tree owes a fresh full-field identity run
   before any exact eval.
3. **Per-frame resume does not exist and is not claimed.** `runtime/entropy/rc64_backend.c` exports no
   state save/restore, so the coder cannot resume mid-stream. Resume granularity is the whole token
   stage — the same granularity the Python path already had. Adding an RC64 state pair plus a proof
   that adding it changed no byte is a separate landing.
4. **x86 is UNVERIFIED.** The AVX2 kernels and the cpuid dispatch are compiled and reviewed but never
   executed: no x86 host was reachable. The scalar row is the fail-closed floor.
5. **The custody problem in §HEADLINE 5 is a submission risk, not a tidiness one.** 24 unversioned
   files on one 93%-full external volume is the whole decoder.
