# ddm_wc2 — decode/eval wall-clock pass: the shipping wall is a LIVE RISK, and the target is the token decode

Date: 2026-08-20 · Owner: ddm_wc2 (Opus arm) · Status: **HOLDING — $0 phase complete; heavy
profiling gated behind MAIN's GO (sub-0.15 T4 confirmation)**

Operator directive (2026-08-20): *"wall pass optimization pass against everything done in upstream
evaluate.py, the inflate.py and decoder and all, after profiling hotspots. Can parallelize and chunk
and multithread and concurrency and embarrassingly parallelize and lower into rust as necessary to
fit within final 30 minute evaluate.py on t4 or contest cpu hardware budgets."*

Operator priority correction (2026-08-20, binding): *"As long as within 30 minutes we are good, and
never want to sacrifice score for wall clock. Wall clock is secondary objective but must also be
fully optimized."*

---

## HEADLINE (all $0, from existing retained receipts — no wc2 compute was run)

1. **The shipping T4 stage split was already in the br1 receipt and nobody had read it.** Of br1's
   1,242.916 s decode, **token decode is 1,186.930 s = 95.50%**. Render is 41.926 s = 3.37%.
2. **wc1 handed wc2 the wrong target.** wc1's verdict says *"the remaining wall is the RENDER"* —
   true on its M5-CPU advisory axis, **false on the shipping axis**, where the render is already
   GPU-batched at 3.4% and the token decode is everything.
3. **The 4.4x token accelerator we already built is REFUSED by every current candidate.**
   `f26_inflate.py:435-441` hard-raises on `native-hpac` because the ddm_rr2 free probability
   corrector was wired into the Python decoder only. The C port was never updated.
4. **That refusal is the plausible cause of a 3.42x decode regression inside the shipping lineage:**
   hv1 T4 inflate **364.76 s** -> br1 T4 inflate **1,246.93 s** on the same axis. (DERIVED — the
   causal attribution is not yet measured.)
5. **The T4 axis is on a knife edge and the CPU axis is projected to FAIL.** br1's 1,246.93 s sits
   inside ua2's CUDA residual band `[822, 1302] s` only at its *best* end. The CPU axis projects to
   1,414–1,913 s against a residual of `[1044, 1332] s` — over budget in every corner.
6. **Our gating harness cannot detect a CI-wall failure.** `contest_auth_eval.py` allows inflate
   alone 1800 s (`--inflate-timeout` default) and never sums setup + inflate + evaluate against the
   job wall. A candidate can pass our Modal gate and still time out in the real CI. **There is a
   measured precedent**: lc2/PR130 hit 1,958 s and returned rc=1 on the *faster* 8-core box.

**Consequence for the sub-0.15 submission:** the score work is not finished when the score lands.
The jg5 candidate has **no measured decode wall clock on either axis**, and its lineage is the one
carrying the Python-only corrector. Harvesting `stage_seconds.token_decode_or_checkpoint_load` from
the T4 row firing now is the single highest-value next action, and it is free.

---

## 0. Scope of this memo

This is the **$0 phase**: budget semantics read at source, prior receipts mined, hotspots MEASURED
FROM EXISTING RECEIPTS where they existed and ranked as hypotheses where they did not, probes
pre-registered. **No profiling decode, no optimization build, no scorer pass was run by this arm.**
Every second below is either read from a retained receipt (with its axis label) or an explicitly
labelled DERIVED projection.

---

## 1. The budget, read at source (m44: never recall from working memory)

| fact | source | exact text / value |
|---|---|---|
| time limit | `upstream/README.md:114` | "The official evaluation has a time limit of 30 minutes." |
| hardware fork | `upstream/README.md:114` | "If your **inflation script** requires a GPU, it will run on a T4 GPU instance (RAM: 26GB, VRAM: 16GB), if it doesn't it will run on a CPU instance (CPU: 4, RAM: 16GB)." |
| timeout scope | `upstream/.github/workflows/eval.yml:30` | `timeout-minutes: 30`, a sibling of `runs-on:`/`steps:` under `jobs.test` — it bounds the **whole job** |
| device fork | `upstream/.github/workflows/eval.yml:32-33` | `UV_GROUP`/`EVAL_DEVICE` derived from the `runner` input (`cu128`/`cuda` vs `cpu`/`cpu`) |

Steps inside the single 30-minute budget (`eval.yml`): `checkout` :36 -> `check_name` :41 ->
`check_gpu` :50 -> `download archive.zip` :56 -> `install_lfs` :62 -> `pull_lfs` :69 ->
`install_uv` :73 -> `install_deps` (`uv sync --group "$UV_GROUP"`) :79 -> `install_ffmpeg` :83 ->
`evaluate` :87 -> `upload-artifact` :94.

Inside `evaluate.sh`: `unzip` :44 -> **`bash inflate.sh` :47 (OURS — the only movable term)** ->
missing-`.raw` assertion :50-62 -> `python evaluate.py` :69 (upstream, READ-ONLY).

### 1.1 The budget identity

```
1800 s  >=  T_ci_setup  +  T_unzip  +  T_inflate(OURS)  +  T_assert  +  T_evaluate(UPSTREAM)  +  T_upload

=>  T_inflate_ceiling  =  1800 s  −  T_ci_setup  −  T_evaluate  −  (small fixed terms)
```

ua2 (`.omx/research/ddm_ua2_upstream_defenses_and_budget_surface_20260731.md:189`) already named
this the compliance predicate and derived the residual:

> `t_inflate <= T_residual(axis, cache_state)`, with **`T_residual(CPU) ∈ [17.4, 22.2] min`** and
> **`T_residual(CUDA) ∈ [13.7, 21.7] min` — not 30 on either axis.**

= **CPU `[1044, 1332] s`**, **CUDA `[822, 1302] s`**. ua2's per-step *seconds* are ESTIMATED; its
per-step *payloads* are measured (checkout 31.6 MB; `git lfs pull` 132,856,531 B; `uv sync --group
cpu` ~78 MB; **`uv sync --group cu128` 3,190,398,780 B = 3.19 GB**). The 40x install asymmetry,
mitigated only by `enable-cache: true`, is what separates the best and worst residual ends.

### 1.2 P0 RESOLVED — the "2.17x headroom" is doubly optimistic

The inherited claim was *"contest-CPU decode 831.5 s, 2.17x headroom"*. Both legs are now audited:

| defect | receipt |
|---|---|
| **Wrong denominator.** `831.534525545 -> 2.1647x` divides by the **full 1800 s**, pricing `T_ci_setup` and `T_evaluate` at zero. Against ua2's CPU residual `[1044, 1332] s` the honest figure is **1.26x–1.60x**. | `.omx/research/ddm_pq1_submission_packet_prep_20260815.md:70` |
| **Wrong hardware.** The 831.5 s row ran on Modal `cpu=8.0, memory=16*1024` (`experiments/modal_auth_eval_cpu.py:1080-1084`) — **8 vCPU**. The contest CPU box is **4** (`README.md:114`). ua2's 8->4 band is 1.7–2.3x. | `.omx/research/ddm_lc2_adjudication_and_cpu_verdict_20260810.md:52-56` |
| **Wrong object.** 831.5 s is **MC36** (`f0ba4bb4…`, 186,269 B), not any shipped candidate. Flagged twice before. | `ddm_rr1_…:243-244`, `ddm_hm1_…:199-211` |

This is the pc2 genus — *the floor you divide by decides the answer* — with a second factor stacked
on it. **Corrected CPU projection: 831.5 x [1.7, 2.3] = 1,414–1,913 s against a residual of
1,044–1,332 s. Over budget in every corner**, for a candidate lighter than the one we intend to
ship. (PROJECTION: the 8->4 factor is ua2's DERIVED band, never measured.)

---

## 2. MEASURED — the shipping stage split (extracted at $0 from the br1 T4 receipt)

Source: `/Volumes/APDataStore/pact/ddm_br1/t4_row_r1/MODAL_REMOTE_RESULT.json`,
`artifacts.contest_auth_eval.stdout.log`, the decoder's own `stage_seconds` (emitted by
`ddm_f26p_f26_inflate_cpu.py:692-699`). Axis **[contest-CUDA T4]**, n600, archive `44e9e650…`,
176,429 B, call `fc-01M0DQECXABB3PBMS4REVT5P76`.

| stage | seconds | share |
|---|---:|---:|
| `archive_setup` | 0.274 | 0.02% |
| **`token_decode_or_checkpoint_load`** | **1186.930** | **95.50%** |
| `neural_render_and_resize` | 41.926 | 3.37% |
| `render_checkpoint_copy` | 0.000 | 0.00% |
| `frame0_selector_and_io` | 3.418 | 0.28% |
| final 3.66 GB `_sha256_file` (residual) | 10.368 | 0.83% |
| **`total_including_raw_sha256`** | **1242.916** | 100% |
| shell prelude (rc64 `.so` compile, brotli gate) — `inflate_elapsed − total` | 4.012 | — |
| **`inflate_elapsed_seconds`** | **1246.928** | — |
| `evaluate_elapsed_seconds` (upstream, `--device cuda`) | 43.181 | — |

Corroborating progress lines in the same log: `rendered masters 600/600 in 32.8s`,
`rendered carriers 600/600 in 41.8s`. `token_cache: {"status": "DISABLED"}` — confirming §4.2.

**Retained** (ALWAYS KEEP THE PAYLOAD — this split had never been read and must not be re-derived):
`/Volumes/APDataStore/pact/ddm_wc2/receipts/br1_t4_shipping_stage_split.json`, 1,723 B, sha256
`2eac38af90b0a67304ee400f5842273e2852f8062f934ba83d1f9a4bef5e643a`. It binds the source receipt sha,
the Modal call id, archive/raw shas, the six `stage_seconds` keys, and the three token-decoder
identity digests used as the §3.1 / §6-item-2 gate:
`decoded_token_sha256 9ba2e52b…`, `corrected_quantized_logit_sha256 562ac652…`,
`corrected_cdf_input_sha256 dd48843b…`.

**Three inherited beliefs die here:**

- **"The remaining wall is the render."** (wc1 verdict §4.) On the shipping axis the render is
  **3.37%**. wc1's claim is correct only for its M5-CPU advisory instrument, where the render cannot
  use CUDA. An axis-scoped finding was carried forward as if it were axis-free.
- **"The 88 s checkpoint copy might leak into shipping."** Measured `render_checkpoint_copy: 0.0`.
  Closed, no action.
- **"Parallelize/chunk the decode."** The token loop is **strictly sequential** and cannot be
  parallelized — see §3.2. The operator's parallelism instruction lands on the render, which is
  already 3.4%.

### 2.1 The lineage regression

| row | axis | archive | inflate s | receipt |
|---|---|---|---:|---|
| hv1 | contest-CUDA T4 | — | **364.762** | `ddm_pq1_…:57-58` |
| **br1** | contest-CUDA T4 | 176,429 B | **1246.928** | `t4_row_r1/MODAL_REMOTE_RESULT.json` |
| jg5 (sub-0.15) | — | 180,625 B | **UNMEASURED** | — |

**3.42x regression within the shipping lineage.** The mechanism is *hypothesised*, not measured:
ddm_rr2 added the free probability corrector to the Python token decoder, which both (a) adds
per-group numpy work inside a 114,000-iteration loop and (b) disqualified the native C path. hv1
predates it. **DERIVED — attribution requires the P1 probe.**

---

## 3. Why the token decode costs 1,187 s

Hot loop: `runtime/residual_archive.py:600-649`. Groups per frame =
`(1 + HPAC_DELTA) * HPAC_PATCH - HPAC_DELTA = 3*64 - 2 = 190` (`cpr1/inflate.py:281`).
**600 frames x 190 groups = 114,000 Python iterations**; `1186.930 / 114000 = 10.41 ms per
iteration` — enormous for the work involved.

Per iteration, read at source:

| # | op | note |
|---|---|---|
| 1 | `sparse.selected_logits(current, context, group)` | torch op **on `device`** — CUDA on the T4 box |
| 2 | `selected.cpu().numpy()` | **GPU->CPU sync, every iteration** |
| 3-5 | `argmax`, feature index, `corrected = base_logits + table.values[feature]` | numpy |
| 6 | `corrected_digest.update(...)` | **sha256 over the full float32 group array** |
| 7 | `_probability_table(corrected, HPAC_LOGIT_PRECISION)` | numpy |
| 8 | `cdf_digest.update(...)` | **second sha256 over a full float32 array** |
| 9 | `corrector.group_state(...)` | numpy FreeCorrector (`free_corrector.py:241-301`, per-class loop :276) |
| 10 | `decoder.decode(corrector.coding_row(state))` | ctypes -> `rc64_backend.so` |
| 11 | `corrector.observe(state, symbols)` | `np.add.at` scatter |
| 12 | `current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)` | **CPU->GPU transfer, every iteration** |

### 3.1 The leading hypothesis: this is GPU-latency-bound, not compute-bound

Steps 1/2/12 form **two host<->device round trips plus one tiny CUDA kernel launch per iteration,
114,000 times**. The per-group work is small; on a T4 that pattern is dominated by launch and sync
latency, not arithmetic. If so, **running the token model on CPU while leaving the render on CUDA
would be faster on the very same T4 box** — a counter-intuitive but cheap and testable claim.

**The identity gate for this already exists and is free.** The decoder emits
`corrected_cdf_input_sha256`, `corrected_quantized_logit_sha256`, and `decoded_token_sha256`. A
device change is admissible **iff** the quantized-logit and decoded-token digests are unchanged;
the float `corrected_digest` will likely differ under any fp reordering, which is exactly why the
quantized digest exists as a separate handle. Under score primacy: digests match -> adopt; digests
differ -> **refuse, regardless of speed.**

### 3.2 Honest negative: the token decode cannot be parallelized

RC64 is a single bitstream, and the model is autoregressive over frames (`previous = current`, :649)
and over groups within a frame (`current.reshape(-1)[device_positions] = …`, :644). Chunking,
multithreading, or embarrassing parallelism are all **structurally unavailable** here without
changing the codec — and changing the codec changes the bytes, which score primacy forbids. The
only routes into this stage are (a) lower it (native C), (b) move it off the latency-bound device,
(c) delete non-load-bearing work inside the loop (the two sha256 digests).

---

## 4. The accelerator we already own, and why it is disconnected

### 4.1 `native-hpac` is hard-refused by every current candidate

`/Volumes/APDataStore/pact/ddm_jg5/candidate_runtime_jg5/runtime/f26_inflate.py:435-441` (identical
at `:436` in `ddm_br1/candidate_runtime_r1` and `ddm_jg4/candidate_runtime_complete`):

```python
if token_decoder != "python":
    raise InflationError(
        "this generation wires the ddm_rr2 free probability corrector into the "
        "python token decoder only; the native-hpac path is unpatched and would "
        "decode a different field, so it is refused rather than trusted"
    )
```

The refusal is **correct** — an unported corrector would decode a different field, i.e. change the
score. But it means the asset is stranded:

- `runtime-rs/native/f26-hpac/f26_hpac_native.c` — 40,911 B fused sparse-int-HPAC + probability +
  RC64 decoder, OpenMP + NEON/AVX2, with a `-DF26_FORCE_SCALAR=1` twin and a full audit bundle
  (`binary_source_audit.md`, `embedded_constants_audit.txt`, `archive_payload_manifest.json`).
- wc1 measured it byte-exact against the scalar twin (raw sha `e5539653…`) at **4.38x** on the token
  stage (642.14 s -> 146.5 s, `[macOS-CPU advisory]`).

**Porting the ddm_rr2 `FreeCorrector` into `f26_hpac_native.c` is the single highest-value shipping
optimization available**, and it is a PORTING ITEM, not a wall (runtime-lift grant). Its identity
gate is pre-built: the scalar-twin + golden-vector pattern wc1 already used, plus the three digests
in §3.1. DERIVED projection, if the 4.38x transfers: token `1186.93 -> ~271 s`, total inflate
`~331 s` — which lands next to hv1's measured 364.76 s, a consistency check that supports the
attribution in §2.1. **The 4.38x is from a pre-RR2 generation on a different axis and must not be
quoted as a shipping number until measured.**

### 4.2 The token cache is LOCAL-ONLY and must never be priced into the shipping budget

wc1's headline 370.4 s depends on a warm content-addressed token cache. The CI job starts from
`checkout`; there is no warm cache, and pre-populating one outside `archive.zip` would be
video-derived payload past the rate term (rule-118). The br1 receipt confirms
`token_cache: {"status": "DISABLED"}` on the shipping path.

- **shipping-relevant wc1 figure = 516.8 s** (cold) — never 370.4 s.
- **local-advisory figure = 370.4 s** (warm).

### 4.3 Correction to wc1's post-cache split (arithmetic)

wc1 verdict §4 reads *"neural render 318 s (86% of the cached total) + checkpoint copy 88 s +
frame0/selector I/O 45 s"* — those sum to 451 s against a stated cached total of 370.4 s. Reconciling
against the two timed rows: 516.8 ≈ 146.5 + 318 + 45 (+7 setup) and 370.4 ≈ 5.9 + 318 + 45. So the
318/45 terms are cache-invariant and belong to both rows, and the **88 s checkpoint copy is not
inside the 370.4 s total**. Recorded because an inherited-but-unreconciled split is exactly the
shape that later gets quoted as measured.

---

## 5. Budget verdict, per axis

### 5.1 contest-CUDA T4 — KNIFE EDGE

| term | value | source |
|---|---:|---|
| job wall | 1800 s | `eval.yml:30` |
| `T_residual(CUDA)` for inflate | **[822, 1302] s** | ua2:189 (PROJECTION; step seconds estimated) |
| br1 measured inflate | **1246.928 s** | br1 T4 receipt |
| margin at best-case residual | **+55.07 s (1.04x)** | DERIVED |
| margin at worst-case residual | **−424.93 s (0.66x) — FAIL** | DERIVED |

The gap between the two ends is dominated by a `uv sync --group cu128` cache miss (3.19 GB inside
the wall). **br1 fits only if the runner's uv cache is warm.** jg5 is unmeasured and its lineage is
heavier.

### 5.2 contest-CPU — PROJECTED FAILURE

| term | value | source |
|---|---:|---|
| `T_residual(CPU)` for inflate | **[1044, 1332] s** | ua2:189 (PROJECTION) |
| MC36 measured inflate, **8 vCPU** | 831.535 s | `experiments/results/ddm_f26r_mc36_contest_cpu_20260814/` |
| MC36 projected on contest **4 vCPU** | **1,414–1,913 s** | DERIVED, ua2 1.7–2.3x band |
| verdict | **over budget in all four corners** | DERIVED |
| measured precedent | lc2/PR130 **1,958 s -> rc=1 timeout** on the *faster* 8-core box | `ddm_lc2_…:44-47` |

MC36 is a **pre-RR2** candidate. A jg5-lineage candidate carrying the Python-only corrector would
be substantially worse. **Recommendation: the submission should target the T4 runner.**

### 5.3 The hardware fork is a lever we must exercise deliberately

`README.md:114` routes by whether **our inflation script requires a GPU**. Today
`inflate.py:54-64` reads `device = "cuda" if torch.cuda.is_available() else "cpu"` — it *uses* CUDA
when present but does not *require* it. A maintainer reading that could route us to the 4-core CPU
box, where §5.2 projects a timeout. **If we intend to be scored on T4, the packet must make its GPU
requirement explicit.** This is a submission-packet decision for MAIN/operator with a score
consequence (CPU and CUDA are separate evidence spaces), not a wc2 call — wc2 supplies the
arithmetic.

### 5.4 The harness cannot see this failure mode

`contest_auth_eval.py:2024` defaults `--inflate-timeout` to 1800 s — the *entire* CI job budget —
and then runs `evaluate.py` on top with its own 5400 s allowance. No code path sums
`T_ci_setup + inflate + evaluate` against 1800 s. **Our gating instrument has no budget gate.** A
candidate can pass our Modal row and time out in the real CI; lc2 is the measured precedent that
this is not theoretical. Cure (structural, not procedural): a budget predicate in the harness that
computes the residual per axis and emits a WARN/REFUSE row. Proposed as a wc2 landing.

---

## 6. Ranked head (score-primacy weighted)

Ranking follows the operator's correction: bit-identity is an **admissibility constraint**, not just
proof discipline; the shipping wall is a **constraint to satisfy with margin**; **local advisory
seconds compound** across every future candidate gate.

| # | item | axis | measured basis | prize | byte risk |
|---|---|---|---|---|---|
| **1** | **Port the ddm_rr2 `FreeCorrector` into `f26_hpac_native.c`, re-enable `native-hpac`** | shipping, both | token = 95.50% of decode | up to ~4.4x on 95.5% of the wall | **gated** — the refusal exists because identity was never proven; scalar-twin + 3 digests are the gate |
| **2** | **Move the token model off CUDA on the T4 box (§3.1)** | shipping T4 | 114,000 x 2 host<->device round trips | unknown, possibly large | **gated** on `corrected_quantized_logit_sha256` + `decoded_token_sha256` |
| **3** | **Budget predicate in `contest_auth_eval.py` (§5.4)** | both, guard | lc2 rc=1 precedent | prevents a silent CI timeout | none (measurement only) |
| **4** | Delete/short-circuit the two in-loop sha256 digests if not load-bearing | both | 2 x 114,000 float32 digests | unmeasured | none if audit-only |
| **5** | Render batching on CPU (`semantic_batch/pose_batch = 1` when not CUDA, `cpr1/inflate.py:312,335`) | **CPU axis only** | 1,200 single-item forwards | CPU-axis only; 3.4% on T4 | **gated** — batch shape is part of the instrument (et4) |
| **6** | Final 3.66 GB `_sha256_file` | shipping | 10.368 s (0.83%) | small | none |
| **7** | Local advisory: parallel render (`F26_ADVISORY_RENDER_WORKERS`, already built, default-off) | local only | render = 86% of the *advisory* wall | compounding on every future gate | already identity-receipted per chunk |

**Deliberately NOT pursued, with reasons:**

- **Parallelising the token decode** — structurally impossible (§3.2). The operator's
  parallelise/chunk/embarrassingly-parallel instruction has no purchase on 95.5% of the shipping
  wall; it applies to the render, which is 3.4% on T4 and already has a built parallel path locally.
- **Rust lowering of the render** — dense conv/GEMM already served by cuDNN/oneDNN; hand-lowering
  will not beat it, and on T4 the whole stage is 41.9 s. The native grant is not a reason to spend
  effort where the prize is 3.4%. The C lowering that matters is item 1, and it already exists.
- **Anything touching `upstream/`** — READ-ONLY. `T_evaluate` is measured (43.181 s CUDA / 113.899 s
  in-harness CPU), never patched. Note upstream runs the scorer **twice** per batch
  (`modules.py:158-159`, GT + ours), so half its cost is redundant across submissions — we cannot
  exploit it, and it is recorded only to explain the term's size.
- **Anything that changes decoded bytes** — inadmissible under score primacy, at any speed.

---

## 7. Pre-registered probe plan (fires only on GO)

- **P0 — denominator audit. DONE, $0.** §1.2 + §5. Both defects found; the CPU axis is projected to
  fail and the T4 axis is a knife edge.
- **P0b — harvest jg5's `stage_seconds` from the T4 row firing now. $0, no new compute.** Extract
  `token_decode_or_checkpoint_load` and `inflate_elapsed_seconds` exactly as §2 did for br1. This
  tells us whether the sub-0.15 candidate is inside or outside the wall. **Requested of MAIN.**
- **P1 — per-group instrumentation of `residual_archive.py:600-649`.** The 114,000-iteration loop is
  currently a single opaque number; the 6 outer `stage_seconds` keys already exist, but nothing
  splits sparse-logits vs `.cpu()` sync vs digests vs FreeCorrector vs rc64 ctypes. This is the only
  instrumentation that must be built. Advisory-only; never in the shipped packet.
- **P2 — device probe for item 2**, validated on the existing digests before any full decode.
- **P3 — the `FreeCorrector` C port (item 1)**, gated by the scalar-twin double-build byte-equality
  driver that already exists (`experiments/ddm_f26r_hpac_hot_stage_final_rung.py`).
- **P4 — budget predicate landing (item 3).** Spec in §7.1.
- **P5 — CPU-axis render batching (item 5)** and the digest/sha256 trims (items 4, 6).

**Identity discipline, every probe:** `sha256(inflated/0.raw)` before == after on the same archive;
for native paths additionally the scalar-twin equality; for eval-side changes, value-identity or a
**new named instrument** with its own calibration receipt against a T4 row — never a silent mutation
of the gating instrument (et4).

### 7.1 P4 spec — the budget predicate, and why its windows must not be bare constants

**Requirement (MAIN, 2026-08-20):** the predicate carries BOTH residual windows —
`CUDA [822, 1302] s` and `CPU [1044, 1332] s` — as named constants with this memo's derivation as
provenance, rather than re-deriving them silently at each call site.

**Refinement (accepted and specified here).** A bare `T_RESIDUAL_CUDA = (822, 1302)` would reproduce
*exactly the defect this arm just caught*. The "2.17x headroom" number was wrong not because anyone
mis-multiplied, but because the denominator had been baked into a quotable scalar whose inputs were
no longer visible — so nobody could see that it priced `T_ci_setup` and `T_evaluate` at zero, or
that it came off an 8-vCPU box. A hardcoded `822` is the same shape one generation later. This is
the constants-are-poison genus (memory: CONSTANTS→LAWS) and the value-provenance ladder (#351).

So the window lands as the **evaluated output of a recorded derivation**, with the constant cached
for callers but carrying its inputs and grade:

| input | value | grade | source |
|---|---|---|---|
| `JOB_WALL_S` | 1800 | **MEASURED at source** | `eval.yml:30` |
| per-step CI seconds | ua2's step table | **ESTIMATED** | ua2:189 |
| `git lfs pull` payload | 132,856,531 B | **MEASURED** | ua2 |
| `uv sync --group cu128` payload | 3,190,398,780 B | **MEASURED** | ua2 |
| `uv sync --group cpu` payload | ~78 MB | **MEASURED** | ua2 |
| `cache_state` | `warm` \| `cold` | **input axis** | `setup-uv` `enable-cache: true` |
| → `T_residual(axis, cache_state)` | CUDA [822,1302] / CPU [1044,1332] | **PROJECTION** | derived here |

The emitted constant must therefore carry `grade: PROJECTION` and
`provenance: {ua2:189, ddm_wc2 §1.1/§5, receipt br1_t4_shipping_stage_split.json}`. Any consumer
quoting the window as MEASURED is a false-authority claim — the step seconds have never been timed
on a real runner, only the payload sizes have.

**Verdict vocabulary — three-valued, never boolean.** The window has two ends *because* the answer
depends on the uv cache state, which is the whole finding. Collapsing that to `fits: true/false`
would erase the dependency the predicate exists to surface (m52: a bool flag is a UI over a
continuum). So:

- **PASS** — `t_inflate + t_evaluate` inside the window's *narrow* (worst-case) end → fits even on a
  cold cache.
- **WARN** — inside the wide end but outside the narrow → fits only if the runner's uv cache is
  warm. **br1 sits here today** (1,246.93 s vs `[822, 1302]`).
- **REFUSE** — outside both → projected timeout. **MC36-on-4-vCPU sits here** (1,414–1,913 s vs
  `[1044, 1332]`).

The predicate takes `axis`, the measured `inflate_elapsed_seconds`, and `evaluate_elapsed_seconds`
from the harness receipt it already writes, so it adds a verdict and no new measurement. It is a
guard on the gating instrument, not a change to the shipped packet.

---

## 8. Ops findings from the $0 pass

1. **`/Volumes/VertigoDataTier` is full** — `892 Mi` free of 1.8 Ti (100% capacity). Any decode
   retaining a 3.66 GB raw there will fail. wc2's workspace is `/Volumes/APDataStore/pact/ddm_wc2/`
   (137 Gi free). Live blocker for **any** arm defaulting its retention to Vertigo.
2. **Host memory pressure `warn`** (~40 GiB available vs a 64 GiB derived floor, swap 36.4 GiB) in
   the last sampled `memory_blackbox` rows — the same condition that `SIGSTOP`-blocked wc1's first
   n600 attempt. P1 must launch governed and check pressure at fire time.
3. **jg5 packet carries AppleDouble `._*` files**; `ddm_wc1_advisory_decode_wallclock.py:456` sweeps
   them before launch. Any packet shipped from macOS needs that sweep or the archive gains junk.
4. **No `wc2` row in `.omx/state/canonical_task_status.jsonl`** (565 rows). To register at landing.
5. **Doctrine drift confirmed by ua2:** `CLAUDE.md:302` and `CLAUDE.md:929` say "the only constraint
   is the 30-min decode budget" — CONFLATED; the 30 min is the whole job. `upstream/README.md:114`
   and `CLAUDE.md:294` are correct. ~20 research memos carry the wrong framing. Correcting the two
   CLAUDE.md lines is a candidate landing (operator call — CLAUDE.md is operator-owned).

---

## 9. Status / NEXT

- **Disposition: HOLDING.** $0 phase complete. P0 resolved and it changed the arm's conclusion:
  this is not a polish exercise — **the shipping wall is a live risk to the sub-0.15 submission on
  both axes**, and the dominant term is the token decode, not the render.
- **No wc2 measurement, build, scorer pass, or pointer movement.** Wall clock is not a score axis
  and this arm will not move the pointer.

### Adjudication + routing (MAIN, 2026-08-20)

| item | disposition |
|---|---|
| axis-scoped-finding catch (wc1 render-86% vs shipping 95.5% token decode); stranded `native-hpac` attribution; corrected CPU budget arithmetic | **BANKED in #835** |
| GPU-requirement packet decision (§5.3) | **routed to operator at the #1111 boundary** |
| **P0b** — harvest `stage_seconds.token_decode_or_checkpoint_load` + `inflate_elapsed_seconds` from the jg5 T4 row, call **`fc-01M0EZ3DR3HVB8HBKWEG2P12CT`** (in flight) | **OWNED BY MAIN**; delivered to wc2 with the GO |
| ranked head (1) rr2 `FreeCorrector` C port behind the scalar-twin driver · (2) token model off CUDA on T4 with sha-provable identity · (3) harness budget predicate | **stands as filed** |
| addition to (3): both residual windows as named constants with wc2's derivation as provenance | **specified in §7.1**, with the constants-are-poison refinement and a three-valued verdict |

### Scope extension (operator, 2026-08-20): *"Harden and polish and optimize the harness and inflate.py as well."*

Three verbs on two surfaces, same GO. **Surface A** = shipping runtime tree (`inflate.py` 69 LOC,
`runtime/f26_inflate.py` 706, `cpr1/inflate.py` 358 = **1,133 LOC**, plus
`f26_hpac_native.c` 40,911 B). **Surface B** = harness (`contest_auth_eval.py`,
`fire_modal_auth_eval.py`, `fire_local_advisory.py`, poller/closer).

**Favourable asymmetry — Surface A polish is free on the score axis.** The rate term charges
`archive.zip` bytes only (`evaluate.py:63`); `inflate.py`/`inflate.sh`/`runtime/` are unsized
(rule-118 boundary, and the LOC cap was deleted 2026-07-21). So L12 single-LOC-per-LOC
reviewability over 1,133 LOC of judge-facing code costs **zero score**. Its only costs are the
re-pin (C1) and the identity proof.

#### Pre-GO reconnaissance — state of each named item ($0, this session)

| item | surface | measured state | action at GO |
|---|---|---|---|
| `gt_lineage` in advisory JSONs (dg1 cure) | B | **ABSENT — 0 occurrences** across all three harness files | **extend**, not "verify wired" |
| instrument tuple (et4) | B | **THIN** — only `torch_version` (`contest_auth_eval.py:1271`); no threads / batch / weights sha in harness provenance | extend |
| bare-`python` emission (si1) | B | **CLEAN, no action** — `fire_local_advisory.py:67-69` writes an exec-wrapper shim with a `sys.prefix` self-test; that is the compliant form, not the defect |
| budget predicate | B | **ABSENT** — `--inflate-timeout` 1800 + `--evaluate-timeout` 1800 (`:2730-2733`) permits **3,600 s, 2x the job wall** | build per §7.1 |
| runtime sha pins | A→B | **PRESENT AND ENFORCED** — `_validate_expected_runtime_tree` :1554, `_validate_expected_runtime_files` :1566, `inflate_script_sha256` :1627 | see **C1** |
| dep self-install bootstrap (e4/rr3) | A | not exercised this session | bare-venv smoke (r5: prove it, don't assume) |

#### Three collisions to resolve before GO, not during

**C1 — polish invalidates the pins, and the coordinator's ordering pays for it twice.**
Every Surface A edit changes the runtime-tree sha; the three validators above will **REFUSE**, and
stored fire manifests carrying expected shas refuse with them. The filed sequencing —
(1) inflate optimize → (2) harness → (3) polish *both* — touches Surface A at step 1 **and** step 3,
so it costs two re-pin cycles **and two full n600 identity-proof decodes** (each proof is a complete
decode; that is the expensive unit here).
**Proposal: fold Surface A polish into step 1.** One Surface A landing carrying optimize + harden +
polish → one re-pin → one identity proof. Harness polish stays at step 3, where it is cheap and
pin-free. Same work, half the decodes.

**C2 — "HARDEN: hash assertions" collides head-on with ranked-head item 4.**
`residual_archive.py:625-634` already runs two sha256 digests per group; item 4 proposed removing
them, which the hardening verb forbids. **Both are satisfiable**: the cost is not the sha256 (the
total bytes hashed are identical either way) — it is the **114,000 `np.ascontiguousarray(...)
.tobytes()` temporary copies**. Keep the digests (hardening intact, final digest value unchanged by
construction, hence self-verifying), kill the per-group copy. Item 4 is rewritten accordingly.

**C3 — hardening asserts must not enter the hot loop.**
Surface A sits on a knife edge (§5.1) and the loop runs 114,000 times, so *anything* per-group is a
budget item. Rule for the hardening pass: fail-closed at **boundaries** — archive parse, section
lengths, exact-consumption asserts, final byte count, final raw sha, atomic writes, storage
preflight — **never per-group**.

### Released build scope (operator, 2026-08-20): *"Port and lower and optimize all necessary."* + *"Recursive fractal optimization and include OS and arch gating and fully leverage all native features possible."*

Scorer-free build + identity proving is RELEASED. Still held for GO-with-numbers: any local scorer
pass, any advisory decode, any timing measurement needing a quiet machine, and any **MEASURED**
speedup claim. Build receipts may say **BUILT + IDENTITY-PROVEN**; every speed figure stays a
labelled PROJECTION until the GO.

#### Architecture consequence — dispatch is a design constraint, not a wrapper

The arch-gating amendment landed *before* any C was written, which changes the port's shape:
`native-hpac` must be a **runtime-dispatched** family with a fail-closed floor, not a compile-time
build flavour.

| level (fractal) | target | gate |
|---|---|---|
| instruction | SIMD width, integer ops | **cpuid AT RUNTIME** — AVX2 vs AVX-512 probed, never assumed (contest silicon unknown); NEON on arm64 |
| loop | the 114,000-iteration token loop: memory layout, branch structure, kill the per-group `tobytes()` copies (C2) | profile-led |
| function | fuse the corrector **into** the decode inner loop rather than calling across it | the port itself |
| stage | overlap decode / render / hash where dependency-free — note §3.2: the token loop itself is **not** parallelisable | dependency analysis |
| pipeline | setup/download overlap in the harness | Surface B |

**Dispatch ladder, fail-closed at every rung:** `AVX-512 → AVX2 → SSE2/scalar-C → NEON (arm64) →
Python`. Fall back to the proven Python path on **compile failure, cpuid miss, or a decode-start
identity check failure**. *A wrong-fast decode is forbidden; a slow-correct one is merely a WARN on
the budget predicate.*

**Determinism is what makes gating legal.** Same archive → bit-identical `0.raw` on every
OS/arch/dispatch path. The token decode is **integer** arithmetic, so SIMD lanes are naturally exact
— keep it that way: **no FP shortcuts in the coder**. The neural render keeps its existing torch
numeric path unless bitwise equivalence is proven. Cross-path identity =
`decoded_token_sha256` + full `0.raw` sha compared across scalar / SIMD / native on the real br1 +
jg5 bodies, **per path, per platform reachable**. Locally reachable now: arm64/NEON + scalar.
x86_64 AVX2/AVX-512 and the T4 sm_75 path are **UNVERIFIED until the GO** — labelled as such, never
assumed.

#### Compile-at-decode is already proven and already priced

The shipping prelude **already compiles `rc64_backend.so` at decode time** and costs **4.012 s**
(§2, `inflate_elapsed − total_including_raw_sha256`). So compile-at-decode is a demonstrated,
budget-visible mechanism, not a new risk. Adding the HPAC native compile extends that prelude; per
the amendment its cost is counted **inside** the budget arithmetic, including the cold-cache CI
corner. Against a projected ~916 s token-stage win, a few seconds of `cc` is trivially favourable —
but it is priced, not waved through.

**Integration point:** because the fallback is fail-closed and therefore *silent by design*, the
budget predicate (§7.1) must read **which decode path actually ran** from the receipt and grade
accordingly — a Python-fallback decode on the contest box is precisely the WARN/REFUSE case the
predicate exists to surface. `decode_path` joins the et4 instrument tuple.

### PORT DESIGN NOTE — the corrector is FLOAT, and that dictates the SIMD axis

**Correction to a premise in the arch-gating amendment.** The amendment states *"the token decode is
integer arithmetic — SIMD lanes are naturally exact."* True of the RC64 coder; **false of the rr2
corrector**, which is the part being ported. `free_corrector.py` runs **float64** on the decision
path (`:229, 269, 274-275, 283`). Its own docstring pins the discipline:

> *"No `log`, `exp`, `pow`, `**` or table anywhere on the decision path — only `+ - * /` and
> comparisons, all IEEE correctly rounded, so the receiver is bit-identical on every conforming
> platform. This is the `ddm_rr2` refusal class (S = 27.83, a libm ULP desynchronising the decoder)."*

Bit-identity therefore rests on IEEE-correctly-rounded `+ - * /` — **not** on integer exactness. FP
addition is not associative, so **any SIMD reduction that changes summation order breaks the
decoder**, and it breaks it the way rr2 already measured: **S = 27.83**, a catastrophic
desynchronisation, not a rounding wobble. This is precisely how a naive port ships a *wrong-fast*
decoder — the class the amendment forbids.

**The cure is already in the Python and is deliberate** (`free_corrector.py:266-279`):

```python
# Sums are taken lane by lane in a fixed order so the reduction is unambiguous
# rather than dependent on a library's pairwise blocking.
big_w = np.zeros(arg.size, dtype=np.float64)
for lane in range(NUM_CLASSES):
    big_w += weighted[:, lane]
```

The author hit this hazard and fixed the order by hand, refusing `sum(axis=1)` and its pairwise
blocking. Note the sister discipline named in the docstring: `e` was made **fixed point** *"so the
sum is order-independent"* — the same hazard, cured the other way, where changing bytes was allowed.

**Consequence — the port has exactly one safe vectorization axis.**

| axis | safe? | why |
|---|---|---|
| across **positions** (`arg.size`) | **YES — exact** | each of the 5 accumulation steps adds `weighted[pos, lane]` for a block of positions; per-position order is untouched |
| across **lanes** (`NUM_CLASSES=5`, horizontal reduction) | **NO — breaks identity** | reorders the very summation the Python fixes by hand |

So the C port keeps the 5 sequential lane steps and SIMDs each one across positions — which is also
the *wider* axis and therefore the better vectorization anyway. Correctness and speed agree here;
the naive choice (horizontally reducing 5 lanes) would have been both wrong and narrower.

**Binding rule for the port:** float64 throughout the corrector, `-ffp-contract=off`, no
`-ffast-math`, no FMA substitution (FMA changes rounding), no reassociation, no horizontal
reductions on the decision path. The decode-start identity check exists to catch exactly this.

### GO — jg5 P0b harvested. The sub-0.15 pointer body does NOT fit the CI wall.

Pointer moved: **S 0.14839100138338618 @ 180,625 B [contest-CUDA T4, n600]**. Shipping stage split
for that body (`/Volumes/APDataStore/pact/ddm_jg5/t4_row_r1/harvested_artifacts/`):

| stage | jg5 s | share of 1401.58 | br1 s (prior) |
|---|---:|---:|---:|
| **token decode** | **1341.540** | **95.72%** | 1186.930 |
| neural render | 54.536 | 3.89% | 41.926 |
| frame0/selector I/O | 5.145 | 0.37% | 3.418 |
| archive setup | 0.358 | 0.03% | 0.274 |
| non-token remainder (raw sha + prelude) | 18.360 | — | 14.380 |
| **inflate elapsed** | **1419.900** | | 1246.928 |
| evaluate (upstream, CUDA) | 51.400 | | 43.181 |

`decoder_bit_position 910,837`, `token_codec rc64`. Identity anchors for the port proof:
`decoded_token_sha256 cc10a7b0…`, `corrected_quantized_logit_sha256 8269fe1a…`,
`corrected_cdf_input_sha256 370a5e2a…`, `raw_sha256 6bf8acf8…`.

**Three reads, all confirming the $0 analysis:**

1. **RR2 attribution — second point CONFIRMED.** Token decode along the corrector lineage:
   hv1 (pre-RR2, *total* inflate) 364.76 s → br1 1186.93 s → **jg5 1341.54 s**. Monotone, and jg5 is
   1.130x br1. §2.1's DERIVED attribution now has two independent points.
2. **Budget verdict on the NEW POINTER BODY: REFUSE.** 1419.90 s against the CUDA residual
   `[822, 1302] s` — over by **117.9 s even at the widest (warm-cache) end**. br1 was WARN; jg5 is
   worse. **The sub-0.15 archive is not CI-shippable as it stands**, and the corrector port is
   formally the submission critical path, not an optimization.
3. **The axis-scoped split holds on jg5**: render 3.89%. wc1's "render is 86%" remains an
   advisory-axis artifact.

#### DERIVED — how much speedup the port actually needs (a modest target)

Non-token inflate is fixed at `1419.900 − 1341.540 = 78.360 s`. So the token stage must fall to:

| goal | token budget | required speedup |
|---|---:|---:|
| exit REFUSE → **WARN** (≤ 1302 s, warm cache) | 1223.640 s | **≥ 1.096x** |
| reach **PASS** (≤ 822 s, fits cold cache) | 743.640 s | **≥ 1.804x** |
| if wc1's 4.38x transfers | 306.3 s | → inflate **384.7 s**, 2.14x–3.38x margin |

**The port does not need to be heroic — 1.81x makes the sub-0.15 archive shippable on a cold
cache.** wc1 measured 4.38x on the pre-RR2 token stage; even a large haircut for the corrector work
clears the bar. All four figures are PROJECTIONS until the port is measured.

#### SELF-CORRECTION — my flat "REFUSE" overstates it. The honest verdict is a range, and the action is unchanged.

Surface B's landing surfaced an algebra point I had not chased, and it lands on my own number.
ua2's `T_residual` **already nets out an ESTIMATED `evaluate.py`** (ua2 step 12: CPU 300–400 s
DERIVED from 176.3 s measured @ 8-core; CUDA 120–180 s). We now hold a **MEASURED** evaluate for the
jg5 body: **51.4 s** on T4. Substituting a measurement for an estimate is strictly better
information, and it *widens* the inflate residual:

```
T_residual_corrected(CUDA) = [822, 1302] + (evaluate_est − evaluate_measured)
                           = [822, 1302] + (120…180 − 51.4)
                           = [890.6, 1430.6] s
```

jg5 inflate **1419.900 s** then sits **inside the wide end by 10.7 s**.

| treatment | jg5 verdict |
|---|---|
| inflate alone vs ua2's published residual (**what §GO claimed**) | REFUSE, over by 117.9 s |
| Surface B's conservative rule (charges inflate+evaluate, double-counting evaluate) | REFUSE |
| **measured-evaluate substitution (most accurate)** | **WARN — fits by 10.7 s, warm cache only** |

**So the strongest defensible claim is WARN, not REFUSE.** I stated REFUSE flatly; that overstated
the danger. This is the same genus I caught in the "2.17x headroom" figure — *the floor you divide
by decides the answer* — now applied to my own arithmetic. NO-FAKE cuts both ways: overstating a
risk is as much a false claim as understating one.

**The action does not change.** 10.7 s of margin, available only on a warm uv cache, on a path whose
fallback is a silent Python decode, is not shippable margin — it is noise against a 1800 s wall. The
corrector port remains the submission critical path, and the derived targets (≥1.096x / ≥1.804x)
are unaffected because they were computed against the published window, which is the conservative
one. br1 is WARN under every treatment.

**Two follow-ons this opens** (neither chased yet): (a) the same substitution is owed on the CPU
axis; (b) three nominally-8-core `evaluate.py` measurements disagree — **176.3 s** and **174.9 s**
corroborate each other, but ft1's MC36 row measured **113.9 s**. That outlier is unexplained and
should not be averaged into anything until it is.

### PORT RESULT (ddm_wc2c, head `1da04b7014`, ledger `ddm_wc2c_native_split_identity_and_speedup_20260820.md`)

#### P0 CUSTODY — the submission's decoder is not in version control

**Verified independently by me** (`git ls-files`): `f26_inflate.py` → **0** repo paths;
`hpac_inference.py` → **0**; `cpr1/` → **0** files. The two apparent hits (`free_corrector.py`,
`residual_archive.py`) are glob collisions with `rr4_free_corrector.py`. The arm's count: **24 of 34
files** in the jg5 candidate tree have no repo source — including `inflate.py`, `inflate.sh`,
`f26_inflate.py`, `residual_archive.py`, `free_corrector.py`, and all of `cpr1/`.

**The decoder for the first sub-0.15 archive exists only on one external SSD at 93% capacity, with
no version control and no second copy.** This outranks every wall-clock item in this memo: a disk
failure loses the submission outright, and no identity receipt can be reproduced without the tree
that produced it. Escalated to MAIN as P0. It is also *why* the refusal lift had to be a stager with
asserted rewrites rather than a hand edit.

#### MY CHARTER NAMED THE WRONG OBJECT

I tasked the arm to *"port the ddm_rr2 FreeCorrector"*. The module the shipping tree actually loads
as `runtime/free_corrector.py` is **`ddm_ma1`'s `Ma1WithinMissCorrector`** — the tip of a four-level
chain `rr4 → fx1 → fx2 → ma1`, **2,121 lines of stateful float64**, not one function. In mitigation,
the stale name came from the shipping code's own refusal string (`f26_inflate.py:435-441` says
"ddm_rr2 free probability corrector") and I propagated it without checking the module header — which
I had in fact read. The arm re-derived instead of confirming and caught it. **Re-derive-don't-confirm
worked, one level down from where I applied it.**

#### The port that was actually right: split, not whole

Profile first (`experiments/ddm_wc2c_token_stage_profile.py`, n600, 114,000 iterations,
`[macOS-CPU advisory]`): **63.8% integer HPAC model / 33.4% float64 corrector / ~3% other**. The arm
lowered the **integer** half — where lanes are exact and addition *is* associative — and left the
float64 half in its audited numpy form, where a reordered reduction has a measured failure mode of
**S = 27.83**. That is the §PORT DESIGN NOTE discipline applied as a partition rather than a
constraint, and it is the better answer.

#### Identity — full n600, PASS

Against the jg5 receipt (archive `f3bce5d2…`, 180,625 B), for **both** the dispatched NEON build and
the intrinsic-free scalar twin: `decoded_token_sha256` `cc10a7b0…`, `corrected_quantized_logit_sha256`
`8269fe1a…`, `corrected_cdf_input_sha256` `370a5e2a…`, `decoder_bit_position` `910837`, and the
retained 117,964,800 B token payload — **all MATCH**. A local macOS-arm64 decode reproduces a
`[contest-CUDA T4]` receipt exactly. Thread counts 1/4/8 agree at n12; 1/4 at n600.

#### Speedup — straddles the bar, and the arm refused to quote the favourable run

| row | n600 s | ratio |
|---|---:|---:|
| pure-Python shipping loop | 578.716 | 1.000x |
| split native, **shipped build**, 4 threads | 326.160 | **1.774x** |
| split native, prior build, 4 threads | 315.614 | 1.834x |

Two runs of byte-identical work differ by **3.3% — wider than the distance to the 1.804x PASS bar.**
So **WARN (≥1.096x) is secured; PASS is unresolved by the local proxy.** Integer model alone:
369.292 s → 109.372 s = **3.376x**.

**This is not a shipping-axis number.** The T4 baseline runs the sparse model on CUDA; this ran it on
CPU torch. Transferring the ratio is the cross-regime-constant genus. **One T4 row resolves it —
fire-order owed to whoever holds the fire path.**

#### Honest negatives and hardening

- Hand NEON kernels are **within 0.4%** of the auto-vectorised scalar build at n600 — retained for
  the runtime x86 dispatch, **not because they measured faster**.
- Dispatch verified: arm64/**NEON** + **scalar**. x86 **AVX2 UNVERIFIED**, T4 **sm_75 UNVERIFIED** —
  compiled and reviewed, never executed.
- Compile-at-decode added **0.203 s** (3 repeats, M5 Max), and is **plausibly net-negative** on
  prelude time because the arm removed an OpenMP dependency that shelled out to `brew --prefix
  libomp` **at decode time**. Worth flagging beyond speed: **`brew` does not exist on a Linux CI
  runner**, so that shell-out is a candidate second reason the native path was stranded — refused by
  flag *and* Linux-broken. HYPOTHESIS, not verified by me.
- Two defects the arm found in its own C during review and fixed: uninvalidated cached geometry
  (out-of-bounds write on a second model) and a worker-pool counter a short job could pollute. Both
  unreachable in this flow — which is why they would have survived and detonated later.
- **Not done, correctly not claimed:** per-frame resume does not exist (`rc64_backend.c` exports no
  state save/restore); resume granularity remains the whole token stage. The staged tree changes the
  runtime-tree sha and decoder fingerprint, so it **owes a fresh full-field identity run** before it
  is submittable.

**GO IS LIVE. Sequencing per the coordinator: (1) corrector port + sha-identity proof against the
jg5 body — critical path; (2) budget predicate; (3) C1-folded Surface-A single landing; (4) harness
lineage/tuple hardening; then recursive-fractal + arch-gating depth.** Fire condition for P1–P5: MAIN's GO carrying the
jg5 `token_decode` + `inflate_elapsed` numbers. Those numbers also settle §2.1 — if jg5's token
decode is at or above br1's 1,186.93 s, the RR2-corrector attribution for the 3.42x regression
gains a second point and the item-1 port becomes the critical path for the submission, not just the
fastest win.
