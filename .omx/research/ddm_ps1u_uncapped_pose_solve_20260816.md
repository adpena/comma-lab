# ddm_ps1u — the uncapped pose solve: the CAP was not the constraint, the TARGET was

- arm: `ddm_ps1u` (ns1 P3 / task #1074; #850 cap-lift the named prerequisite)
- date: 2026-08-16
- axis: **`[macOS-CPU advisory frozen CPU-torch PoseNet; exact CP135 receiver render]`**,
  `score_claim=false`, `promotable=false`. Frontier pointer UNMOVED. No `upstream/evaluate.py`
  row was produced by this arm and no archive was byte-closed — see §6, stated as a limit.
- code: `experiments/ddm_ps1u_uncapped_pose_solve.py`
- store: `/Volumes/APDataStore/pact/ddm_ps1u_uncapped_pose_20260816/`
  (`n64/rows_shard*.jsonl` + `n64/retained/codes/pair_*_final_codes.int32.npy` — every solved
  code vector retained; the rendered frames are deterministically rebuildable from those codes
  plus the pinned basis, and the rebuild recipe is the module itself)

STORES CONSULTED: `.omx/research/ddm_pg1_pose_gn_convergence_20260802.md` ·
`ddm_pk4_optimal_form_frame0_pose_verdict_20260813.md` ·
`ddm_qs5_verdict_and_no_toy_enforcement_20260813.md` ·
`ddm_mp2_relay_base_advisory_row_20260815.md` ·
`ddm_hv1_ep0634_t4_fire_execution_20260815.md` ·
`ddm_ns1_negative_signal_audit_and_missing_patterns_20260816.md` §P3 ·
`.omx/state/canonical_task_status.jsonl` (#850 completion row) ·
`/Volumes/VertigoDataTier/pact/ddm_pj2_20260802/pj2_report.json` ·
`/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json` ·
`/Volumes/APDataStore/pact/ddm_mt1_t4_sign_gate_20260814_custody/.../cp135_hard/` ·
`upstream/evaluate.py:92` + `upstream/modules.py:82-84`.

---

## 0. My charter's premise was STALE. Re-derived at source first.

| charter claim | status | evidence |
|---|---|---|
| "every pose GN solve in the corpus was HARD-CAPPED at 2–3 relinearizations with NO convergence test" | **TRUE for three v4d-lineage solvers, ALREADY CURED for two, and NEVER TRUE for the solver that governs this vehicle** | census in §1 |
| "still descending 13–23%/iter when stopped" | **FALSE on this vehicle — measured here at 0.07%** | §3 |
| "#850's cap-lift is the named prerequisite" | **#850 is CLOSED** (completed 2026-08-03, `tools/pj2_pose_scale_joint_solve.py`), and `ddm_pg1` had already falsified the 13–23% figure on 2026-08-02 ("that rate belongs to relins 1–3, which the shipped bound already buys") | task ledger + pg1 §2 |

The charter's own branch (4) — *"IF the descent flattens right past the cap — say so plainly
with the curve"* — is the branch the instrument selected. It is answered in §3.

**But the arm did not stop there**, because re-deriving the mechanism surfaced a genuinely
un-run measurement with a large prize (§2, §4).

---

## 1. The cap census (exact, $0, file:line)

| solver | vehicle | loop | convergence test |
|---|---|---|---|
| `experiments/ddm_su2_qa43_tail_solver.py:148` | v4d tail | `:1072 for iteration in range(config.relinearizations)` | **NONE** — `:148` structurally refuses any value outside `(2, 3)`; `:1819` argparse `choices=(2,3)`. This is the literal #850 cap. |
| `experiments/ddm_pfs1_ep_warp_pose_solve.py:183` | ep warp 6-dim | `for _ in range(relins)`, `:454` default 4 | **NONE** |
| `experiments/ddm_pu2_pose_tail_floor_probe.py:255` | pose tail | `for _ in range(relins)`, `:678` default 5 | **NONE** |
| `experiments/ddm_v4c_resolve.py:197` | photometric (a,b) | `relins=GN_RELINS_PHOTO` | **CURED by pg1**: `AB_RELINS_DERIVED=32`, `AB_DAMP_LEVELS_DERIVED=12`, realized acceptance |
| `tools/pj2_pose_scale_joint_solve.py:355/495` | v4d pose6+scale | inner relins + outer sweeps | **CURED by #850**: split exits, `sweep_relative_gain_below_tol` |
| **`experiments/ddm_qs1_frame0_schur_coupled_solve.py:490`** | **this vehicle's frame-0 carrier** | **`while True`** | **UNCAPPED ALREADY** — stops on one full non-improving singleton pass |

**The cap never applied to the actuator that governs the hv1/cp135 decode.** #850's premise
is a v4d-lineage property that was carried into a charter for a different vehicle.

---

## 2. What WAS un-run: the objective, not the iteration budget

`qs1` owns the exact frame-0 actuator at optimal form — it renders CP135's real signed-int12
frame-0 carrier (600×12), scores through the frozen CPU-torch PoseNet, and descends uncapped
on the shipped lattice. Its objective is

    || pose6(candidate_frame0, EDITED frame_1) − pose6(base_frame0, base_frame_1) ||²

— **cancellation** of the pose leak a frame-1 seg edit introduced. `qs1:718` loads the GT pose
only to REPORT; it never steers. That is a defensive use of an actuator that has never been
pointed at the score.

`ddm_ps1u` changes **exactly one thing**, the target:

    || pose6(candidate_frame0, BASE frame_1) − gt_pose6 ||²   ( = the pair's own d_pose )

Same actuator, same renderer, same frozen scorer, same lattice, same uncapped stop.

**This is not pk4.** pk4 FIT a linear low-knot overlay across pairs and failed heldout at every
rate. This is a per-pair EXACT SOLVE with **realized acceptance**: every candidate is int12,
rendered through the exact receiver, and scored; acceptance is the pair's realized d_pose. There
is no fit, so there is no generalization gap — and pk4's own verdict says its ceiling does not
bind the exact-solve sibling.

---

## 3. THE ANSWER TO #850 — the descent flattens, and the curve says so

n=29 seeded-random pairs (seed 20260816; **never a prefix** — m88/m96: pose prefixes measure
2.5–4.2× ANTI-conservatively). Sweep = one relinearization + damped-LS step + uncapped descent.

| cap | mass-weighted gain FORFEITED by stopping there |
|---|---:|
| stop after 1 sweep | 2.08% |
| stop after 2 sweeps (the su2 minimum) | 1.24% |
| **stop after 3 sweeps (the su2 default)** | **0.19%** |

28 of 29 pairs terminated on `sweep_no_improvement` and 1 on
`sweep_relative_gain_below_tol` — both convergence proofs, not bounds. Mean sweeps ~2.6,
max 5. **Zero pairs hit the runaway guard.**

**The 0.19% is not tautological.** 6 of 29 pairs actually ran past 3 sweeps; on those the
mass-weighted gain past cap-3 is **3.16%**. ⚠ That sub-population figure is **NOT converged** —
it moved 0.91% (n=13) → 0.72% (n=21) → 3.16% (n=29) as heavy pairs entered, so treat it as
`verdict_scope: instance` and do not quote it. The **population** figure (0.19%) is the robust
one and is what closes #850: uncapping buys ~0.2% of the achievable reduction. **The iteration
cap was worth ~nothing on this actuator.**

This reproduces pg1's independent finding on a different rung ("the descent is front-loaded, so
the bound was not the main cost") and pj2's on a third (18.1% / 6.4% / 5.9% / 4.5% / 3.4%, 588
of 600 pairs converging on tolerance). **Three solvers, three vehicles, one law: the pose GN
descent is front-loaded and the relinearization budget is not where the pose prize lives.**

`verdict_scope: **formulation**` — relinearization-budget uncapping on per-pair pose GN solves.
This closes #850's remaining question. It is a NEGATIVE on the cap, and it is the whole reason
the arm pivoted to the target.

---

## 4. THE PRIZE IS IN THE TARGET — 94.9% realized d_pose reduction

Same n=29, same instrument, realized through the real render + uint8 + frozen PoseNet.
**Snapshot note:** the fleet continued past this snapshot; at n=50/64 the figures are
mass-weighted reduction **92.00%**, gain past a 3-sweep cap **0.155%**, worse-than-base **0** —
both conclusions (cap ≈ worthless, retarget large) stable across n=13/21/29/50.


| quantity | value |
|---|---|
| mass-weighted d_pose | **1.428705e-04 → 7.349168e-06** |
| **mass-weighted reduction** | **94.86%** |
| per-pair reduction | mean 75.1% · median 94.0% · min 0.0% · max 99.9% |
| pairs ending worse than base | **0** (realized acceptance is structural) |
| accepted code delta | median 10 of 12 dims nonzero; \|Δ\|max median 5, p95 19, max 20 |
| cost | ~350–2,200 scorer evals/pair, ~30 s/pair on 2 threads |

The subset's base mass-weighted d_pose (1.4287e-04) sits within 3% of the n600 mean
(1.4747e-04), so the sample is representative in scale. Mass-weighted reduction (94.9%) exceeds
the per-pair mean (75.1%) because **the heavy pairs solve best** — the favourable direction for a waterfill.

The reduction is **structural, not noise-fitting**: it concentrates in pose dim 0 (the dim that
carries ~81% of the base error), it is produced by a coherent multi-dimension photometric change
of the carrier, and the instrument's own reproducibility is 2.4e-7 — **two orders below the
per-pair gains** (max live-vs-retained base drift 2.40e-07 against gains of ~1e-04).

### What it is worth, on the instrument that measured the base

If the n600 reduction matched this subset (**an extrapolation, not a measurement** — n=29):

| bytes spent | net ΔS |
|---:|---:|
| 2,400 | **−0.0281** |
| 4,800 | **−0.0265** |
| 7,200 | −0.0249 |
| 12,000 | −0.0217 |

against an admission bar of −3.5e-6. Byte **cost model**: the empirical order-0 entropy of a
delta symbol is 3.548 bits → **5.32 B/pair** over 12 dims. Per the td1 law, *an entropy estimate
is NOT a price*: the byte half of any admitted row must come from re-running the real coder and
diffing real archive bytes. Note the shipped `Q2C1` overlay **cannot carry this** — it caps at
15 pairs with deltas in [−3, 4] (`ddm_qs2_compensation_overlay_runtime.py:52-58`), while these
deltas reach ±30 across ~10 dims. A new coder is owed.

---

## 5. THE #1054 PHENOMENON, MECHANISM-NARROWED — device-dependent decode is the survivor

**This is the known #1054 result, not a new discrepancy.** MAIN's receipt (2026-08-14): the
first contest-CPU row on the MC36 frontier bytes — same F26/mc36 lineage as hv1 — measured
contest-CPU S 0.20513189 vs CUDA 0.16193, "pose 21× CPU-degraded", on real Modal CPU vs T4,
same archive bytes. **The T4 frontier row is NOT in question** (measured twice, repeat-identical);
nothing here says the frontier is mis-priced.

What this arm ADDS is the mechanism narrowing: three candidate explanations, two eliminated.

Four measurements, all on archive **`80d9c8c6…` @182,759 B** or on frames proven byte-identical
to it:

| source | d_pose | n |
|---|---:|---|
| hv1 frontier row, `[contest-CUDA] T4` (pose contribution 0.0082946) | **6.88e-06** | 600 |
| hv1 base advisory `upstream/evaluate.py` (`contest_auth_eval.json`, `avg_posenet_dist`) | **1.4747e-04** | 600 |
| this arm's CPU instrument on the retained cp135 pose vectors | **1.474653e-04** | 600 |
| `ddm_mt1` **T4 CUDA** run, `cp135_hard` arm | 1.179e-04 | 32 |

I eliminated the two obvious explanations:

1. **Not a CPU-vs-CUDA instrument gap.** On the mt1 T4 arm's 32 pairs, CPU d_pose 1.179117e-04
   vs CUDA 1.179279e-04 — ratio **0.9999**, per-pair correlation **1.0000**, and the GT pose
   vectors are **bit-identical** (max abs diff 0.0). PoseNet CPU ≡ CUDA on this workload.
2. **Not a definition difference.** `upstream/modules.py:82-84` is a plain MSE over the first 6
   pose dims with no normalization; `evaluate.py:92` takes `sqrt(10·posenet_dist)`. That is
   exactly what this arm computes.

And I closed the identity chain: **the frame_0 bytes the T4 run actually scored are
byte-identical to this arm's rendered frame_0 and to the retained cp135 raw** (0 mismatches,
3/3 pairs checked). So a real T4 job, on exactly these frames, measured ~1.18e-04.

The remaining explanation is that **the hv1 CUDA decode and the hv1 CPU decode produce different
frames** — the one hypothesis this arm did not have the hardware to test, and the one §5b stages. The fire memo supports this reading: the frontier's components were *inherited* —
"Components expected: seg 0.029611 (identical decode) · pose 0.0082946 (identical decode)" — and
the decode identity was proven **on the CPU axis only** (`ddm_hv1_ep0634_t4_fire_execution_20260815.md`,
"Local full-raw decode proven byte-identical to the incumbent's decode (sha e5539653…, CPU
axis…)"). CPU-decode identity does not establish CPU-decode ≡ CUDA-decode. The seg component
disagrees the same way (0.029611 vs 0.042714, 1.44×).

**Direct mechanism pin (new here).** The T4 run's own receipt
(`experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json`)
records `inflate_device_policy = "auto"` with `scorer_device = "cuda"` on a Tesla T4 — so the
**inflate itself ran on CUDA**, while the advisory chain inflated on CPU. The two chains are not
merely scoring differently; they are decoding on different devices.

**Consequences, stated plainly:**
1. **§4 is an ADVISORY-AXIS instance.** If the CPU decode manufactures most of the 1.4747e-04,
   then the 94.86% reduction largely cancels CPU-decode-manufactured error. **Its CUDA-axis value
   is UNMEASURED and bounded above by the full pose contribution, 0.0083 S** — not by §4's
   −0.028. Every §4 number is `verdict_scope: instance` on the advisory axis and must not be
   quoted as a CUDA prize.
2. **A compliance-grade defect, independent of scoring axis.** Our deterministic-decode
   non-negotiable requires "same `archive.zip` → bit-identical inflate output every run/host."
   A device-dependent decode violates that whether or not either score is "right".

**No candidate should be compiled from §4 until the reconciliation in §5b closes.**

---

## 5b. RECONCILIATION — **ANSWERED at $0. DEVICE-DEPENDENT DECODE CONFIRMED.**

The dispatch was never needed: the T4 run's own receipt already carried the CUDA-side raw sha.
Both sides re-derived here rather than inherited.

| axis | inflate device | raw `0.raw` sha256 | bytes |
|---|---|---|---|
| advisory chain | **cpu** | `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` | 3,662,409,600 |
| T4 r2 chain | **cuda** (`inflate_device_policy=auto`) | `9a6b75e55268a68ed7e1b59d9ee871f99b89b0960bd63efae12ca2aa3e8f2339` | 3,662,409,600 |

**One archive (`80c8…`→`80d9c8c6…`), identical geometry, two devices, two different decodes.**
CPU side: this arm's manifest, `frames_concat_sha256 == raw_sha256` (proves the frame slicing
tiles the file exactly), 1200/1200 unique frame hashes. CUDA side:
`ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json` →
`artifacts.inflated_outputs_manifest.json` → `files[0].sha256` (call
`fc-01M036FY225QC9A75CM0Y7X7NP`, Tesla T4, 421.6 s, PASSED).

**VERDICT `DEVICE_DEPENDENT_DECODE_CONFIRMED`** — receipt
`DECODE_AXIS_VERDICT.json` (`ddm_ps1u_decode_axis_aggregate_verdict.v1`), emitted by
`ddm_ps1u_decode_axis_reconciliation.py verdict-aggregate`, which refuses fail-closed if the two
sides differ in raw geometry (a shape change would confound a decode change).

**This is a VEHICLE DEFECT, independent of which score is "right".** Our deterministic-decode
non-negotiable requires *"same `archive.zip` → bit-identical inflate output every run/host."*
The shipped vehicle violates it. `verdict_scope: **formulation**` — the hv1/e480b receiver
family on the CPU-vs-CUDA axis, on this archive.

**Consequence for §4, restated sharply.** The CPU decode and the CUDA decode are *different
objects*. §4's 94.86% was solved against the CPU-decode object; its CUDA-axis value is
**unmeasured and bounded above by the 0.0083 S pose contribution** — never by §4's advisory
−0.028. The advisory instrument remains valid for what it is (it reproduces the base row to 5
s.f.); it is simply measuring a different decode than the one that ships.

**Localization: rides free, no dedicated dispatch.** Per-pair localization needs the CUDA-side
per-frame hashes. Enable the hash-request fields and run this module's `manifest` step remotely
on the NEXT T4 row we buy; `diff` then names which pairs and which frame (f0/f1) diverge, and —
with both raws local — the first divergent byte offset, pixel coordinate and both values. The
`spec` subcommand carries the exact remote step; it fires nothing.

**Cure direction (the engineering ask).** Port the decode to the portable native /
`runtime-rs` program so ONE deterministic implementation runs on every host. Engineer it to
**PRESERVE the CUDA-favorable frames — the frontier rides them** — rather than naively
CPU-pinning, which would lock in the degraded decode and cost ~0.03 S. Ranked suspects for where
the divergence enters, from the receiver code: (1) the device-adaptive block, the only
intentional CPU/CUDA branch, recorded UNCHANGED in the hv1 fire memo; (2) `torch` bicubic
`interpolate` 384×512 → 874×1164 then `round()` to uint8 — CUDA/CPU kernels are not bit-identical
and a half-ULP straddle flips a pixel; (3) clamp/round ordering and fp32 accumulation in the
HPAC/neural render; (4) native-decoder threading (the CPU lift runs 4 workers).

Instrument retained at `/Volumes/APDataStore/pact/ddm_ps1u_uncapped_pose_20260816/`:
`CPU_DECODE_MANIFEST.json` (112.9 KB, 94.6 s) · `DECODE_AXIS_VERDICT.json` ·
`CUDA_DECODE_DISPATCH_SPEC.json` (retained as the piggyback recipe) · `SELFTEST_DIFF.json`
(positive control: CPU manifest vs itself → IDENTICAL 0/600). Diff tool self-tested 4/4
including both refusal branches.

---

## 6. What I did NOT do, and why

* **No byte-closed archive, no `upstream/evaluate.py` row, no T4 fire.** §5 gates it: compiling a
  candidate whose prize is stated on a chain that disagrees 21.4× with the frontier row would be
  a row I could not interpret in either direction. The gate spent $0 saying wait.
* **No n600.** n=29 of a seeded-random 64 at the time of writing; shards continue and are
  resumable. Every magnitude in §4 is `verdict_scope: **instance**` and is **not** a finding —
  only §3's cap answer and §5's discrepancy are scope-`formulation`. A partial-n mean presented
  as a finding is exactly the failure this program has a rule against (pg1 §7).
* **No coder.** The Q2C1 overlay cannot carry these deltas (§4); designing one before §5 resolves
  would be building infrastructure for a row that may not exist.

## 7. My own round-1 adversarial review (a fix is unreviewed new code)

1. **Is "0.074% past cap-3" tautological?** Partly — pairs converging in ≤3 sweeps have it 0 by
   construction. Caught it, and reported the non-tautological subset separately (6 pairs that
   ran past 3 sweeps: 3.16%), and flagged that the sub-population figure is still moving. Without that split the number would have been a fake.
2. **Is the reduction instrument noise?** Measured rather than assumed: live-vs-retained base
   drift is 2.40e-07 against gains of ~1e-04 (two orders). Also checked the shape — the gain is
   concentrated in the dominant dim and produced by coherent multi-dim deltas, not ±1 wiggles.
3. **Does the actuator control the shipped object?** Verified byte-identity three ways rather
   than assumed: rendered frame_0 == retained cp135 raw == the frames a real T4 job scored.
4. **I nearly reported a "21× CPU-vs-CUDA pose instrument gap" as the headline.** The mt1 T4
   custody refuted it (correlation 1.0000, GT bit-identical). The difference between that
   sentence and §5's is the whole credibility of this memo.
5. **Redundant work in the solver**: `solve_pair` re-executes the pinned joint-solver module per
   pair and re-evaluates `current` after each relinearization. Both are wasteful, neither is
   wrong; noted rather than silently fixed mid-measurement.
6. **Unverified**: the per-pair wall-clock is contended (4 shards, other arms live); the
   scorer-eval count is the reliable cost figure, not seconds.

---

## 8. THE CUDA-AXIS CANDIDATE — fleet COMPLETE, receiver VERIFIED, container writer OWED

**Fleet 60/60.** Top-mass selection (48.19% of advisory pose mass), solved with the retargeted
uncapped exact int12 carrier solve, realized acceptance throughout (0 pairs worse than base).

| quantity | value |
|---|---|
| pairs edited | **60 / 60** |
| counted section (`P1D1`) | **626 B** = 10.43 B/pair, sha `307894b674b172ed…` |
| rate | 0.12169172 → **0.12210854** (**+0.000417 S**) |
| pose ceiling (CUDA) | **0.008298 S** |
| advisory reduction on edited pairs | 86.77% |
| advisory n600 d_pose | 1.4747e-04 → 8.5810e-05 (41.8% of total) |

### PRE-REGISTERED ADMISSION (recomputed at the final 626 B)

> **ADMIT iff** the T4 row's recomputed S < **0.15959729295498598** with the pose leg
> **MEASURED**. Equivalently **CUDA d_pose must fall by more than 9.79%**.
> Expected components: **seg 0.029611 unchanged** (seg-hold — the edit touches only the frame-0
> carrier, and SegNet reads the LAST frame, which this candidate does not touch; the assertion
> is OWED at compile as advisory Δd_seg == 0 or priced exactly) · **rate
> (182,759 + 626)·25/37,545,489 = 0.12210854** · **pose = the measured unknown**.
> The advisory row is DIRECTIONAL ONLY and cannot admit (§5b).

### Receiver half: BUILT and VERIFIED against live objects

`experiments/ddm_ps1u_receiver_p1d1_dispatch.py` — `RECEIVER_CONTRACT_VERIFIED`, **8/8 required
checks evaluated** (the denominator is reported, because a skipped check that still reads green
is the vacuity trap). Verified against the REAL shipped `runtime` package and the REAL archive:
the shipped Q2C1 path is unchanged (apply matches the runtime element-for-element, 7 pairs
touched); **byte-identity control holds at the section level** — `selector + overlay` rejoins to
the original tail byte-for-byte and an ABSENT overlay round-trips to the bare selector; P1D1
splits from and applies through the SAME slot; and all four refusals (truncation, bad magic,
trailing bytes, int12 overflow) fire at the receiver boundary. The pinned generation was **not
mutated** — runtime custody is MAIN-owned.

An honest trap caught in my own review: the first run resolved the selector tail to `None`, so
checks 2–3 silently skipped while `failed_checks` stayed empty. Cured by reconstructing the tail
from the packed carrier and making a non-evaluated required check FAIL LOUD.

### Container writer: BUILT and PROVEN — the fire order is SEALED

`experiments/ddm_ps1u_container_writer.py` → **`CONTAINER_WRITER_PROVEN`**.

**The structural fact that made it tractable** (measured at source): the frame-0 overlay is the
*trailing slice* of the brotli-decompressed carrier body — `PACKED_CAP1_SECTION_BYTES` = 22,183
of packed CAP1 metadata, then the overlay to the end (22,219), and the shipped bytes at exactly
that offset are `Q2C1`, 36 B. So a swap is a splice plus one recompression.

**The recompression identity is exact and was measured, not guessed:**
`brotli.compress(body, quality=11, lgwin=24)` reproduces the shipped 22,161 B carrier stream
**byte-for-byte**. Quality 9 and 10 miss by length; quality 11 at lgwin 22 matches in *length*
but not in bytes. The writer pins (11, 24) and **re-proves the round-trip on every run** rather
than trusting the constant.

| proof | result |
|---|---|
| **(a) byte-identity control, through the WRITER path** | writer with the ORIGINAL overlay reproduces **182,759 B / sha `80d9c8c6…`** exactly; `member_identical = true` |
| **(b) parse-back** | written archive → real receiver → recovered overlay **626 B byte-equal** to the retained P1D1; 60 pairs; **deltas equal the solved deltas exactly** |
| **(c) repeat-identical** | two independent writer runs → **same sha** `97048f9f…`, same 183,347 B |

**Candidate archive: 183,347 B, sha `97048f9fe1845a2b0b602dbdaf5f85e87fb19dee0e6cc57503fe5fd60096bef8`.**
The archive delta is **+588 B**, not the section's 626 B — brotli absorbs part of the addition,
which is why the rate leg is priced from **real archive bytes** and never from the section size.

**Receiver graft lives in a NEW generation.** The shipped receiver refuses `P1D1` by design
(`_decode_rx1_models` accepts only `Q2C1` at the overlay offset) — a correct fail-closed refusal,
and the reason the first parse-back attempt failed. The graft (a `decode_frame0_overlay` magic
dispatch plus magic acceptance, vendored generic code) went into a COPY at
`retained/candidate_generation`; **the pinned generation was not mutated** (verified: its archive
sha and overlay-module sha are unchanged).

### SEALED FIRE ORDER — `SEALED_T4_FIRE_ORDER.json`

| leg | value |
|---|---|
| rate | 0.12169172 → **0.12208324** (**+0.000392 S**, from +588 real archive bytes) |
| seg | **0.029611 asserted decode-identical** — the edit touches only the frame-0 carrier; SegNet reads the LAST frame, untouched. Any T4 seg drift is signal. |
| pose | **the measured unknown**, ceiling 0.008295 S |

> **ADMIT iff** the T4 row's recomputed S < **0.15959729295498598** with the pose leg
> **MEASURED**. Equivalently: **CUDA d_pose must fall by more than 9.21%** (from 6.885643e-06 to
> below **6.251199e-06**). The advisory row is DIRECTIONAL ONLY and cannot admit (§5b).

Both decode hash-request fields and the remote `manifest` step are in the order, so per-pair
decode localization rides along at $0 marginal. MAIN fires; this arm fired nothing.

## NEXT_IF_RESUMED

1. **MAIN fires the sealed T4 row** (`SEALED_T4_FIRE_ORDER.json`, ~$0.16) on candidate
   `97048f9f…` @183,347 B. Admission is pre-registered: **CUDA d_pose must fall >9.21%**.
   Enable both decode hash-request fields + the remote `manifest` step so per-pair decode
   localization rides along free.
2. **On ADMIT**: the pointer moves; then re-solve the NEXT mass tranche (pairs 61–120 carry the
   following ~15% of mass) and re-price — the marginal B/pair is flat at ~10.4 so the exchange
   rate is known. On REFUSE: the transfer question is answered with a real number and the pose
   axis routes to joint/nonlinear training (js8) with that number in hand.
3. **Cure the decode defect (§5b) independent of scoring** — portable native / `runtime-rs`
   decode, engineered to **PRESERVE the CUDA-favorable frames** (the frontier rides them), never
   naive CPU-pinning, which would lock in ~0.03 S of degradation.
4. **Promote the P1D1 receiver graft** into the canonical runtime lineage if the row admits; it
   is generic code (no video-derived data) and is already contract-verified 8/8.
5. **Do not re-open the relinearization cap** (§3). **Do not quote §4's advisory −0.028 as a
   prize** — the CUDA ceiling is 0.0083 S.
