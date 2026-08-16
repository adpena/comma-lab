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

## 8. THE CUDA-AXIS CANDIDATE — assembled and priced; byte-close is OWED

The honest CUDA-axis test of §4, built exactly as designed: spend the byte budget where the
pose mass is, price it against the **0.0083 S ceiling** (never against §4's advisory −0.028).

**Selection.** The top-mass pairs by retained base-vs-GT error — a deliberate non-random rule,
because the candidate's bytes must go where the mass is. Top-60 captures **48.19%** of the
advisory-axis pose mass; reproducible from retained vectors with no scorer call
(`top_mass_pairs`, recorded with the share it captures).

**Counted section.** `Q2C1` cannot carry this payload — it admits 1–15 pairs with deltas in
[−3, 4], while the solve produces 60 pairs with deltas in **[−29, 48]** over ~8.2 of 12 dims.
So `P1D1` (`experiments/ddm_ps1u_carrier_delta_codec.py`): delta-coded 10-bit pair gaps +
12-bit support mask + order-0 Exp-Golomb of zigzag values. Exp-Golomb rather than a fixed width
because the magnitude distribution is heavy-headed (median |Δ| ~5) with a long tail — a 7-bit
field truncates, a 12-bit field wastes ~40%. **No trained table ships**; the code is a fixed
generic construction, so nothing here is video-derived. `encode` self-verifies its own
round-trip; decode refuses truncation, disorder, zero-support, zero-in-support, nonzero padding,
trailing bytes, and any overlay that drives a coefficient outside signed-int12. Self-tested.

| quantity | value |
|---|---|
| pairs edited | **39** (fleet continuing to 60; the bundle regenerates deterministically) |
| counted section | **405 B** = 10.38 B/pair, sha `d21043d4c8ba03f1…` |
| rate cost | **+0.000270 S** |
| pose prize CEILING (CUDA) | **0.008298 S** |
| advisory-axis reduction on edited pairs | 84.78% |
| advisory-axis n600 d_pose | 1.474653e-04 → 1.087900e-04 |

### PRE-REGISTERED ADMISSION (stated before the row exists)

> **ADMIT iff** the T4 row's recomputed S < **0.15959729295498598** with the pose leg
> **MEASURED**, not projected. Equivalently: **CUDA d_pose must fall by more than
> 6.39%** to pay for the 405 B section.
> **The advisory row is DIRECTIONAL ONLY and cannot admit** — §5b proved it decodes a different
> object than the shipping chain.

**Transfer is the measurement, and I am not assuming it.** The edit is to the int12 **codes**,
which both decoders consume; the qs2/re1 precedent is that CPU-solved candidates realized on T4
to ~6 significant figures despite the decode divergence — which is why this is worth one row.
That precedent is not a guarantee. `verdict_scope: instance` on every advisory number here.

### What is OWED, named precisely (I did not complete it)

The **byte-close**. The path exists and I located it rather than hand-waving:
`ddm_qs1_frame0_schur_coupled_solve._candidate_physical_carrier` ("Encode selected CP135 int12
changes through the banked CPR1/CAP1 producers") and `_compile_one` (which already carries a
`repeat` flag for the repeat-identical build). **But it compiles into the CP135 container
(186,252 B), not hv1 (182,759 B).** The hv1 integration point is its **compensation slot**:
hv1's retained decode carries `outer_compensation.bin` = a 36 B `Q2C1` overlay whose own
docstring says it "applies those deltas to the real 600×12 signed-int12 CP135 code lattice" —
so **hv1 already carries this lattice and already has an overlay consumer**. `P1D1` is that
consumer's widened sibling. The owed build is: teach the hv1 receiver's compensation slot to
parse `P1D1`, re-encode, verify parse-back, confirm repeat-identical bytes. Well-scoped, not
done here, and **no fire-order can be sealed without it** — a sealed order needs a byte-closed
archive sha, and inventing one would be a fake.

**Fire-order status: TEMPLATE, precondition UNMET.** `CANDIDATE_FIRE_ORDER_TEMPLATE.json`
carries the admission rule, the runtime pins, and — per the §5b piggyback plan — **both decode
hash-request fields enabled plus the remote `manifest` step**, so per-pair decode localization
rides along free on whatever T4 row MAIN eventually fires. Its `preconditions_unmet` field names
the byte-close explicitly.

---

## NEXT_IF_RESUMED

1. **BYTE-CLOSE the §8 candidate — the one owed build.** Teach the hv1 receiver's compensation
   slot to parse `P1D1` (it already parses `Q2C1` over the same 600×12 int12 lattice), re-encode,
   verify parse-back, confirm repeat-identical bytes. Then seal the fire order from
   `CANDIDATE_FIRE_ORDER_TEMPLATE.json` — its `preconditions_unmet` names exactly this — and MAIN
   fires ONE T4 row (~$0.16). **Admission is pre-registered: CUDA d_pose must fall >6.39%** (at
   the 405 B section; recompute if the fleet finishes 60 pairs). Enable both decode hash-request
   fields so per-pair decode localization rides along at $0 marginal.
2. **Finish the fleet** (resumable, strided, ~30 s/pair). `--top-mass 60` for the candidate set;
   `--n-pairs 64 --seed 20260816` for the characterization set; `--all-pairs` for n600 **only if
   the T4 row shows CUDA-axis signal**:
   ```
   .venv/bin/python experiments/ddm_ps1u_uncapped_pose_solve.py \
       --out <store>/cand60 --top-mass 60 --shard <k> --shards 4
   ```
3. **Cure the decode defect (§5b), independent of scoring.** Port the decode to the portable
   native / `runtime-rs` program so ONE deterministic implementation runs on every host —
   engineered to **PRESERVE the CUDA-favorable frames** (the frontier rides them), never naively
   CPU-pinned, which would lock in the degraded decode at ~0.03 S. Suspects ranked in §5b.
4. **Do not re-open the relinearization cap** on any pose GN. §3 closes it at formulation scope;
   pg1 and pj2 agree on two other vehicles.
5. **Do not quote §4's advisory −0.028 as a prize.** The CUDA ceiling is 0.0083 S. §5b proved the
   advisory chain decodes a different object; every §4 magnitude is `verdict_scope: instance`.
