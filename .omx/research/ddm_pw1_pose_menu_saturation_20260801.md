---
schema: ddm_pw1_pose_menu_saturation.v1
date_utc: 2026-08-01
arm: ddm_pw1
axis: "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE"
pointer: "own-vehicle v4d 0.9639878 [macOS-CPU advisory] UNMOVED; bar 0.172141"
score_claim: false
promotable: false
pointer_moved: false
research_only: true
council_predicted_mission_contribution: frontier_breaking
verdict_scope: INSTANCE
consumes: [ddm_p3v2_optimal_form_pose_resolve_20260729, ddm_pfs1_posefield_and_recompose_20260729,
  ddm_bc1_qa24_compose_and_fire_20260731, ddm_gd1_generic_default_census_20260731,
  ddm_cn3_week_coherence_audit_20260731, ddm_cr1_composition_row_827_20260801, QA78_v4d_gate]
consumers: [ddm_deferral_queue_ledger_QA01, "#827_composition_row", "#861", "#864", ddm_gd5_grade5_detector]
---

# ddm_pw1 — the live v4d pose solve saturated TWO of its own bounds

**STORES CONSULTED:** `tools/corpus_query.py` over research/equations/memory/DAG/council/tasks/docs
(the pose/frame_0/warp-base/terminal-GN topic); `.omx/state/main_hot_state.md`;
`.omx/state/canonical_frontier_pointer.json`; the primary code (`experiments/ddm_v4c_resolve.py`,
`ddm_v4d_resolve.py`, `ddm_v4d_build_composed_archive.py`, `inflate_runner_v4d.py`,
`tools/pb1_terminal_pose_gn_600.py`, `src/tac/optimization/terminal_pose_gn.py`); the shipped
receipts on `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/`. NOT loaded: the burn/b4s seg line
(out of scope), the su2/QA43 tail-solver internals (named as an open rung, not measured here).

## §0 HEADLINE — the task I was handed was aimed at a retired instrument

My charter named three stale choices in "the LIVE production pose tool"
`tools/pb1_terminal_pose_gn_600.py`. **Re-derived at source: that tool is not live.** It is the
2026-07-28/29 pb1/eg1 instrument. The live chain is
`ddm_v4c_resolve.py` → `ddm_v4d_resolve.py` → `ddm_v4d_build_composed_archive.py` →
`inflate_runner_v4d.py`, and it already carries better forms of the first two items. The third —
my own #850 relinearization fix in `terminal_pose_gn.py` — is wired to a function **the live chain
never calls**.

So I re-aimed at the live chain and found the same *class* of defect there, unowned:
**two hardcoded search bounds that the shipped n600 solution runs into**, both rate-free to remove.

MEASURED, n600, monotone-safe continuation of the shipped solve
(`[macOS-CPU frozen-PoseNet advisory]`, canary exact):

| arm | what was removed | d_pose mean | pose contribution | ΔS vs shipped |
|---|---|---:|---:|---:|
| shipped v4d | — | 0.00858133 | 0.292939 | — |
| A | dim0 ±0.048 coarse bound | 0.00808000 | 0.284253 | **−0.008686** |
| B | 3-entry beta menu, sign forced from yaw | 0.00807876 | 0.284232 | **−0.008708** |
| **AB** | **both** | **0.00764506** | **0.276497** | **−0.016442** |

Pointer honesty, first: **the exact contest pointer is UNMOVED.** The own-vehicle advisory line is
v4d `0.9639878` and the bar is `0.172141`, so the gap is `0.7918468`; this move is **2.07%** of that
gap. It is a real measured move on the live vehicle at essentially zero rate, and it compounds into
every future pose re-solve — including the #827 post-burn re-solve, which is the named HEAD blocker
and which would otherwise have paid these two bounds again.

## §1 What I re-derived vs what I could not (the seed was a pointer, not evidence)

| seed claim | verdict | source |
|---|---|---|
| `pb1_terminal_pose_gn_600.py` is the live production pose tool | **FALSE** | the live archive's `manifest.pose_carrier = two_plane_static_photo_beta_v4d`; `pb1` is unreferenced by any v4d script |
| frame_0 START = `zeros`, d_pose ~88 | **superseded on the live vehicle** | `inflate_runner_v4d.py:69` `FRAME0_POLICY="warp_two_plane_static_photo_beta_v4d"`; frame_0 is a two-plane ground-homography warp + photometric (a,b) + rolling-shutter blend (`:158-189`) |
| basis = `eg1_generic_low_frequency_six_v1`, rank-deficient | **superseded on the live vehicle** | the live basis is warp-pose6 (Rodrigues SO(3) + `s_t` 11-grid), `pfs1_warp_receiver.py:44-49`; the cosine selector appears ONLY in `tools/pb1_*` and `tools/rehearse_terminal_pose_gn.py` |
| "stored composed render scores d_pose 10.22, better than the converged cosine solve" | **not transferable** — could not be re-derived on the live vehicle | p3v2's numbers were solved on the ct1 FRAME_ROOT frames; `ddm_pfs1_…:§1` measured those differ from the shipped render at `max_abs 255` and named it the staleness confound. The live render's d_pose is 0.00858, 3 orders below 10.22. |
| relinearization cap (my #850 fix) | landed, but **wired to a dead call path** | `solve_terminal_pose_gn` callers are only `pb1_*`, `rehearse_*` and its own tests |
| gd1 P6: FD Jacobian "RACED-SUPERSEDED" by the tt1 analytic-STE twin | **the cited race shows PARITY, not supersession** | `ddm_bc1_…:§69-70` — FD-LM-GN plateau ~30, analytic-LM-GN plateau ~29, both under the same (self-reported `s_t=1.0`) conditions, and bc1's own verdict is "ALL FOUR plateau … This is FUNDAMENTAL, not solver quality". The census row promoted bc1's *motivating hypothesis* and dropped bc1's table. |

**Could NOT re-derive:** p3v2's `stored render 10.22` and `warp base 0.59` on the live vehicle (they
are ct1-frame numbers and the frames no longer exist in the live chain); any current definition of
"Path A" (see §6); whether the su2/QA43 tail solver would beat this (unwired, not measured here).

## §2 The finding — two bounds, both saturated, both rate-free (MEASURED)

**dim0.** `_refine_dim0` (`ddm_v4d_resolve.py:177-184`) searched `d0 ± 0.048` coarse then `± 0.006`
fine. Total-move histogram over the shipped n600 solution decays monotonically and then jumps at
the bound — the textbook clipping signature, not a smooth tail:

```
|move| bin   .000  .003  .009  .015  .021  .027  .033  .039  .045 | >=.0475
count         103    93    67    39    51    34    46    28    15 |   124
```

124/600 pairs at the bound; they carry **37.4%** of the population d_pose mass and have **2.3×** the
mean d_pose of the interior pairs (0.01552 vs 0.00677).

**beta.** `_beta_select` picked a magnitude from `BETA_MAGS=(0.0,0.5,1.0)` with the **sign forced
from yaw**. The live manifest ships `beta_idx_counts [459, 65, 76]` — **76 pairs at the top entry**,
carrying **26.4%** of the mass, with nothing above it in the table.

**The in-run control — a third menu that does NOT saturate.** `s_t` is an 11-point grid
(`ddm_pfs1_ep_warp_pose_solve.py:61`) and is the obvious third suspect. MEASURED over the same
shipped n600 solution, it is **strictly interior**: every pair lands in indices 6–9, with **zero**
at the top entry (0.24) and zero at indices 0–5.

```
s_t index    0     1     2     3     4     5     6     7     8     9    10
count        0     0     0     0     0     0    22   364   156    58     0
%pose mass  0.0   0.0   0.0   0.0   0.0   0.0  16.1  70.2  12.5   1.2   0.0
```

So the finding is not "menus are generically too small" — two specific menus bind and a third does
not. (Incidental, not pursued: 6 of the 11 `s_t` entries are never selected, which is an
over-provisioned symbol alphabet, a rate observation rather than a distortion one.)

Both saturating bounds are rate-free to remove, and the receiver already supports the removal:
`inflate_runner_v4d.py:127` reads `self.beta_mags = manifest.get("rs_beta_mags", DEFAULT_BETA_MAGS)`
and `:177-180` applies `beta_mags[idx] * yaw_sign`, so **negative and >1.0 entries need no receiver
change**; `beta_idx` is a `uint8` column and dim0 is an f16 residual off a manifest offset, so
neither widening costs per-pair bytes. The capability was built; only the solver's constants pinned
it.

## §3 The measurement — arms, canary, noise floor, attribution

`tools/pw1_pose_menu_saturation_ab.py`, n600, one change per arm, every arm a **monotone-safe
continuation** of the shipped per-pair solution (accept only a strict improvement at the same
realized scorer), so no arm can report a win it did not realize.

- **CANARY (positive control):** CTRL re-scores the shipped `(pose,a,b,beta)`; it reproduced the
  shipped `d_final` **exactly (0.0)** on 577/600 pairs.
- **NOISE FLOOR (P2):** 23 pairs disagreed by ≤ `1.08e-05`. Cause diagnosed, not waved off — my
  scorer routes `g=0` through the beta-blend path, which computes `(1-α)x + αx`; that is not
  bit-identical to `x` in floating point and can flip a pixel across the uint8 rounding boundary.
  Every arm-vs-shipped delta carries that floor; the arm-vs-CTRL deltas are measured entirely inside
  the instrument and are floor-free. Both are reported. The wins are 3–4 orders above the floor.
- **ATTRIBUTION (arm A):** the shipped search could reach at most `0.048 + 0.006 = 0.054` from its
  v4c start. ****97.89%** of arm A's gain lies OUTSIDE that reach** — unreachable by the shipped
  search at any budget, so it is attributable to the removed bound and not to "more search".
- **DECOMPOSITION (arm B), from my own round-1 review:** my first framing of arm B was wrong. The
  bracket explores the *signed* beta line, so it conflated two mechanisms. Decomposed:

  | mechanism | wins | d_pose gain |
  |---|---:|---:|
  | magnitude extension only (`g > 1.0`) | 31 | 0.05561 |
  | sign freedom only (`-1.0 ≤ g < 0`) | 31 | 0.02631 |
  | **both** (`g < -1.0`) | **29** | **0.21963** |

  The dominant win needs **both**. So "sign from yaw" is not a free modelling choice — it is a
  binding constraint — and a longer table alone would not have bought it.

## §4 The wiring (and the guard that proves it is safe)

1. `_refine_dim0` and `_beta_select` (`ddm_v4d_resolve.py`) keep the shipped sweep verbatim as a
   prefix, then continue with a self-terminating Swann bracket: probe both directions at the
   search's own native step, commit to the improving one, double until a probe fails to improve.
   **Not a bigger constant.** Termination is a proof: an accepted step strictly decreases a quantity
   bounded below by 0, and the doubling step leaves the float16 range after at most
   `ceil(log2(65504/step0))` doublings (23 for dim0, 17 for beta), after which the candidate cannot
   move on the lattice. Cost on a pair whose bound did not bind is exactly **2 evaluations**
   (measured, `test_bracket_zero_cost_claim_is_two_evaluations`).
2. `derive_beta_table` (`ddm_v4d_build_composed_archive.py`) builds `rs_beta_mags` from the
   magnitudes the solve actually chose instead of pinning it to the seed menu, and remaps
   `beta_idx` into it.
3. **REGRESSION GUARD, measured not asserted:** rebuilding from the *existing pre-pw1*
   `final_refine.jsonl` produces a **BYTE-IDENTICAL** archive — `360,238 B`, sha
   `f1f3288062468e97c090ffe88ac81a6d6f76925743bd83aecb15307c0314a220`, the shipped v4d bytes. When
   the chosen set is the seed menu the derived table and the index remap are both the identity.
   Verified twice (before and after the helper extraction).
4. 9 tests in `src/tac/tests/test_ddm_pw1_pose_bracket.py` guard behaviour, not constants: the
   bracket never returns a worse point under an adversarial evaluator, reaches a minimum that
   requires expansion, terminates under unbounded improvement, stays on the quantization lattice,
   costs 2 evaluations when the bound does not bind; and the derived beta table reduces to the seed
   menu with identity indices on legacy rows, absorbs negative/extended magnitudes, and fails closed
   past 256 entries.

## §5 Byte-closed row

The arm-AB solution was emitted as a v4d final JSONL and pushed through the **real builder and the
real receiver** — not reported as a d_pose delta with an assumed rate.

| | shipped v4d | pw1 arm-AB |
|---|---:|---:|
| archive bytes | 360,238 | **360,323** (+85) |
| archive sha256 | `f1f3288062…` | `0ef9ff7129…` |
| d_seg (frame_1 untouched) | 0.00431179 | 0.00431179 |
| d_pose | 0.00858133 | **0.00764541** |
| pose contribution | 0.292939 | **0.276503** |
| rate term | 0.2398677 | 0.2399243 (+0.0000566) |
| **composed S** | **0.9639858** | **0.9476066** |

**Net composed ΔS = −0.0163792**, i.e. the +85 bytes cost 0.0000566 S and bought 0.016436 S of pose.
The 85 bytes are the 13-entry `rs_beta_mags` table in the manifest plus the beta section growing
105 → 151 B as its symbol alphabet widened; per-pair byte counts are unchanged.

**Parse-back proof (#417), all four checks pass on the extended archive:** every `pose_warp` byte is
consumed (no counted-but-inert bytes); decoded pose / (a,b) / selector / **beta magnitudes** are
bit-exact against the JSONL's encode chain; an independent compose recompute is byte-exact on 7
sampled pairs including a beta≠0 pair; and the rebuild sha is stable. The same verifier still passes
on the ORIGINAL shipped archive, so the change is backward-compatible at the decode surface too.

**This is NOT a gate row and I did not fire one.** `experiments/stage_v4d_realized_gate.sh:3` says
"DO NOT self-fire — MAIN fires ONE candidate at a time when the n600 scorer slot is idle", and I
respected it. The archive is staged and ready; the gate is ~11 min (`real 10m37s` on the v4d fire
log) and the command is:

```
bash experiments/stage_v4d_realized_gate.sh cpu pw1
```

The predicted composed S above carries a MEASURED fidelity anchor on this exact vehicle: the v4d
gate's prediction residual was **1.82e-6** (QA78: predicted 0.963986 vs `evaluate.py`-measured
0.9639878179), and the d_pose prediction residual was 1.2e-7. So the prediction is strong evidence,
and the gate is confirmation rather than discovery — but it is still a prediction and is labelled
one.

## §6 Answer to the ddm_wr2 relay + a correction to the gd5 positive control

**(1) The KD-warm-start retirement does NOT conflict with anything in the live chain.** Verified by
grep over all six files of the live TR1→v4c→v4d token+pose chain
(`train_tr1_partition_renderer_mlx.py`, `ddm_v4c_resolve.py`, `ddm_v4d_resolve.py`,
`ddm_v4d_build_composed_archive.py`, `inflate_runner_v4d.py`, `ddm_tr1_runtime.py`): **zero** hits
for `kd_warm|HNeRV|torch_vehicle|build_frozen_teacher`. wr2's mechanism claim re-derives —
`src/tac/torch_vehicle/driver.py:1703-1705` passes `vendored_decoder_cls=self.v.HNeRVDecoder` — and
that file is a different vehicle the live chain never imports. The live TR1 trainer's only init
modes are `zero` and `solve_project` (`:1396`). **Scope caveat, stated because negative-existence
claims are the day's dominant error class:** I did not find a current definition of "Path A" in
`.omx/research` (the only hits are 2026-05 artifacts on unrelated subjects), so I can confirm only
for the live chain, not for whatever ddm_sb2 meant by Path A. If Path A is the burn/b4s seg line,
that trainer is `train_tr1_partition_renderer_mlx.py` and it is in the verified set.

**(2) CORRECTION — the gd5 grade-5 detector's positive control is stale on BOTH sides.** gd1 P5
records it as "live cosine `eg1_generic_low_frequency_six_v1` @ d_pose 15.29 / 7,295 B vs unwired
RACED-better warp-pose6 @ 0.393 / 194 B". Measured:

- the cosine basis is **not live** — it appears only in `tools/pb1_terminal_pose_gn_600.py:47`,
  `tools/rehearse_terminal_pose_gn.py:50`, `tools/pb1_p5_byte_close_and_eval.py:264`;
- warp-pose6 is **not unwired** — it *is* the live carrier and has been since 2026-07-30
  (`inflate_runner_v4d.py:57-64`);
- the live pose row is **d_pose 0.00858145 at 8,621 B** (`tp 6,378 + st 189 + sel 79 + ab 1,838 +
  beta 105`), i.e. 46× better than the "unwired better" 0.393 and at 8,621 B rather than 194 B.

A detector trained on that pair would be trained on a pattern that no longer exists. The honest
replacement positive control is the pair this memo measured: a live mechanism with a *measured
recipient* and a measured successor.

## §7 Verdict scopes + still-open reformulations

- The result is **INSTANCE**-scoped: it prices two specific bounds on one shipped solution, on ONE
  base (`celldrop50`). Whether the same bounds bind on a different base is untested — `ddm_cr1`
  measured that a seg-only-trained base degrades this carrier 6.36×, so base-independence must not
  be assumed. It is not a claim that the pose axis is solved — the residual remains extremely tail-concentrated
  (**10 pairs carry 62.1%** of the population mass, top-44 carry 84.6%, spread 40,549×), which is
  the CONTENT-limited signature cn3 §452 already typed.
- **Arithmetic that bounds the whole family:** bringing the top-44 pairs to the population median
  would be ΔS −0.1753, and the top-88 −0.1945. So the tail is worth ~10× what these two bounds were.
  This move does not touch that; it removes two obstacles in front of it.
- **Named next rungs, not built here:** (a) dims 1–5 are inherited verbatim from v4c
  (`ddm_v4d_resolve.py:311`) and are never re-solved at the refined dim0/(a,b) operating point — a
  stale linearization point on a coupled 6-equation system; (b) `s_t` is read from the v4c row and
  never re-searched, and `ST_GRID` is itself an 11-point menu with the same saturation question
  unasked; (c) the su2/QA43 tail solver (`experiments/ddm_su2_qa43_tail_solver.py`) exists, targets
  exactly this tail, and is unwired — but it needs a QA43 manifest section the live archive does not
  carry, and per the C7 constraint `ddm_tr1_runtime.py:90-95` `SECTION_CONTRACT` is a closed 4-tuple,
  so that path must be priced as a schema bump, not assumed free.
- **gd1 P6 (tt1 analytic Jacobian) should be re-labelled from RACED-SUPERSEDED to RACED-PARITY-ON-
  d_pose** on the evidence it cites: bc1 measured FD ~30 vs analytic ~29 on the same parameterization
  under the same conditions. **In fairness to the row, that is not the whole story** — bc1 also
  records the analytic arm going "200→48 in 1 relin", i.e. a much faster initial contraction before
  it stalls at the same plateau. So the honest label is *equal endpoint, plausibly better
  convergence rate*, and since wall-clock is a co-equal objective the adoption is not worthless —
  it is simply not the d_pose lever the census row implies. Its adoption is additionally near-moot
  on the live chain today: v4d's refine uses a 1-D grid for dim0 and a 2-parameter GN for (a,b), so
  there is almost no Jacobian surface for it to act on. Reopen trigger: a live 6-DOF GN re-solve
  (next rung (a) above), where the Jacobian becomes load-bearing again — and where the wall-clock
  claim could finally be measured rather than inferred.

## §8 Wire-in (Catalog #125) + boundaries

- sensitivity-map N/A · Pareto: the §0 arms are new advisory (d_pose, bytes) points ·
  bit-allocator N/A · cathedral N/A · continual-learning: this memo + the receipt +
  the ledger note · probe-disambiguator: the saturation histogram IS the disambiguator (clipping
  pile-up vs smooth tail), and the reach-attribution test separates "removed bound" from
  "more search".
- `[no-triality]`: a solver-bound removal + measurement on the v4d instrument chain; no DSL lever
  and no canonical-equation surface changed. The measured law here (a menu that saturates is a
  priced constant) is recorded in this memo and the receipt, not as a new equation.
- **d_seg is untouched by construction**: frame_1 is never modified and SegNet reads `x[:, -1]`
  only (`upstream/modules.py:108`) — the same factorization p3v2 §3 measured directly.
- Receipts (SSD custody): `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/pw1/`
  (`pw1_receipt.json`, `pw1_arms.jsonl` with every probe), `pw1_n600.log`, and the byte-identity
  rebuild archives `v4d_composed_pw1_rebuildcheck{,2}_archive.zip`.
- Axis on every number above: `[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`,
  `promotable=false`. No contest-hardware row was produced and the exact pointer did not move.
