---
schema: ddm_lg2_binary_inventory.v1
date_utc: 2026-08-02
arm: ddm_lg2 (row C / task #871 — EXTENSION of ddm_bs2's 84-row sweep, not a re-run)
research_only: true
score_claim: false
promotion_eligible: false
axis: "[macOS-CPU advisory — source reads + occupancy reads against persisted receipts. NO training, NO scorer job, NO pointer mutation]"
consumes:
  - .omx/research/ddm_bs2_lane_guard_schedule_and_binary_occupancy_sweep_20260801.md (§5, the 84-row sweep this extends)
  - .omx/research/ddm_pw1_pose_menu_saturation_20260801.md (the discriminator)
  - /Volumes/VertigoDataTier/pact/ddm_v4c_20260730/photo_celldrop50_resolve.partial.jsonl
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/pw1/{final_pw1.jsonl,pw1_arms.jsonl}
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pw1_archive.zip
consumers: [MAIN, ddm_bs2, the #871 owner, ddm_pw1 successor]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_lg2 row C — the binary/discrete inventory, its occupancy-feasibility column, and one measured NEGATIVE

Companion to `.omx/research/ddm_lg2_arity_mismatch_three_rows_20260802.md` §4.
**Pointer UNMOVED.** Everything here is `[macOS-CPU advisory]`, `score_claim=false`.

## §0 RELATION TO ddm_bs2 — this EXTENDS, it does not re-run

`ddm_bs2` §5 inventoried **84** discrete choice points over **10** live-chain files and measured 5.
This sweep opened **13** files (11 read in substantial part, 2 constants-grepped), found candidates in
**12**, and returned **64** rows — a different scoping (bs2 counted mode-strings and accept/reject
rules; this one includes the TR1 trainer's argparse surface, which bs2's live-chain-only scope
excluded). The two are complementary. bs2's five measured rows are **not** re-derived here.

## §1 CHAIN-STATE CORRECTIONS (mechanism, verified at source)

1. **pw1's fix has landed in the live file.** `experiments/ddm_v4d_resolve.py:66` now reads
   `BETA_MAGS = (0.0, 0.5, 1.0)  # the SEED menu; ddm_pw1 extends it per pair`, with
   `DIM0_MAX_DOUBLINGS = 23` / `BETA_MAX_DOUBLINGS = 17` (`:78-79`). The shipped manifest carries a
   **13-entry** `rs_beta_mags`. Any future charter naming those constants as live bounds is stale.
2. **`experiments/ddm_pfs1_warp_receiver.py` does not exist** under that or any name in the repo. The
   live receiver module is vendored on the external volume
   (`…/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1/pfs1_warp_receiver.py`, 101 lines) and staged
   into the eval submission by `experiments/stage_v4d_realized_gate.sh:41-44`.
3. **`src/tac/optimization/terminal_pose_gn.py` is NOT live** — `solve_terminal_pose_gn` callers are
   only `tools/pb1_*`, `tools/rehearse_*`, and its own tests. Reproduces pw1 §1 independently.

## §2 THE ONE NEW OCCUPANCY READ I TOOK — and it is a NEGATIVE

The sweep's highest-ranked *readable-today* candidate was **`RS_GLOBAL_G = (0.5, 1.0)`**
(`experiments/ddm_v4c_resolve.py:93`, used at `:800`) — the UPSTREAM rung-A shear menu that seeds the
same quantity pw1 fixed downstream. Its per-pair occupancy is already persisted as `rungA_by_g`.

MEASURED, n=600, all rows carrying the identical keyset `('0.0','0.5','1.0')`:

| admissible \|g\| | pairs | % | % of rung-AB d mass |
|---|---:|---:|---:|
| 0.0 (rung-A inactive) | 415 | 69.17 | 63.17 |
| 0.5 (interior) | 84 | 14.00 | 8.79 |
| **1.0 (TOP entry)** | **101** | **16.83** | **28.05** |

The pw1 signature is present **at the seed**: monotone decay 415 → 84, then a **jump at the terminal
bin** (84 → 101 in count, **3.19× in mass**), with the top entry carrying 1.67× the population's
mean per-pair mass.

**But it does NOT survive to the shipped solution, and reporting it as clipping would be the error.**
Joining those 101 pairs to `final_pw1.jsonl`:

- **22 / 101 (21.8%)** escaped past \|β\| = 1.0 in the shipped solve;
- **79 / 101 (78.2%)** stayed at or inside 1.0 — but pw1's self-terminating bracket *did* probe them
  outward (`BETA_STEP0 = 0.5` from the seed) and the outward probe failed to improve.

Shipped \|β\| histogram: `{0.0: 420, 0.5: 81, 1.0: 52, 1.5: 23, 2.0: 1, 2.5: 8, 3.5: 6, 4.5: 4, 7.5: 5}`
— a decaying tail, no terminal jump.

**VERDICT: `SATURATED SEED, NOT A BINDING BOUND`.** `verdict_scope: INSTANCE` (this base, this solve).
pw1's bracket is exactly the cure for this menu and it already applies. **Reported because a sweep
that reports only positives commits the selection defect it audits** (bs2's rule, honored).

Incidental, measured in the same pass: `yaw_sign` occupancy is **333 / 267** — genuinely two-sided, so
the weld in §3 row A6/C2 is not a degenerate binary; it is a real DOF that was routed around.

## §3 RANKED RESIDUE — what is left, by whether an occupancy read is possible

**Rank 1 — instrumented in the LIBRARY, DISCARDED at the live call site (≈5 LOC).**
`experiments/ddm_v4d_resolve.py:372` binds `a_q, b_q, d_ab, ab_trace = _refit_ab(…)` and `ab_trace`
appears nowhere else; the `rec` dict written at `:381-406` has no `ab_*` key. The signal is produced
and thrown away. This is the orphan-grade class, at the cheapest possible price.

**Rank 2 — instrumented in CODE, ABSENT from the shipped artifact (a re-run, no new code).**
`ddm_v4c_resolve.py:818-822` writes `ab_stop` / `ab_relins` / `ab_damp_used` / `ab_start`, but a census
over all 600 rows of the shipped `photo_celldrop50_resolve.partial.jsonl` returns
**`ab_stop: {'ABSENT': 600}`**. Re-running `--mode photo` populates it. This is the cheapest un-taken
occupancy read in the chain.

**Rank 3 — the welding-bools, where the two settings are NOT a 2-point sample of one continuum:**

| row | site | what the flag WELDS |
|---|---|---|
| B10 | `ddm_v4c_resolve.py:98,:936` `AB_START_POLICIES = ("neutral","derived")`, default `neutral` | `neutral` = ONE start; `derived` = up to FOUR heuristic starts (`derive_ab_starts:213-215`). The flag welds *how many restarts* to *which restarts* — two axes on one switch |
| A6 / C2 | `ddm_v4d_resolve.py:277` and `inflate_runner_v4d.py:180` `beta = beta_mag * (1.0 if pose[5] >= 0.0 else -1.0)` | pw1 **routed around** this weld (by letting the table carry negatives), it did not remove it. A pair whose optimum is (\|β\|, +) with yaw < 0 must still be spelled as a negative table entry. Sign is still derived from a hard 0-threshold on `pose[5]` |
| G7 | `train_tr1…:1377` `--token-ste {round,dither}` | two gradient ESTIMATORS, not a knob. Still unswept (bs2 concurs) |
| G6 | `train_tr1…:1374` `--token-temporal-mode {shared_base,independent}` | advection-tied vs fully independent per-pair tokens |
| G9 | `train_tr1…:1415` `--token-init-mode {zero,solve_project}` | analytic-projection init vs zero control |
| G8 | `train_tr1…:1390` `--adam-bias-correction {off,on}` | self-described "reset-race ARM SELECTOR" |

**Rank 4 — 2-point quantizations of a genuine continuum, receiver-change but ZERO payload bytes:**

- **A7** rolling shutter: `ddm_v4d_resolve.py:283-287` samples a *continuous per-row* rot_scale with
  exactly **two** warps (`1 − β/2`, `1 + β/2`) blended linearly by `alpha_row`. Mirrored at
  `ddm_v4c_resolve.py:802-806` and `inflate_runner_v4d.py:181-184`. N is generic code — 0 counted bytes
  — but `Decoder.f0` must change.
- **B15** the far/ground split `v_row = geometric_horizon_row(K)` is a **hard 1-row step**
  (`ddm_v4c_resolve.py:277-279`, mirrored `inflate_runner_v4d.py:150-152`); a soft band is 0 bytes.

**Rank 5 — asymmetries and defaults sitting ON a ceiling:**

- **C5** the manifest already carries an `st_grid` key, but `inflate_runner_v4d.py:146` reads the
  imported module constant instead. So widening `ST_GRID` needs a **receiver change**, unlike
  `rs_beta_mags` which is manifest-driven (`:127`). Same family, two different schema postures.
- **E1** `ST_GRID` occupancy re-measured independently: `[0,0,0,0,0,0,22,364,156,58,0]` — strictly
  interior, **6 of 11 entries never selected** (an over-provisioned alphabet: a rate observation, not
  a distortion one). Reproduces pw1 and bs2 at source.
- **G4 vs F3** `--token-quant-levels default=16` sits **exactly on** `_R7_SMEVR_MAX_LEVELS = 16`
  (`src/tac/optimization/ddm_tr1_runtime.py:83`). bs2 classified this CLIPPING-SUSPECTED and could not
  resolve it at $0; the mechanism is now named — *the default IS the ceiling*.
- **D4** the s_t index stream is **copied verbatim from the base**
  (`ddm_v4d_build_composed_archive.py:85,:152`); neither v4c nor v4d ever re-picks `s_t`. Re-solving it
  requires writing a new `st_coded` section. (pw1 §7 named the same rung from the other side.)
- **A14 / B12** `build_oracle(base, s_r=1.0)` pins `s_r` while the D1 receiver shipped `s_r = 0`
  (dims 3–5 INERT). A pinned continuum parameter, not a menu.
- **D2** `if len(table) > 256: raise` — the beta table's uint8 ceiling. Live tables: pw1 **13**,
  `mq1_partial` **44**. Not near the bound.

**Rank 6 — no record exists; new instrumentation required before any read:** the pose-GN bounds
(`RELINS = 4`, `for _damp in range(4)`, `for scale in (1.0, 0.5)` at three sites with *different*
damping constants `0.33/1e-3` vs `0.3/1e-4` and `4.0×` vs `8.0×`), the FD steps (`GAIN_FD/BIAS_FD`,
`FD_STEPS`), and the dim0 coarse/fine grids' index-level occupancy.

## §4 SCOPE SEARCHED / NOT SEARCHED

**Opened and read:** `experiments/{ddm_v4c_resolve,ddm_v4d_resolve,ddm_v4d_build_composed_archive,
inflate_runner_v4d,ddm_pfs1_ep_warp_pose_solve,train_tr1_partition_renderer_mlx,ddm_r7_token_coder,
ddm_v4d_verify_decode}.py`, `experiments/stage_v4d_realized_gate.sh`, the vendored
`pfs1_warp_receiver.py`, plus constants-block greps over
`src/tac/optimization/{ddm_tr1_runtime,terminal_pose_gn}.py`.

**No additional menu, bound, welding-bool or sign/threshold was found beyond the 64 rows — IN THAT
FILE LIST.** No claim is made about files outside it. Explicitly **NOT** searched, named so the next
arm can re-aim: the rest of `src/tac/optimization/*.py`; `experiments/ddm_su2_qa43_tail_solver.py`
(live/dead status unverified); `experiments/{ddm_ps1_pose_stage,ddm_tt1_twin,ddm_ck1_build_composed_archive,
inflate_runner_v4c,ddm_composed_s_verdict}.py` (all reference the vendored receiver, all unexamined);
`experiments/repair_entropy_coder_runtime_adapters.py` (staged into the decode path at
`stage_v4d_realized_gate.sh:42`, **not opened** — a decode-path file left unread is the sharpest gap
here); `experiments/ddm_kl1_pose_field_receiver.py`; the `mq1_*` source that produced the 44-entry
`rs_beta_mags`; ~1900 lines of the TR1 trainer outside its argparse block; ~1100 of 1296 lines of
`terminal_pose_gn.py`.

## §5 verdict_scope

| claim | scope |
|---|---|
| RS_GLOBAL_G occupancy 415/84/101, 28.05% mass at the top entry | **MEASURED**, n=600, this base/solve |
| 22/101 escaped, 79/101 probed-and-settled ⇒ saturated SEED, not a binding BOUND | **MEASURED**; `verdict_scope: INSTANCE` |
| `ab_stop: {'ABSENT': 600}`; `ab_trace` bound-and-dropped at `:372` | **VERIFIED AT SOURCE + MEASURED** |
| ST_GRID interior, 6/11 entries unused | **MEASURED** (independent reproduction of pw1/bs2) |
| the 64-row inventory is complete | **SCOPED to the 13 files named in §4** — never a repo-wide claim |

No prior negative re-opened. No score claim. Pointer `0.1910828242 [contest-CPU]` UNMOVED.
`[no-triality] [p0-ledger-ok]`
