# owed-16 BOUNDED basis-reality A/B + pinned v752 dry-start — 2026-07-10

**Task:** v7.5.2 PRE-DECISION BUNDLE (operator GO 2026-07-09 "Anything we should run prior to
deciding on 385? … you have GO to run and queue"). Two de-risk items for the #385 which-to-run
decision. **Axis:** `[macOS-CPU advisory · governed-launch attempted · NON-PROMOTABLE]`.
**Pointer contest-CPU 0.19110 UNMOVED — this is MEANS.** git `31ea1df1a`.

**STORES CONSULTED:** `docs/operating_manual_craft_handoff.md` (§4 re-derive-from-primary · §5
label MEASURED/DERIVED/INFERRED · §8 no-borrowed-number) · `.omx/research/negaudit_retests_c1_e5_20260709.md`
(C1 = the owed-16 subject; "decisive re-test = self-orient-ON-vs-OFF trained-through-R n600 A/B,
heavy, operator-GO") · `.omx/research/philosophy_pass_v752_20260709.md` (P9 **BLOCKING**: owed-16 is
the named fix) · the mod32cap run dir (read-only) · `tools/{launch_witness_run,safe_run,witness_memory_preflight}.py`
· `tac.witness_autoconfig.derive_crucible_v752_config` · MEMORY L25 (basis-match PRIMARY, realized
transfer unmeasured) · L70 (fused-R bit-identity) · CLAUDE.md P0 machine-crash gate (override FORBIDDEN).

---

## HEADLINE (pointer-first)

**The exact pointer did NOT move (0.19110). Both heavy items are governed-launch BLOCKED** — the
`safe_run` SUM-over-RAM crash guard REFUSED all self-orient n600 launches **3×** because the machine
is under **sustained concurrent operator-agent load** (a sibling `palette_probe_driver.py` agent
cycling ~22–25 GiB MLX jobs). This is the P0 crash guard **working correctly**; overriding it is
FORBIDDEN. I did NOT bypass. What I DID land: (1) a **measured-from-artifact sec/ep anchor** that
replaces the borrowed 42 s/ep provenance and shows it was a **~2.8× under-estimate**; (2) a
**fully-built, preflighted, load-path-validated** owed-16 A/B, queue-ready with exact commands; (3) a
**new #385 cost-facet — self-orient costs ~47 GiB RAM** (the reason it cannot be admitted under
coexistence). The owed-16 verdict cell (realized directional Δd_seg) remains **OWED-BLOCKED**.

---

## GOVERNOR EVENTS (the REFUSEs — information, not obstacles)

| # | launch attempted | current used | active-growth | new (proj) | adaptive ceiling | verdict |
|---|---|---:|---:|---:|---:|---|
| 1 | v752 dry-start (item 2) | 36.6 | 0.0 | 72.0 | 103.3 | **REFUSE** (over 5.3) |
| 2 | v752 dry-start retry | 28.6 | 25.0 | 72.0 | 93.1 | **REFUSE** (over 32.5) |
| 3 | owed-16 OFF n8 boot-val (25 GiB) | 62.9 | 25.0 | 25.0 | 64.0 (floor clamp) | **REFUSE** (over 48.9) |

The ceiling is *adaptive*: it tightens (103.3 → 93.1 → 64.0-floor) as the gate detects concurrent
active-growth. Even at the sibling's idle floor, `used ≈ 36 GiB` baseline (my claude + a 2nd claude
agent + 2 marimo viz servers + macOS) already leaves `< 72 GiB` under the 103.3 ceiling — a 72 GiB
self-orient run is **structurally short by ~5 GiB regardless of the sibling probe**. The machine is
genuinely not single-workload; self-orient n600 (≥72 GiB) cannot coexist.

---

## ITEM 2 — pinned v752 dry-start (sec/ep anchor)

**Status: governed-launch BLOCKED (REFUSE #1/#2). Substitute measured anchor LANDED.**

The launcher path is a deliberate dead-end here: `compile_crucible_v752_config`'s docstring states
**"NO launcher dispatch wiring (that is a P8-wall item …)"** — `--config crucible_v752` is *by design*
not in `tools/launch_witness_run.py::derive_named_config` (known configs stop at `crucible_v7`), so
wiring it would preempt the very #385 decision this bundle informs. I did **not** wire it. The
compliant path is the DSL-compiled v752 argv under `safe_run --timeout 1500` + `TAC_GOVERNED_ADMISSION=1`
+ memory-preflight. Memory-preflight passed (**SAFE 71.5 GiB**); the runtime admission gate REFUSED
(concurrency).

**Measured-from-artifact sec/ep anchor (the missing #385 wall-clock cell).** The mod32cap run
(`levelset_n600_witness_mod32cap_20260706T115554Z`, self-orient-ON n600, the A/B's parent) preserves
per-stage checkpoints whose mtimes bound the wall-clock rate:

| stage span | Δwall (s) | Δep | s/ep |
|---|---:|---:|---:|
| ep299 → ep726 (tau_softplus→Muon) | 50 716 | 427 | **118.8** |
| ep726 → ep1000 (Muon) | 33 392 | 274 | **121.9** |

**~120 s/ep** across-stage — **MEASURED** on a real self-orient-ON n600 run. Caveats (honest, §5
label): this is an **UPPER bound** on steady compute (it folds in eval-every-25 async n600 through-R
verdicts, ckpt I/O, and any coexistence idle), and v752 is **heavier** than mod32cap (adds pose-carrier
+ w_pose=1 + AA-ipe + lane-band + persistence), so **v752 sec/ep ≳ 120 s/ep**. The borrowed **42 s/ep**
provenance the #385 brief carried is a **~2.8× UNDER-estimate** vs this real-run floor. The pinned
25-min dry-start remains OWED for the tight steady number, but the wall-clock column should adopt
**≳120 s/ep** now, not 42.

---

## ITEM 1 — owed-16 self-orient ON-vs-OFF realized-through-R A/B

**Status: BUILT · preflighted · load-path validated · governed-launch BLOCKED (ON arm). Verdict cell = OWED.**

This is the P9-BLOCKING resolver (philosophy_pass §P9 + auditor-A C1): the −48% directional basis is a
**direct-partition** advisory number (170–350× off realized-through-R, transfer UNVERIFIED), and
owed-15 *exempts* it — owed-16 is the FRESH-matched-arm realized-through-R isolation that most needs it.

**Method (built, not run):**
- **Warm-start** both arms **weights-only** (fresh matched AdamW) from mod32cap **ep650 BEST**
  (`levelset_witness_ema_BEST.npz`, d_seg 0.003366) via `--resume-from … --warm-start-weights-only
  --warm-start-epoch 650`. Identical seed 0 / lr / curriculum / schedule (tau@300, Muon@726) — the
  **only** difference is the directional input channels.
- **OFF arm** loads a **sliced** checkpoint (`owed16_ab_off_ckpt/…_selforient_OFF.npz`, sha
  `b1c31667…`): `in_proj.weight` truncated (96,96)→**(96,80)** (drop the 16 directional columns =
  4·n_dir_freqs), `__cfg_in_feat=80`, `__cfg_self_orient=0`; **all 18 param tensors otherwise
  byte-identical** to the ON BEST (verified param-key parity = True). This is the clean
  channel-ablation warm-start.
- **Verdict** = the trainer's own async **through-R n600** CPU-torch verdict (`--verdict-pairs 0` =
  ALL 600, `--verdict-batch 32` per the #205 OOM chunk-fix) → `levelset_best.json` d_seg per arm; the
  Δ(OFF − ON) is the realized directional contribution. `--fused-r-kernel` NOT set (mirrors mod32cap;
  the sub-permille GPU cross-proc non-bit-identity, L70, is dominated by the claimed lever magnitude).

**Load-path validation (CPU-only, $0, no admission gate):** OFF sliced npz → OFF model builds
`in_feat=80`, `in_proj (96,80)` → shape match; `_load_resume_state` routes flat deploy keys → `live`;
arch-drift missing-key guard = ∅ (self-orient adds no param KEYS); ON npz (96,96) → match. The old C1
crash (`directional_fourier_feats` render-grid 98304 vs seg-grid 196608) was in the **retired**
`dir_iso_axis_check` path, NOT the levelset self-orient path used here — it does not recur. **The
config is validated as launchable; only the live MLX execution is blocked (RAM).**

**Memory preflight (the #385 cost-facet — MEASURED):**

| arm | self_orient | in_feat | cf_mx_cache | projected peak |
|---|---|---:|---:|---:|
| **ON** | True | 96 | **47.13 GiB** | **71.5 GiB** |
| **OFF** | False | 80 | 0.07 GiB | **24.5 GiB** |

**Self-orient's realized cost is ~47 GiB of RAM** — the per-pair directional feature cache at n600. It
nearly **TRIPLES** the footprint (24.5→71.5 GiB) and is the single reason self-orient runs cannot be
admitted under any coexistence. This is a **genuine new #385 input**: the −48% direct-partition upside
must be weighed against a ~47 GiB memory tax that (a) forces single-workload scheduling and (b) is what
blocked this very A/B. The OFF arm alone fits under admission — but a lone OFF arm is not an A/B, and a
matched pair requires the ON arm's 71.5 GiB window.

**Verdict (owed-16 — the brief's missing cell):**

> **OWED-BLOCKED.** Realized directional Δd_seg (ON vs OFF, through-R n600) **NOT MEASURED** — the ON
> arm is governor-blocked under sustained concurrency. Scope FORMULATION/bounded-warm-start (≠
> from-scratch). C1's grade STANDS unchanged: −48% is a confirmed direct-partition number whose
> realized transfer remains UNVERIFIED. The A/B is BUILT + preflighted + load-validated + QUEUE-READY;
> it fires the moment the machine is single-workload.

---

## QUEUE-READY RETRY (fire when the machine is single-workload; ≳120 s/ep ⇒ size N to budget)

```bash
# item 2 — pinned dry-start (sec/ep): re-attempt when admission admits
.venv/bin/python tools/safe_run.py --timeout 1500 --rss-mb 96000 --projected-gib 72 \
  --label v752_drystart -- bash experiments/results/v752_drystart_<TS>/launch.sh   # regenerate launch.sh via compile_crucible_v752_config

# item 1 — owed-16 A/B (SEQUENTIAL; regenerate with final N from measured sec/ep):
.venv/bin/python <scratch>/gen_owed16_ab.py <N>     # N=fine-tune epochs past ep650; ≳120 s/ep ⇒ N≈50–100 to fit ≤4h for 2 arms
.venv/bin/python tools/safe_run.py --timeout <2h> --rss-mb 96000 --projected-gib 72 -- bash .../owed16_ab_selforient_ON_<TS>/launch.sh
.venv/bin/python tools/safe_run.py --timeout <2h> --rss-mb 40000 --projected-gib 25 -- bash .../owed16_ab_selforient_OFF_<TS>/launch.sh
# verdict = levelset_best.json d_seg per arm; Δ(OFF−ON) = realized directional contribution
```
Artifacts preserved: `owed16_ab_off_ckpt/` (sliced OFF ckpt), `owed16_ab_selforient_{ON,OFF}_*/launch.sh`,
`owed16_bootval_{ON,OFF}/launch.sh`. Disposable dry-start dir (16 KB, launch.sh + REFUSE logs only) is
scratch — no bulk, disk-hygiene clean.

---

## LAUNCH-GATE LIST UPDATE (for the #385 philosophy-conformance column)

- **owed-16 (P9 resolver):** status **BUILT · QUEUE-READY · verdict OWED-BLOCKED**. The realized-through-R
  directional isolation is *implemented and validated* but *unmeasured* (governor-blocked). **P9 remains
  BLOCKING** — the −48% still lacks a realized-authority number; do not claim the self-orient trunk
  "optimal-form" on a direct-partition advisory grade.
- **wall-clock column:** replace borrowed **42 s/ep** with **measured ≳120 s/ep** (mod32cap artifact) —
  a ~2.8× correction that materially lengthens the v752 launch projection.
- **NEW cost-facet:** self-orient = **~47 GiB RAM tax** (11-item philosophy "every comparison carries its
  cost"): weigh the direct-partition upside against the coexistence-blocking footprint.

---

## Triality legs
- **DAG:** FEED-owed16 (this landing) appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **Equations:** EmpiricalAnchor `owed16_realized_transfer_blocked_selforient_47gib_20260710` appended to
  `curvelet_directional_basis_dseg_reduction_v1` (realized transfer STILL owed + measured 47 GiB tax).
- **Verdict:** `tac.verdicts.emit_verdict` → `.omx/research/owed16_verdict_20260710.json` (OWED-BLOCKED,
  FORMULATION scope, 3 MeasurementRows: sec/ep ×1, ON/OFF memory ×2).
- **Bulletin:** 3 `gate_ruled` REFUSE events posted to the session bus.

**Pointer 0.19110 UNMOVED (means/apparatus). No fakes: nothing was run that was not run; every number is
MEASURED-from-artifact or a labeled projection; the owed-16 measured cell is honestly OWED.**

---

## MEASURED VERDICT (2026-07-10, appended — supersedes OWED-BLOCKED above)

**Both arms RAN. Axis `[macOS-CPU advisory · through-R n600 · NON-PROMOTABLE]`. Pointer contest-CPU
0.19110 UNMOVED — this is MEANS.** Execution: ON admitted ~03:29Z, safe_run TIMEOUT-KILLED at ~ep687
(exit 124, 9000 s, peak RSS 81.0 GiB); its ep650/ep675 verdicts completed BEFORE the kill (clean cells).
OFF admitted ~08:00Z after 1 governed REFUSE (sibling active-growth 50 GiB window), ran 650→700 + final
verdict clean (10 224 s, peak RSS 23.95 GiB, exit 0). ON resume for the ep700 cell: **BLOCKED — the
resume projects 93.8 GiB (TAC_MEM_PROBE), so baseline ~25 + 93.8 = ~119 GiB exceeds even the idle
adaptive ceiling (~103); 6× governed REFUSE (attempts 10:44–11:09Z), never bypassed. The ep700 ON cell
is structurally un-admittable without freeing ~15 GiB of baseline.**

**Matched-cell Δ table (identical config except the 16 directional input channels; seed 0 both;
config-diff verified directional-flags-only):**

| cell | ON (self-orient) d_seg | OFF (ablated) d_seg | Δ(OFF−ON) | as % of ON |
|---|---:|---:|---:|---:|
| ep649 (pre-step, raw loaded weights) | 0.208537 | 0.208281 | −0.000256 | −0.12% |
| ep650 (zero-shot, EMA re-init) | 0.006384 | 0.006295 | −0.000089 | −1.39% |
| **ep675 (25 trained ep; ON BEST)** | **0.004259** | **0.004244** | **−0.000015** | **−0.35%** |
| ep700 (final) | BLOCKED (resume refused ×6) | 0.004181 (OFF BEST) | — | — |

**Realized directional contribution ≈ ZERO.** OFF is marginally BETTER at every matched cell (all
|Δ| ≤ 1.4% of ON — INSTANCE-level, single-seed, noise floor UNMEASURED per P2). The −48%
direct-partition advisory predicts OFF ≈ +100% of ON; observed |Δ| ≤ 1.4% — the claim-direction verdict
is robust to any plausible noise floor (>70× separation). Per-class rows near-identical at every cell
(lane ep675: ON 0.27789 vs OFF 0.27860; road 0.00612 vs 0.00597). Live RAM: ON RSS ~67 GiB vs OFF
~10 GiB — the ~47–57 GiB directional-cache tax bought no measured realized d_seg.

**Scope (FORMULATION — the honest boundary):** bounded warm-start fine-tune from a parent TRAINED WITH
self-orient ON for 650 ep. This measures the MARGINAL contribution of the directional channels during
fine-tune; the parent trunk may have internalized directional structure that persists in the OFF arm.
From-scratch contribution NOT covered (reformulation queue row 1). ON's kill+resume asymmetry
(fresh-seeded RNG on resume; OFF had no resume) would have applied ONLY to the ep700 ON cell (which is
BLOCKED anyway); ep649/650/675 are clean matched, no-resume.

**P9 disposition:** the BLOCKING objection ("MEASURED −48% n600-class" grade rests on a direct-partition
proxy) is now RESOLVED-BY-MEASUREMENT — and the measurement says the realized number is ~0 at this
formulation. C1's proxy-mirage grade CONFIRMED at the realized axis. Composition row (P12):
self-orient × warm-start-from-self-orient-parent = REDUNDANT (measured).

**Decision context:** the #385 addendum v2 + operator GO ask already went out on the ep675 cell (commit
0a550f9e2); this landing is the completeness + canonical-record chain, not the decision trigger.
Verdict artifact: `.omx/research/owed16_verdict_20260710.json` · anchor
`owed16_realized_transfer_measured_zero_20260710` on `curvelet_directional_basis_dseg_reduction_v1`.

---

## OWED-16 v2 REBALANCE (2026-07-10, appended — the "wrong-allocation" hypothesis, MEASURED + REFUTED)

**Operator: "There is a self orient fix right under our noses."** owed-16's ON arm ran `--freq-across 32
--freq-along 8` — a **4:1 allocation AGAINST** the MEASURED 3.2× along-tangent dash deficit (L25/L65).
The hypothesis: owed-16 refuted THAT allocation, not the oriented basis; the along-heavy allocation
might beat OFF. **Tested it. It does not.**

**Arm:** `owed16v2_rebalanced_ON_20260710T114759Z` — identical to owed-16 ON EXCEPT `--freq-along 8→26`
(the #335 `anisotropic_basis_two_regime_allocation_v1` `lane_carried` derived optimum
`min(across, round(8·3.2))=26`; regime = lane_carried because this config carries the lane, NO
`--lane-render-band`). **Semantics VERIFIED** (`lever_b_generator.py:150-166`): `freq_along` modulates
`u_t` = tangent-parallel (ALONG the edge, where dashes oscillate); `freq_across` modulates `u_n` = normal
(ACROSS). NOT reversed. **CLEAN single-run** (NO resume — the mid-run "reap" alert was a FALSE ALARM;
the process survived a session-shell SIGURG by reparenting to init, was never relaunched/resumed),
seed 0, `SAFE_RUN exit=0 elapsed=13202s peak_rss=74.2GiB`. `[macOS-CPU advisory · through-R n600 · NON-PROMOTABLE]`.

**Matched cells (rebal along-26 vs owed-16 ON along-8 vs OFF ablated):**

| cell | rebal (along-26) | ON (along-8) | OFF (ablated) | Δ(rebal−OFF) |
|---|---:|---:|---:|---:|
| ep649 v0 | 0.207618 | 0.208537 | 0.208281 | −0.32% |
| ep650 zero-shot | 0.006409 | 0.006384 | 0.006295 | +1.81% |
| ep675 (25 trained) | 0.004286 | 0.004259 | 0.004244 | +0.99% |
| ep700 (final) | 0.004213 | BLOCKED | 0.004181 | +0.77% |

Rebalanced is marginally **WORSE than OFF at every trained cell**, and **worst on the LANE class** it
targeted (ep675 lane 0.28146 > ON 0.27789 > OFF 0.27860). All |Δ| ≤ 1.81% = INSTANCE noise (single-seed,
floor unmeasured, P2); disconfirmation direction robust (the −48% claim predicts along-heavy ≫ OFF).

**VERDICT (NO-GO, FORMULATION scope):** the along-tangent rebalance provides NO realized d_seg benefit;
the "wrong-allocation" explanation is **REFUTED** — realized directional contribution ≈ 0 is **ROBUST to
allocation**; the −48%→~0 direct-partition→realized gap is NOT an allocation problem. Only uncovered
directional formulation remains **from-scratch**. **Gate-1 → recommend the v7.5.2 launch use SELF-ORIENT-OFF.**
Verdict artifact: `.omx/research/owed16v2_verdict_20260710.json` · anchor
`owed16v2_rebalanced_allocation_measured_no_benefit_20260710` on `curvelet_directional_basis_dseg_reduction_v1`.
**Pointer 0.19110 UNMOVED (MEANS).**
