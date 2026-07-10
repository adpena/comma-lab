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
