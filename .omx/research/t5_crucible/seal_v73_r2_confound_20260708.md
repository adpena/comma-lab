# SEAL v7.3 ROUND-2 — CONFOUND lens (2026-07-08)

Hunting DEFAULT-HARMFUL × SILENT × MEASUREMENT-CORRUPTING in the v7.3 composition (the
spike-guard-freeze class). Round-1 caught the event-muon fire-epoch not persisting (crash-resume would
silently shift the schedule); this round hunts that class in the composed launch candidate.

**VERDICT: NOT_CLEAN** — 2 MAJOR · 1 MAJOR (event-mode-conditional) · 3 REVISE. **verdict_scope:
FORMULATION** (the v7.3 composition's observability + candidate-selection + attribution surfaces).
None is a BLOCKER: none corrupts the PRIMARY d_seg measurement of the launch candidate — all are
attribution / orphaned-signal / observability gaps at the composition layer. Pointer **0.19110 UNMOVED**
— every item here is MEANS; only a byte-closed n600 `upstream/evaluate.py` row < 0.19110 moves it.

## STORES CONSULTED
- CLAUDE.md non-negotiables (Confound self-protection: DEFAULT-HARMFUL×SILENT×MEASUREMENT-CORRUPTING +
  the 3-layer immune system; "Bugs must be permanently fixed AND self-protected against"; NO-FAKE
  surrogate≠authority) · docs/operating_manual_craft_handoff.md (attack your own conclusion; point-fix ≠
  class-fix; label MEASURED/DERIVED/INFERRED).
- SYNTHESIS_seal_v7_round1 · crucible_v73_compile · resume_registry_canonical · resume_registry_fold_nongate ·
  r7_finishers · d16_metal_kernels (all 20260708).
- CODE (read + cheap local checks, run dir + pid 63069 UNTOUCHED): `resume_registry.py` ·
  `event_wirings.py:106-221` (EventBackstopGate.state_arrays/update) · `mlx_safe_compile.py`
  (`resolve_enabled_regions` L983-1019, `manifest_fingerprint_ok`) · `safe_compile_device_bitidentity_20260708.py`
  (the ADMIT law) · `metal_persistence_pool.py` (`persistence_pool_metal_enabled` L148-158) ·
  `persistence_topology_loss.py:479-505` (`_smooth_density_mlx` dispatch) ·
  `witness_byte_close_and_eval.py` (weights_arm L589) · trainer L7093-7116 (safe-compile resolve),
  L7411-7423/7539/7572 (gate updates), L8177-8258 (verdict cadence `eval_every`) ·
  `witness_autoconfig.py` L1044-1069 (all_levers persistence_loss_weight=1.0), L1146-1148 (v7←all_levers) ·
  `launch_witness_run.py` L620-690 (rc=8), L1365-1390 (b2), L809-857 governor MEMORY budget ·
  `system_memory_governor.py` (grep: no wall-clock coupling).
- LOCAL CHECK RUN: registry manifest-stamp for an all-unfired event-active gate set (surface 6, below).

review_status: fresh-eyes on the composed v7.3; all six surfaces reasoned from source, not memo prose.
Two task premises were EMPIRICALLY CORRECTED (surface 4 "80-min stale"; surface 5 governor coupling).

---

## FINDING 1 (surface 1) — Polyak candidate is ORPHANED; the "picks the better candidate" consumer does not exist. **MAJOR**
- **MEASURED:** v7 arms `R7_polyak_finisher` (`witness_autoconfig.py:1795`) → exports
  `levelset_witness_polyak_mlx.npz`. `grep` over `tools/` for `levelset_witness_polyak` / `polyakM__` /
  `__pta_` → **0 hits**: NO stop-time / byte-close tool consumes the polyak npz.
  `witness_byte_close_and_eval.py` records `weights_arm ∈ {"ema","live"}` (L589) — a **2-way** selection;
  the polyak arm is silently absent.
- **The claim it breaks:** `r7_finishers_20260708.md` (and the r7 module docstring) assert *"the
  byte-close/eval stop-time checklist MEASURES d_seg/d_pose/rate and picks the better candidate"* — that
  consumer **does not exist**. The candidate is produced but never measured or ranked.
- **Corruption scenario:** v7 spends the finishing-window compute building a polyak tail-average; the
  stop-time verdict compares only {ema, live}; any "we shipped the best candidate" statement is FALSE
  (2 of 3 arms measured), and the campaign's "EMA is the right shadow" prior is updated on an incomplete
  arm-set **with no record that polyak was skipped**. This is orphaned-signal + incomplete-selection
  attribution (NOT corruption of the ema/live numbers themselves — those are honestly recorded).
- **FIX:** either (a) add a polyak arm to `witness_byte_close_and_eval.py` — measure d_seg/d_pose/rate on
  `levelset_witness_polyak_mlx.npz`, record the **3-way** selection + the margin by which the winner beat
  each loser; or (b) if polyak stays a duty-to-measure, STRIKE the "picks the better candidate" claim from
  the r7 memo and register polyak in the activation-ledger (#247) with a NAMED consumer so it is a tracked
  queue state, not a silent orphan (per the "off is a tracked queue" non-negotiable).

## FINDING 2 (surface 2) — D16 persistence-pool kernel is LIVE in v7 with NO fingerprint gate + FAIL-OPEN silent fallback + no compute-path provenance. **MAJOR**
- **MEASURED it is LIVE (not inert):** v7 builds from `all_levers=True` (`witness_autoconfig.py:1146-1148`
  via `sealed_205`), and `_all_levers_base` sets `persistence_loss_weight=1.0` (L1069). So the persistence
  loss is ON → `_smooth_density_mlx` is on the active forward path → with `TAC_MLX_CUSTOM_PERSISTENCE_POOL=1`
  in the GLOBAL `PERF_ENV_PREFIX`, the fused kernel is genuinely dispatched every step.
- **Three self-protection gaps vs its safe-compile sibling:**
  1. **FAIL-OPEN silent fallback.** `persistence_topology_loss.py:488-502`:
     `try: if persistence_pool_metal_enabled(): <kernel> ... except Exception: pass` → the plain-MLX path.
     ANY kernel failure (mlx incompat, compile error, device introspection) is swallowed → pure-MLX with
     **NO telemetry marker**. A mid-campaign compute-path flip is SILENT.
  2. **NO fingerprint/version gate.** `persistence_pool_metal_enabled()` (L148-158) checks only
     "flag set AND `mx.default_device().type == mx.gpu`" — unlike `mlx_safe_compile.resolve_enabled_regions`
     which fingerprint-**fail-closes** on chip/os/mlx change. A future mlx/brew/uv bump that changes the
     kernel's fp output STILL dispatches it (flag+GPU both still true) — no recert, no refuse.
  3. **NO per-value compute-path provenance.** The persistence telemetry carries no kernel-vs-pureMLX
     marker → cross-run comparability (run-1 v6 computed persistence WITHOUT the kernel; v7 WITH) is
     **unverifiable from telemetry**.
- **Currently benign, precisely-scoped:** bit-identity is MEASURED max|Δ|=0 on THIS host/mlx (d16 memo),
  and the wiring is forward-only + `mx.stop_gradient`'d → the flip is score-NEUTRAL *by construction today*.
- **Corruption scenario:** a mid-campaign mlx/chip change breaks bit-identity; the global flag stays on,
  no fingerprint gate → the kernel silently emits a divergent density-smoothing prior. Blast radius is
  bounded (stop-grad, forward-only, a density weight), but it IS on v7's live path and the divergence is
  undetectable from telemetry.
- **FIX:** mirror the safe-compile per-chip trust — add a fingerprint/version fail-closed check to
  `persistence_pool_metal_enabled()` (or a manifest); turn the `except Exception: pass` into a LOUD
  one-shot `confound_alarm`-class marker (kernel→pure-MLX flip is never silent); stamp a compute-path tag
  on the persistence telemetry row so cross-run comparability is auditable.

## FINDING 3 (surface 3) — safe-compile stale-fingerprint refusal is a real silent flip, but DEFANGED on the measurement axis. **REVISE (defused; MINOR residual)**
- **The silent flip is real at the trainer level:** `resolve_enabled_regions` returns `frozenset()` on
  stale fingerprint / device-mismatch and DISCARDS the reason (launcher L1378 `ok, _reason = ...`; trainer
  L7102-7103 discards). The trainer's `safe_compile` json row (L7111) prints `enabled_regions: []`
  identically whether armed-but-refused or never-armed — discriminable only by `spec != none`, NOT an alarm.
- **Why it is NOT a measurement-corrupting confound:** a compiled region is CERTIFIED **bit-identical** to
  uncompiled by the ADMIT law (`safe_compile_hosc_device_bitidentity_v1`, GPU max|Δ|=0). So a
  compiled→uncompiled flip across a resume boundary is **score-NEUTRAL** — it cannot move d_seg; it changes
  only SPEED. Even the worst case (mlx bump → recert marks hosc FAILED → trains uncompiled post-resume)
  leaves both halves bit-identical to the SAME uncompiled reference → no cross-resume numerical
  discontinuity.
- **And the launcher IS loud:** b2 (`launch_witness_run.py:1382-1386`) fails **rc=4 + ERROR** on a stale/
  absent manifest when resumed via `tools/launch_witness_run.py`.
- **Residual (MINOR):** a RAW `python experiments/train_...` resume bypassing the launcher gets no loud b2;
  the trainer prints a passive `enabled_regions:[]` with no distinct ARMED-but-REFUSED alarm.
- **FIX (minor):** emit a typed `confound_alarm` row from the trainer when `_safe_compile_spec` is non-none
  but `resolve_enabled_regions` returns ∅ (surface the discarded `_reason`).

## FINDING 4 (surface 4) — event-mode fire telemetry records the FIRE epoch, not the SENSOR-DATA epoch. **MAJOR (event-mode-conditional); task premise CORRECTED.** 
- **PREMISE CORRECTION (MEASURED):** the sensors (`_muon_meat` / lane / chroma) read the FAST in-loop GPU
  verdict `history`, which updates every `eval_every=25` epochs (`witness_autoconfig.py:1008`; trainer
  L8177 `if ep % args.eval_every == 0`). They do **NOT** read the ~80-min CPU anchor — the task's "80-min
  stale" is a misconception. Real staleness is **≤ 25 epochs + async-verdict lag** (`_schedule_async_verdict`).
- **The genuine confound:** `EventBackstopGate.update` fires with the CURRENT epoch (trainer L7420/7539/7572
  `.update(ep, event_fired=...)`) and the `start_event_fired` telemetry (`event_wirings.py:164-170`) records
  only `"epoch": ep`. There is **NO** `sensor_data_epoch` / `last_verdict_epoch` field. A transition
  attributed to epoch `ep` was actually DECIDED on trajectory data up to `≤ ep − (ep mod 25)` — the fire
  epoch and the decision-granularity epoch differ by up to 25 (+ async lag). Per-epoch attribution ("the
  transition fired because d_seg plateaued at epoch ep") is off by up to a full verdict cadence.
- **SCOPE:** run-1 is recommended CLOCK mode (round-1 synthesis) → event gates OFF →
  `if _muon_gate.event_mode:` is skipped → NO sensor read → NO staleness. This bites run-2 / event-override
  ONLY — which is exactly why round-1 recommended clock-first (event couples schedules to a never-run
  sensor and confounds attribution). Continuity with round-1's catch: round-1 = fire-epoch not PERSISTED
  (resume); this = sensor-data-epoch not RECORDED (attribution) — same fire-epoch-fidelity class.
- **FIX:** stamp the last-landed-verdict epoch (and the async-pending flag) into the `start_event_fired` /
  `cap_fired_before_event` telemetry so event-mode attribution names the epoch whose STATE caused the fire,
  not merely the epoch at which the gate latched.

## FINDING 5 (surface 5) — budget 3.62 does NOT over-reserve the governor; premise unsupported. **REVISE**
- **MEASURED orthogonality:** the wall-clock `budget_days` (8.673) is used ONLY in the launcher's rc=8
  REFUSE projection (`_run_throughput_gate` → `project_launch_wall_clock` vs the declared budget,
  L664-689). The GOVERNOR's "budget" is a MEMORY budget (`training_budget = TOTAL_RAM − baseline − margin`,
  `system_memory_governor.py:809/853`) — grep confirms **no wall-clock coupling** in the governor. D1
  stop-window probes are gated by MEMORY admission, not wall-clock days. A conservative 8.673d makes rc=8
  **MORE permissive** (more headroom), it does NOT reserve governor capacity and does NOT "block D1 for
  longer." The task's resource-attribution premise is not supported by the code.
- **The genuine (low) risk is the OPPOSITE and fail-safe:** 3.62 measured under fleet contention +
  startup over-estimates min/ep → rc=8 could FALSELY refuse a run that would finish in time on an
  uncontended machine — a LOUD false-refuse with an operator `--accept-wall-clock` escape (L667-679), not a
  silent measurement corruption. No action required beyond noting the anchor's contention-inflation is a
  fail-safe (refuse-side) bias.

## FINDING 6 (surface 6) — event-active run stamps the manifest from checkpoint 1; no silent-legacy-restore window. **REVISE (defused; VERIFIED)**
- **VERIFIED (local check, all three gates event-active, NONE fired — the crash-before-any-gate-fires
  case):** `build_gate_resume_registry([...]).state_arrays()` emits
  `['__cbg_fired_by','__cbg_fired_epoch','__lbg_*','__mg_*','__resume_registry_manifest']` →
  **MANIFEST PRESENT = True** with all three listed `event: True`.
- **Why:** an event-mode-ON but UNFIRED `EventBackstopGate.state_arrays` (`event_wirings.py:197-204`)
  emits the `fired_epoch = -1` sentinel keys (non-empty) → the registry counts it as an event-active
  "wrote" → `any(w["event"])` True (`resume_registry.py:191`) → the manifest is stamped. So there is **NO
  window** where an event-mode run writes no manifest and gets silently legacy-restored (gates re-armed
  from scratch — the exact bug the registry exists to prevent). The "only event-active writers" stamp rule
  is consistent: the unfired sentinel IS a write.
- **Residual (already NAMED in the canonical memo, not new):** the static gate covers the gate class + the
  4 folded non-gate producers; a NEW controller of a NEW shape outside the registry could still be
  forgotten at the WRITE surface. Unchanged from the fold memo's stated residual risk.

---

## SUMMARY TABLE
| # | surface | severity | axis |
|---|---|---|---|
| 1 | polyak vs EMA selection | **MAJOR** | orphaned-signal / incomplete-selection attribution |
| 2 | D16 pool global env | **MAJOR** | silent fail-open + no fingerprint gate + no compute-path provenance |
| 3 | safe-compile refusal across resume | REVISE (MINOR residual) | defused by bit-identity; raw-resume has no alarm |
| 4 | event sensor staleness | **MAJOR** (event-mode only) | fire-epoch ≠ sensor-data-epoch attribution; premise "80-min"→"≤25-ep" corrected |
| 5 | budget 3.62 governor reserve | REVISE | premise unsupported; wall-clock ⟂ governor memory; fail-safe refuse-bias only |
| 6 | registry manifest stamp | REVISE (VERIFIED defused) | unfired event sentinel stamps manifest at ckpt-1 |

Pointer **0.19110 UNMOVED**. This is a confound audit of APPARATUS — a MEANS. The END is the byte-closed
n600 exact row < 0.19110; nothing in this report moves the pointer.
