# SEAL v7.4 ROUND-3 — CONFOUND + BUGS lens on the fix-wave DIFF (2026-07-08)

Round 3 reviews ONLY the fix-wave diff `106e77b84..HEAD` (HEAD `d7b6d4f8d`). Fixes are UNREVIEWED NEW
CODE — reviewed here fresh-eyes, not trusted from the landing memos. Pointer **0.19110 UNMOVED** — every
item is APPARATUS/MEANS; the END is a byte-closed n600 `upstream/evaluate.py` row < 0.19110 AFTER the run.

## VERDICT: NOT_CLEAN — 0 BLOCKER · 0 MAJOR · 3 MINOR. (fix-all policy: all reported.)
Every round-2 item (A1–A8, B1–B6) is INDEPENDENTLY VERIFIED GENUINELY FIXED (re-derivations below). The
3 MINOR are fix-INTRODUCED / housekeeping residuals; NONE corrupts the primary d_seg measurement, NONE
blocks launch. verdict_scope on each finding is FORMULATION or narrower.

## STORES CONSULTED
- `SYNTHESIS_seal_v73_round2_20260708.md` (the fix-wave charter) + the 2 round-2 lens reports it cites
  (`seal_v73_r2_{confound,bugs}_20260708.md`) + both landing memos (`r2_fixwave_{A,B}_20260708.md`).
- CLAUDE.md non-negotiables (NO-FAKE tests-verify-behavior-not-constants class 2 · Confound self-protection
  3-layer · value-provenance ladder · fix-all) + `docs/operating_manual_craft_handoff.md` (§4 RE-DERIVE
  from primary artifacts · §6 attack-your-own-conclusion: a fix is not safer than the bug).
- CODE read from source (not memo prose): `witness_autoconfig.py` (A1/A2/A5/A7 override block L2280-2305,
  `crucible_v7_polyak_start_provenance`, `crucible_v7_registered_off_levers`, `_CRUCIBLE_V7_HOSC_BETA_END_EVENT`)
  · `tau_advance.py:254-285` (the octave_fraction β coupling — A1 mechanism) · `curriculum_dsl.py`
  (`persistence_classes_for_basis_regime`) · trainer L4056-4061 (persist_classes PARSE), L5639-5646/7415-7600
  (B3 wiring), L6760-6763 (`_lever_epoch`), L7281 (loop range) · `persistence_topology_loss.py:485-522`
  (B2 dispatch) · `metal_persistence_pool.py` (B2 fingerprint+marker) · `typed_config.py:101-140` (B4) ·
  `event_wirings.py` + `resume_registry.py` (B3/B5) · `levelset_byte_close_and_eval.py` (B1 select) ·
  `scorer_throughput_gate.py` (A2) · `tail_stop_forfeit_floor_20260708.py` (A8).
- MEASURED/EXECUTED myself: run-1 log `levelset_n600_crucible_v6_run1_20260708T095730Z/run.log` verdict
  `ts` timestamps (A2 re-derivation); fresh `_build_crucible_v7` emit (A1/A5/A7 values); B4 probe; B5
  sentinel probe; B3 lag probe; 222 affected tests; ruff F/E9 on all 11 touched source + 8 test files.
- review_status: fresh-eyes round-3; every value re-derived from the PRIMARY artifact (log / source / emit),
  never the memo's asserted number.

---

## PART 1 — CONFOUND lens (does each fix close its confound; does any fix INTRODUCE one?)

### A1 β_end 3.177 (BLOCKER) — CLOSED, re-derived from source. ✓
Independently re-derived the event-frozen β from `tau_advance.py:280-285`: linear shape
`β = β_start + (β_end − β_start)·octave_fraction()`, `octave_fraction = rung/n_octaves ∈ [0,1]`
(`:254-257`). At the ladder floor (Muon switch, prog=1.0) **β = β_end EXACTLY** — so freezing at β_end is
CONFIRMED against source, and β_end is a strict CEILING for any earlier-firing muon (prog<1 ⇒ β<β_end).
The intended frozen value = the mod32cap control's linear β(726) on its den-1000:
`1 + (4−1)·725/999 = 1 + 3·0.7257257 = 3.17718 ≈ 3.177` — arithmetic reproduced. 3.177 ≤ 4.0 divergence
bound; [1,10] GPU bit-cert is a superset (bit-identity is β-value-invariant). **Fresh `_build_crucible_v7`
emit MEASURED `--hosc-beta-end = 3.177`** (not 10.0). No other config path still assumes β_end=10 (the
only `hosc_beta_end 10.0` occurrences at emit are inside NOTE strings, not emitted flags; the v6
`d6["hosc_beta_end"]=10.0` base is OVERRIDDEN at L2285). CLOSED.

### B1 arm-selection (F1 MAJOR) — CLOSED, no silent wrong-winner. ✓
Each arm byte-closes through the IDENTICAL `run(ckpt_dir, npz_name=_ARM_NPZ[arm], **run_kwargs)` — same
kwargs, only the weights npz differs; rate term is each arm's OWN `archive_zip_bytes` (correct basis, not
a shared one). Ranked by `implied_S_advisory` ascending; an UNSCORED arm sorts LAST via `(1, inf)` so it
can never beat a scored arm; all-unscored falls back to order-stable ema with S=None RECORDED (visible,
not silent). NO-FAKE refusals FIRE: `--skip-parity` → ValueError (reads `run_kwargs["skip_parity"]`, which
IS in `_run_kwargs` — verified), `--npz-name`+`--select-arms` → SystemExit, unknown `--arms` → SystemExit,
no-arm-present → FileNotFoundError. `npz_name` is passed SEPARATELY (not in `_run_kwargs`) so no
duplicate-kwarg TypeError. Test `test_select_records_three_way_selection_and_winner` asserts the WINNER +
ranked order + margin ARITHMETIC (0.08/0.06) on a fake-run — BEHAVIOR, not constants.

### B2 D16 pool fingerprint gate (F2 MAJOR) — CLOSED, no silent-disable on the matching host. ✓
On the MATCHING host: no cert env set (default) → `persistence_pool_certified_fingerprint()` returns None
→ `_fingerprint_ok_uncached` returns `(True, permissive)` → kernel dispatches (unchanged); a cert that
MATCHES → `fingerprint_matches` True → dispatches. There is NO path that silently disables the pool on a
matching host (the perf-regression the prompt worried about cannot happen silently). The per-step verdict
is CACHED per cert-env string (`_FP_OK_CACHE`) so `host_fingerprint()`'s sysctl/sw_vers subprocess does
NOT run every step (own-review catch, verified present). Marker fires **one-time per (path,reason)** —
`confound_alarm=True` on any non-metal path; the FIRST flip away from the kernel is always LOUD. Note
(NOT a defect): an intermittent kernel that flips metal↔pure repeatedly records each direction ONCE, so
the flip COUNT is lost — but provenance ("a flip happened") is preserved, which is the confound-alarm
contract (CLAUDE.md L1 "one-time"). Score-neutral by construction (forward-only, `stop_gradient`).

### B3 sensor-data-epoch (F4 MAJOR, event-mode) — CLOSED for muon/chroma; MINOR frame-mix for lane (see F-1).
`sensor_lag_epochs = ep − sensor_data_epoch`. PROBED: muon fire@700 sde=675 → lag 25; cap-fire@500 sde=475
→ lag 25; no-sde → lag None; `sensor_async_pending` is a SEPARATE flag (does not enter the lag arithmetic
— correct, it qualifies the lag as a lower bound). Trainer wiring: all new sde/async computation is inside
`if <gate>.event_mode:` with `_muon_sde/_muon_async_pending` defaulted None/False BEFORE the block, so a
clock/OFF run (run-1) passes the defaults and the OFF branch ignores them — byte-identical (`event_mode =
sensor is not None`; a clock gate has sensor=None). Persist/restore: fired-with-known-sde writes a 3rd
additive key; unfired/no-sde keeps the EXACT legacy 2 keys (probed) → strict key-set regression green;
pre-fix sidecar restores sde=None. **The residual (F-1, MINOR):** the lane gate fires on `_lever_epoch(ep)`
while its sde is the real-epoch `lane_ev_epoch`, so `sensor_lag_epochs` is a cross-frame subtraction when
re-anchor is ON (memo's disclosed "known nuance").

### B5 pre-fire manifest window (REVISE-2) — CLOSED, sentinel-write claim RE-PROBED TRUE. ✓
Re-ran the probe MYSELF: `build_gate_resume_registry([muon,lane_band,seg_chroma_boundary])` all event-mode
ON, NONE fired → `state_arrays()` → **`__resume_registry_manifest` PRESENT = True**, `__mg_fired_epoch =
−1` sentinel. So an event-mode run has NO manifest-free window: the unfired sentinel IS a write, stamped
from ckpt 1 → every co-writing non-event controller is vanish-protected. Sister cap-only registry stays
manifest-free (byte-identity contract). The `resume_registry.py` change is DOC-ONLY (a docstring); no
behavior change (matches the memo). B5 is the honest close: PROOF + documentation.

### Confound INTRODUCED audit (attack-your-own-conclusion): only F-1 (lane frame-mix, MINOR). No fix
introduces a DEFAULT-HARMFUL×SILENT×MEASUREMENT-CORRUPTING confound on d_seg. B2/B3 telemetry is
forward-only/pure-telemetry (never read into training). A5 is a design coupling (F-3), not a confound.

---

## PART 2 — BUGS lens (line-level correctness of the diff)

### A2 budget 3.39 / 8.122d — RE-DERIVED from run-1's log MYSELF, consistent. ✓
Parsed the verdict `ts` timestamps from `run.log` (launch 09:57:30Z) directly — reproduced the memo's
table EXACTLY: ep0=24.43, ep25=137.77, ep50=219.02, ep75=312.33, ep100=396.62 min. My r_ss(ep75→100)
= (396.62−312.33)/25 = **3.371**; S = 396.62 − 3.371·100 = **59.48**; amortized(3000) = 3.371 +
59.48/3000 = **3.391 → 3.39**; budget = 3.39·3000/1440·1.15 = **8.122d** (emitted config value =
8.122 — verified via fresh emit). The reasoned deviation from the synthesis's 3.12 is SOUND: 3.12 rested
on the memo's r_ss=3.1 LOWER BOUND, which run-1's measured 3.371 steady slope contradicts (value-provenance
ladder forbids anchoring on a bound). Observation (non-blocking): per-interval slopes vary 3.25–3.73
(fleet noise); the ep75→100 pick is defensible (latest, most startup-decayed) and the failure mode of a
slightly-optimistic anchor is a LOUD false-REFUSE with `--accept-wall-clock` escape + the at-admission
real SegNet bench as final arbiter — fail-safe, not silent.

### A3 Polyak fenceposts (MINOR-1/MINOR-2) — both fixed, tests are BEHAVIORAL. ✓
- Degenerate: `polyak_start_epoch = epochs+1`; trainer loop `range(start, epochs+1)` max-ep = epochs < epochs+1
  → observe never fires. Test `..._genuinely_inert_over_the_real_loop` ARMS `PolyakTailAverager(start=4)`,
  runs `range(1,4)`, asserts `count==0` — BEHAVIOR (closes the round-2 NO-FAKE class-2 finding; the old
  test asserted only the constant). ✓
- Non-degenerate: window = round(0.2·(3000−726)) = round(454.8) = 455; `abs_start = 3000−455+1 = 2546`;
  loop `range(2546,3001)` observes 455 epochs (inclusive fencepost). Test `..._averages_exactly_window_epochs`
  counts the real range == 455 — BEHAVIOR. `polyak_relative_start_epoch = 2546−726 = 1820`. All match emit.

### A5 persistence_classes='3' (M1, THE HIGHEST-RISK ITEM) — class indexing VERIFIED CORRECT. ✓
Traced the CONSUMER: trainer L4061 `persist_classes = tuple(int(x) for x in _pc.split(","))` → `"3"` →
`(3,)` = class INDEX 3. Canonical comma10k order (Road0/Lane1/Undriv2/Movable3/MyCar4) ⇒ index 3 =
**Movable**. So `--persistence-classes 3` targets Movable ONLY and EXCLUDES Lane (index 1) — which IS the
intended lane_offloaded semantics. It does NOT mean "first 3 classes" (that would wrongly include
Road/Lane/Undriv). Confirmed `persist_classes` flows as class INDICES into `persistence_topology_loss_mlx`
(L4507), same type as the `"auto"` path's `detect_persistence_tail_classes` output. `persistence_classes_for_basis_regime`
fail-closes on an unknown regime (ValueError, tested). The A5 mechanism is correct; the residual is a
cross-surface coupling (F-3, MINOR), NOT a class-indexing bug.

### A7 --per-group-grad-clip — present, gated on grad-clip>0. ✓  A8 tail_stop — doc-only reactivation
extension (ν stale on rebalanced basis), no behavior change. ✓

### B4 perf-env token boundary (REVISE-1) — RE-PROBED, fixed. ✓
Re-ran: `missing_perf_env_vars('NAME=10 NAME2=10 x')` now returns BOTH required (pre-fix returned `[]` —
the measured false-pass); exact `=1` satisfied; `=100` vs required `=10` caught. Uses `_parse_perf_env`
(the SAME parser the required manifest DERIVES from) → NAME→VALUE map compare, no substring collision.
Regression test locks all three directions. ✓

### B6 closed-loop guard (REVISE-3) — structural, not re-pointed. ✓
`_sync_verdict_branch_offset` uses a whitespace-tolerant regex `v\s*=\s*realized_verdict\s*\(` with an
exactly-one-match assert (fails LOUD on ambiguity) — arg-list/reformat tolerant (the round-1 stale-token
class structurally fixed). NEW `test_sync_verdict_branch_passes_epoch_ast` asserts the CONTRACT
(`realized_verdict(ep=…)` exists) via the parse tree — source-text-free BEHAVIOR test. ✓

### Absorption-repair completeness + hygiene — HEAD CONSISTENT. ✓
**222 tests GREEN at HEAD** across all 8 affected suites (crucible_v7_config, weights_arm, metal_persistence_pool,
event_wirings, resume_registry, wallclock/perfenv, v7_compute_exploitation, closed_loop) — the B4
serializer-absorption incident (test briefly landed under A's body ahead of its fix) is fully repaired;
every test's fix is present. **ruff F/E9 clean** on all 11 touched source + 8 test files.

---

## FINDINGS (fix-all: all reported; all MINOR)

| # | sev | file:loc | finding | verdict_scope |
|---|---|---|---|---|
| F-1 | MINOR | `train_levelset_witness…mlx.py:7596` + `event_wirings.py:189` | **Lane gate `sensor_lag_epochs` mixes epoch frames.** Lane gate fires on `_lever_epoch(ep)` (re-anchored frame) but its `sensor_data_epoch = lane_ev_epoch` is the REAL-epoch frame, so `lag = _lever_epoch(ep) − lane_ev_epoch` is a cross-frame subtraction when BOTH lane-event-mode AND re-anchor are ON (≈0 or nonsense, not the true sensor lag). Muon/chroma gates are unaffected (they fire on real `ep`). Telemetry-only (never read into training) → score-neutral; disclosed as a "known nuance" in the B3 memo. Fix: compute lag in ONE consistent frame (e.g. stash the lever-frame sde) OR omit the derived `sensor_lag_epochs` for the lane gate and record only the two raw epochs. | formulation (event-mode lane attribution; not run-1) |
| F-2 | MINOR | stale artifact `experiments/results/levelset_n600_witness_20260708T173144Z/launch.sh` | **A pre-fix crucible_v7 launch.sh (mtime 12:31 < fix 13:17) carries `--hosc-beta-end 10.0`** — the BLOCKER value. The FRESH emit is correct (3.177, verified), so this is a stale artifact, not a fix failure; but if a launch reuses this dir instead of re-emitting, the BLOCKER re-enters. Fix: the launch package must re-emit from config (confirm launcher regenerates launch.sh, not reuses) OR delete/quarantine stale pre-fix v7 dirs. | housekeeping / launch hygiene |
| F-3 | MINOR | `witness_autoconfig.py:2287` (A5 override) | **A5 couples training to an unenforced byte-close band.** Dropping lane from the training persistence recall (`--persistence-classes 3`) assumes the byte-close renders the analytic lane band (`--lane-render-band`), but that is a SEPARATE downstream flag not structurally linked to this training config. If the eventual byte-close omits it, lane gets NEITHER training recall NOR analytic band → lane d_seg regression. Blast radius small (run-1 Lane flip_share 0.0018) and mitigated by the registered `lane_carried` counter-arm + Road/Lane-jitter watch-list, but the coupling is not gated. Fix: assert/record the byte-close band requirement alongside the training regime (co-emit or a launch-package invariant). | formulation (lane regime coherence) |

## What I RE-DERIVED (not read from the memos)
- A1: β = β_end at octave floor, CONFIRMED against `tau_advance.py:280-285`; 3.177 = 1+3·725/999 reproduced.
- A2: full cadence table + r_ss 3.371 + S 59.48 + amortized 3.39 + budget 8.122d, from run-1 `ts` timestamps.
- A3: 2546→455-epoch inclusive count; degenerate epochs+1 → count 0 over the real loop.
- A5: `"3"` → class-index-3 = Movable (parse traced to trainer L4061), excludes Lane — intent correct.
- Fresh `_build_crucible_v7` emit: `--hosc-beta-end 3.177 · --persistence-classes 3 · --per-group-grad-clip True · budget 8.122d`.
- B4/B5/B3 probes re-run live; 222 tests green; ruff clean.

Pointer **0.19110 UNMOVED** — this round-3 audit is APPARATUS/MEANS. The END is the byte-closed n600
exact row < 0.19110 AFTER the run.
